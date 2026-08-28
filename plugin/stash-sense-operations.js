/**
 * Stash Sense Operations Module
 * Operations tab UI for job queue visibility and controls
 */
(function() {
  'use strict';

  const SS = window.StashSense;
  if (!SS) {
    console.error('[Stash Sense] Core module not loaded');
    return;
  }

  // ==================== Queue API ====================

  async function apiCall(mode, params = {}) {
    const settings = await SS.getSettings();
    const result = await SS.runPluginOperation(mode, {
      sidecar_url: settings.sidecarUrl,
      ...params,
    });
    if (result.error) {
      throw new Error(result.error);
    }
    return result;
  }

  const QueueAPI = {
    async getStatus() { return apiCall('queue_status'); },
    async getJobs(status) { return apiCall('queue_list', status ? { status } : {}); },
    async getTypes() { return apiCall('queue_types'); },
    async submit(type, cursor = null) {
      const payload = { type };
      if (cursor !== null) payload.cursor = cursor;
      return apiCall('queue_submit', payload);
    },
    async cancel(jobId) { return apiCall('queue_cancel', { job_id: jobId }); },
    async stop(jobId) { return apiCall('queue_stop', { job_id: jobId }); },
    async retry(jobId) { return apiCall('queue_retry', { job_id: jobId }); },
    async clearHistory() { return apiCall('queue_clear_history'); },
  };

  // ==================== State ====================

  let pollInterval = null;
  let jobTypes = null;
  let historyExpanded = true;
  const FORCE_FULL_SCAN_USER_JOB_TYPES = new Set(['scene_fingerprint_match', 'upstream_scene_changes']);

  // Recent-rate tracking for ETA. A lifetime average (items_processed /
  // time since started_at) goes badly stale across a phase change within
  // one job -- e.g. fingerprint generation skips thousands of
  // already-complete scenes in seconds, then hits real work; the lifetime
  // average stays anchored near that fast burst for the rest of the run,
  // so the ETA can read "1s remaining" for the entire slow phase. Track a
  // short rolling window of (time, items_processed) samples per job instead.
  const progressHistory = new Map(); // job.id -> [{ t, processed }, ...]
  const PROGRESS_HISTORY_WINDOW_MS = 30000;

  function recordProgressSamples(jobs) {
    const now = Date.now();
    const activeIds = new Set();
    for (const job of jobs) {
      if (job.status !== 'running' || job.items_processed == null) continue;
      activeIds.add(job.id);
      const history = progressHistory.get(job.id) || [];
      history.push({ t: now, processed: job.items_processed });
      const cutoff = now - PROGRESS_HISTORY_WINDOW_MS;
      while (history.length > 2 && history[1].t < cutoff) {
        history.shift();
      }
      progressHistory.set(job.id, history);
    }
    // Drop history for jobs no longer running (finished, or no longer returned).
    for (const id of progressHistory.keys()) {
      if (!activeIds.has(id)) progressHistory.delete(id);
    }
  }

  function ensurePolling(container) {
    if (!pollInterval) {
      startPolling(container);
    }
  }

  // ==================== Rendering ====================

  function createOperationsContainer() {
    return SS.createElement('div', {
      id: 'ss-operations',
      className: 'ss-operations-page',
    });
  }

  async function renderOperations(container) {
    container.innerHTML = '<div class="ss-operations-loading">Loading operations...</div>';

    try {
      const [statusResult, jobsResult, typesResult] = await Promise.all([
        QueueAPI.getStatus(),
        QueueAPI.getJobs(),
        QueueAPI.getTypes(),
      ]);
      jobTypes = typesResult.types || [];
      renderContent(container, statusResult, jobsResult.jobs || []);
      startPolling(container);
    } catch (e) {
      container.innerHTML = '';
      const errorDiv = SS.createElement('div', { className: 'ss-operations-error' });
      errorDiv.appendChild(SS.createElement('h3', { textContent: 'Failed to load operations' }));
      errorDiv.appendChild(SS.createElement('p', { textContent: e.message }));
      errorDiv.appendChild(SS.createElement('button', {
        className: 'ss-btn ss-btn-primary',
        textContent: 'Retry',
        events: { click: () => renderOperations(container) },
      }));
      container.appendChild(errorDiv);
    }
  }

  function renderContent(container, status, jobs) {
    recordProgressSamples(jobs);

    container.innerHTML = '';

    // Header
    const header = SS.createElement('div', { className: 'ss-operations-header' });
    header.appendChild(SS.createElement('h1', { textContent: 'Operations' }));
    header.appendChild(SS.createElement('p', {
      className: 'ss-operations-subtitle',
      textContent: `${status.queued} queued \u00b7 ${status.running} running`,
    }));
    container.appendChild(header);

    // Quick Actions
    renderQuickActions(container);

    // Active Jobs (running)
    const running = jobs.filter(j => j.status === 'running' || j.status === 'stopping');
    if (running.length > 0) {
      renderSection(container, 'Active Jobs', running, true);
    }

    // Queue (pending)
    const queued = jobs.filter(j => j.status === 'queued');
    if (queued.length > 0) {
      renderSection(container, 'Queue', queued, false);
    }

    // History (completed/failed/cancelled - collapsible)
    const history = jobs.filter(j => ['completed', 'failed', 'cancelled'].includes(j.status));
    if (history.length > 0) {
      renderHistory(container, history);
    }

    if (running.length === 0 && queued.length === 0 && history.length === 0) {
      container.appendChild(SS.createElement('div', {
        className: 'ss-operations-empty',
        textContent: 'No jobs in the queue. Use Quick Actions to run an operation.',
      }));
    }
  }

  function renderQuickActions(container) {
    if (!jobTypes || jobTypes.length === 0) return;

    const section = SS.createElement('div', { className: 'ss-operations-section' });
    section.appendChild(SS.createElement('h2', { textContent: 'Quick Actions' }));

    const grid = SS.createElement('div', { className: 'ss-quick-actions' });

    for (const type of jobTypes) {
      const btn = SS.createElement('button', {
        className: 'ss-btn ss-btn-secondary ss-quick-action-btn',
        attrs: { 'data-type': type.type_id },
        events: {
          click: async (e) => {
            const button = e.currentTarget;
            button.disabled = true;
            button.textContent = 'Submitting...';
            try {
              // "Fingerprint Missing" and "Refresh Outdated" are separate
              // job types (fingerprint_generation / fingerprint_refresh_outdated)
              // with scope fixed by type, not a cursor -- see
              // fingerprint_job.py's docstring -- so this generic "run it"
              // button needs no special-casing for either of them, only
              // for the pre-existing force-full-scan pair below.
              const cursor = FORCE_FULL_SCAN_USER_JOB_TYPES.has(type.type_id) ? '__full__' : null;
              await QueueAPI.submit(type.type_id, cursor);
              await refreshContent(container);
              ensurePolling(container);
            } catch (err) {
              if (err.message.includes('409') || err.message.includes('already')) {
                button.textContent = 'Already Running';
                setTimeout(() => { button.textContent = type.display_name; button.disabled = false; }, 2000);
              } else {
                button.textContent = 'Error';
                setTimeout(() => { button.textContent = type.display_name; button.disabled = false; }, 2000);
              }
              return;
            }
            button.textContent = type.display_name;
            button.disabled = false;
          },
        },
      });

      // Resource badge -- effective_resource reflects what a new run would
      // actually use right now (GPU types can run CPU-only, depending on
      // the gpu_enabled setting and real GPU availability); falls back to
      // the static classification for non-GPU types.
      const resource = type.effective_resource || type.resource;
      const badge = SS.createElement('span', {
        className: `ss-resource-badge ss-resource-${resource}`,
        textContent: resource.toUpperCase(),
      });
      btn.appendChild(document.createTextNode(type.display_name + ' '));
      btn.appendChild(badge);
      grid.appendChild(btn);
    }

    section.appendChild(grid);
    container.appendChild(section);
  }

  function renderSection(container, title, jobs, isActive) {
    const section = SS.createElement('div', { className: 'ss-operations-section' });
    section.appendChild(SS.createElement('h2', { textContent: title }));

    const list = SS.createElement('div', { className: 'ss-job-list' });
    for (const job of jobs) {
      list.appendChild(renderJobCard(job, isActive));
    }
    section.appendChild(list);
    container.appendChild(section);
  }

  function renderJobCard(job, isActive, isHistory = false) {
    const card = SS.createElement('div', {
      className: `ss-job-card ss-job-${job.status}`,
    });

    // Job header row
    const headerRow = SS.createElement('div', { className: 'ss-job-header' });

    const typeInfo = jobTypes ? jobTypes.find(t => t.type_id === job.type) : null;
    const displayName = typeInfo ? typeInfo.display_name : job.type;

    headerRow.appendChild(SS.createElement('span', {
      className: 'ss-job-name',
      textContent: displayName,
    }));

    // Resource badge -- job.resource_used is the actual device that
    // specific run used (recorded when it started; frozen from then on,
    // so history stays accurate even if the setting changes later).
    // Falls back to effective_resource for a not-yet-started queued job,
    // then the static classification for non-GPU types.
    if (typeInfo) {
      const resource = job.resource_used || typeInfo.effective_resource || typeInfo.resource;
      headerRow.appendChild(SS.createElement('span', {
        className: `ss-resource-badge ss-resource-${resource}`,
        textContent: resource.toUpperCase(),
      }));
    }

    // Status badge
    headerRow.appendChild(SS.createElement('span', {
      className: `ss-status-badge ss-status-${job.status}`,
      textContent: job.status,
    }));

    card.appendChild(headerRow);

    // Progress bar for active jobs
    if (isActive && job.items_total && job.items_total > 0) {
      const pct = Math.round((job.items_processed / job.items_total) * 100);
      const progressWrap = SS.createElement('div', { className: 'ss-progress-wrap' });
      const progressBar = SS.createElement('div', { className: 'ss-progress-bar ss-ops-progress-bar' });
      const progressFill = SS.createElement('div', {
        className: 'ss-progress-fill',
        styles: { width: `${pct}%` },
      });
      progressBar.appendChild(progressFill);
      progressWrap.appendChild(progressBar);
      const eta = getETA(job);
      const etaText = eta ? ` \u00b7 ${eta} remaining` : '';
      const labelText = job.progress_label ? ` \u00b7 ${job.progress_label}` : '';
      progressWrap.appendChild(SS.createElement('span', {
        className: 'ss-ops-progress-text',
        textContent: `${job.items_processed} / ${job.items_total} (${pct}%)${etaText}${labelText}`,
      }));
      card.appendChild(progressWrap);
    } else if (isActive && job.status === 'running') {
      // Indeterminate state: running but no items_total yet
      const progressWrap = SS.createElement('div', { className: 'ss-progress-wrap' });
      const progressBar = SS.createElement('div', { className: 'ss-progress-bar ss-ops-progress-bar ss-progress-indeterminate' });
      progressWrap.appendChild(progressBar);
      const indeterminateLabelText = job.progress_label ? ` \u00b7 ${job.progress_label}` : '';
      progressWrap.appendChild(SS.createElement('span', {
        className: 'ss-ops-progress-text',
        textContent: `Analyzing${indeterminateLabelText}\u2026`,
      }));
      card.appendChild(progressWrap);
    }

    // Meta row
    const metaRow = SS.createElement('div', { className: 'ss-job-meta' });
    const triggeredByDisplay = job.triggered_by
      ? job.triggered_by.charAt(0).toUpperCase() + job.triggered_by.slice(1)
      : 'Unknown';
    if (isHistory) {
      const triggeredAt = formatTimestamp(job.created_at || job.started_at);
      const completedDuration = getCompletionDuration(job);
      const triggeredText = triggeredAt ? `Triggered by ${triggeredByDisplay} on ${triggeredAt}` : `Triggered by ${triggeredByDisplay}`;
      const durationText = completedDuration ? `Finished in ${completedDuration}` : 'Finished';
      metaRow.appendChild(SS.createElement('span', {
        textContent: triggeredText,
      }));
      metaRow.appendChild(SS.createElement('span', {
        textContent: durationText,
      }));
    } else {
      metaRow.appendChild(SS.createElement('span', {
        textContent: `Triggered by ${triggeredByDisplay}`,
      }));
      if (job.started_at) {
        const elapsed = getElapsed(job.started_at);
        metaRow.appendChild(SS.createElement('span', { textContent: elapsed }));
      }
    }
    card.appendChild(metaRow);

    // Result summary (completed jobs) or error message (failed jobs)
    if (job.result_summary && job.status === 'completed') {
      card.appendChild(SS.createElement('div', {
        className: 'ss-job-result',
        textContent: job.result_summary,
      }));
    }
    if (job.error_message) {
      card.appendChild(SS.createElement('div', {
        className: 'ss-job-error',
        textContent: job.error_message,
      }));
    }

    // Action buttons
    const actions = SS.createElement('div', { className: 'ss-job-actions' });

    function actionHandler(button, action, label, pendingLabel) {
      return async () => {
        button.disabled = true;
        button.textContent = pendingLabel;
        try {
          await action();
          const container = button.closest('#ss-operations');
          if (container) await refreshContent(container);
        } catch (err) {
          button.textContent = 'Error';
          setTimeout(() => { button.textContent = label; button.disabled = false; }, 2000);
        }
      };
    }

    if (job.status === 'running') {
      const btn = SS.createElement('button', {
        className: 'ss-btn ss-btn-danger ss-btn-sm',
        textContent: 'Stop',
      });
      btn.addEventListener('click', actionHandler(btn, () => QueueAPI.stop(job.id), 'Stop', 'Stopping\u2026'));
      actions.appendChild(btn);
    } else if (job.status === 'stopping') {
      const btn = SS.createElement('button', {
        className: 'ss-btn ss-btn-danger ss-btn-sm',
        textContent: 'Force Cancel',
      });
      btn.addEventListener('click', actionHandler(btn, () => QueueAPI.cancel(job.id), 'Force Cancel', 'Cancelling\u2026'));
      actions.appendChild(btn);
    } else if (job.status === 'queued') {
      const btn = SS.createElement('button', {
        className: 'ss-btn ss-btn-secondary ss-btn-sm',
        textContent: 'Cancel',
      });
      btn.addEventListener('click', actionHandler(btn, () => QueueAPI.cancel(job.id), 'Cancel', 'Cancelling\u2026'));
      actions.appendChild(btn);
    } else if (job.status === 'failed' || job.status === 'cancelled') {
      const btn = SS.createElement('button', {
        className: 'ss-btn ss-btn-primary ss-btn-sm',
        textContent: 'Retry',
      });
      btn.addEventListener('click', actionHandler(btn, () => QueueAPI.retry(job.id), 'Retry', 'Retrying\u2026'));
      actions.appendChild(btn);
    }
    if (actions.children.length > 0) {
      card.appendChild(actions);
    }

    return card;
  }

  function renderHistory(container, jobs) {
    const section = SS.createElement('div', { className: 'ss-operations-section' });

    const headerRow = SS.createElement('div', { className: 'ss-setting-row-header' });

    const toggle = SS.createElement('h2', {
      className: 'ss-collapsible-header',
      events: {
        click: () => {
          historyExpanded = !historyExpanded;
          list.style.display = historyExpanded ? '' : 'none';
          toggle.classList.toggle('ss-collapsed');
        },
      },
    });
    toggle.textContent = `History (${jobs.length})`;
    if (!historyExpanded) {
      toggle.classList.add('ss-collapsed');
    }
    headerRow.appendChild(toggle);

    const clearBtn = SS.createElement('button', {
      className: 'ss-btn ss-btn-secondary ss-btn-sm',
      textContent: 'Clear History',
      events: {
        click: async (e) => {
          e.stopPropagation();
          const button = e.currentTarget;
          button.disabled = true;
          button.textContent = 'Clearing...';
          try {
            await QueueAPI.clearHistory();
            await refreshContent(container);
          } catch (err) {
            button.textContent = 'Error';
            setTimeout(() => { button.textContent = 'Clear History'; button.disabled = false; }, 2000);
          }
        },
      },
    });
    headerRow.appendChild(clearBtn);

    section.appendChild(headerRow);

    const list = SS.createElement('div', {
      className: 'ss-job-list',
    });
    if (!historyExpanded) {
      list.style.display = 'none';
    }
    for (const job of jobs.slice(0, 20)) {
      list.appendChild(renderJobCard(job, false, true));
    }
    section.appendChild(list);
    container.appendChild(section);
  }

  function parseUtcDate(value) {
    if (!value) return null;
    const normalized = value.includes('T') ? value : value.replace(' ', 'T');
    const withTimezone = /Z$|[+-]\d\d:\d\d$/.test(normalized) ? normalized : `${normalized}Z`;
    const date = new Date(withTimezone);
    return Number.isNaN(date.getTime()) ? null : date;
  }

  function formatTimestamp(value) {
    const date = parseUtcDate(value);
    if (!date) return '';
    return date.toLocaleString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  }

  function formatExactDuration(secs) {
    if (!Number.isFinite(secs) || secs < 0) return '';
    const total = Math.floor(secs);
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    if (h > 0) return `${h}h ${m}m ${s}s`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
  }

  function getCompletionDuration(job) {
    const completed = parseUtcDate(job.completed_at);
    if (!completed) return '';
    const start = parseUtcDate(job.created_at) || parseUtcDate(job.started_at);
    if (!start) return '';
    const secs = (completed.getTime() - start.getTime()) / 1000;
    return formatExactDuration(secs);
  }

  function getElapsed(startedAt) {
    if (!startedAt) return '';
    const start = parseUtcDate(startedAt);
    if (!start) return '';
    const now = new Date();
    const secs = Math.floor((now - start) / 1000);
    if (secs < 60) return `${secs}s`;
    if (secs < 3600) return `${Math.floor(secs / 60)}m ${secs % 60}s`;
    return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
  }

  function formatDuration(secs) {
    if (secs < 60) return `${Math.ceil(secs)}s`;
    if (secs < 3600) return `~${Math.ceil(secs / 60)}m`;
    return `~${Math.floor(secs / 3600)}h ${Math.ceil((secs % 3600) / 60)}m`;
  }

  function getETA(job) {
    if (!job.items_total || !job.items_processed || job.items_processed <= 0) return null;
    const remaining = job.items_total - job.items_processed;
    if (remaining <= 0) return null;

    // Prefer a recent-window rate (see recordProgressSamples) so the
    // estimate tracks the job's *current* pace, not its pace since it
    // started -- important for jobs like fingerprint generation that skip
    // already-complete items fast before hitting real work.
    const history = progressHistory.get(job.id);
    if (history && history.length >= 2) {
      const oldest = history[0];
      const latest = history[history.length - 1];
      const dt = (latest.t - oldest.t) / 1000;
      const dItems = latest.processed - oldest.processed;
      if (dt >= 3 && dItems > 0) {
        return formatDuration(remaining / (dItems / dt));
      }
      // Full window with zero progress (a single slow item, or genuinely
      // stalled) -- don't fall back to a stale lifetime average that could
      // under-report; just omit the ETA rather than mislead.
      if (dt >= PROGRESS_HISTORY_WINDOW_MS / 1000 - 1) return null;
    }

    // Not enough recent samples yet (job just started) -- lifetime average
    // is a reasonable bootstrap estimate until the rolling window fills in.
    if (!job.started_at) return null;
    const start = parseUtcDate(job.started_at);
    if (!start) return null;
    const elapsed = (new Date() - start) / 1000;
    if (elapsed < 3) return null;
    const rate = job.items_processed / elapsed;
    return formatDuration(remaining / rate);
  }

  // ==================== Polling ====================

  function startPolling(container) {
    stopPolling();
    pollInterval = setInterval(() => refreshContent(container), 3000);
  }

  function stopPolling() {
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
  }

  async function refreshContent(container) {
    // Bail out if container has been removed from DOM (navigation away)
    if (!document.contains(container)) {
      stopPolling();
      return;
    }
    try {
      const [statusResult, jobsResult] = await Promise.all([
        QueueAPI.getStatus(),
        QueueAPI.getJobs(),
      ]);
      renderContent(container, statusResult, jobsResult.jobs || []);
    } catch (e) {
      console.error('[Stash Sense] Operations poll error:', e);
    }
  }

  // ==================== Tab Injection ====================

  function injectOperationsTab() {
    const route = SS.getRoute();
    if (route.type !== 'plugin') return;

    // Wait for tab bar to exist (created by settings module)
    const dashboard = document.getElementById('ss-recommendations');
    if (!dashboard) return;

    const tabBar = dashboard.querySelector('.ss-page-tabs');
    if (!tabBar) return;

    // Already injected?
    if (document.getElementById('ss-operations')) return;
    if (tabBar.querySelector('[data-tab="operations"]')) return;

    // Check if we should start on this tab
    const initialTab = SS.getTabFromUrl();

    // Create Operations tab button -- insert BEFORE Settings tab
    const operationsTab = SS.createElement('button', {
      className: `ss-page-tab ${initialTab === 'operations' ? 'active' : ''}`,
      textContent: 'Operations',
      attrs: { 'data-tab': 'operations' },
    });

    const settingsTabBtn = tabBar.querySelector('[data-tab="settings"]');
    if (settingsTabBtn) {
      tabBar.insertBefore(operationsTab, settingsTabBtn);
    } else {
      tabBar.appendChild(operationsTab);
    }

    // Create operations panel
    const operationsPanel = createOperationsContainer();
    operationsPanel.style.display = initialTab === 'operations' ? '' : 'none';
    operationsPanel.setAttribute('data-panel', 'operations');

    // Insert before settings panel
    const settingsPanel = document.getElementById('ss-settings');
    if (settingsPanel) {
      settingsPanel.parentElement.insertBefore(operationsPanel, settingsPanel);
    } else {
      dashboard.appendChild(operationsPanel);
    }

    // If starting on operations tab, hide other panels and lazy load
    if (initialTab === 'operations') {
      // Deactivate other tabs
      tabBar.querySelectorAll('.ss-page-tab').forEach(t => {
        if (t !== operationsTab) t.classList.remove('active');
      });
      // Hide other panels
      const recPanel = dashboard.querySelector('.ss-page-panel[data-panel="recommendations"]');
      if (recPanel) recPanel.style.display = 'none';
      if (settingsPanel) settingsPanel.style.display = 'none';
      // Load content
      operationsPanel.dataset.loaded = 'true';
      renderOperations(operationsPanel);
    }

    // Patch the existing tab click handler to include operations
    tabBar.addEventListener('click', (e) => {
      const btn = e.target.closest('.ss-page-tab');
      if (!btn) return;
      const tabName = btn.dataset.tab;

      // Update URL
      SS.setTabInUrl(tabName);

      // Show/hide operations panel
      operationsPanel.style.display = tabName === 'operations' ? '' : 'none';

      // Lazy load on first view
      if (tabName === 'operations' && !operationsPanel.dataset.loaded) {
        operationsPanel.dataset.loaded = 'true';
        renderOperations(operationsPanel);
      } else if (tabName === 'operations') {
        // Returning to an already-loaded Operations tab: resume live polling
        // and refresh immediately so queued jobs transition to running/progress.
        ensurePolling(operationsPanel);
        refreshContent(operationsPanel);
      }

      // Stop polling when leaving operations tab
      if (tabName !== 'operations') {
        stopPolling();
      }
    });
  }

  // ==================== Initialization ====================

  function cleanup() {
    stopPolling();
    const operations = document.getElementById('ss-operations');
    if (operations) operations.remove();
    jobTypes = null;
  }

  function init() {
    const tryInject = () => {
      if (SS.getRoute().type === 'plugin') {
        setTimeout(injectOperationsTab, 800);
      }
    };

    tryInject();

    SS.onNavigate((route) => {
      if (route.type === 'plugin') {
        setTimeout(injectOperationsTab, 800);
      } else {
        stopPolling();
      }
    });

    // Clean up when leaving plugin page
    SS.onLeavePlugin(cleanup);

    console.log(`[${SS.PLUGIN_NAME}] Operations module loaded`);
  }

  window.StashSenseOperations = {
    refresh: () => {
      const container = document.getElementById('ss-operations');
      if (container) renderOperations(container);
    },
    init,
  };

  init();
})();
