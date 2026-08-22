/**
 * Stash Sense Settings Module
 * Settings tab UI for the plugin page — hardware info, sidecar configuration
 */
(function() {
  'use strict';

  const SS = window.StashSense;
  if (!SS) {
    console.error('[Stash Sense] Core module not loaded');
    return;
  }

  // ==================== Settings API ====================

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

  const SettingsAPI = {
    async getAll() {
      return apiCall('settings_get_all');
    },
    async update(key, value) {
      const args = { key, value, plugin_version: SS.PLUGIN_VERSION };
      return apiCall('settings_update', args);
    },
    async reset(key) {
      return apiCall('settings_reset', { key });
    },
    async getSystemInfo() {
      return apiCall('system_info');
    },
    async getDatabaseInfo() {
      return apiCall('database_info');
    },
    async getFingerprintStatus() {
      return apiCall('fp_status');
    },
    async startFingerprintGeneration() {
      return apiCall('queue_submit', { type: 'fingerprint_generation' });
    },
    async getFingerprintJobs() {
      return apiCall('queue_list', { type: 'fingerprint_generation' });
    },
    async checkUpdate() {
      return apiCall('db_check_update');
    },
    async getLogs() {
      return apiCall('logs_list');
    },
    async downloadLog(filename) {
      return apiCall('logs_download', { filename });
    },
    async deleteLog(filename) {
      return apiCall('logs_delete', { filename });
    },
    async deleteAllLogs() {
      return apiCall('logs_delete_all');
    },
    async downloadAllLogs() {
      return apiCall('logs_download_all');
    },
  };

  // ==================== State ====================

  let settingsData = null;
  let systemInfo = null;
  let saveTimeouts = {};
  const UPSTREAM_GENDER_SETTINGS = [
    { key: 'upstream_scene_gender_female_enabled', label: 'Female' },
    { key: 'upstream_scene_gender_male_enabled', label: 'Male' },
    { key: 'upstream_scene_gender_transgender_female_enabled', label: 'Transgender Female' },
    { key: 'upstream_scene_gender_transgender_male_enabled', label: 'Transgender Male' },
    { key: 'upstream_scene_gender_intersex_enabled', label: 'Intersex' },
    { key: 'upstream_scene_gender_non_binary_enabled', label: 'Non-Binary' },
    { key: 'upstream_scene_gender_unknown_enabled', label: 'Unknown' },
  ];

  // ==================== Rendering ====================

  function createSettingsContainer() {
    const container = SS.createElement('div', {
      id: 'ss-settings',
      className: 'ss-settings-page',
    });
    return container;
  }

  async function renderSettings(container) {
    container.innerHTML = '<div class="ss-settings-loading">Loading settings...</div>';

    try {
      const [settingsResult, infoResult] = await Promise.all([
        SettingsAPI.getAll(),
        SettingsAPI.getSystemInfo(),
      ]);
      settingsData = settingsResult;
      systemInfo = infoResult;
    } catch (e) {
      const errorDiv = SS.createElement('div', { className: 'ss-settings-error' });
      errorDiv.appendChild(SS.createElement('h3', { textContent: 'Failed to load settings' }));
      errorDiv.appendChild(SS.createElement('p', { textContent: e.message }));
      errorDiv.appendChild(SS.createElement('button', {
        className: 'ss-btn ss-btn-primary',
        textContent: 'Retry',
        events: { click: () => window.StashSenseSettings.refresh() },
      }));
      container.innerHTML = '';
      container.appendChild(errorDiv);
      return;
    }

    container.innerHTML = '';

    // Header
    const header = SS.createElement('div', { className: 'ss-settings-header' });
    header.innerHTML = `
      <h1>Settings</h1>
      <p class="ss-settings-subtitle">Sidecar configuration and hardware info</p>
    `;
    container.appendChild(header);

    // Hardware banner
    if (systemInfo?.hardware) {
      container.appendChild(renderHardwareBanner(systemInfo));
    }

    // Identification Database section (above models)
    container.appendChild(await renderIdDatabaseSection());

    // Models section
    container.appendChild(await renderModelsCategory());

    // Endpoint priority ordering
    container.appendChild(await renderEndpointPriorityCategory());

    // Settings categories
    if (settingsData?.categories) {
      const cats = Object.entries(settingsData.categories)
        .sort((a, b) => (a[1].order || 0) - (b[1].order || 0));

      for (const [catKey, cat] of cats) {
        if (catKey === 'upstream_sync') continue;
        const section = renderCategory(catKey, cat);
        container.appendChild(section);
        if (catKey === 'diagnostics') {
          patchAnonymizeToggle(section);
          container.appendChild(await renderLogFilesSection());
        }
      }
    }

    // Display settings (from plugin-local user settings)
    container.appendChild(await renderDisplayCategory());

    // Upstream sync field monitoring
    container.appendChild(await renderUpstreamSyncCategory());

    // Job Schedules section
    await renderSchedulesCategory(container);
  }

  // ==================== Debug Log File Management ====================

  function formatFileSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }

  function formatLogDate(unixTimestamp) {
    const d = new Date(unixTimestamp * 1000);
    return d.toLocaleString(undefined, {
      year: 'numeric', month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });
  }

  async function renderLogFilesSection() {
    const section = SS.createElement('div', {
      className: 'ss-settings-category ss-log-files-section',
      attrs: { id: 'ss-log-files-section' },
    });

    section.innerHTML = `
      <div class="ss-settings-cat-header">
        <h2>Debug Log Files</h2>
      </div>
      <div class="ss-log-files-body">
        <div class="ss-log-files-loading">
          <div class="ss-spinner" style="width:16px;height:16px;"></div>
          <span>Loading...</span>
        </div>
      </div>
    `;

    await refreshLogFilesList(section);
    return section;
  }

  async function refreshLogFilesList(section) {
    const body = section.querySelector('.ss-log-files-body');
    if (!body) return;

    let files = [];
    try {
      const result = await SettingsAPI.getLogs();
      files = result.files || [];
    } catch (e) {
      body.innerHTML = `<p class="ss-log-files-empty">Failed to load log files: ${e.message}</p>`;
      return;
    }

    body.innerHTML = '';

    if (files.length === 0) {
      body.innerHTML = '<p class="ss-log-files-empty">No debug log files found.</p>';
      return;
    }

    const table = SS.createElement('div', { className: 'ss-log-files-table' });

    for (const file of files) {
      const row = SS.createElement('div', { className: 'ss-log-file-row' });
      row.innerHTML = `
        <span class="ss-log-file-name">${file.filename}</span>
        <span class="ss-log-file-size">${formatFileSize(file.size_bytes)}</span>
        <span class="ss-log-file-date">${formatLogDate(file.modified_at)}</span>
        <span class="ss-log-file-actions"></span>
      `;

      const actions = row.querySelector('.ss-log-file-actions');

      const downloadBtn = SS.createElement('button', {
        className: 'ss-btn ss-btn-secondary ss-btn-xs',
        textContent: 'Download',
        events: {
          click: async (e) => {
            e.stopPropagation();
            downloadBtn.disabled = true;
            downloadBtn.textContent = '...';
            try {
              const result = await SettingsAPI.downloadLog(file.filename);
              if (result.error) throw new Error(result.error);
              const binary = atob(result.content_b64);
              const bytes = new Uint8Array(binary.length);
              for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
              const blob = new Blob([bytes], { type: 'text/plain' });
              const url = URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = file.filename;
              document.body.appendChild(a);
              a.click();
              document.body.removeChild(a);
              URL.revokeObjectURL(url);
            } catch (err) {
              console.error('[Stash Sense] Log download failed:', err);
            } finally {
              downloadBtn.disabled = false;
              downloadBtn.textContent = 'Download';
            }
          },
        },
      });

      const deleteBtn = SS.createElement('button', {
        className: 'ss-btn ss-btn-danger ss-btn-xs',
        textContent: 'Delete',
        events: {
          click: async (e) => {
            e.stopPropagation();
            deleteBtn.disabled = true;
            deleteBtn.textContent = '...';
            try {
              await SettingsAPI.deleteLog(file.filename);
              await refreshLogFilesList(section);
            } catch (err) {
              deleteBtn.disabled = false;
              deleteBtn.textContent = 'Delete';
              console.error('[Stash Sense] Log delete failed:', err);
            }
          },
        },
      });

      actions.appendChild(downloadBtn);
      actions.appendChild(deleteBtn);
      table.appendChild(row);
    }

    body.appendChild(table);

    const footer = SS.createElement('div', { className: 'ss-log-files-footer' });

    const downloadAllBtn = SS.createElement('button', {
      className: 'ss-btn ss-btn-secondary ss-btn-sm',
      textContent: 'Download All Logs',
      events: {
        click: async () => {
          downloadAllBtn.disabled = true;
          downloadAllBtn.textContent = '...';
          try {
            const result = await SettingsAPI.downloadAllLogs();
            if (result.error) throw new Error(result.error);
            const binary = atob(result.content_b64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
            const blob = new Blob([bytes], { type: 'application/zip' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = result.filename || 'stash_sense_logs.zip';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
          } catch (err) {
            console.error('[Stash Sense] Download all logs failed:', err);
          } finally {
            downloadAllBtn.disabled = false;
            downloadAllBtn.textContent = 'Download All Logs';
          }
        },
      },
    });

    const deleteAllBtn = SS.createElement('button', {
      className: 'ss-btn ss-btn-danger ss-btn-sm',
      textContent: 'Delete All Logs',
      events: {
        click: async () => {
          deleteAllBtn.disabled = true;
          deleteAllBtn.textContent = '...';
          try {
            await SettingsAPI.deleteAllLogs();
            await refreshLogFilesList(section);
          } catch (err) {
            deleteAllBtn.disabled = false;
            deleteAllBtn.textContent = 'Delete All Logs';
            console.error('[Stash Sense] Delete all logs failed:', err);
          }
        },
      },
    });

    footer.appendChild(downloadAllBtn);
    footer.appendChild(deleteAllBtn);
    body.appendChild(footer);
  }

  function patchAnonymizeToggle(diagnosticsSection) {
    // Find the toggle row for debug_logging_anonymize inside the diagnostics section.
    // The label text uniquely identifies it; we replace the checkbox's change handler
    // to show a warning when logs exist before saving.
    const rows = diagnosticsSection.querySelectorAll('.ss-setting-row');
    for (const row of rows) {
      const label = row.querySelector('.ss-setting-label');
      if (!label || !label.textContent.includes('Anonymise')) continue;

      const checkbox = row.querySelector('input[type="checkbox"]');
      if (!checkbox) break;

      // Clone → replace to strip the existing change listener
      const fresh = checkbox.cloneNode(true);
      checkbox.replaceWith(fresh);

      fresh.addEventListener('change', async () => {
        if (!fresh.checked) {
          // Turning off — no confirmation needed
          debouncedSave('debug_logging_anonymize', false, row);
          return;
        }

        // Turning on — check if logs exist first
        let logs = [];
        try {
          const result = await SettingsAPI.getLogs();
          logs = result.files || [];
        } catch (_) {}

        if (logs.length === 0) {
          debouncedSave('debug_logging_anonymize', true, row);
          return;
        }

        // Show confirmation overlay
        const overlay = SS.createElement('div', { className: 'ss-modal-overlay' });
        overlay.innerHTML = `
          <div class="ss-modal-content" style="max-width:420px;padding:20px;">
            <h3 style="margin:0 0 12px;">Enable Anonymisation?</h3>
            <p style="margin:0 0 16px;color:#ccc;">
              Enabling anonymisation will <strong>delete ${logs.length} existing debug log file${logs.length !== 1 ? 's' : ''}</strong>
              to prevent un-anonymised data from being mixed with anonymised entries.
              Do you wish to proceed and delete the existing debug logs?
            </p>
            <div style="display:flex;gap:10px;justify-content:flex-end;">
              <button class="ss-btn ss-btn-secondary" id="ss-anon-cancel">Cancel</button>
              <button class="ss-btn ss-btn-danger" id="ss-anon-confirm">Delete Logs</button>
            </div>
          </div>
        `;
        document.body.appendChild(overlay);

        overlay.querySelector('#ss-anon-cancel').addEventListener('click', () => {
          fresh.checked = false;
          overlay.remove();
        });
        overlay.addEventListener('click', (e) => {
          if (e.target === overlay) { fresh.checked = false; overlay.remove(); }
        });
        overlay.querySelector('#ss-anon-confirm').addEventListener('click', async () => {
          overlay.remove();
          try {
            await SettingsAPI.deleteAllLogs();
          } catch (_) {}
          // Refresh the log files list if visible
          const logSection = document.getElementById('ss-log-files-section');
          if (logSection) refreshLogFilesList(logSection);
          debouncedSave('debug_logging_anonymize', true, row);
        });
      });

      break;
    }
  }

  async function renderIdDatabaseSection() {
    const section = SS.createElement('div', { className: 'ss-settings-category ss-id-database-settings-section' });
    section.innerHTML = `
      <div class="ss-settings-cat-header">
        <h2 class="ss-settings-cat-title">Identification Database</h2>
        <p class="ss-settings-cat-desc">Face recognition database used for performer identification and duplicate detection.</p>
      </div>
      <div class="ss-settings-cat-body">
        <div class="ss-id-db-group">
          <h3 class="ss-id-db-group-title">Performer Database</h3>
          <div class="ss-id-database-stats ss-id-database-stats-performer ss-id-database-loading">
            <div class="ss-spinner" style="width:16px;height:16px;margin:0 auto;"></div>
          </div>
        </div>
        <div class="ss-id-db-group">
          <h3 class="ss-id-db-group-title">Fingerprint Database</h3>
          <div class="ss-id-database-stats ss-id-database-stats-fingerprint ss-id-database-loading">
            <div class="ss-spinner" style="width:16px;height:16px;margin:0 auto;"></div>
          </div>
        </div>
      </div>
    `;

    const performerStatsEl = section.querySelector('.ss-id-database-stats-performer');
    const fingerprintStatsEl = section.querySelector('.ss-id-database-stats-fingerprint');

    try {
      const [fpStatus, dbInfo, updateInfo, fpJobs] = await Promise.all([
        SettingsAPI.getFingerprintStatus().catch(() => null),
        SettingsAPI.getDatabaseInfo().catch(() => null),
        SettingsAPI.checkUpdate().catch(() => null),
        SettingsAPI.getFingerprintJobs().catch(() => null),
      ]);

      const fpJobActive = !!(fpJobs && fpJobs.jobs || []).find(
        j => j.status === 'queued' || j.status === 'running' || j.status === 'stopping'
      );

      const fpCoverage = (fpStatus && fpStatus.complete_fingerprints) || 0;
      const totalScenes = fpStatus && fpStatus.total_scenes != null ? fpStatus.total_scenes : null;
      const dbVersion = (fpStatus && fpStatus.current_db_version) || dbInfo?.version || 'N/A';
      const performerCount = dbInfo?.performer_count || 0;
      const faceCount = dbInfo?.face_count || 0;

      // Row 1: performer database (the ~2GB dataset downloaded via Database Update)
      performerStatsEl.className = 'ss-id-database-stats ss-id-database-stats-performer';
      performerStatsEl.innerHTML = `
        <div class="ss-db-stat">
          <span class="ss-db-stat-value">${dbVersion}</span>
          <span class="ss-db-stat-label">Version</span>
          ${updateInfo && updateInfo.update_available ? `
            <div class="ss-update-badge">
              <span class="ss-update-badge-text">v${updateInfo.latest_version} available${
                updateInfo.delta_available
                  ? ` \u2014 ${updateInfo.delta_download_size_mb} MB via delta (${updateInfo.delta_chain_length} release${updateInfo.delta_chain_length === 1 ? '' : 's'} behind)`
                  : updateInfo.download_size_mb
                    ? ` \u2014 ${updateInfo.download_size_mb} MB full download`
                    : ''
              }</span>
            </div>
          ` : ''}
        </div>
        <div class="ss-db-stat">
          <span class="ss-db-stat-value">${performerCount.toLocaleString()}</span>
          <span class="ss-db-stat-label">Performers</span>
        </div>
        <div class="ss-db-stat">
          <span class="ss-db-stat-value">${faceCount.toLocaleString()}</span>
          <span class="ss-db-stat-label">Faces</span>
        </div>
      `;

      // Row 2: this scene library's own fingerprint coverage (stash_sense.db,
      // local to this install -- not part of the downloaded performer
      // database). "Missing" replaces the old "Need Refresh" figure: with
      // cached signals, a performer-database update no longer invalidates
      // existing fingerprints, so the only gap worth surfacing here is
      // scenes that have never been fingerprinted at all.
      const missing = totalScenes != null ? Math.max(0, totalScenes - fpCoverage) : null;
      fingerprintStatsEl.className = 'ss-id-database-stats ss-id-database-stats-fingerprint';
      fingerprintStatsEl.innerHTML = `
        <div class="ss-db-stat">
          <span class="ss-db-stat-value">${totalScenes != null ? totalScenes.toLocaleString() : 'N/A'}</span>
          <span class="ss-db-stat-label">Scenes</span>
        </div>
        <div class="ss-db-stat">
          <span class="ss-db-stat-value">${fpCoverage.toLocaleString()}</span>
          <span class="ss-db-stat-label">Fingerprints</span>
        </div>
        ${missing > 0 ? `
        <div class="ss-db-stat ss-db-stat-warning">
          <span class="ss-db-stat-value">${missing.toLocaleString()}</span>
          <span class="ss-db-stat-label">Missing</span>
        </div>
        <button class="ss-btn ss-btn-primary ss-btn-sm ss-start-fingerprinting-btn" id="ss-start-fingerprinting-btn" ${fpJobActive ? 'disabled' : ''}>
          ${fpJobActive ? 'Fingerprinting in Progress' : 'Start Fingerprinting'}
        </button>
        ` : ''}
      `;

      const startBtn = fingerprintStatsEl.querySelector('#ss-start-fingerprinting-btn');
      if (startBtn) {
        startBtn.addEventListener('click', async () => {
          startBtn.disabled = true;
          startBtn.textContent = 'Starting…';
          try {
            await SettingsAPI.startFingerprintGeneration();
            startBtn.textContent = 'Fingerprinting in Progress';
          } catch (e) {
            const msg = String(e.message || '');
            const alreadyRunning = msg.includes('409') || msg.includes('already');
            startBtn.textContent = alreadyRunning ? 'Fingerprinting in Progress' : 'Error';
            if (!alreadyRunning) {
              setTimeout(() => {
                startBtn.textContent = 'Start Fingerprinting';
                startBtn.disabled = false;
              }, 2500);
            }
          }
        });
      }
    } catch (e) {
      const errHtml = `<span style="color:#888;font-size:13px;">Failed to load: ${e.message}</span>`;
      performerStatsEl.innerHTML = errHtml;
      fingerprintStatsEl.innerHTML = errHtml;
    }

    return section;
  }

  // Re-fetch and swap in fresh stats without re-rendering the whole settings
  // page. The settings panel is only built once (see injectSettingsTab's
  // `dataset.loaded` guard) and just toggled visible/hidden after that, so
  // these numbers would otherwise stay frozen at whatever they were on
  // first load -- e.g. still showing stale "Missing" count after switching
  // away to run fingerprint generation and back once it finishes.
  async function refreshIdDatabaseSection() {
    const existing = document.querySelector('#ss-settings .ss-id-database-settings-section');
    if (!existing) return;
    const fresh = await renderIdDatabaseSection();
    existing.replaceWith(fresh);
  }

  function renderHardwareBanner(info) {
    const hw = info.hardware;
    const banner = SS.createElement('div', { className: 'ss-hw-banner' });

    const tierClass = hw.tier === 'gpu-high' ? 'ss-tier-high' :
                      hw.tier === 'gpu-low' ? 'ss-tier-low' : 'ss-tier-cpu';

    const gpuText = hw.gpu_available
      ? `${hw.gpu_name || 'GPU'} (${hw.gpu_vram_mb ? Math.round(hw.gpu_vram_mb / 1024) + 'GB VRAM' : 'unknown VRAM'})`
      : 'No GPU detected';

    const uptimeMin = Math.floor((info.uptime_seconds || 0) / 60);
    const uptimeStr = uptimeMin < 60
      ? `${uptimeMin}m`
      : `${Math.floor(uptimeMin / 60)}h ${uptimeMin % 60}m`;

    banner.innerHTML = `
      <div class="ss-hw-banner-content">
        <div class="ss-hw-main">
          <span class="ss-hw-tier ${tierClass}">${hw.tier}</span>
          <span class="ss-hw-gpu">${gpuText}</span>
        </div>
        <div class="ss-hw-details">
          <span>${hw.memory_total_mb ? Math.round(hw.memory_total_mb / 1024) + 'GB RAM' : ''}</span>
          <span>${hw.cpu_cores || '?'} cores</span>
          <span>v${info.version || '?'}</span>
          <span>up ${uptimeStr}</span>
        </div>
      </div>
    `;
    return banner;
  }

  // ==================== Models Category ====================

  async function renderModelsCategory() {
    const section = SS.createElement('div', { className: 'ss-settings-category' });

    const header = SS.createElement('div', {
      className: 'ss-settings-cat-header',
      innerHTML: '<h2>Models</h2>',
    });
    section.appendChild(header);

    const body = SS.createElement('div', { className: 'ss-settings-cat-body' });

    try {
      const [modelsResult, capabilitiesResult] = await Promise.all([
        apiCall('models_status'),
        apiCall('capabilities'),
      ]);

      const modelsObj = modelsResult.models || {};
      const models = Object.entries(modelsObj).map(([key, info]) => ({
        name: key,
        ...info,
      }));

      if (models.length === 0) {
        const emptyRow = SS.createElement('div', {
          className: 'ss-setting-row ss-setting-hint',
          textContent: 'No models found.',
        });
        body.appendChild(emptyRow);
        section.appendChild(body);
        return section;
      }

      // Group models by their group field
      const groups = {};
      for (const model of models) {
        const group = model.group || 'other';
        if (!groups[group]) groups[group] = [];
        groups[group].push(model);
      }

      let hasNotInstalled = false;

      for (const [groupKey, groupModels] of Object.entries(groups)) {
        // Group sub-header
        const groupLabel = groupKey
          .replace(/_/g, ' ')
          .replace(/\b\w/g, c => c.toUpperCase());

        const groupHeader = SS.createElement('div', {
          className: 'ss-setting-row ss-models-group-header',
        });
        groupHeader.innerHTML = `<span class="ss-setting-label" style="margin-bottom:0">${groupLabel}</span>`;
        body.appendChild(groupHeader);

        for (const model of groupModels) {
          const row = SS.createElement('div', { className: 'ss-setting-row' });

          const modelName = (model.name || '')
            .replace(/_/g, ' ')
            .replace(/\b\w/g, c => c.toUpperCase());

          const sizeMB = model.size
            ? `${(model.size / (1024 * 1024)).toFixed(0)} MB`
            : '';

          const info = SS.createElement('div', { className: 'ss-setting-info' });
          info.innerHTML = `
            <label class="ss-setting-label">${modelName}</label>
            <span class="ss-setting-desc">${sizeMB}</span>
          `;

          const control = SS.createElement('div', { className: 'ss-setting-control' });

          // Status badge
          const statusClass = model.status === 'installed' ? 'ss-model-installed'
            : model.status === 'corrupted' ? 'ss-model-corrupted'
            : 'ss-model-not_installed';

          const statusLabel = model.status === 'installed' ? 'Installed'
            : model.status === 'corrupted' ? 'Corrupted'
            : 'Not Installed';

          const badge = SS.createElement('span', {
            className: `ss-model-status ${statusClass}`,
            textContent: statusLabel,
          });
          control.appendChild(badge);

          // Download button for non-installed models
          if (model.status !== 'installed') {
            hasNotInstalled = true;
            const dlBtn = SS.createElement('button', {
              className: 'ss-btn ss-btn-primary ss-btn-sm',
              textContent: 'Download',
            });

            dlBtn.addEventListener('click', async () => {
              dlBtn.disabled = true;
              dlBtn.textContent = 'Starting...';

              try {
                await apiCall('models_download', { model_name: model.name });

                // Poll for progress with timeout
                const pollStart = Date.now();
                const POLL_TIMEOUT_MS = 300000; // 5 minutes
                let pollErrors = 0;
                const pollInterval = setInterval(async () => {
                  // Timeout: stop polling if exceeded
                  if (Date.now() - pollStart > POLL_TIMEOUT_MS) {
                    clearInterval(pollInterval);
                    dlBtn.textContent = 'Timeout';
                    dlBtn.disabled = false;
                    dlBtn.className = 'ss-btn ss-btn-danger ss-btn-sm';
                    return;
                  }
                  try {
                    const progress = await apiCall('models_progress');
                    pollErrors = 0;
                    const dl = progress.progress?.[model.name] || progress.current;

                    if (dl && dl.percent !== undefined) {
                      dlBtn.textContent = `${Math.round(dl.percent)}%`;
                    }

                    // Check if complete
                    const isComplete = dl?.status === 'complete'
                      || dl?.percent >= 100;

                    if (isComplete || (!dl && progress.status !== 'downloading')) {
                      clearInterval(pollInterval);
                      badge.className = 'ss-model-status ss-model-installed';
                      badge.textContent = 'Installed';
                      dlBtn.remove();
                    }

                    // Check if failed
                    if (dl?.status === 'failed' || dl?.status === 'error') {
                      clearInterval(pollInterval);
                      dlBtn.textContent = 'Retry';
                      dlBtn.disabled = false;
                      dlBtn.className = 'ss-btn ss-btn-danger ss-btn-sm';
                    }
                  } catch (pollErr) {
                    pollErrors++;
                    if (pollErrors >= 5) {
                      clearInterval(pollInterval);
                      dlBtn.textContent = 'Error';
                      dlBtn.disabled = false;
                      dlBtn.className = 'ss-btn ss-btn-danger ss-btn-sm';
                    }
                  }
                }, 1000);
              } catch (err) {
                dlBtn.textContent = 'Error';
                dlBtn.disabled = false;
                dlBtn.className = 'ss-btn ss-btn-danger ss-btn-sm';
                console.error(`[Stash Sense] Failed to download ${model.name}:`, err);
              }
            });

            control.appendChild(dlBtn);
          }

          row.appendChild(info);
          row.appendChild(control);
          body.appendChild(row);
        }
      }

      // Download All button
      if (hasNotInstalled) {
        const downloadAllRow = SS.createElement('div', {
          className: 'ss-setting-row',
          style: 'justify-content: flex-end; margin-top: 8px;',
        });

        const downloadAllBtn = SS.createElement('button', {
          className: 'ss-btn ss-btn-primary',
          textContent: 'Download All',
        });

        downloadAllBtn.addEventListener('click', async () => {
          downloadAllBtn.disabled = true;
          downloadAllBtn.textContent = 'Downloading...';

          try {
            await apiCall('models_download_all');

            // Poll for progress with timeout
            const pollStart = Date.now();
            const POLL_TIMEOUT_MS = 300000; // 5 minutes
            let pollErrors = 0;
            const pollInterval = setInterval(async () => {
              if (Date.now() - pollStart > POLL_TIMEOUT_MS) {
                clearInterval(pollInterval);
                downloadAllBtn.textContent = 'Timeout';
                downloadAllBtn.disabled = false;
                downloadAllBtn.className = 'ss-btn ss-btn-danger';
                return;
              }
              try {
                const progressResult = await apiCall('models_progress');
                pollErrors = 0;
                const entries = Object.values(progressResult.progress || {});

                if (entries.length > 0) {
                  const totalBytes = entries.reduce((s, e) => s + (e.total_bytes || 0), 0);
                  const dlBytes = entries.reduce((s, e) => s + (e.downloaded_bytes || 0), 0);
                  const pct = totalBytes > 0 ? Math.round((dlBytes / totalBytes) * 100) : 0;
                  downloadAllBtn.textContent = `Downloading... ${pct}%`;

                  const allDone = entries.every(e => e.status === 'complete');
                  const anyFailed = entries.some(e => e.status === 'failed');

                  if (allDone) {
                    clearInterval(pollInterval);
                    downloadAllBtn.textContent = 'Done';
                    downloadAllBtn.className = 'ss-btn ss-btn-success';

                    // Refresh the entire settings page after a brief delay
                    setTimeout(() => {
                      if (window.StashSenseSettings) {
                        window.StashSenseSettings.refresh();
                      }
                    }, 1500);
                  } else if (anyFailed) {
                    clearInterval(pollInterval);
                    downloadAllBtn.textContent = 'Retry';
                    downloadAllBtn.disabled = false;
                    downloadAllBtn.className = 'ss-btn ss-btn-danger';
                  }
                }
              } catch (pollErr) {
                pollErrors++;
                if (pollErrors >= 5) {
                  clearInterval(pollInterval);
                  downloadAllBtn.textContent = 'Error';
                  downloadAllBtn.disabled = false;
                  downloadAllBtn.className = 'ss-btn ss-btn-danger';
                }
              }
            }, 1000);
          } catch (err) {
            downloadAllBtn.textContent = 'Error';
            downloadAllBtn.disabled = false;
            downloadAllBtn.className = 'ss-btn ss-btn-danger';
            console.error('[Stash Sense] Failed to download all models:', err);
          }
        });

        downloadAllRow.appendChild(downloadAllBtn);
        body.appendChild(downloadAllRow);
      }
    } catch (e) {
      const errorRow = SS.createElement('div', {
        className: 'ss-setting-row ss-setting-hint',
        textContent: `Failed to load model status: ${e.message}`,
      });
      errorRow.style.color = '#ef4444';
      body.appendChild(errorRow);
    }

    section.appendChild(body);
    return section;
  }

  function renderCategory(catKey, cat) {
    const section = SS.createElement('div', { className: 'ss-settings-category' });

    const header = SS.createElement('div', {
      className: 'ss-settings-cat-header',
      innerHTML: `<h2>${cat.label}</h2>`,
    });

    const body = SS.createElement('div', { className: 'ss-settings-cat-body' });

    for (const [key, setting] of Object.entries(cat.settings)) {
      body.appendChild(renderSetting(key, setting));
    }

    section.appendChild(header);
    section.appendChild(body);
    return section;
  }

  function renderSetting(key, setting) {
    const row = SS.createElement('div', { className: 'ss-setting-row' });

    const info = SS.createElement('div', { className: 'ss-setting-info' });
    info.innerHTML = `
      <label class="ss-setting-label">${setting.label}</label>
      <span class="ss-setting-desc">${setting.description}</span>
      ${key === 'detection_size' ? `
        <span class="ss-setting-warning">
          ⚠️ Changing this invalidates the cached face signals used by
          Stash Sense's fingerprint fast-path. Every scene will need to
          be fully re-fingerprinted (frame extraction + detection +
          embedding, not just re-matched) to use the new resolution.
        </span>
      ` : ''}
    `;

    const control = SS.createElement('div', { className: 'ss-setting-control' });

    if (setting.type === 'bool') {
      control.appendChild(renderToggle(key, setting));
    } else if (setting.type === 'int' || setting.type === 'float') {
      control.appendChild(renderNumberInput(key, setting));
    } else {
      control.appendChild(renderTextInput(key, setting));
    }

    // Reset button (only when overridden)
    if (setting.is_override) {
      const resetBtn = SS.createElement('button', {
        className: 'ss-setting-reset',
        textContent: 'Reset',
        events: {
          click: async () => {
            try {
              const result = await SettingsAPI.reset(key);
              setting.value = result.value;
              setting.is_override = false;
              // Re-render this row
              const parent = row.parentNode;
              const newRow = renderSetting(key, setting);
              parent.replaceChild(newRow, row);
              showSaveIndicator(newRow, 'Reset');
            } catch (e) {
              console.error(`Failed to reset ${key}:`, e);
            }
          },
        },
      });
      control.appendChild(resetBtn);
    }

    row.appendChild(info);
    row.appendChild(control);
    return row;
  }

  function renderToggle(key, setting) {
    const toggle = SS.createElement('label', { className: 'ss-toggle' });
    const input = SS.createElement('input', {
      attrs: { type: 'checkbox' },
    });
    input.checked = !!setting.value;
    input.addEventListener('change', () => {
      const onSave = key === 'debug_logging_enabled' ? _refreshLogFilesIfVisible : null;
      debouncedSave(key, input.checked, toggle.closest('.ss-setting-row'), onSave);
    });

    const slider = SS.createElement('span', { className: 'ss-toggle-slider' });
    toggle.appendChild(input);
    toggle.appendChild(slider);
    return toggle;
  }

  function renderNumberInput(key, setting) {
    const wrapper = SS.createElement('div', { className: 'ss-number-input' });
    const input = SS.createElement('input', {
      attrs: {
        type: 'number',
        value: String(setting.value),
        step: setting.type === 'float' ? '0.5' : '1',
      },
    });
    if (setting.min !== undefined) input.setAttribute('min', setting.min);
    if (setting.max !== undefined) input.setAttribute('max', setting.max);

    input.addEventListener('input', () => {
      const val = setting.type === 'float' ? parseFloat(input.value) : parseInt(input.value, 10);
      if (!isNaN(val)) {
        debouncedSave(key, val, wrapper.closest('.ss-setting-row'));
      }
    });

    wrapper.appendChild(input);

    if (setting.min !== undefined || setting.max !== undefined) {
      const range = SS.createElement('span', {
        className: 'ss-setting-range',
        textContent: `${setting.min ?? ''}–${setting.max ?? ''}`,
      });
      wrapper.appendChild(range);
    }

    return wrapper;
  }

  function renderTextInput(key, setting) {
    const input = SS.createElement('input', {
      attrs: { type: 'text', value: String(setting.value || '') },
    });
    input.addEventListener('input', () => {
      debouncedSave(key, input.value, input.closest('.ss-setting-row'));
    });
    return input;
  }

  // ==================== Endpoint Priority ====================

  async function renderEndpointPriorityCategory() {
    const section = SS.createElement('div', { className: 'ss-settings-category' });

    const header = SS.createElement('div', {
      className: 'ss-settings-cat-header',
      innerHTML: '<h2>Endpoint Priority</h2>',
    });
    section.appendChild(header);

    const body = SS.createElement('div', { className: 'ss-settings-cat-body' });

    const desc = SS.createElement('div', {
      className: 'ss-setting-row ss-setting-hint',
      textContent: 'Set the priority order for stash-box endpoints. Higher priority endpoints are checked first for upstream changes. Disabled endpoints are excluded from all upstream analysis.',
    });
    body.appendChild(desc);

    try {
      const result = await apiCall('endpoint_priorities_get');
      const endpoints = result.endpoints || [];
      const disabledEndpoints = result.disabled || [];

      if (endpoints.length === 0 && disabledEndpoints.length === 0) {
        const emptyRow = SS.createElement('div', {
          className: 'ss-setting-row ss-setting-hint',
          textContent: 'No stash-box endpoints configured.',
        });
        body.appendChild(emptyRow);
        section.appendChild(body);
        return section;
      }

      // === Enabled section ===
      const enabledLabel = SS.createElement('div', {
        className: 'ss-setting-row ss-setting-hint',
        textContent: 'Enabled',
        attrs: { style: 'font-weight:600;margin-top:0.5rem;' },
      });
      body.appendChild(enabledLabel);

      const list = SS.createElement('div', { className: 'ss-priority-list' });
      const saveStatus = SS.createElement('span', { className: 'ss-setting-hint' });

      function renderEnabledList() {
        list.innerHTML = '';
        if (endpoints.length === 0) {
          const emptyHint = SS.createElement('div', {
            className: 'ss-setting-row ss-setting-hint',
            textContent: 'All endpoints are disabled.',
          });
          emptyHint.style.color = '#888';
          list.appendChild(emptyHint);
          return;
        }
        endpoints.forEach((ep, i) => {
          const row = SS.createElement('div', { className: 'ss-priority-row' });

          const rank = SS.createElement('span', {
            className: 'ss-priority-rank',
            textContent: `${i + 1}`,
          });

          const name = SS.createElement('span', {
            className: 'ss-priority-name',
            textContent: ep.name || ep.domain || ep.endpoint,
          });

          const actions = SS.createElement('div', { className: 'ss-priority-arrows' });

          const upBtn = SS.createElement('button', {
            className: 'ss-btn ss-btn-sm ss-priority-arrow',
            textContent: '\u25B2',
            attrs: { title: 'Move up', disabled: i === 0 ? 'true' : undefined },
          });
          upBtn.addEventListener('click', () => {
            if (i > 0) {
              [endpoints[i - 1], endpoints[i]] = [endpoints[i], endpoints[i - 1]];
              renderEnabledList();
              savePriorities();
            }
          });

          const downBtn = SS.createElement('button', {
            className: 'ss-btn ss-btn-sm ss-priority-arrow',
            textContent: '\u25BC',
            attrs: { title: 'Move down', disabled: i === endpoints.length - 1 ? 'true' : undefined },
          });
          downBtn.addEventListener('click', () => {
            if (i < endpoints.length - 1) {
              [endpoints[i], endpoints[i + 1]] = [endpoints[i + 1], endpoints[i]];
              renderEnabledList();
              savePriorities();
            }
          });

          const disableBtn = SS.createElement('button', {
            className: 'ss-btn ss-btn-sm',
            textContent: 'Disable',
            attrs: { title: 'Disable this endpoint' },
          });
          disableBtn.style.cssText = 'color:#999;border:1px solid #555;background:transparent;margin-left:0.5rem;';
          disableBtn.addEventListener('click', () => showDisableConfirmation(ep));

          actions.appendChild(upBtn);
          actions.appendChild(downBtn);
          actions.appendChild(disableBtn);
          row.appendChild(rank);
          row.appendChild(name);
          row.appendChild(actions);
          list.appendChild(row);
        });
      }

      let saveTimeout;
      async function savePriorities() {
        clearTimeout(saveTimeout);
        saveTimeout = setTimeout(async () => {
          try {
            await apiCall('endpoint_priorities_set', {
              endpoints: endpoints.map(ep => ep.endpoint),
            });
            saveStatus.textContent = 'Saved';
            saveStatus.style.color = '#22c55e';
            setTimeout(() => { saveStatus.textContent = ''; }, 2000);
          } catch (e) {
            saveStatus.textContent = `Error: ${e.message}`;
            saveStatus.style.color = '#ef4444';
          }
        }, 300);
      }

      async function showDisableConfirmation(ep) {
        const displayName = ep.name || ep.domain || ep.endpoint;
        const overlay = document.createElement('div');
        overlay.className = 'ss-modal-overlay';
        overlay.innerHTML = `
          <div class="ss-modal" style="max-width:420px;">
            <h3>Disable ${SS.escapeHtml(displayName)}</h3>
            <p style="margin:0.75rem 0;color:#aaa;">This endpoint will be excluded from all upstream analysis until re-enabled.</p>
            <p style="margin:0.75rem 0;color:#aaa;">Clear existing recommendations and snapshots from this endpoint?</p>
            <div style="display:flex;flex-direction:column;gap:0.5rem;margin-top:1rem;">
              <button class="ss-btn ss-btn-danger" id="ss-disable-clear">Disable & Clear Data</button>
              <button class="ss-btn ss-btn-secondary" id="ss-disable-keep">Disable & Keep Data</button>
              <button class="ss-btn" id="ss-disable-cancel" style="margin-top:0.25rem;">Cancel</button>
            </div>
          </div>
        `;
        document.body.appendChild(overlay);

        const handleDisable = async (clearRecs) => {
          overlay.querySelector('.ss-modal').innerHTML = '<div class="ss-loading-inline"><div class="ss-spinner"></div></div><p style="text-align:center;margin-top:0.5rem;">Disabling...</p>';
          try {
            await apiCall('endpoint_disable', { endpoint: ep.endpoint, clear_recommendations: clearRecs });
            // Move from enabled to disabled locally
            const idx = endpoints.findIndex(e => e.endpoint === ep.endpoint);
            if (idx !== -1) {
              disabledEndpoints.push(endpoints.splice(idx, 1)[0]);
            }
            overlay.remove();
            renderEnabledList();
            renderDisabledList();
          } catch (e) {
            overlay.remove();
            saveStatus.textContent = `Failed: ${e.message}`;
            saveStatus.style.color = '#ef4444';
          }
        };

        overlay.querySelector('#ss-disable-clear').addEventListener('click', () => handleDisable(true));
        overlay.querySelector('#ss-disable-keep').addEventListener('click', () => handleDisable(false));
        overlay.querySelector('#ss-disable-cancel').addEventListener('click', () => overlay.remove());
        overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
      }

      renderEnabledList();
      body.appendChild(list);
      body.appendChild(saveStatus);

      // === Disabled section ===
      const disabledLabel = SS.createElement('div', {
        className: 'ss-setting-row ss-setting-hint',
        textContent: 'Disabled',
        attrs: { style: 'font-weight:600;margin-top:1.5rem;' },
      });
      body.appendChild(disabledLabel);

      const disabledList = SS.createElement('div', { className: 'ss-priority-list ss-disabled-list' });

      function renderDisabledList() {
        disabledList.innerHTML = '';
        if (disabledEndpoints.length === 0) {
          const emptyHint = SS.createElement('div', {
            className: 'ss-setting-row ss-setting-hint',
            textContent: 'No disabled endpoints.',
          });
          emptyHint.style.color = '#888';
          disabledList.appendChild(emptyHint);
          return;
        }
        disabledEndpoints.forEach((ep) => {
          const row = SS.createElement('div', { className: 'ss-priority-row ss-priority-row-disabled' });

          const name = SS.createElement('span', {
            className: 'ss-priority-name',
            textContent: ep.name || ep.domain || ep.endpoint,
          });
          name.style.color = '#666';

          const enableBtn = SS.createElement('button', {
            className: 'ss-btn ss-btn-sm ss-btn-primary',
            textContent: 'Enable',
            attrs: { title: 'Re-enable this endpoint' },
          });
          enableBtn.addEventListener('click', async () => {
            enableBtn.disabled = true;
            enableBtn.textContent = 'Enabling...';
            try {
              await apiCall('endpoint_enable', { endpoint: ep.endpoint });
              // Move from disabled to enabled locally
              const idx = disabledEndpoints.findIndex(e => e.endpoint === ep.endpoint);
              if (idx !== -1) {
                endpoints.push(disabledEndpoints.splice(idx, 1)[0]);
              }
              renderEnabledList();
              renderDisabledList();
            } catch (e) {
              enableBtn.textContent = 'Failed';
              enableBtn.disabled = false;
            }
          });

          row.appendChild(name);
          row.appendChild(enableBtn);
          disabledList.appendChild(row);
        });
      }

      renderDisabledList();
      body.appendChild(disabledList);
    } catch (e) {
      const errorRow = SS.createElement('div', {
        className: 'ss-setting-row ss-setting-hint',
        textContent: `Failed to load endpoint priorities: ${e.message}`,
      });
      errorRow.style.color = '#ef4444';
      body.appendChild(errorRow);
    }

    section.appendChild(body);
    return section;
  }

  // ==================== Display Settings ====================

  async function renderDisplayCategory() {
    const section = SS.createElement('div', { className: 'ss-settings-category' });

    const header = SS.createElement('div', {
      className: 'ss-settings-cat-header',
      innerHTML: '<h2>Display</h2>',
    });
    section.appendChild(header);

    const body = SS.createElement('div', { className: 'ss-settings-cat-body' });

    // Normalize Enum Display
    const row = SS.createElement('div', { className: 'ss-setting-row' });
    const info = SS.createElement('div', { className: 'ss-setting-info' });
    info.innerHTML = `
      <label class="ss-setting-label">Normalize Enum Display</label>
      <span class="ss-setting-desc">Show ALL_CAPS values as Title Case (e.g. BROWN → Brown)</span>
    `;

    const control = SS.createElement('div', { className: 'ss-setting-control' });
    const toggle = SS.createElement('label', { className: 'ss-toggle' });
    const input = SS.createElement('input', { attrs: { type: 'checkbox' } });

    try {
      const val = await apiCall('user_get_setting', { key: 'normalize_enum_display' });
      input.checked = val.value !== false;
    } catch (e) {
      console.error('[Stash Sense] Failed to load normalize_enum_display setting:', e);
      input.checked = true;
    }

    input.addEventListener('change', async () => {
      try {
        await apiCall('user_set_setting', { key: 'normalize_enum_display', value: input.checked });
        showSaveIndicator(row, 'Saved');
      } catch (e) {
        showSaveIndicator(row, 'Error', true);
        console.error('[Stash Sense] Failed to save display setting:', e);
      }
    });

    const slider = SS.createElement('span', { className: 'ss-toggle-slider' });
    toggle.appendChild(input);
    toggle.appendChild(slider);
    control.appendChild(toggle);
    row.appendChild(info);
    row.appendChild(control);
    body.appendChild(row);

    section.appendChild(body);
    return section;
  }

  // ==================== Upstream Sync Settings ====================

  function getUpstreamGenderSettingValue(settingKey) {
    return settingsData?.categories?.upstream_sync?.settings?.[settingKey]?.value ?? true;
  }

  function getAvailableUpstreamGenderSettings() {
    const availableSettings = settingsData?.categories?.upstream_sync?.settings || {};
    return UPSTREAM_GENDER_SETTINGS.filter(setting =>
      Object.prototype.hasOwnProperty.call(availableSettings, setting.key)
    );
  }

  function renderPerformerGenderSettingsRow() {
    const row = SS.createElement('div', { className: 'ss-setting-row-vertical' });

    const heading = SS.createElement('div', {
      className: 'ss-upstream-subheading',
      textContent: 'Performer genders',
    });
    row.appendChild(heading);

    const hint = SS.createElement('div', {
      className: 'ss-setting-hint',
      textContent: 'Only selected genders are considered for upstream scene performer additions, removals, and alias changes.',
    });
    row.appendChild(hint);

    const fieldsGrid = SS.createElement('div', {
      className: 'ss-upstream-fields-grid',
      attrs: { style: 'margin-top:10px;' },
    });

    const availableGenderSettings = getAvailableUpstreamGenderSettings();
    if (availableGenderSettings.length === 0) {
      const unavailable = SS.createElement('div', {
        className: 'ss-setting-hint',
        textContent: 'Gender filters are unavailable on the current sidecar version.',
      });
      row.appendChild(unavailable);
      return row;
    }

    for (const setting of availableGenderSettings) {
      const label = SS.createElement('label', { className: 'ss-upstream-field-label' });
      const cb = SS.createElement('input', { attrs: { type: 'checkbox' } });
      cb.checked = !!getUpstreamGenderSettingValue(setting.key);
      cb.addEventListener('change', () => {
        if (settingsData?.categories?.upstream_sync?.settings?.[setting.key]) {
          settingsData.categories.upstream_sync.settings[setting.key].value = cb.checked;
        }
        debouncedSave(setting.key, cb.checked, row);
      });
      label.appendChild(cb);
      label.appendChild(document.createTextNode(' ' + setting.label));
      fieldsGrid.appendChild(label);
    }

    row.appendChild(fieldsGrid);
    return row;
  }

  async function renderUpstreamSyncCategory() {
    const section = SS.createElement('div', { className: 'ss-settings-category' });

    const header = SS.createElement('div', {
      className: 'ss-settings-cat-header',
      innerHTML: '<h2>Upstream Sync</h2>',
    });
    section.appendChild(header);

    const body = SS.createElement('div', { className: 'ss-settings-cat-body' });
    body.appendChild(renderPerformerGenderSettingsRow());

    const loadingRow = SS.createElement('div', {
      className: 'ss-setting-row ss-setting-hint',
      textContent: 'Loading stash-box endpoints...',
    });
    body.appendChild(loadingRow);
    section.appendChild(body);

    // Load endpoints asynchronously
    try {
      const [configResult, priorityResult] = await Promise.all([
        SS.stashQuery(`
          query { configuration { general { stashBoxes { endpoint name } } } }
        `),
        apiCall('endpoint_priorities_get'),
      ]);
      const allEndpoints = configResult?.configuration?.general?.stashBoxes || [];
      const disabledUrls = new Set((priorityResult.disabled || []).map(d => d.endpoint));
      const endpoints = allEndpoints.filter(ep => !disabledUrls.has(ep.endpoint));

      body.removeChild(loadingRow);

      if (allEndpoints.length === 0) {
        const emptyRow = SS.createElement('div', {
          className: 'ss-setting-row ss-setting-hint',
          textContent: 'No stash-box endpoints configured. Configure them in Stash Settings \u2192 Metadata Providers.',
        });
        body.appendChild(emptyRow);
      } else if (endpoints.length === 0) {
        const emptyRow = SS.createElement('div', {
          className: 'ss-setting-row ss-setting-hint',
          textContent: 'All endpoints are disabled. Enable endpoints in the Endpoint Priority section above.',
        });
        body.appendChild(emptyRow);
      } else {
        const helpRow = SS.createElement('div', {
          className: 'ss-setting-row ss-setting-hint',
          textContent: 'Configure which fields are monitored for upstream changes per endpoint. Disabled endpoints are hidden.',
        });
        body.appendChild(helpRow);

        for (const ep of endpoints) {
          body.appendChild(renderEndpointFieldConfig(ep));
        }
      }
    } catch (e) {
      loadingRow.textContent = `Could not load stash-box endpoints: ${e.message}`;
    }

    return section;
  }

  function renderEndpointFieldConfig(ep) {
    const displayName = ep.name || (() => { try { return new URL(ep.endpoint).hostname; } catch (_) { return ep.endpoint; } })();
    const row = SS.createElement('div', { className: 'ss-setting-row-vertical' });

    // Header row with name and toggle
    const headerRow = SS.createElement('div', { className: 'ss-setting-row-header' });

    const info = SS.createElement('div', { className: 'ss-setting-info' });
    info.innerHTML = `
      <label class="ss-setting-label">${displayName}</label>
      <span class="ss-setting-desc">${ep.endpoint}</span>
    `;
    headerRow.appendChild(info);

    const toggleBtn = SS.createElement('button', {
      className: 'ss-setting-reset',
      textContent: 'Show Fields',
    });
    headerRow.appendChild(toggleBtn);
    row.appendChild(headerRow);

    // Fields wrapper (hidden initially)
    const fieldsWrapper = SS.createElement('div', { className: 'ss-upstream-fields-wrapper' });
    fieldsWrapper.style.display = 'none';
    row.appendChild(fieldsWrapper);

    let fieldsLoaded = false;

    toggleBtn.addEventListener('click', async () => {
      const isHidden = fieldsWrapper.style.display === 'none';
      fieldsWrapper.style.display = isHidden ? 'block' : 'none';
      toggleBtn.textContent = isHidden ? 'Hide Fields' : 'Show Fields';

      if (isHidden && !fieldsLoaded) {
        fieldsLoaded = true;
        const loading = SS.createElement('div', {
          className: 'ss-setting-hint',
          textContent: 'Loading field config...',
        });
        fieldsWrapper.appendChild(loading);

        try {
          const fieldConfig = await apiCall('rec_get_field_config', { endpoint: ep.endpoint });
          loading.remove();

          const fieldsGrid = SS.createElement('div', {
            className: 'ss-upstream-fields-grid',
          });

          const sortedFields = Object.entries(fieldConfig.fields).sort(([, a], [, b]) => a.label.localeCompare(b.label));

          for (const [fieldName, config] of sortedFields) {
            const label = SS.createElement('label', { className: 'ss-upstream-field-label' });
            const cb = SS.createElement('input', { attrs: { type: 'checkbox' } });
            cb.dataset.field = fieldName;
            cb.checked = config.enabled;
            label.appendChild(cb);
            label.appendChild(document.createTextNode(' ' + config.label));
            fieldsGrid.appendChild(label);
          }
          fieldsWrapper.appendChild(fieldsGrid);

          // Save button
          const actionsDiv = SS.createElement('div', { className: 'ss-setting-control' });
          const saveBtn = SS.createElement('button', {
            className: 'ss-btn ss-btn-primary ss-btn-sm',
            textContent: 'Save Field Config',
          });
          const saveStatus = SS.createElement('span', { className: 'ss-setting-hint' });

          saveBtn.addEventListener('click', async () => {
            saveBtn.disabled = true;
            saveStatus.textContent = 'Saving...';
            const configs = {};
            fieldsGrid.querySelectorAll('input[type="checkbox"]').forEach(cb => {
              configs[cb.dataset.field] = cb.checked;
            });
            try {
              await apiCall('rec_set_field_config', { endpoint: ep.endpoint, field_configs: configs });
              saveStatus.textContent = 'Saved!';
              saveStatus.style.color = '#22c55e';
              setTimeout(() => { saveStatus.textContent = ''; saveStatus.style.color = ''; }, 2000);
            } catch (e) {
              saveStatus.textContent = `Error: ${e.message}`;
              saveStatus.style.color = '#ef4444';
            }
            saveBtn.disabled = false;
          });

          actionsDiv.appendChild(saveBtn);
          actionsDiv.appendChild(saveStatus);
          fieldsWrapper.appendChild(actionsDiv);

        } catch (e) {
          loading.textContent = `Error loading field config: ${e.message}`;
        }
      }
    });

    return row;
  }

  // ==================== Job Schedules ====================

  let scheduleTimeouts = {};

  function saveSchedule(type, enabled, intervalHours, row) {
    if (scheduleTimeouts[type]) {
      clearTimeout(scheduleTimeouts[type]);
    }
    scheduleTimeouts[type] = setTimeout(async () => {
      try {
        await apiCall('queue_update_schedule', {
          type,
          enabled,
          interval_hours: intervalHours,
        });
        showSaveIndicator(row, 'Saved');
      } catch (e) {
        showSaveIndicator(row, 'Error', true);
        console.error(`Failed to save schedule ${type}:`, e);
      }
    }, 500);
  }

  async function renderSchedulesCategory(container) {
    const section = SS.createElement('div', { className: 'ss-settings-category' });

    const header = SS.createElement('div', {
      className: 'ss-settings-cat-header',
      innerHTML: '<h2>Job Schedules</h2>',
    });
    section.appendChild(header);

    const body = SS.createElement('div', { className: 'ss-settings-cat-body' });

    const desc = SS.createElement('div', {
      className: 'ss-setting-row ss-setting-hint',
      textContent: 'Configure automatic job schedules. Jobs will run at the specified intervals.',
    });
    body.appendChild(desc);

    try {
      const [schedulesResult, typesResult] = await Promise.all([
        apiCall('queue_schedules'),
        apiCall('queue_types'),
      ]);

      const schedules = schedulesResult.schedules || [];
      const types = typesResult.types || [];

      if (schedules.length === 0) {
        const emptyRow = SS.createElement('div', {
          className: 'ss-setting-row ss-setting-hint',
          textContent: 'No schedules configured. They will be created automatically on next sidecar restart.',
        });
        body.appendChild(emptyRow);
        section.appendChild(body);
        container.appendChild(section);
        return;
      }

      for (const schedule of schedules) {
        const typeInfo = types.find(t => t.type_id === schedule.type);
        const displayName = typeInfo ? typeInfo.display_name : schedule.type;
        const description = typeInfo?.description || '';
        const allowedIntervals = typeInfo?.allowed_intervals || [];

        const row = SS.createElement('div', { className: 'ss-setting-row' });

        const info = SS.createElement('div', { className: 'ss-setting-info' });
        info.innerHTML = `
          <label class="ss-setting-label">${displayName}</label>
          <span class="ss-setting-desc">${description}</span>
        `;
        row.appendChild(info);

        const control = SS.createElement('div', { className: 'ss-setting-control' });

        // Enable toggle
        const toggle = SS.createElement('label', { className: 'ss-toggle' });
        const checkbox = SS.createElement('input', {
          attrs: { type: 'checkbox' },
        });
        checkbox.checked = !!schedule.enabled;
        const slider = SS.createElement('span', { className: 'ss-toggle-slider' });
        toggle.appendChild(checkbox);
        toggle.appendChild(slider);
        control.appendChild(toggle);

        // Interval select dropdown
        const select = SS.createElement('select', { className: 'ss-select' });
        const currentHours = schedule.interval_hours || 24;

        for (const interval of allowedIntervals) {
          const option = SS.createElement('option', {
            attrs: { value: String(interval.hours) },
            textContent: interval.label,
          });
          if (interval.hours === currentHours) {
            option.selected = true;
          }
          select.appendChild(option);
        }

        // If current value isn't in the allowed list, add it as a fallback
        if (allowedIntervals.length > 0 && !allowedIntervals.some(i => i.hours === currentHours)) {
          const fallback = SS.createElement('option', {
            attrs: { value: String(currentHours), selected: 'selected' },
            textContent: `Every ${currentHours} hours`,
          });
          select.insertBefore(fallback, select.firstChild);
        }

        select.disabled = !schedule.enabled;
        control.appendChild(select);

        // Auto-save on toggle change
        checkbox.addEventListener('change', () => {
          select.disabled = !checkbox.checked;
          saveSchedule(schedule.type, checkbox.checked, parseFloat(select.value), row);
        });

        // Auto-save on interval change
        select.addEventListener('change', () => {
          saveSchedule(schedule.type, checkbox.checked, parseFloat(select.value), row);
        });

        row.appendChild(control);
        body.appendChild(row);
      }
    } catch (e) {
      const errorRow = SS.createElement('div', {
        className: 'ss-setting-row ss-setting-hint',
        textContent: `Failed to load schedules: ${e.message}`,
      });
      errorRow.style.color = '#ef4444';
      body.appendChild(errorRow);
    }

    section.appendChild(body);
    container.appendChild(section);
  }

  // ==================== Save Logic ====================

  function debouncedSave(key, value, row, onSave = null) {
    if (saveTimeouts[key]) {
      clearTimeout(saveTimeouts[key]);
    }
    saveTimeouts[key] = setTimeout(async () => {
      try {
        await SettingsAPI.update(key, value);
        showSaveIndicator(row, 'Saved');
        if (onSave) onSave();
      } catch (e) {
        showSaveIndicator(row, 'Error', true);
        console.error(`Failed to save ${key}:`, e);
      }
    }, 500);
  }

  function _refreshLogFilesIfVisible() {
    const logSection = document.getElementById('ss-log-files-section');
    if (logSection) refreshLogFilesList(logSection);
  }

  function showSaveIndicator(row, text, isError = false) {
    if (!row) return;
    // Remove any existing indicator
    const existing = row.querySelector('.ss-save-indicator');
    if (existing) existing.remove();

    const indicator = SS.createElement('span', {
      className: `ss-save-indicator ${isError ? 'ss-save-error' : 'ss-save-ok'}`,
      textContent: text,
    });
    row.appendChild(indicator);
    setTimeout(() => indicator.remove(), 2000);
  }

  // ==================== Plugin Page Integration ====================

  function injectSettingsTab() {
    const route = SS.getRoute();
    if (route.type !== 'plugin') return;

    // Wait for the recommendations module to create the dashboard
    const existing = document.getElementById('ss-recommendations');
    if (!existing) {
      // Recommendations not yet injected, wait
      return;
    }

    // Check if settings tab already exists
    if (document.getElementById('ss-settings')) return;

    const dashboard = existing;

    // Look for or create a top-level tab bar
    let tabBar = dashboard.querySelector('.ss-page-tabs');
    if (!tabBar) {
      const initialTab = SS.getTabFromUrl();

      // Create page-level tabs (Recommendations | Settings)
      tabBar = SS.createElement('div', { className: 'ss-page-tabs' });

      const recTab = SS.createElement('button', {
        className: `ss-page-tab ${initialTab === 'recommendations' ? 'active' : ''}`,
        textContent: 'Recommendations',
        attrs: { 'data-tab': 'recommendations' },
      });

      const settingsTab = SS.createElement('button', {
        className: `ss-page-tab ${initialTab === 'settings' ? 'active' : ''}`,
        textContent: 'Settings',
        attrs: { 'data-tab': 'settings' },
      });

      tabBar.appendChild(recTab);
      tabBar.appendChild(settingsTab);

      // Insert tab bar after the app header
      const appHeader = dashboard.querySelector('.ss-app-header');
      if (appHeader) {
        appHeader.after(tabBar);
      } else {
        dashboard.insertBefore(tabBar, dashboard.firstChild);
      }

      // Create settings panel
      const settingsPanel = createSettingsContainer();
      settingsPanel.style.display = initialTab === 'settings' ? '' : 'none';
      dashboard.appendChild(settingsPanel);

      // Wrap content (skip app header) as the "recommendations" panel
      const recContent = SS.createElement('div', {
        className: 'ss-page-panel',
        attrs: { 'data-panel': 'recommendations' },
      });
      const children = Array.from(dashboard.children);
      for (const child of children) {
        if (child !== tabBar && child !== settingsPanel && !child.classList.contains('ss-app-header')) {
          recContent.appendChild(child);
        }
      }
      dashboard.insertBefore(recContent, settingsPanel);

      // Show/hide based on initial tab
      recContent.style.display = initialTab === 'recommendations' ? '' : 'none';

      // Lazy load settings if starting on that tab
      if (initialTab === 'settings' && !settingsPanel.dataset.loaded) {
        settingsPanel.dataset.loaded = 'true';
        renderSettings(settingsPanel);
      }

      // If starting on settings tab, begin periodic log file refresh
      let logRefreshInterval = null;
      function startLogRefresh() {
        if (logRefreshInterval) return;
        _refreshLogFilesIfVisible();
        logRefreshInterval = setInterval(_refreshLogFilesIfVisible, 5000);
      }
      function stopLogRefresh() {
        if (logRefreshInterval) {
          clearInterval(logRefreshInterval);
          logRefreshInterval = null;
        }
      }
      if (initialTab === 'settings') {
        // Delay slightly so renderSettings() populates the log section first
        setTimeout(startLogRefresh, 500);
      }

      // Tab switching
      tabBar.addEventListener('click', (e) => {
        const btn = e.target.closest('.ss-page-tab');
        if (!btn) return;

        const tabName = btn.dataset.tab;

        // Update active state
        tabBar.querySelectorAll('.ss-page-tab').forEach(t => t.classList.remove('active'));
        btn.classList.add('active');

        // Update URL
        SS.setTabInUrl(tabName);

        // Show/hide panels
        recContent.style.display = tabName === 'recommendations' ? '' : 'none';
        settingsPanel.style.display = tabName === 'settings' ? '' : 'none';

        if (tabName === 'settings') {
          if (!settingsPanel.dataset.loaded) {
            // First view: full render.
            settingsPanel.dataset.loaded = 'true';
            renderSettings(settingsPanel);
          } else {
            // Already rendered once -- just refresh fingerprint/database
            // coverage stats rather than the whole page, so e.g. running
            // fingerprint generation from the Operations tab and switching
            // back here shows current numbers instead of whatever was
            // loaded the first time this tab was opened.
            refreshIdDatabaseSection();
          }
          // Refresh log file list immediately and start 5-second polling
          setTimeout(startLogRefresh, 200);
        } else {
          stopLogRefresh();
        }

        // Refresh recommendation counters when switching back to that tab
        if (tabName === 'recommendations' && window.StashSenseRecommendations) {
          window.StashSenseRecommendations.refreshCounts();
        }
      });

      // Stop polling when navigating away from the plugin page entirely
      SS.onLeavePlugin(stopLogRefresh);
    }
  }

  // ==================== Initialization ====================

  function cleanup() {
    const settings = document.getElementById('ss-settings');
    if (settings) settings.remove();

    // Clear pending save timeouts
    for (const key of Object.keys(saveTimeouts)) {
      clearTimeout(saveTimeouts[key]);
    }
    saveTimeouts = {};

    for (const key of Object.keys(scheduleTimeouts)) {
      clearTimeout(scheduleTimeouts[key]);
    }
    scheduleTimeouts = {};

    settingsData = null;
    systemInfo = null;
  }

  function init() {
    // Try to inject after recommendations module
    const tryInject = () => {
      if (SS.getRoute().type === 'plugin') {
        setTimeout(injectSettingsTab, 600);
      }
    };

    tryInject();

    SS.onNavigate((route) => {
      if (route.type === 'plugin') {
        setTimeout(injectSettingsTab, 600);
      }
    });

    // Clean up when leaving plugin page
    SS.onLeavePlugin(cleanup);

    console.log(`[${SS.PLUGIN_NAME}] Settings module loaded`);
  }

  // Export
  window.StashSenseSettings = {
    API: SettingsAPI,
    refresh: () => {
      const container = document.getElementById('ss-settings');
      if (container) renderSettings(container);
    },
    init,
  };

  init();
})();
