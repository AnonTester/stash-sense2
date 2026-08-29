/**
 * Stash Sense Recommendations Module
 * Dashboard UI for viewing and acting on recommendations
 */
(function() {
  'use strict';

  const SS = window.StashSense;
  if (!SS) {
    console.error('[Stash Sense] Core module not loaded');
    return;
  }

  // ==================== Recommendations API ====================
  // All API calls go through the Python backend to bypass CSP restrictions

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

  const RecommendationsAPI = {
    async getCounts() {
      return apiCall('rec_counts');
    },

    async getList(params = {}) {
      return apiCall('rec_list', {
        status: params.status,
        type: params.type,
        limit: params.limit || 100,
        offset: params.offset || 0,
      });
    },

    async getOne(id) {
      return apiCall('rec_get', { rec_id: id });
    },

    async resolve(id, action, details = null) {
      return apiCall('rec_resolve', {
        rec_id: id,
        action,
        details,
      });
    },

    async dismiss(id, reason = null) {
      return apiCall('rec_dismiss', {
        rec_id: id,
        reason,
      });
    },

    async getAnalysisTypes() {
      return apiCall('rec_analysis_types');
    },

    async runAnalysis(type, full = false) {
      return apiCall('rec_run_analysis', { analysis_type: type, full });
    },

    async getAnalysisRuns(type = null, limit = 10) {
      return apiCall('rec_analysis_runs', {
        analysis_type: type,
        limit,
      });
    },

    async getStashStatus() {
      return apiCall('rec_stash_status');
    },

    async getSidecarStatus() {
      const settings = await SS.getSettings();
      try {
        const health = await SS.runPluginOperation('health', {
          sidecar_url: settings.sidecarUrl,
        });
        return {
          connected: !health.error,
          url: settings.sidecarUrl,
          error: health.error || null,
          version: health.version || null,
        };
      } catch (e) {
        return {
          connected: false,
          url: settings.sidecarUrl,
          error: e.message,
        };
      }
    },

    // Actions
    async mergePerformers(destinationId, sourceIds) {
      return apiCall('rec_merge_performers', {
        destination_id: destinationId,
        source_ids: sourceIds,
      });
    },

    async deleteSceneFiles(sceneId, fileIdsToDelete, keepFileId, allFileIds) {
      return apiCall('rec_delete_files', {
        scene_id: sceneId,
        file_ids_to_delete: fileIdsToDelete,
        keep_file_id: keepFileId,
        all_file_ids: allFileIds,
      });
    },

    async mergeScenes(destinationId, sourceIds) {
      return apiCall('rec_merge_scenes', {
        destination_id: destinationId,
        source_ids: sourceIds,
      });
    },

    async deleteScene(sceneId, deleteFile = false) {
      return apiCall('rec_delete_scene', {
        scene_id: sceneId,
        delete_file: deleteFile,
      });
    },

    async getSceneDetail(sceneId) {
      return apiCall('rec_get_scene', { scene_id: sceneId });
    },

    async mergeDuplicateSceneGroup(sourceSceneId, selectedMatchSceneIds, selectedRecommendationIds, unselectedRecommendationIds) {
      return apiCall('rec_merge_duplicate_scene_group', {
        source_scene_id: sourceSceneId,
        selected_match_scene_ids: selectedMatchSceneIds,
        selected_recommendation_ids: selectedRecommendationIds,
        unselected_recommendation_ids: unselectedRecommendationIds || [],
      });
    },

    async deleteDuplicateSceneGroup(sourceSceneId, recommendationIds, deleteFile = false) {
      return apiCall('rec_delete_duplicate_scene_group', {
        source_scene_id: sourceSceneId,
        recommendation_ids: recommendationIds,
        delete_file: deleteFile,
      });
    },

    async deleteDuplicateSceneMatch(sourceSceneId, matchSceneId, recommendationId, deleteFile = false) {
      return apiCall('rec_delete_duplicate_scene_match', {
        source_scene_id: sourceSceneId,
        match_scene_id: matchSceneId,
        recommendation_id: recommendationId,
        delete_file: deleteFile,
      });
    },

    async mergeSourceIntoDuplicateSceneMatch(sourceSceneId, keeperMatchSceneId, keeperRecommendationId, otherMatches = []) {
      return apiCall('rec_merge_source_into_duplicate_scene_match', {
        source_scene_id: sourceSceneId,
        keeper_match_scene_id: keeperMatchSceneId,
        keeper_recommendation_id: keeperRecommendationId,
        other_matches: otherMatches,
      });
    },

    async dismissDuplicateSceneGroup(recommendationIds, reason = null) {
      return apiCall('rec_dismiss_duplicate_scene_group', {
        recommendation_ids: recommendationIds,
        reason,
      });
    },

    // Fingerprint operations
    async getFingerprintStatus() {
      return apiCall('fp_status');
    },

    async startFingerprintGeneration(options = {}) {
      // Only send params that are explicitly provided; sidecar defaults from face_config.py
      const params = { refresh_outdated: options.refreshOutdated ?? true };
      if (options.numFrames != null) params.num_frames = options.numFrames;
      if (options.minFaceSize != null) params.min_face_size = options.minFaceSize;
      if (options.maxDistance != null) params.max_distance = options.maxDistance;
      return apiCall('fp_generate', params);
    },

    async getFingerprintProgress() {
      return apiCall('fp_progress');
    },

    async stopFingerprintGeneration() {
      return apiCall('fp_stop');
    },

    // Upstream sync operations
    async updatePerformer(performerId, fields) {
      return apiCall('rec_update_performer', { performer_id: performerId, fields });
    },

    async updateTag(tagId, fields) {
      return apiCall('rec_update_tag', { tag_id: tagId, fields });
    },

    async updateStudio(studioId, fields, endpoint) {
      return apiCall('rec_update_studio', { studio_id: studioId, fields, endpoint });
    },

    async dismissUpstream(recId, reason, permanent) {
      return apiCall('rec_dismiss_upstream', { rec_id: recId, reason, permanent: !!permanent });
    },

    async batchDismiss(type, permanent) {
      return apiCall('rec_batch_dismiss', { type, permanent: !!permanent });
    },

    async searchEntities(entityType, query, endpoint) {
      return apiCall('rec_search_entities', { entity_type: entityType, query, endpoint });
    },

    async findLinkedEntity(entityType, endpoint, stashboxId) {
      return apiCall('rec_find_linked_entity', {
        entity_type: entityType,
        endpoint,
        stashbox_id: stashboxId,
      });
    },

    async getUpstreamScenePreview(endpoint, stashboxId) {
      return apiCall('rec_upstream_scene_preview', {
        endpoint,
        stashbox_id: stashboxId,
      });
    },

    async linkEntity(entityType, entityId, endpoint, stashboxId) {
      return apiCall('rec_link_entity', { entity_type: entityType, entity_id: entityId, endpoint, stashbox_id: stashboxId });
    },

    async createPerformer(stashboxData, endpoint, stashboxId) {
      return apiCall('rec_create_performer', { stashbox_data: stashboxData, endpoint, stashbox_id: stashboxId });
    },

    async createTag(stashboxData, endpoint, stashboxId) {
      return apiCall('rec_create_tag', { stashbox_data: stashboxData, endpoint, stashbox_id: stashboxId });
    },

    async createStudio(stashboxData, endpoint, stashboxId) {
      return apiCall('rec_create_studio', { stashbox_data: stashboxData, endpoint, stashbox_id: stashboxId });
    },

    async updateScene(sceneId, fields, performerIds, tagIds, studioId) {
      return apiCall('rec_update_scene', {
        scene_id: sceneId,
        fields,
        performer_ids: performerIds || null,
        tag_ids: tagIds || null,
        studio_id: studioId || null,
      });
    },

    async getFieldConfig(endpoint) {
      return apiCall('rec_get_field_config', { endpoint });
    },

    async setFieldConfig(endpoint, fieldConfigs) {
      return apiCall('rec_set_field_config', { endpoint, field_configs: fieldConfigs });
    },

    // Database info
    async getDatabaseInfo() {
      const settings = await SS.getSettings();
      return SS.runPluginOperation('database_info', { sidecar_url: settings.sidecarUrl });
    },

    // Database update operations
    async checkUpdate() {
      const settings = await SS.getSettings();
      return SS.runPluginOperation('db_check_update', { sidecar_url: settings.sidecarUrl });
    },
    async startUpdate() {
      const settings = await SS.getSettings();
      return SS.runPluginOperation('db_update', { sidecar_url: settings.sidecarUrl });
    },
    async getUpdateStatus() {
      const settings = await SS.getSettings();
      return SS.runPluginOperation('db_update_status', { sidecar_url: settings.sidecarUrl });
    },

    // User settings
    async getUserSetting(key) {
      const result = await apiCall('user_get_setting', { key });
      return result.value;
    },

    async setUserSetting(key, value) {
      return apiCall('user_set_setting', { key, value });
    },

    async getAllUserSettings() {
      const result = await apiCall('user_get_all_settings');
      return result.settings || {};
    },

    async acceptFingerprintMatch(recommendationId, sceneId, endpoint, stashId) {
      return apiCall('rec_accept_fingerprint_match', {
        recommendation_id: recommendationId,
        scene_id: sceneId,
        endpoint: endpoint,
        stash_id: stashId,
      });
    },

    async acceptAllFingerprintMatches(endpoint) {
      return apiCall('rec_accept_all_fingerprint_matches', {
        endpoint: endpoint || null,
      });
    },

    async acceptSceneFaceMatches(sceneId, recommendationIds) {
      return apiCall('rec_accept_scene_face_matches', {
        scene_id: sceneId,
        selections: recommendationIds.map(id => ({ recommendation_id: id })),
      });
    },

    async rejectAllSceneFaceMatches(sceneId) {
      return apiCall('rec_reject_all_scene_face_matches', { scene_id: sceneId });
    },

    async dismissSceneFaceMatch(recId) {
      return apiCall('rec_dismiss_scene_face_match', { rec_id: recId });
    },

    async acceptAllSceneTagOnlyChanges() {
      return apiCall('rec_accept_all_scene_tag_only_changes', {});
    },

    async acceptAllPerformerUrlOnlyChanges() {
      return apiCall('rec_accept_all_performer_url_only_changes', {});
    },

    async acceptSceneTagOnlyChange(recId) {
      return apiCall('rec_accept_scene_tag_only_change', { rec_id: recId });
    },

    async acceptSceneChange(recId) {
      return apiCall('rec_accept_scene_change', { rec_id: recId });
    },

    async sceneTagOnlyStats() {
      return apiCall('rec_scene_tag_only_stats', {});
    },
    async performerUrlOnlyStats() {
      return apiCall('rec_performer_url_only_stats', {});
    },
    async fingerprintMatchStats() {
      return apiCall('rec_fingerprint_match_stats', {});
    },
    async bulkAcceptStats(type) {
      return apiCall('rec_bulk_accept_stats', { type });
    },
  };

  /**
   * Convert ALL_CAPS enum values to Title Case for display.
   * e.g. "BROWN" -> "Brown", "EYE_COLOR" -> "Eye Color", "NATURAL" -> "Natural"
   * Returns original value if not an ALL_CAPS string.
   */
  function formatRecTimestamp(isoStr) {
    if (!isoStr) return '';
    try {
      const d = new Date(isoStr.endsWith('Z') ? isoStr : isoStr + 'Z');
      if (isNaN(d)) return isoStr;
      return d.toLocaleString(undefined, {
        year: 'numeric', month: 'short', day: 'numeric',
        hour: '2-digit', minute: '2-digit',
      });
    } catch {
      return isoStr;
    }
  }

  function normalizeEnumValue(val) {
    if (typeof val !== 'string') return val;
    // Only transform if the string is ALL_CAPS (with optional underscores)
    if (!/^[A-Z][A-Z0-9_]*$/.test(val)) return val;
    return val
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ');
  }

  function isNullishString(val) {
    return typeof val === 'string' && val.trim().toLowerCase() === 'null';
  }

  function isEmptyLikeValue(val) {
    return val === null || val === undefined || val === '' || val === 0 || isNullishString(val);
  }

  // Filter out false-positive changes where values are effectively equal
  function filterRealChanges(changes) {
    return (changes || []).filter(change => {
      const local = change.local_value;
      const upstream = change.upstream_value;
      // Both null/undefined/empty string/zero/"null"
      if (isEmptyLikeValue(local) && isEmptyLikeValue(upstream)) return false;
      // String case-insensitive comparison
      if (typeof local === 'string' && typeof upstream === 'string') {
        const localNorm = local.trim().toLowerCase();
        const upstreamNorm = upstream.trim().toLowerCase();
        if (localNorm === upstreamNorm) return false;
        if ((localNorm === '' || localNorm === 'null') && (upstreamNorm === '' || upstreamNorm === 'null')) return false;
      }
      // List comparison (alias_list, urls) - case-insensitive set equality with trailing slash normalization
      if (Array.isArray(local) && Array.isArray(upstream)) {
        const norm = (v) => {
          const text = String(v).trim().toLowerCase().replace(/\/+$/, '');
          return text === 'null' ? '' : text;
        };
        const localSet = new Set(local.map(norm).filter(Boolean));
        const upstreamSet = new Set(upstream.map(norm).filter(Boolean));
        if (localSet.size === upstreamSet.size && [...localSet].every(v => upstreamSet.has(v))) return false;
      }
      // Strict equality for other types
      if (local === upstream) return false;
      return true;
    });
  }

  function isTagUrlCodeOnlySceneChangeDetails(details) {
    if (!details || typeof details !== 'object') return false;

    const simpleChanges = filterRealChanges(details.changes || []);
    const allowedFields = new Set(['code', 'urls']);
    if (simpleChanges.some(change => !allowedFields.has(change.field))) return false;

    if (details.studio_change) return false;

    const performerChanges = details.performer_changes || {};
    if ((performerChanges.added || []).length > 0) return false;
    if ((performerChanges.removed || []).length > 0) return false;

    const tagChanges = details.tag_changes || {};
    const hasTagChanges = (tagChanges.added || []).length > 0 || (tagChanges.removed || []).length > 0;
    const hasSimpleChanges = simpleChanges.length > 0;
    return hasTagChanges || hasSimpleChanges;
  }

  function isUrlOnlyPerformerChangeDetails(details) {
    if (!details || typeof details !== 'object') return false;
    const changes = filterRealChanges(details.changes || []);
    if (changes.length === 0) return false;
    return changes.every(change => change.field === 'urls');
  }

  function parseConfidencePercent(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return null;
    if (numeric <= 1) return numeric * 100;
    return numeric;
  }

  function getDuplicateSceneConfidencePercent(rec) {
    const detailConfidence = parseConfidencePercent(rec?.details?.confidence);
    if (detailConfidence !== null) return detailConfidence;
    const recommendationConfidence = parseConfidencePercent(rec?.confidence);
    return recommendationConfidence !== null ? recommendationConfidence : 0;
  }

  // Session cache for entities created during this session
  // Maps "endpoint|stashbox_id" -> local_id
  const entityCache = new Map();

  // ==================== State ====================

  let currentState = {
    view: 'dashboard', // dashboard, list, detail
    type: null,
    status: 'pending',
    page: 0,
    selectedRec: null,
    counts: null,
    // Cache of the last-fetched list page, keyed by "type:status:page" --
    // lets returning from a detail view (plain Back, or after resolving/
    // dismissing/matching one recommendation) skip a full re-fetch +
    // re-sort of the whole list. See renderList()'s cache-hit branch and
    // removeFromListCache(). Any real change of type/status/page just
    // naturally misses the cache since the key won't match.
    listCache: null,
  };

  // (Polling for analysis/fingerprint progress now handled by Operations tab)

  // ==================== Dashboard Container ====================

  function createDashboardContainer() {
    const existing = document.getElementById('ss-recommendations');
    if (existing) existing.remove();

    const container = SS.createElement('div', {
      id: 'ss-recommendations',
      className: 'ss-recommendations',
    });

    // Persistent app header (stays above tabs)
    const appHeader = SS.createElement('div', {
      className: 'ss-app-header',
    });
    appHeader.innerHTML = `
      <div class="ss-app-header-left">
        <h1>${SS.PLUGIN_NAME}</h1>
        <p class="ss-dashboard-subtitle">Library analysis and curation tools</p>
      </div>
      <div class="ss-app-header-right" id="ss-status-area"></div>
    `;
    container.appendChild(appHeader);

    // Content wrapper (views render inside this)
    const content = SS.createElement('div', {
      className: 'ss-dashboard-content',
    });
    container.appendChild(content);

    return container;
  }

  // ==================== Dashboard View ====================

  function updateStatusArea(sidecarStatus) {
    const statusArea = document.getElementById('ss-status-area');
    if (!statusArea) return;
    const connected = sidecarStatus?.connected || false;
    const sidecarVersion = sidecarStatus?.version || null;
    const pluginVersion = SS.PLUGIN_VERSION || null;

    let versionHtml = '';
    if (pluginVersion || sidecarVersion) {
      // Flag the sidecar only when it's actually below what this plugin
      // needs (MIN_SIDECAR_VERSION) -- not on any plugin/sidecar version
      // string mismatch. Versions drift independently by design (sidecar
      // only needs bumping when a plugin change actually requires a new
      // sidecar feature), so e.g. plugin v0.14.4 next to a perfectly
      // compatible sidecar v0.14.3 is not an error.
      const outdated = sidecarVersion && SS.compareVersions(sidecarVersion, SS.MIN_SIDECAR_VERSION) < 0;
      const pVer = pluginVersion ? `Plugin v${pluginVersion}` : '';
      const sVer = sidecarVersion ? `Sidecar v${sidecarVersion}` : '';

      if (outdated) {
        const pHtml = pVer ? `<span>${pVer}</span>` : '';
        const sHtml = sVer ? `<span class="ss-version-mismatch">${sVer}</span>` : '';
        versionHtml = [pHtml, sHtml].filter(Boolean).join(' <span class="ss-status-sep">-</span> ');
      } else {
        versionHtml = `<span>${[pVer, sVer].filter(Boolean).join(' - ')}</span>`;
      }
    }

    statusArea.className = `ss-app-header-right ${connected ? 'connected' : 'disconnected'}`;
    statusArea.innerHTML = `
      ${versionHtml}
      <span class="ss-status-dot"></span>
      <span class="ss-status-label">${connected ? 'Connected' : 'Disconnected'}</span>
      ${sidecarStatus?.error ? `<span class="ss-status-error">${sidecarStatus.error}</span>` : ''}
    `;
  }

  async function renderDashboard(mainContainer, content) {
    content.innerHTML = `
      <div class="ss-dashboard-loading">
        <div class="ss-spinner"></div>
        <p>Loading recommendations...</p>
      </div>
    `;

    let sidecarStatus = null;
    try {
      const [rawCounts, _sidecarStatus] = await Promise.all([
        RecommendationsAPI.getCounts().catch(() => null),
        RecommendationsAPI.getSidecarStatus(),
      ]);
      sidecarStatus = _sidecarStatus;

      const counts = rawCounts || {};
      currentState.counts = counts;

      // Update the persistent status area in the app header
      updateStatusArea(sidecarStatus);

      // Build type cards HTML
      const typeConfigs = {
        duplicate_performer: {
          title: 'Duplicate Performers',
          icon: `<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M16 11c1.66 0 2.99-1.34 2.99-3S17.66 5 16 5c-1.66 0-3 1.34-3 3s1.34 3 3 3zm-8 0c1.66 0 2.99-1.34 2.99-3S9.66 5 8 5C6.34 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z"/></svg>`,
          description: 'Performers sharing the same StashDB ID',
        },
        duplicate_scenes: {
          title: 'Duplicate Scenes',
          icon: `<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M4 6H2v14c0 1.1.9 2 2 2h14v-2H4V6zm16-4H8c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H8V4h12v12zm-6-1l4-4-1.4-1.4-1.6 1.6V6h-2v6.2l-1.6-1.6L10 12l4 4z"/></svg>`,
          description: 'Scenes that may be duplicates based on stash-box ID, faces, or metadata',
        },
        duplicate_scene_files: {
          title: 'Duplicate Scene Files',
          icon: `<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M4 6H2v14c0 1.1.9 2 2 2h14v-2H4V6zm16-4H8c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-8 12.5v-9l6 4.5-6 4.5z"/></svg>`,
          description: 'Scenes with multiple files attached',
        },
        upstream_performer_changes: {
          title: 'Upstream Performer Changes',
          icon: `<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M12 6V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z"/></svg>`,
          description: 'Performer fields updated on StashDB since last sync',
        },
        upstream_tag_changes: {
          title: 'Upstream Tag Changes',
          icon: `<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M21.41 11.58l-9-9C12.05 2.22 11.55 2 11 2H4c-1.1 0-2 .9-2 2v7c0 .55.22 1.05.59 1.42l9 9c.36.36.86.58 1.41.58.55 0 1.05-.22 1.41-.59l7-7c.37-.36.59-.86.59-1.41 0-.55-.23-1.06-.59-1.42zM5.5 7C4.67 7 4 6.33 4 5.5S4.67 4 5.5 4 7 4.67 7 5.5 6.33 7 5.5 7z"/></svg>`,
          description: 'Tag fields updated on StashDB since last sync',
        },
        upstream_studio_changes: {
          title: 'Upstream Studio Changes',
          icon: `<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M12 7V3H2v18h20V7H12zM6 19H4v-2h2v2zm0-4H4v-2h2v2zm0-4H4V9h2v2zm0-4H4V5h2v2zm4 12H8v-2h2v2zm0-4H8v-2h2v2zm0-4H8V9h2v2zm0-4H8V5h2v2zm10 12h-8v-2h2v-2h-2v-2h2v-2h-2V9h8v10zm-2-8h-2v2h2v-2zm0 4h-2v2h2v-2z"/></svg>`,
          description: 'Studio fields updated on StashDB since last sync',
        },
        upstream_scene_changes: {
          title: 'Upstream Scene Changes',
          icon: `<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M18 4l2 4h-3l-2-4h-2l2 4h-3l-2-4H8l2 4H7L5 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V4h-4z"/></svg>`,
          description: 'Scene fields and relationships updated on StashDB since last sync',
        },
        scene_fingerprint_match: {
          title: 'Scene Stash-Box Tagger',
          icon: `<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M3.9 12c0-1.71 1.39-3.1 3.1-3.1h4V7H7c-2.76 0-5 2.24-5 5s2.24 5 5 5h4v-1.9H7c-1.71 0-3.1-1.39-3.1-3.1zM8 13h8v-2H8v2zm9-6h-4v1.9h4c1.71 0 3.1 1.39 3.1 3.1s-1.39 3.1-3.1 3.1h-4V17h4c2.76 0 5-2.24 5-5s-2.24-5-5-5z"/></svg>`,
          description: 'Searches untagged Scenes on all Stash-Box endpoints',
        },
        scene_face_match: {
          title: 'Face Recommendations',
          icon: `<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4zm9-6l1.5-3L24 4l-1.5-1L21 0l-1 1.5L18 3l1.5 1L21 8zm-3.5-1.5L19 3l-1.5-.5L17 1l-.5 1.5L15 3l1.5.5L17 5z"/></svg>`,
          description: 'Performers identified in scenes that have none assigned yet',
        },
      };

      content.innerHTML = `
        <div class="ss-dashboard-types">
          <div class="ss-section-header">
            <h2>Recommendations</h2>
            <span class="ss-count-badge">${counts.total_pending}</span>
            <span id="ss-info-types"></span>
          </div>
          <div class="ss-type-cards"></div>
        </div>
      `;

      const typesInfoSlot = content.querySelector('#ss-info-types');
      if (typesInfoSlot) typesInfoSlot.appendChild(createInfoIcon(() => showHelpModal('Recommendations', HELP_REC_TYPES)));

      // Render type cards
      const typeCards = content.querySelector('.ss-type-cards');

      // Ensure all types are shown, even if no counts yet
      const allTypes = Object.keys(typeConfigs);
      for (const type of allTypes) {
        const config = typeConfigs[type];
        const typeCounts = counts.counts?.[type] || {};
        const pending = typeCounts.pending || 0;
        const resolved = typeCounts.resolved || 0;
        const dismissed = typeCounts.dismissed || 0;

        const card = SS.createElement('div', {
          className: 'ss-type-card',
          attrs: { 'data-type': type, tabindex: '0', role: 'button', 'aria-label': `View ${config.title}` },
          innerHTML: `
            <div class="ss-type-card-header">
              <span class="ss-type-icon">${config.icon}</span>
              <div class="ss-type-title-block">
                <h3>${config.title}</h3>
                <p>${config.description}</p>
              </div>
            </div>
            <div class="ss-type-card-footer">
              <div class="ss-type-counts">
                <div class="ss-count-item ss-count-pending">
                  <span class="ss-count-number">${pending}</span>
                  <span class="ss-count-label">pending</span>
                </div>
                <div class="ss-count-item ss-count-resolved">
                  <span class="ss-count-number">${resolved}</span>
                  <span class="ss-count-label">resolved</span>
                </div>
                <div class="ss-count-item ss-count-dismissed">
                  <span class="ss-count-number">${dismissed}</span>
                  <span class="ss-count-label">dismissed</span>
                </div>
              </div>
            </div>
          `,
        });

        const openType = () => {
          currentState.type = type;
          currentState.view = 'list';
          renderCurrentView(mainContainer);
        };
        card.addEventListener('click', openType);
        card.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            openType();
          }
        });

        typeCards.appendChild(card);
      }

    } catch (e) {
      updateStatusArea(sidecarStatus || { connected: false });

      content.innerHTML = `
        <div class="ss-error-state">
          <div class="ss-error-icon">
            <svg viewBox="0 0 24 24" width="48" height="48" fill="currentColor">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
            </svg>
          </div>
          <h2>Connection Error</h2>
          <p>Make sure the ${SS.PLUGIN_NAME} sidecar is running and configured correctly.</p>
          <button class="ss-btn ss-btn-primary" id="ss-retry-btn">Retry</button>
        </div>
      `;
      const retryBtn = content.querySelector('#ss-retry-btn');
      if (retryBtn) retryBtn.addEventListener('click', () => location.reload());
    }
  }

  // ==================== Batch Accept All ====================

  function computeBatchChanges(recommendations) {
    function smartDefault(localVal, upstreamVal) {
      const upstreamEmpty = isEmptyLikeValue(upstreamVal);
      if (!upstreamEmpty) return 'upstream';
      return 'local';
    }

    const results = [];
    // Recs where filterRealChanges() found a real local/upstream difference,
    // but every field's smart default still resolves to "keep local" (e.g.
    // upstream cleared a field local already has a value for -- smartDefault
    // deliberately never overwrites local with emptiness). Nothing to apply,
    // but unlike genuinely no-diff recs, these never hit the single-item
    // detail view's own filterRealChanges()===0 auto-resolve either, so
    // they'd otherwise sit pending forever, never touched by Accept All or
    // anything else. Caller resolves these as accepted_no_changes.
    const noOpRecIds = [];

    for (const rec of recommendations) {
      // Scenes use a dedicated per-rec accept endpoint with full relational resolution;
      // they are handled separately in the acceptAllBtn handler for upstream_scene_changes.
      if (rec.type === 'upstream_scene_changes') continue;

      const details = rec.details;
      const rawChanges = details.changes || [];
      const changes = filterRealChanges(rawChanges);
      if (changes.length === 0) continue;

      const isTag = rec.type === 'upstream_tag_changes';
      const isStudio = rec.type === 'upstream_studio_changes';
      const entityName = isStudio ? details.studio_name : (isTag ? details.tag_name : details.performer_name);
      const entityType = isStudio ? 'Studio' : (isTag ? 'Tag' : 'Performer');
      const entityId = isStudio ? details.studio_id : (isTag ? details.tag_id : details.performer_id);

      const fields = {};
      const fieldSummaries = [];

      for (const change of changes) {
        const mergeType = change.merge_type || 'simple';
        const fieldKey = change.field;

        if (mergeType === 'alias_list') {
          const allAliases = new Set();
          if (Array.isArray(change.local_value)) change.local_value.forEach(a => allAliases.add(a));
          if (Array.isArray(change.upstream_value)) change.upstream_value.forEach(a => allAliases.add(a));
          const merged = [...allAliases];
          fields[fieldKey] = merged;
          const localCount = (change.local_value || []).length;
          const newCount = merged.length - localCount;
          if (newCount > 0) {
            const itemLabel = fieldKey === 'urls' ? 'URLs' : 'aliases';
            fieldSummaries.push({ field: change.field_label || fieldKey, desc: `+${newCount} ${itemLabel} merged` });
          }
        } else {
          const choice = smartDefault(change.local_value, change.upstream_value);
          const resultVal = choice === 'upstream' ? change.upstream_value : change.local_value;
          const localStr = formatFieldValue(change.local_value) === '(empty)' ? '' : String(change.local_value || '');
          const resultStr = formatFieldValue(resultVal) === '(empty)' ? '' : String(resultVal || '');

          if (resultStr === localStr) {
            if (mergeType === 'name' && choice !== 'upstream') {
              if (change.upstream_value) {
                fields['_alias_add'] = fields['_alias_add'] || [];
                fields['_alias_add'].push(String(change.upstream_value));
                fieldSummaries.push({ field: 'Alias', desc: `+ "${change.upstream_value}"` });
              }
            }
            continue;
          }

          fields[fieldKey] = resultStr;

          if (mergeType === 'name' && choice === 'upstream') {
            if (change.local_value) {
              fields['_alias_add'] = fields['_alias_add'] || [];
              fields['_alias_add'].push(String(change.local_value));
            }
          }

          const fromDisplay = formatFieldValue(change.local_value);
          const toDisplay = formatFieldValue(resultVal);
          fieldSummaries.push({
            field: change.field_label || fieldKey,
            desc: `${fromDisplay} \u2192 ${toDisplay}`,
          });
        }
      }

      if (fieldSummaries.length > 0) {
        results.push({ rec, entityName, entityType, entityId, fields, changes: fieldSummaries });
      } else {
        noOpRecIds.push(rec.id);
      }
    }

    return { batchChanges: results, noOpRecIds };
  }

  function showAcceptAllModal(batchChanges) {
    return new Promise((resolve) => {
      const totalChanges = batchChanges.reduce((sum, item) => sum + item.changes.length, 0);

      const overlay = document.createElement('div');
      overlay.className = 'ss-modal-overlay';
      overlay.innerHTML = `
        <div class="ss-accept-all-modal">
          <div class="ss-modal-header">
            <h3>Accept All Changes</h3>
            <button class="ss-modal-close">&times;</button>
          </div>
          <div class="ss-modal-body">
            <p>This will apply smart defaults to <strong>${batchChanges.length}</strong> ${batchChanges.length === 1 ? 'entity' : 'entities'} (${totalChanges} field ${totalChanges === 1 ? 'change' : 'changes'}). Upstream values are preferred when available; alias lists are merged.</p>
            ${batchChanges.map(item => `
              <div class="ss-batch-entity-group">
                <span class="ss-batch-entity-name">${escapeHtml(item.entityName)}</span>
                <span class="ss-batch-entity-type">${escapeHtml(item.entityType)}</span>
                <ul class="ss-batch-changes-list">
                  ${item.changes.map(c => `<li><span class="ss-batch-field-name">${escapeHtml(c.field)}:</span> ${escapeHtml(c.desc)}</li>`).join('')}
                </ul>
              </div>
            `).join('')}
          </div>
          <div class="ss-modal-footer">
            <button class="ss-btn ss-btn-secondary" id="ss-modal-cancel">Cancel</button>
            <button class="ss-accept-all-btn" id="ss-modal-confirm">Accept ${batchChanges.length} ${batchChanges.length === 1 ? 'Change' : 'Changes'}</button>
          </div>
        </div>
      `;

      document.body.appendChild(overlay);

      function close(result) {
        if (!result) overlay.remove();
        resolve(result);
      }

      overlay.querySelector('.ss-modal-close').addEventListener('click', () => close(false));
      overlay.querySelector('#ss-modal-cancel').addEventListener('click', () => close(false));
      overlay.querySelector('#ss-modal-confirm').addEventListener('click', () => close(true));
      overlay.addEventListener('click', (e) => {
        if (e.target === overlay) close(false);
      });
    });
  }

  async function processBatchChanges(batchChanges, modalOverlay) {
    const modal = modalOverlay.querySelector('.ss-accept-all-modal');
    const body = modal.querySelector('.ss-modal-body');
    const footer = modal.querySelector('.ss-modal-footer');

    footer.innerHTML = '';
    body.innerHTML = `
      <div class="ss-batch-progress">
        <div class="ss-batch-progress-bar">
          <div class="ss-batch-progress-fill" style="width: 0%"></div>
        </div>
        <div class="ss-batch-progress-text">Processing 0 / ${batchChanges.length}...</div>
      </div>
    `;

    const progressFill = body.querySelector('.ss-batch-progress-fill');
    const progressText = body.querySelector('.ss-batch-progress-text');

    let succeeded = 0;
    const failed = [];

    for (let i = 0; i < batchChanges.length; i++) {
      const item = batchChanges[i];
      const pct = Math.round(((i + 1) / batchChanges.length) * 100);
      progressText.textContent = `Processing ${i + 1} / ${batchChanges.length} — ${item.entityName}...`;
      progressFill.style.width = `${pct}%`;

      try {
        if (item.entityType === 'Tag') {
          await RecommendationsAPI.updateTag(item.entityId, item.fields);
        } else if (item.entityType === 'Studio') {
          await RecommendationsAPI.updateStudio(item.entityId, item.fields, item.rec.details.endpoint);
        } else {
          await RecommendationsAPI.updatePerformer(item.entityId, item.fields);
        }
        await RecommendationsAPI.resolve(item.rec.id, 'accepted', { batch: true });
        succeeded++;
      } catch (e) {
        failed.push({ entityName: item.entityName, error: e.message });
      }
    }

    progressFill.style.width = '100%';
    if (failed.length === 0) {
      progressText.textContent = `Done! ${succeeded} ${succeeded === 1 ? 'change' : 'changes'} applied successfully.`;
    } else {
      progressText.textContent = `${succeeded} applied, ${failed.length} failed.`;
    }

    return { succeeded, failed };
  }

  // ==================== List View ====================

  async function renderList(container, _retried = false) {
    const typeConfigs = {
      duplicate_performer: 'Duplicate Performers',
      duplicate_scenes: 'Duplicate Scenes',
      duplicate_scene_files: 'Duplicate Scene Files',
      upstream_performer_changes: 'Upstream Performer Changes',
      upstream_tag_changes: 'Upstream Tag Changes',
      upstream_studio_changes: 'Upstream Studio Changes',
      upstream_scene_changes: 'Upstream Scene Changes',
      scene_fingerprint_match: 'Scene Stash-Box Tagger',
      scene_face_match: 'Face Recommendations',
    };

    container.innerHTML = `
      <div class="ss-list-header">
        <button class="ss-btn ss-btn-back" id="ss-back-btn">
          <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
            <path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z"/>
          </svg>
          Back
        </button>
        <h1>${typeConfigs[currentState.type] || currentState.type}</h1>
      </div>

      <div class="ss-list-filters">
        <div class="ss-filter-tabs">
          <button class="ss-filter-tab ${currentState.status === 'pending' ? 'active' : ''}" data-status="pending">
            Pending
          </button>
          <button class="ss-filter-tab ${currentState.status === 'resolved' ? 'active' : ''}" data-status="resolved">
            Resolved
          </button>
          <button class="ss-filter-tab ${currentState.status === 'dismissed' ? 'active' : ''}" data-status="dismissed">
            Dismissed
          </button>
        </div>
        ${currentState.status === 'pending' ? `
        <div class="ss-list-actions">
          ${currentState.type === 'upstream_performer_changes' || currentState.type === 'upstream_tag_changes' || currentState.type === 'upstream_studio_changes' || currentState.type === 'upstream_scene_changes'
            ? '<button class="ss-accept-all-btn" id="ss-accept-all-btn" style="display:none;">Accept All Changes</button>'
            : ''
          }
          ${currentState.type === 'scene_fingerprint_match'
            ? '<button class="ss-accept-all-btn" id="ss-accept-all-fp-btn" style="display:none;">Accept All High-Confidence</button>'
            : ''
          }
          ${currentState.type === 'upstream_scene_changes'
            ? '<button class="ss-accept-all-btn" id="ss-accept-all-tag-url-code-btn" style="display:none;">Accept All Tag/URL/Code Only Changes</button>'
            : ''
          }
          ${currentState.type === 'upstream_performer_changes'
            ? '<button class="ss-accept-all-btn" id="ss-accept-all-performer-url-btn" style="display:none;">Accept All URL Only Changes</button>'
            : ''
          }
          ${currentState.type === 'duplicate_scenes'
            ? '<button class="ss-dismiss-selected-btn" id="ss-dismiss-selected-btn" style="display:none;">Dismiss Selected</button>'
            : ''
          }
          <button class="ss-dismiss-all-btn" id="ss-dismiss-all-btn" style="display:none;">Dismiss All</button>
        </div>
        ` : ''}
      </div>

      <div class="ss-list-content">
        <div class="ss-loading-inline">
          <div class="ss-spinner"></div>
        </div>
      </div>
    `;

    // Back button -- returning to the main dashboard (as opposed to a
    // detail view's own Back, which goes to 'list' and keeps the surgical
    // single-item cache removal in showSuccessAndReturn/removeFromListCache
    // for snappy re-entry into the *same* list) invalidates the cached list
    // page. Otherwise re-opening this same type/status/page later would hit
    // the exact same cache key and silently show stale rows/tab counts --
    // e.g. a resolved/dismissed count that doesn't reflect actions taken
    // since this page was first cached (only the currently-viewed status's
    // count gets patched by removeFromListCache; the ones an item moved
    // *into* never do). Dashboard itself always re-fetches counts fresh
    // (renderDashboard), so this only needs to clear the list-row cache.
    container.querySelector('#ss-back-btn').addEventListener('click', () => {
      invalidateListCache();
      currentState.view = 'dashboard';
      currentState.type = null;
      renderCurrentView(container);
    });

    // Filter tabs -- switching status is a different list, not "the
    // particular list you came from" (that's whatever status was active
    // before), so it must not reuse a cache entry from an earlier visit to
    // this status/page in the same session. Same principle as the dashboard
    // Back button above, just triggered by a different navigation.
    container.querySelectorAll('.ss-filter-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        invalidateListCache();
        currentState.status = tab.dataset.status;
        currentState.page = 0;
        renderCurrentView(container);
      });
    });

    // Accept All Changes button
    const acceptAllBtn = container.querySelector('#ss-accept-all-btn');
    if (acceptAllBtn) {
      acceptAllBtn.addEventListener('click', async () => {
        acceptAllBtn.disabled = true;
        acceptAllBtn.textContent = 'Loading...';

        // Upstream scene changes: dedicated per-rec accept loop with full relational resolution
        if (currentState.type === 'upstream_scene_changes') {
          try {
            const allPending = await RecommendationsAPI.getList({
              type: 'upstream_scene_changes',
              status: 'pending',
              limit: 10000,
              offset: 0,
            });
            const recs = allPending.recommendations || [];
            const total = recs.length;
            if (total === 0) {
              acceptAllBtn.textContent = 'No pending scene changes';
              setTimeout(() => {
                acceptAllBtn.textContent = 'Accept All Changes';
                acceptAllBtn.disabled = false;
              }, 1600);
              return;
            }

            const STALL_TIMEOUT_MIN_MS = 120000;
            let accepted = 0;
            let cleaned = 0;
            let failed = 0;
            let ensuredPerformers = 0;
            let ensuredTags = 0;
            let processed = 0;
            let avgMsPerItem = 0;
            let stallTimeoutMs = STALL_TIMEOUT_MIN_MS;
            let currentItemStart = 0;
            let progressTicker = null;
            const failureDetails = [];

            const setProgressText = () => {
              const elapsedSec = Math.max(0, Math.floor((Date.now() - currentItemStart) / 1000));
              acceptAllBtn.textContent = `Accepting ${processed}/${total} (${elapsedSec}s)...`;
            };

            for (const rec of recs) {
              currentItemStart = Date.now();
              setProgressText();
              progressTicker = setInterval(setProgressText, 1000);

              try {
                const response = await Promise.race([
                  RecommendationsAPI.acceptSceneChange(rec.id),
                  new Promise((_, reject) => setTimeout(() => reject(new Error(`Timed out after ${Math.ceil(stallTimeoutMs / 1000)}s`)), stallTimeoutMs)),
                ]);
                if (response?.action === 'deleted_stale_scene') {
                  cleaned += 1;
                } else {
                  accepted += 1;
                  ensuredPerformers += (response?.ensured_performers_count || 0);
                  ensuredTags += (response?.ensured_tags_count || 0);
                }
              } catch (e) {
                failed += 1;
                const msg = String(e.message || e || 'unknown error');
                failureDetails.push(`rec ${rec.id}: ${msg}`);
                console.warn('[Stash Sense] Accept scene change failed:', { rec_id: rec.id, error: msg });
              } finally {
                if (progressTicker) { clearInterval(progressTicker); progressTicker = null; }
                processed += 1;
                const duration = Date.now() - currentItemStart;
                avgMsPerItem = avgMsPerItem === 0 ? duration : Math.round((avgMsPerItem * (processed - 1) + duration) / processed);
                stallTimeoutMs = Math.max(STALL_TIMEOUT_MIN_MS, avgMsPerItem * 8);
                setProgressText();
              }
            }

            if (failed > 0) {
              const suffix = cleaned > 0 ? `, ${cleaned} stale removed` : '';
              acceptAllBtn.textContent = `${accepted}/${total} accepted${suffix}, ${failed} failed — see browser console`;
              acceptAllBtn.classList.add('ss-btn-error');
              console.warn('[Stash Sense] Accept all scene changes failures:', failureDetails);
            } else {
              const parts = [`Accepted ${accepted}/${total}`];
              if (ensuredPerformers > 0) parts.push(`${ensuredPerformers} performers created`);
              if (ensuredTags > 0) parts.push(`${ensuredTags} tags created`);
              if (cleaned > 0) parts.push(`${cleaned} stale removed`);
              acceptAllBtn.textContent = parts.join(', ');
              acceptAllBtn.classList.add('ss-btn-success');
            }

            invalidateListCache();
            setTimeout(() => {
              renderCurrentView(document.getElementById('ss-recommendations'));
            }, 1500);
          } catch (e) {
            acceptAllBtn.textContent = `Error: ${e.message}`;
            acceptAllBtn.classList.add('ss-btn-error');
            acceptAllBtn.disabled = false;
          }
          return;
        }

        // Performer / tag / studio changes: smart-default batch with confirmation modal
        try {
          // Fetch ALL pending (high limit to get everything)
          const allPending = await RecommendationsAPI.getList({
            type: currentState.type,
            status: 'pending',
            limit: 10000,
            offset: 0,
          });

          RecommendationsAPI.bulkAcceptStats(currentState.type).catch(() => null);
          const { batchChanges, noOpRecIds } = computeBatchChanges(allPending.recommendations);

          // Recs with a real local/upstream difference but nothing the smart
          // default would actually change (e.g. upstream cleared a field
          // local already has a value for) never show up in the batch and
          // never auto-resolve on their own -- without this they'd sit
          // pending forever, permanently excluded from both Accept All and
          // the per-item auto-resolve check. Clear them out silently; there's
          // nothing to confirm since nothing is being applied.
          let autoResolved = 0;
          if (noOpRecIds.length > 0) {
            const results = await Promise.allSettled(
              noOpRecIds.map(id => RecommendationsAPI.resolve(id, 'accepted_no_changes', {
                note: 'Upstream had no additional information over the local value',
                batch: true,
              }))
            );
            autoResolved = results.filter(r => r.status === 'fulfilled').length;
          }

          if (batchChanges.length === 0) {
            acceptAllBtn.textContent = autoResolved > 0
              ? `${autoResolved} had nothing new, marked reviewed`
              : 'No changes to apply';
            invalidateListCache();
            setTimeout(() => {
              acceptAllBtn.textContent = 'Accept All Changes';
              acceptAllBtn.disabled = false;
              renderCurrentView(document.getElementById('ss-recommendations'));
            }, 2000);
            return;
          }

          const confirmed = await showAcceptAllModal(batchChanges);
          if (!confirmed) {
            acceptAllBtn.textContent = 'Accept All Changes';
            acceptAllBtn.disabled = false;
            if (autoResolved > 0) {
              invalidateListCache();
              renderCurrentView(document.getElementById('ss-recommendations'));
            }
            return;
          }

          const overlay = document.querySelector('.ss-modal-overlay');
          const result = await processBatchChanges(batchChanges, overlay);

          overlay.remove();

          const autoResolvedSuffix = autoResolved > 0 ? `, ${autoResolved} had nothing new` : '';
          if (result.failed.length === 0) {
            acceptAllBtn.textContent = `Done! ${result.succeeded} applied${autoResolvedSuffix}`;
            acceptAllBtn.classList.add('ss-btn-success');
          } else {
            acceptAllBtn.textContent = `${result.succeeded} applied${autoResolvedSuffix}, ${result.failed.length} failed`;
            acceptAllBtn.classList.add('ss-btn-error');
            console.warn('[Stash Sense] Accept all changes failures:', result.failed);
          }

          invalidateListCache();
          setTimeout(() => {
            renderCurrentView(document.getElementById('ss-recommendations'));
          }, 2000);
        } catch (e) {
          acceptAllBtn.textContent = `Error: ${e.message}`;
          acceptAllBtn.disabled = false;
        }
      });
    }

    // Accept All High-Confidence fingerprint matches button
    const acceptAllFpBtn = container.querySelector('#ss-accept-all-fp-btn');
    if (acceptAllFpBtn) {
      acceptAllFpBtn.addEventListener('click', async () => {
        acceptAllFpBtn.disabled = true;
        acceptAllFpBtn.textContent = 'Accepting...';
        try {
          RecommendationsAPI.fingerprintMatchStats().catch(() => null);
          const result = await RecommendationsAPI.acceptAllFingerprintMatches();
          acceptAllFpBtn.textContent = `Accepted ${result.accepted_count}!`;
          acceptAllFpBtn.classList.add('ss-btn-success');
          invalidateListCache();
          setTimeout(() => {
            renderCurrentView(document.getElementById('ss-recommendations'));
          }, 1500);
        } catch (e) {
          acceptAllFpBtn.textContent = `Failed: ${e.message}`;
          acceptAllFpBtn.classList.add('ss-btn-error');
          acceptAllFpBtn.disabled = false;
        }
      });
    }

    // Accept All Tag-Only Scene Changes button
    const acceptAllTagOnlyBtn = container.querySelector('#ss-accept-all-tag-url-code-btn');
    if (acceptAllTagOnlyBtn) {
      acceptAllTagOnlyBtn.addEventListener('click', async () => {
        acceptAllTagOnlyBtn.disabled = true;
        acceptAllTagOnlyBtn.textContent = 'Loading...';
        try {
          const allPending = await RecommendationsAPI.getList({
            type: 'upstream_scene_changes',
            status: 'pending',
            limit: 10000,
            offset: 0,
          });

          const allSceneChanges = allPending.recommendations || [];
          const tagUrlCodeOnlyRecs = allSceneChanges.filter(rec => isTagUrlCodeOnlySceneChangeDetails(rec.details));
          const total = tagUrlCodeOnlyRecs.length;
          RecommendationsAPI.sceneTagOnlyStats().catch(() => null);
          if (total === 0) {
            acceptAllTagOnlyBtn.textContent = 'No tag/URL/code-only changes';
            setTimeout(() => {
              acceptAllTagOnlyBtn.textContent = 'Accept All Tag/URL/Code Only Changes';
              acceptAllTagOnlyBtn.disabled = false;
            }, 1600);
            return;
          }

          const STALL_TIMEOUT_MIN_MS = 120000;
          let accepted = 0;
          let cleaned = 0;
          let failed = 0;
          let ensuredTags = 0;
          let processed = 0;
          let avgMsPerItem = 0;
          let stallTimeoutMs = STALL_TIMEOUT_MIN_MS;
          let currentItemStart = 0;
          let progressTicker = null;
          const failureDetails = [];

          const setProgressText = () => {
            const elapsedSec = Math.max(0, Math.floor((Date.now() - currentItemStart) / 1000));
            acceptAllTagOnlyBtn.textContent = `Accepting ${processed}/${total} (${elapsedSec}s)...`;
          };

          for (const rec of tagUrlCodeOnlyRecs) {
            currentItemStart = Date.now();
            setProgressText();
            progressTicker = setInterval(setProgressText, 1000);

            try {
              const response = await Promise.race([
                RecommendationsAPI.acceptSceneTagOnlyChange(rec.id),
                new Promise((_, reject) => setTimeout(() => reject(new Error(`Timed out after ${Math.ceil(stallTimeoutMs / 1000)}s`)), stallTimeoutMs)),
              ]);
              if (response?.action === 'deleted_stale_scene') {
                cleaned += 1;
              } else {
                accepted += 1;
                ensuredTags += (response?.ensured_tags_count || 0);
              }
            } catch (e) {
              failed += 1;
              const msg = String(e.message || e || 'unknown error');
              failureDetails.push(`rec ${rec.id}: ${msg}`);
              console.warn('[Stash Sense] Accept tag/url/code-only scene change failed:', { rec_id: rec.id, error: msg });
            } finally {
              if (progressTicker) {
                clearInterval(progressTicker);
                progressTicker = null;
              }
              processed += 1;
              const duration = Date.now() - currentItemStart;
              avgMsPerItem = avgMsPerItem === 0 ? duration : Math.round((avgMsPerItem * (processed - 1) + duration) / processed);
              // Dynamic stall timeout grows with observed item duration.
              stallTimeoutMs = Math.max(STALL_TIMEOUT_MIN_MS, avgMsPerItem * 8);
              setProgressText();
            }
          }

          if (failed > 0) {
            const cleanedSuffix = cleaned > 0 ? `, ${cleaned} stale removed` : '';
            acceptAllTagOnlyBtn.textContent = `${accepted}/${total} accepted${cleanedSuffix}, ${failed} failed — see browser console for details`;
            acceptAllTagOnlyBtn.classList.add('ss-btn-error');
            console.warn('[Stash Sense] Tag/URL/code-only accept failures:', failureDetails);
          } else {
            const parts = [`Accepted ${accepted}/${total}`];
            if (ensuredTags > 0) parts.push(`${ensuredTags} tags created/linked`);
            if (cleaned > 0) parts.push(`${cleaned} stale removed`);
            acceptAllTagOnlyBtn.textContent = parts.join(', ');
            acceptAllTagOnlyBtn.classList.add('ss-btn-success');
          }

          invalidateListCache();
          setTimeout(() => {
            renderCurrentView(document.getElementById('ss-recommendations'));
          }, 1500);
        } catch (e) {
          acceptAllTagOnlyBtn.textContent = `Failed: ${e.message}`;
          acceptAllTagOnlyBtn.classList.add('ss-btn-error');
          acceptAllTagOnlyBtn.disabled = false;
        }
      });
    }

    // Accept All URL Only Performer Changes button
    const acceptAllPerformerUrlBtn = container.querySelector('#ss-accept-all-performer-url-btn');
    if (acceptAllPerformerUrlBtn) {
      acceptAllPerformerUrlBtn.addEventListener('click', async () => {
        acceptAllPerformerUrlBtn.disabled = true;
        acceptAllPerformerUrlBtn.textContent = 'Loading...';
        try {
          const allPending = await RecommendationsAPI.getList({
            type: 'upstream_performer_changes',
            status: 'pending',
            limit: 10000,
            offset: 0,
          });

          const urlOnlyRecs = (allPending.recommendations || []).filter(rec => isUrlOnlyPerformerChangeDetails(rec.details));
          const total = urlOnlyRecs.length;
          RecommendationsAPI.performerUrlOnlyStats().catch(() => null);
          if (total === 0) {
            acceptAllPerformerUrlBtn.textContent = 'No URL-only changes';
            setTimeout(() => {
              acceptAllPerformerUrlBtn.textContent = 'Accept All URL Only Changes';
              acceptAllPerformerUrlBtn.disabled = false;
            }, 1600);
            return;
          }

          const result = await RecommendationsAPI.acceptAllPerformerUrlOnlyChanges();
          const accepted = result.accepted_count || 0;
          const failed = result.failed_count || 0;
          if (failed > 0) {
            acceptAllPerformerUrlBtn.textContent = `${accepted} accepted, ${failed} failed`;
            acceptAllPerformerUrlBtn.classList.add('ss-btn-error');
          } else {
            acceptAllPerformerUrlBtn.textContent = `Accepted ${accepted}`;
            acceptAllPerformerUrlBtn.classList.add('ss-btn-success');
          }
          invalidateListCache();
          setTimeout(() => {
            renderCurrentView(document.getElementById('ss-recommendations'));
          }, 1500);
        } catch (e) {
          acceptAllPerformerUrlBtn.textContent = `Failed: ${e.message}`;
          acceptAllPerformerUrlBtn.classList.add('ss-btn-error');
          acceptAllPerformerUrlBtn.disabled = false;
        }
      });
    }

    // Dismiss All button
    const dismissAllBtn = container.querySelector('#ss-dismiss-all-btn');
    if (dismissAllBtn) {
      dismissAllBtn.addEventListener('click', async () => {
        // Show confirmation modal with permanent/temporary options
        const overlay = document.createElement('div');
        overlay.className = 'ss-modal-overlay';
        overlay.innerHTML = `
          <div class="ss-modal" style="max-width:420px;">
            <h3>Dismiss All</h3>
            <p style="margin:0.75rem 0;color:#aaa;">How would you like to dismiss all pending ${typeConfigs[currentState.type] || currentState.type} recommendations?</p>
            <div style="display:flex;flex-direction:column;gap:0.5rem;margin-top:1rem;">
              <button class="ss-btn ss-btn-secondary" id="ss-dismiss-temp">Dismiss until next analysis</button>
              <button class="ss-btn ss-btn-danger" id="ss-dismiss-perm">Never show again</button>
              <button class="ss-btn" id="ss-dismiss-cancel" style="margin-top:0.25rem;">Cancel</button>
            </div>
          </div>
        `;
        document.body.appendChild(overlay);

        const handleDismiss = async (permanent) => {
          overlay.querySelector('.ss-modal').innerHTML = '<div class="ss-loading-inline"><div class="ss-spinner"></div></div><p style="text-align:center;margin-top:0.5rem;">Dismissing...</p>';
          try {
            const result = await RecommendationsAPI.batchDismiss(currentState.type, permanent);
            overlay.remove();
            dismissAllBtn.textContent = `Dismissed ${result.dismissed_count}!`;
            dismissAllBtn.disabled = true;
            invalidateListCache();
            setTimeout(() => {
              renderCurrentView(document.getElementById('ss-recommendations'));
            }, 1500);
          } catch (e) {
            overlay.remove();
            dismissAllBtn.textContent = `Failed: ${e.message}`;
            setTimeout(() => { dismissAllBtn.textContent = 'Dismiss All'; dismissAllBtn.disabled = false; }, 2000);
          }
        };

        overlay.querySelector('#ss-dismiss-temp').addEventListener('click', () => handleDismiss(false));
        overlay.querySelector('#ss-dismiss-perm').addEventListener('click', () => handleDismiss(true));
        overlay.querySelector('#ss-dismiss-cancel').addEventListener('click', () => overlay.remove());
        overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
      });
    }

    // Load recommendations -- reuse the cached last-fetched page instead of
    // re-fetching and re-sorting from scratch when returning from a detail
    // view with nothing to invalidate (plain Back button), or after a
    // single recommendation was resolved/dismissed/matched there
    // (removeFromListCache() already surgically removed it from the cache
    // before navigating back). Any real change of type/status/page
    // naturally misses the cache since the key below won't match.
    //
    // Tab counts are fetched fresh every time regardless of the row-list
    // cache hit/miss above -- a surgical single-item removal only knows how
    // to decrement the status you were just viewing (pending), never the
    // status the item transitioned *into* (resolved/dismissed), so a count
    // cached alongside the row list can never be trusted after any action.
    // This is a small, cheap request on its own, so there's no real cost to
    // always doing it live.
    const PAGE_SIZE = 25;
    const cacheKey = `${currentState.type}:${currentState.status}:${currentState.page}`;
    const cached = currentState.listCache;
    try {
      let recommendations, total, typeCounts;
      const countsPromise = RecommendationsAPI.getCounts()
        .then(r => r.counts?.[currentState.type] || {})
        .catch(() => ({}));

      if (cached && cached.key === cacheKey) {
        ({ recommendations, total } = cached);
        typeCounts = await countsPromise;
      } else {
        const [result, resolvedTypeCounts] = await Promise.all([
          RecommendationsAPI.getList({
            type: currentState.type,
            status: currentState.status,
            limit: PAGE_SIZE,
            offset: currentState.page * PAGE_SIZE,
          }),
          countsPromise,
        ]);

        typeCounts = resolvedTypeCounts;
        recommendations = result.recommendations;
        total = result.total;

        // Defensive client-side ordering for Scene Stash-Box Tagger:
        // high-confidence first, then confidence descending.
        if (currentState.type === 'scene_fingerprint_match') {
          recommendations.sort((a, b) => {
            const aHigh = a?.details?.high_confidence ? 1 : 0;
            const bHigh = b?.details?.high_confidence ? 1 : 0;
            if (aHigh !== bHigh) return bHigh - aHigh;
            const aConf = Number(a?.confidence || 0);
            const bConf = Number(b?.confidence || 0);
            if (aConf !== bConf) return bConf - aConf;
            return Number(b?.id || 0) - Number(a?.id || 0);
          });
        } else if (currentState.type === 'duplicate_scenes' && currentState.status === 'pending') {
          // Defensive client-side ordering for pending items: high confidence first.
          // Dismissed/resolved use server-side recency sort — don't override it.
          recommendations.sort((a, b) => {
            const aConf = getDuplicateSceneConfidencePercent(a);
            const bConf = getDuplicateSceneConfidencePercent(b);
            if (aConf !== bConf) return bConf - aConf;
            return Number(b?.id || 0) - Number(a?.id || 0);
          });
        }

        currentState.listCache = { key: cacheKey, recommendations, total };
      }

      // The current page came back empty but recommendations still remain
      // overall -- either every entry on this page was surgically removed
      // from the cache (removeFromListCache, e.g. resolving/dismissing
      // items one by one) while the page index itself stayed 0 (the common
      // case, since that's the default and where most people work), or this
      // is a stale cache from a previous visit to this same type/status/page
      // (e.g. Back to the dashboard and straight back into the same type,
      // which doesn't reset page/status and so hits the exact same cache
      // key). Either way, force a real re-fetch instead of showing a false
      // "no recommendations" empty state -- snapping back to the last valid
      // page first if the current page index is genuinely beyond the real
      // total. _retried guards against ever looping more than once. Applies
      // to every recommendation type/status, since this is the one shared
      // list renderer they all go through.
      if (recommendations.length === 0 && total > 0 && !_retried) {
        const maxPage = Math.max(0, Math.ceil(total / PAGE_SIZE) - 1);
        if (currentState.page > maxPage) {
          currentState.page = maxPage;
        }
        currentState.listCache = null;
        return renderList(container, true);
      }

      // Update tab labels with counts
      container.querySelectorAll('.ss-filter-tab').forEach(tab => {
        const status = tab.dataset.status;
        const count = typeCounts[status] || 0;
        const label = status.charAt(0).toUpperCase() + status.slice(1);
        tab.textContent = `${label} (${count})`;
      });

      const listContent = container.querySelector('.ss-list-content');

      if (recommendations.length === 0) {
        // Nothing pending to act on -- hide these rather than just disabling
        // them, since a greyed-out "Accept All"/"Dismiss All" reads as a
        // broken button rather than "there's nothing here."
        container.querySelectorAll('.ss-accept-all-btn, .ss-dismiss-all-btn').forEach(btn => { btn.style.display = 'none'; });
        listContent.innerHTML = `
          <div class="ss-empty-state">
            <p>No ${currentState.status} recommendations found.</p>
          </div>
        `;
        return;
      }

      // There's something to act on -- reveal the Accept All / Dismiss All
      // buttons now (they start hidden in the template to avoid a
      // flash-then-hide on page load, before we know whether anything's
      // actually pending).
      container.querySelectorAll('.ss-accept-all-btn, .ss-dismiss-all-btn').forEach(btn => { btn.style.display = ''; });

      // Pagination controls -- built once, rendered both above and below
      // the card listing (top copy saves a scroll-to-top on long lists;
      // bottom copy matches where people naturally look after reading
      // through a page).
      const totalPages = Math.ceil(total / PAGE_SIZE);
      const buildPaginationEl = () => {
        const pagination = document.createElement('div');
        pagination.style.cssText = 'display:flex;align-items:center;justify-content:center;gap:12px;padding:16px 0;';

        const prevBtn = document.createElement('button');
        prevBtn.className = 'ss-btn ss-btn-secondary';
        prevBtn.textContent = '← Prev';
        prevBtn.disabled = currentState.page === 0;
        prevBtn.style.cssText = 'min-width:80px;';
        prevBtn.addEventListener('click', () => {
          currentState.page--;
          renderCurrentView(container);
        });

        const pageText = document.createElement('span');
        pageText.style.cssText = 'color:var(--ss-text-secondary, #aaa);font-size:14px;';
        pageText.textContent = `Page ${currentState.page + 1} of ${totalPages}`;

        const nextBtn = document.createElement('button');
        nextBtn.className = 'ss-btn ss-btn-secondary';
        nextBtn.textContent = 'Next →';
        nextBtn.disabled = currentState.page >= totalPages - 1;
        nextBtn.style.cssText = 'min-width:80px;';
        nextBtn.addEventListener('click', () => {
          currentState.page++;
          renderCurrentView(container);
        });

        pagination.appendChild(prevBtn);
        pagination.appendChild(pageText);
        pagination.appendChild(nextBtn);
        return pagination;
      };

      listContent.innerHTML = '';

      if (totalPages > 1) {
        listContent.appendChild(buildPaginationEl());
      }

      for (const rec of recommendations) {
        const card = renderRecommendationCard(rec);
        if (rec.status !== 'pending' && rec.updated_at) {
          const dateBadge = document.createElement('div');
          dateBadge.className = 'ss-rec-date-badge';
          dateBadge.textContent = formatRecTimestamp(rec.updated_at);
          card.appendChild(dateBadge);
        }
        card.addEventListener('click', (e) => {
          if (e.target.closest('.ss-dup-scene-select')) return;
          currentState.selectedRec = rec;
          currentState.view = 'detail';
          renderCurrentView(container);
        });
        listContent.appendChild(card);
      }

      // Duplicate-scenes bulk selection: toggle "Dismiss Selected" visibility
      // as checkboxes change, and wire the dismiss action itself.
      if (currentState.type === 'duplicate_scenes' && currentState.status === 'pending') {
        const dismissSelectedBtn = container.querySelector('#ss-dismiss-selected-btn');
        if (dismissSelectedBtn) {
          listContent.addEventListener('change', (e) => {
            if (!e.target.matches('.ss-dup-scene-select-cb')) return;
            const anyChecked = listContent.querySelector('.ss-dup-scene-select-cb:checked') != null;
            dismissSelectedBtn.style.display = anyChecked ? '' : 'none';
          });

          dismissSelectedBtn.addEventListener('click', async () => {
            const checkedBoxes = Array.from(listContent.querySelectorAll('.ss-dup-scene-select-cb:checked'));
            if (checkedBoxes.length === 0) return;

            const allGroupIds = [];
            checkedBoxes.forEach(cb => {
              const recId = Number(cb.dataset.recId);
              const rec = recommendations.find(r => r.id === recId);
              allGroupIds.push(...(rec ? getDuplicateSceneGroupIds(rec) : [recId]));
            });

            dismissSelectedBtn.disabled = true;
            dismissSelectedBtn.textContent = 'Dismissing...';
            try {
              const result = await RecommendationsAPI.dismissDuplicateSceneGroup(allGroupIds, 'User dismissed');
              dismissSelectedBtn.textContent = `Dismissed ${result.dismissed_count}!`;
              invalidateListCache();
              setTimeout(() => {
                renderCurrentView(document.getElementById('ss-recommendations'));
              }, 1200);
            } catch (e) {
              dismissSelectedBtn.textContent = `Failed: ${e.message}`;
              setTimeout(() => {
                dismissSelectedBtn.textContent = 'Dismiss Selected';
                dismissSelectedBtn.disabled = false;
              }, 2000);
            }
          });
        }
      }

      if (totalPages > 1) {
        listContent.appendChild(buildPaginationEl());
      }

    } catch (e) {
      container.querySelector('.ss-list-content').innerHTML = `
        <div class="ss-error-state">
          <p>Failed to load recommendations: ${e.message}</p>
        </div>
      `;
    }
  }

  // A duplicate-scenes list card bundles multiple underlying recommendation
  // rows (one per candidate match) into a single card via
  // details.duplicate_matches -- dismissing "this card" means dismissing
  // every row in that bundle, not just rec.id. Mirrors the fallback used in
  // renderDuplicateScenesDetail() for recs with no duplicate_matches array.
  function getDuplicateSceneGroupIds(rec) {
    const details = rec.details || {};
    const rawMatches = Array.isArray(details.duplicate_matches) && details.duplicate_matches.length > 0
      ? details.duplicate_matches
      : [{ recommendation_id: rec.id }];
    return rawMatches
      .map(m => Number(m && m.recommendation_id))
      .filter(id => Number.isFinite(id));
  }

  // Surgically drops one recommendation from the cached list page (see
  // renderList()'s cache-hit branch) instead of invalidating the whole
  // cache -- called right before navigating back to the list from a
  // detail view whose recommendation was just resolved/dismissed/matched,
  // or found stale (source/target no longer exists). No-ops harmlessly if
  // there's no cache yet, or the id isn't on the cached page (e.g. it was
  // resolved from a different page/filter than what's cached).
  function removeFromListCache(recId) {
    // Row list only -- tab counts are never cached (always fetched fresh in
    // renderList), so there's nothing to patch here for them.
    const cache = currentState.listCache;
    if (!cache || recId == null) return;
    const idx = cache.recommendations.findIndex(r => r.id === recId);
    if (idx === -1) return;
    cache.recommendations.splice(idx, 1);
    cache.total = Math.max(0, cache.total - 1);
  }

  // Bulk actions (Accept All / Dismiss All) change an unknown, unbounded
  // number of recommendations server-side in one go -- surgical per-item
  // removal doesn't scale to that, so just drop the whole cached page and
  // let the next renderList() do a real fetch.
  function invalidateListCache() {
    currentState.listCache = null;
  }

  function renderRecommendationCard(rec) {
    const details = rec.details;

    if (rec.type === 'duplicate_performer') {
      const performers = details.performers || [];
      const keeper = performers.find(p => p.is_suggested_keeper);
      const others = performers.filter(p => !p.is_suggested_keeper);

      return SS.createElement('div', {
        className: 'ss-rec-card ss-rec-performer',
        innerHTML: `
          <div class="ss-rec-performers">
            ${performers.map(p => `
              <div class="ss-rec-performer-thumb ${p.is_suggested_keeper ? 'keeper' : ''}">
                ${p.image_path ? `<img src="${relativeUrl(p.image_path)}" alt="${p.name}" loading="lazy" onerror="this.style.display='none'" />` : ''}
                <span class="ss-performer-name">${p.name}</span>
                <span class="ss-performer-count">${p.scene_count} scenes</span>
              </div>
            `).join('')}
          </div>
          <div class="ss-rec-summary">
            <span class="ss-rec-type-badge">Duplicate</span>
            <span>${performers.length} performers share StashDB ID</span>
          </div>
        `,
      });
    }

    if (rec.type === 'duplicate_scene_files') {
      const files = details.files || [];

      return SS.createElement('div', {
        className: 'ss-rec-card ss-rec-scene-files',
        innerHTML: `
          <div class="ss-rec-scene-info">
            <h4>${details.scene_title}</h4>
            <div class="ss-rec-scene-meta">
              ${details.studio?.name ? `<span>${details.studio.name}</span>` : ''}
              ${details.performers?.length ? `<span>${details.performers.map(p => p.name).join(', ')}</span>` : ''}
            </div>
          </div>
          <div class="ss-rec-files-summary">
            <span class="ss-rec-type-badge">${files.length} files</span>
            <span class="ss-rec-savings">Save ${details.potential_savings_formatted}</span>
          </div>
        `,
      });
    }

    if (rec.type === 'duplicate_scenes') {
      const d = details;
      const conf = getDuplicateSceneConfidencePercent(rec);
      const confColor = conf >= 80 ? '#28a745' : conf >= 60 ? '#ffc107' : '#6c757d';
      const sb = d.signal_breakdown || {};
      const primarySignal = sb.stashbox_match ? 'Stash-box match'
        : sb.phash_distance != null && sb.phash_distance <= 10 ? `Phash (dist ${sb.phash_distance})`
        : sb.metadata_score > 0 ? 'Metadata'
        : 'Face analysis';

      const sumA = d.source_summary || d.scene_a_summary || {};
      const titleA = sumA.title || `Scene ${d.source_scene_id || d.scene_a_id}`;
      const studio = sumA.studio || '';
      const performers = (sumA.performers || []).join(', ');
      const contextParts = [studio, performers].filter(Boolean);
      const contextLine = contextParts.length ? contextParts.join(' · ') : `Scene ID: ${d.source_scene_id || d.scene_a_id}`;
      const matchCount = Math.max(1, Number(d.match_count || (d.duplicate_matches || []).length || 1));
      const duplicateLabel = `${matchCount} possible duplicate${matchCount !== 1 ? 's' : ''}`;
      // Selection checkboxes only make sense where bulk-dismissing applies --
      // swap in the plain icon on the Resolved/Dismissed tabs.
      const iconHtml = currentState.status === 'pending'
        ? `<label class="ss-rec-tag-icon ss-dup-scene-select">
             <input type="checkbox" class="ss-dup-scene-select-cb" data-rec-id="${rec.id}" />
           </label>`
        : `<div class="ss-rec-tag-icon">
             <svg viewBox="0 0 24 24" width="32" height="32" fill="currentColor"><path d="M4 6H2v14c0 1.1.9 2 2 2h14v-2H4V6zm16-4H8c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H8V4h12v12z"/></svg>
           </div>`;

      return SS.createElement('div', {
        className: 'ss-rec-card ss-rec-dup-scenes',
        innerHTML: `
          <div class="ss-rec-card-header">
            ${iconHtml}
            <div class="ss-rec-card-info">
              <div class="ss-rec-card-title">${SS.escapeHtml(titleA)}</div>
              <div class="ss-rec-card-subtitle">
                <span style="color: ${confColor}">${Math.round(conf)}% confidence</span>
                &middot; ${duplicateLabel}
                &middot; ${primarySignal}
              </div>
              <div class="ss-rec-card-fields">${SS.escapeHtml(contextLine)}</div>
            </div>
          </div>
        `,
      });
    }

    if (rec.type === 'upstream_performer_changes') {
      const upstreamStatus = details.upstream_status || 'active';
      const realChanges = filterRealChanges(details.changes);
      const changeCount = realChanges.length;
      const changedFields = realChanges.map(c => c.field_label).join(', ');
      let subtitle = `${changeCount} field${changeCount !== 1 ? 's' : ''} changed · ${details.endpoint_name || ''}`;
      if (upstreamStatus === 'merged') {
        subtitle = `Merged upstream performer · relink required · ${details.endpoint_name || ''}`;
      } else if (upstreamStatus === 'deleted') {
        subtitle = `Deleted upstream performer · review required · ${details.endpoint_name || ''}`;
      }
      const fieldSummary = changedFields
        || (upstreamStatus === 'deleted'
          ? 'Upstream performer was deleted'
          : (upstreamStatus === 'merged' ? 'Upstream performer merged into a new ID' : ''));

      return SS.createElement('div', {
        className: 'ss-rec-card ss-rec-upstream',
        innerHTML: `
          <div class="ss-rec-card-header">
            <img src="${relativeUrl(details.performer_image_path) || ''}" class="ss-rec-thumb" onerror="this.style.display='none'"/>
            <div class="ss-rec-card-info">
              <div class="ss-rec-card-title">Upstream Changes: ${details.performer_name || 'Unknown'}</div>
              <div class="ss-rec-card-subtitle">
                ${subtitle}
              </div>
              <div class="ss-rec-card-fields">${fieldSummary}</div>
            </div>
          </div>
        `,
      });
    }

    if (rec.type === 'upstream_tag_changes') {
      const realChanges = filterRealChanges(details.changes);
      const changeCount = realChanges.length;
      const changedFields = realChanges.map(c => c.field_label).join(', ');

      return SS.createElement('div', {
        className: 'ss-rec-card ss-rec-upstream',
        innerHTML: `
          <div class="ss-rec-card-header">
            <div class="ss-rec-tag-icon">
              <svg viewBox="0 0 24 24" width="32" height="32" fill="currentColor"><path d="M21.41 11.58l-9-9C12.05 2.22 11.55 2 11 2H4c-1.1 0-2 .9-2 2v7c0 .55.22 1.05.59 1.42l9 9c.36.36.86.58 1.41.58.55 0 1.05-.22 1.41-.59l7-7c.37-.36.59-.86.59-1.41 0-.55-.23-1.06-.59-1.42zM5.5 7C4.67 7 4 6.33 4 5.5S4.67 4 5.5 4 7 4.67 7 5.5 6.33 7 5.5 7z"/></svg>
            </div>
            <div class="ss-rec-card-info">
              <div class="ss-rec-card-title">Upstream Changes: ${details.tag_name || 'Unknown'}</div>
              <div class="ss-rec-card-subtitle">
                ${changeCount} field${changeCount !== 1 ? 's' : ''} changed · ${details.endpoint_name || ''}
              </div>
              <div class="ss-rec-card-fields">${changedFields}</div>
            </div>
          </div>
        `,
      });
    }

    if (rec.type === 'upstream_studio_changes') {
      const realChanges = filterRealChanges(details.changes);
      const changeCount = realChanges.length;
      const changedFields = realChanges.map(c => c.field_label).join(', ');

      return SS.createElement('div', {
        className: 'ss-rec-card ss-rec-upstream',
        innerHTML: `
          <div class="ss-rec-card-header">
            <div class="ss-rec-tag-icon">
              <svg viewBox="0 0 24 24" width="32" height="32" fill="currentColor"><path d="M12 7V3H2v18h20V7H12zM6 19H4v-2h2v2zm0-4H4v-2h2v2zm0-4H4V9h2v2zm0-4H4V5h2v2zm4 12H8v-2h2v2zm0-4H8v-2h2v2zm0-4H8V9h2v2zm0-4H8V5h2v2zm10 12h-8v-2h2v-2h-2v-2h2v-2h-2V9h8v10zm-2-8h-2v2h2v-2zm0 4h-2v2h2v-2z"/></svg>
            </div>
            <div class="ss-rec-card-info">
              <div class="ss-rec-card-title">Upstream Changes: ${details.studio_name || 'Unknown'}</div>
              <div class="ss-rec-card-subtitle">
                ${changeCount} field${changeCount !== 1 ? 's' : ''} changed · ${details.endpoint_name || ''}
              </div>
              <div class="ss-rec-card-fields">${changedFields}</div>
            </div>
          </div>
        `,
      });
    }

    if (rec.type === 'upstream_scene_changes') {
      const simpleChanges = filterRealChanges(details.changes || []);
      const rawPerfChanges = details.performer_changes || { added: [], removed: [], alias_changed: [] };
      const perfChanges = { ...rawPerfChanges, alias_changed: [] };
      const tagChanges = details.tag_changes || { added: [], removed: [] };
      const studioChange = details.studio_change;

      const parts = [];
      if (simpleChanges.length > 0) parts.push(`${simpleChanges.length} field${simpleChanges.length !== 1 ? 's' : ''}`);
      if (studioChange) parts.push('studio');
      const perfTotal = perfChanges.added.length + perfChanges.removed.length;
      if (perfTotal > 0) parts.push(`${perfTotal} performer${perfTotal !== 1 ? 's' : ''}`);
      const tagTotal = tagChanges.added.length + tagChanges.removed.length;
      if (tagTotal > 0) parts.push(`${tagTotal} tag${tagTotal !== 1 ? 's' : ''}`);
      const summary = parts.join(', ') || 'changes detected';

      return SS.createElement('div', {
        className: 'ss-rec-card ss-rec-upstream',
        innerHTML: `
          <div class="ss-rec-card-header">
            <div class="ss-rec-tag-icon">
              <svg viewBox="0 0 24 24" width="32" height="32" fill="currentColor"><path d="M18 4l2 4h-3l-2-4h-2l2 4h-3l-2-4H8l2 4H7L5 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V4h-4z"/></svg>
            </div>
            <div class="ss-rec-card-info">
              <div class="ss-rec-card-title">${details.scene_name || 'Unknown'}</div>
              <div class="ss-rec-card-subtitle">
                ${summary} · ${details.endpoint_name || ''}
              </div>
            </div>
          </div>
        `,
      });
    }

    if (rec.type === 'scene_fingerprint_match') {
      const d = details;
      const matchColor = d.match_percentage >= 100 ? '#28a745' :
                         d.match_percentage >= 66 ? '#ffc107' : '#dc3545';

      return SS.createElement('div', {
        className: 'ss-rec-card',
        innerHTML: `
          <div class="ss-rec-card-header">
            <div class="ss-rec-tag-icon">
              <svg viewBox="0 0 24 24" width="32" height="32" fill="currentColor"><path d="M3.9 12c0-1.71 1.39-3.1 3.1-3.1h4V7H7c-2.76 0-5 2.24-5 5s2.24 5 5 5h4v-1.9H7c-1.71 0-3.1-1.39-3.1-3.1zM8 13h8v-2H8v2zm9-6h-4v1.9h4c1.71 0 3.1 1.39 3.1 3.1s-1.39 3.1-3.1 3.1h-4V17h4c2.76 0 5-2.24 5-5s-2.24-5-5-5z"/></svg>
            </div>
            <div class="ss-rec-card-info">
              <div class="ss-rec-card-title">
                ${escapeHtml(d.local_scene_title || 'Unknown Scene')}
                ${d.high_confidence ? '<span class="ss-badge ss-badge-success">High Confidence</span>' : ''}
              </div>
              <div class="ss-rec-card-subtitle">
                &rarr; ${escapeHtml(d.stashbox_scene_title || 'Unknown')}
                ${d.stashbox_studio ? ` &middot; ${escapeHtml(d.stashbox_studio)}` : ''}
              </div>
              <div class="ss-rec-card-fields" style="color: ${matchColor}">
                ${d.match_count}/${d.total_local_fingerprints} fingerprints
                ${d.has_exact_hash ? '(exact)' : '(phash only)'}
                &middot; ${escapeHtml(d.endpoint_name || d.endpoint || '')}
              </div>
            </div>
          </div>
        `,
      });
    }

    if (rec.type === 'scene_face_match') {
      const d = details;
      const personCount = (d.persons || []).length;
      const candidateCount = d.candidate_count || (d.candidates || []).length;
      const topNames = (d.candidates || [])
        .filter(c => c.is_best_match)
        .map(c => c.name)
        .filter(Boolean);

      return SS.createElement('div', {
        className: 'ss-rec-card',
        innerHTML: `
          <div class="ss-rec-card-header">
            <div class="ss-rec-tag-icon">
              <svg viewBox="0 0 24 24" width="32" height="32" fill="currentColor"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/></svg>
            </div>
            <div class="ss-rec-card-info">
              <div class="ss-rec-card-title">${escapeHtmlBreakable(d.scene_title || 'Unknown Scene')}</div>
              <div class="ss-rec-card-subtitle">
                ${personCount} person${personCount !== 1 ? 's' : ''} detected &middot; ${candidateCount} candidate${candidateCount !== 1 ? 's' : ''}
              </div>
              ${topNames.length ? `<div class="ss-rec-card-fields">${escapeHtml(topNames.join(', '))}</div>` : ''}
            </div>
          </div>
        `,
      });
    }

    // Fallback for unknown types
    return SS.createElement('div', {
      className: 'ss-rec-card',
      innerHTML: `
        <div class="ss-rec-generic">
          <span class="ss-rec-type-badge">${rec.type}</span>
          <span>Target: ${rec.target_type} ${rec.target_id}</span>
        </div>
      `,
    });
  }

  // ==================== Detail View ====================

  async function renderDetail(container) {
    let rec = currentState.selectedRec;
    if (!rec) {
      currentState.view = 'list';
      renderCurrentView(container);
      return;
    }

    // Refresh recommendation from backend on open so stale scene references
    // can be pruned server-side after list load.
    if (rec.id != null) {
      try {
        rec = await RecommendationsAPI.getOne(rec.id);
        currentState.selectedRec = rec;
      } catch (e) {
        const msg = String(e?.message || e || '');
        const stale = /recommendation not found|recommendation removed because referenced scene no longer exists/i.test(msg);
        showToast(
          stale
            ? 'Recommendation removed because source/target scene no longer exists.'
            : `Failed to open recommendation: ${msg}`,
          stale ? 'warning' : 'error',
        );
        if (stale) removeFromListCache(rec.id);
        currentState.view = 'list';
        currentState.selectedRec = null;
        renderCurrentView(document.getElementById('ss-recommendations') || container);
        return;
      }
    }

    container.innerHTML = `
      <div class="ss-detail-header">
        <button class="ss-btn ss-btn-back" id="ss-back-btn">
          <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
            <path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z"/>
          </svg>
          Back
        </button>
      </div>
      <div class="ss-detail-content"></div>
    `;

    container.querySelector('#ss-back-btn').addEventListener('click', () => {
      currentState.view = 'list';
      currentState.selectedRec = null;
      renderCurrentView(container);
    });

    const content = container.querySelector('.ss-detail-content');

    if (rec.type === 'duplicate_performer') {
      renderDuplicatePerformerDetail(content, rec);
    } else if (rec.type === 'duplicate_scene_files') {
      renderDuplicateSceneFilesDetail(content, rec);
    } else if (rec.type === 'duplicate_scenes') {
      await renderDuplicateScenesDetail(content, rec);
    } else if (rec.type === 'upstream_performer_changes') {
      await renderUpstreamPerformerDetail(content, rec);
    } else if (rec.type === 'upstream_tag_changes') {
      await renderUpstreamTagDetail(content, rec);
    } else if (rec.type === 'upstream_studio_changes') {
      await renderUpstreamStudioDetail(content, rec);
    } else if (rec.type === 'upstream_scene_changes') {
      await renderUpstreamSceneDetail(content, rec);
    } else if (rec.type === 'scene_fingerprint_match') {
      await renderFingerprintMatchDetail(content, rec);
    } else if (rec.type === 'scene_face_match') {
      await renderSceneFaceMatchDetail(content, rec);
    } else {
      content.innerHTML = `<p>Unknown recommendation type: ${escapeHtml(rec.type)}</p>`;
    }
  }

  function renderDuplicatePerformerDetail(container, rec) {
    const details = rec.details;
    const performers = details.performers || [];

    container.innerHTML = `
      <div class="ss-detail-duplicate-performer">
        <h2>Duplicate Performers <span id="ss-info-dup-perf"></span></h2>
        <p class="ss-detail-subtitle">
          These performers share StashDB ID:
          <a href="https://stashdb.org/performers/${details.stash_id}" target="_blank" rel="noopener">
            ${details.stash_id.substring(0, 8)}...
          </a>
        </p>

        <div class="ss-performer-grid">
          ${performers.map(p => `
            <div class="ss-performer-option ${p.is_suggested_keeper ? 'suggested' : ''}" data-id="${p.id}">
              <div class="ss-performer-image">
                ${p.image_path ? `<img src="${relativeUrl(p.image_path)}" alt="${p.name}" loading="lazy" onerror="this.style.display='none'" />` : '<div class="ss-no-image">No Image</div>'}
                ${p.is_suggested_keeper ? '<span class="ss-suggested-badge">Suggested Keeper</span>' : ''}
              </div>
              <div class="ss-performer-details">
                <h3>
                  <a href="/performers/${p.id}" target="_blank">${p.name}</a>
                </h3>
                <ul class="ss-performer-stats">
                  <li><strong>${p.scene_count}</strong> scenes</li>
                  <li><strong>${p.image_count}</strong> images</li>
                  <li><strong>${p.gallery_count}</strong> galleries</li>
                </ul>
                <label class="ss-radio-label">
                  <input type="radio" name="keeper" value="${p.id}" ${p.is_suggested_keeper ? 'checked' : ''} />
                  Keep this performer
                </label>
              </div>
            </div>
          `).join('')}
        </div>

        <div class="ss-detail-actions">
          <button class="ss-btn ss-btn-primary ss-btn-merge" id="ss-merge-btn">
            Merge Performers
          </button>
          <button class="ss-btn ss-btn-secondary" id="ss-dismiss-btn">
            Dismiss
          </button>
        </div>
      </div>
    `;

    // Add info icon
    const dpInfoSlot = container.querySelector('#ss-info-dup-perf');
    if (dpInfoSlot) dpInfoSlot.appendChild(createInfoIcon(() => showHelpModal('Duplicate Performers', HELP_DUP_PERFORMER)));

    // Click anywhere on card to select radio (except links)
    container.querySelectorAll('.ss-performer-option').forEach(card => {
      card.style.cursor = 'pointer';
      card.addEventListener('click', (e) => {
        if (e.target.closest('a')) return;
        const radio = card.querySelector('input[type="radio"]');
        if (radio) radio.checked = true;
      });
    });

    // Merge action
    container.querySelector('#ss-merge-btn').addEventListener('click', async () => {
      const keeperId = container.querySelector('input[name="keeper"]:checked')?.value;
      if (!keeperId) {
        showConfirmModal('Please select a performer to keep.', () => {});
        return;
      }

      const sourceIds = performers.filter(p => p.id !== keeperId).map(p => p.id);
      const btn = container.querySelector('#ss-merge-btn');

      try {
        btn.disabled = true;
        btn.textContent = 'Merging...';

        await RecommendationsAPI.mergePerformers(keeperId, sourceIds);
        await RecommendationsAPI.resolve(rec.id, 'merged', { keeper_id: keeperId });

        showSuccessAndReturn(btn, 'Merged!');
      } catch (e) {
        btn.textContent = `Failed: ${e.message}`;
        btn.classList.add('ss-btn-error');
        btn.disabled = false;
      }
    });

    // Dismiss action
    container.querySelector('#ss-dismiss-btn').addEventListener('click', async () => {
      const btn = container.querySelector('#ss-dismiss-btn');
      try {
        btn.disabled = true;
        btn.textContent = 'Dismissing...';
        await RecommendationsAPI.dismiss(rec.id, 'User dismissed');
        removeFromListCache(rec.id);
        currentState.view = 'list';
        currentState.selectedRec = null;
        renderCurrentView(document.getElementById('ss-recommendations'));
      } catch (e) {
        btn.textContent = `Failed: ${e.message}`;
        btn.disabled = false;
      }
    });
  }

  function renderDuplicateSceneFilesDetail(container, rec) {
    const details = rec.details;
    const files = details.files || [];

    container.innerHTML = `
      <div class="ss-detail-scene-files">
        <h2>${details.scene_title} <span id="ss-info-dup-files"></span></h2>
        <p class="ss-detail-subtitle">
          <a href="/scenes/${rec.target_id}" target="_blank">View Scene</a>
          ${details.studio?.name ? ` | ${details.studio.name}` : ''}
        </p>

        <div class="ss-files-summary">
          <span>Total: ${details.total_size_formatted}</span>
          <span class="ss-potential-savings">Potential savings: ${details.potential_savings_formatted}</span>
        </div>

        <div class="ss-files-list">
          ${files.map((f, i) => `
            <div class="ss-file-option ${f.is_suggested_keeper ? 'suggested' : ''}" data-id="${f.id}">
              <label class="ss-radio-label">
                <input type="radio" name="keeper" value="${f.id}" ${f.is_suggested_keeper ? 'checked' : ''} />
                <div class="ss-file-info">
                  <div class="ss-file-name">${f.basename}</div>
                  <div class="ss-file-meta">
                    <span class="ss-file-resolution">${f.resolution}</span>
                    <span class="ss-file-codec">${f.video_codec}</span>
                    <span class="ss-file-size">${f.size_formatted}</span>
                    <span class="ss-file-duration">${f.duration_formatted}</span>
                  </div>
                  <div class="ss-file-path">${f.path}</div>
                </div>
              </label>
              ${f.is_suggested_keeper ? '<span class="ss-suggested-badge">Best Quality</span>' : ''}
            </div>
          `).join('')}
        </div>

        <div class="ss-detail-actions">
          <button class="ss-btn ss-btn-danger ss-btn-delete" id="ss-delete-btn">
            Delete Other Files
          </button>
          <button class="ss-btn ss-btn-secondary" id="ss-dismiss-btn">
            Dismiss
          </button>
        </div>
      </div>
    `;

    // Add info icon
    const dfInfoSlot = container.querySelector('#ss-info-dup-files');
    if (dfInfoSlot) dfInfoSlot.appendChild(createInfoIcon(() => showHelpModal('Duplicate Scene Files', HELP_DUP_SCENE_FILES)));

    // Click anywhere on file card to select radio (except links)
    container.querySelectorAll('.ss-file-option').forEach(card => {
      card.style.cursor = 'pointer';
      card.addEventListener('click', (e) => {
        if (e.target.closest('a')) return;
        const radio = card.querySelector('input[type="radio"]');
        if (radio) radio.checked = true;
      });
    });

    // Delete action
    container.querySelector('#ss-delete-btn').addEventListener('click', async () => {
      const keeperId = container.querySelector('input[name="keeper"]:checked')?.value;
      if (!keeperId) {
        showConfirmModal('Please select a file to keep.', () => {});
        return;
      }

      const fileIdsToDelete = files.filter(f => f.id !== keeperId).map(f => f.id);
      const allFileIds = files.map(f => f.id);

      showConfirmModal(
        `Delete ${fileIdsToDelete.length} file(s)? This cannot be undone.`,
        async () => {
          const btn = container.querySelector('#ss-delete-btn');

          try {
            btn.disabled = true;
            btn.textContent = 'Deleting...';

            await RecommendationsAPI.deleteSceneFiles(
              rec.target_id,
              fileIdsToDelete,
              keeperId,
              allFileIds
            );
            await RecommendationsAPI.resolve(rec.id, 'deleted', {
              kept_file_id: keeperId,
              deleted_file_ids: fileIdsToDelete,
            });

            showSuccessAndReturn(btn, 'Deleted!');
          } catch (e) {
            const errMsg = e.message || '';
            // If file already deleted, resolve the recommendation anyway
            if (errMsg.includes('no rows in result set') || errMsg.includes('not found')) {
              try {
                await RecommendationsAPI.resolve(rec.id, 'deleted', {
                  kept_file_id: keeperId,
                  deleted_file_ids: fileIdsToDelete,
                  note: 'Files already deleted',
                });
                showSuccessAndReturn(btn, 'Already deleted - resolved');
                return;
              } catch (_) { /* fall through to error display */ }
            }
            btn.textContent = `Failed: ${errMsg}`;
            btn.classList.add('ss-btn-error');
            btn.disabled = false;
          }
        },
        { showDontAsk: true, storageKey: 'delete-scene-files' }
      );
    });

    // Dismiss action
    container.querySelector('#ss-dismiss-btn').addEventListener('click', async () => {
      const btn = container.querySelector('#ss-dismiss-btn');
      try {
        btn.disabled = true;
        btn.textContent = 'Dismissing...';
        await RecommendationsAPI.dismiss(rec.id, 'User dismissed');
        removeFromListCache(rec.id);
        currentState.view = 'list';
        currentState.selectedRec = null;
        renderCurrentView(document.getElementById('ss-recommendations'));
      } catch (e) {
        btn.textContent = `Failed: ${e.message}`;
        btn.disabled = false;
      }
    });
  }

  function disableAllActions(container) {
    container.querySelectorAll('.ss-detail-actions .ss-btn').forEach(function(b) { b.disabled = true; });
  }

  function enableAllActions(container) {
    container.querySelectorAll('.ss-detail-actions .ss-btn').forEach(function(b) { b.disabled = false; });
  }

  async function renderDuplicateScenesDetail(container, rec) {
    const details = rec.details;
    const sourceSceneId = String(details.source_scene_id || details.scene_a_id || '');
    const rawMatches = Array.isArray(details.duplicate_matches) && details.duplicate_matches.length > 0
      ? details.duplicate_matches
      : [{
          recommendation_id: rec.id,
          target_id: rec.target_id,
          match_scene_id: String(details.scene_b_id || ''),
          confidence: getDuplicateSceneConfidencePercent(rec),
          reasoning: details.reasoning || [],
          signal_breakdown: details.signal_breakdown || {},
          source_summary: details.source_summary || details.scene_a_summary || {},
          match_summary: details.match_summary || details.scene_b_summary || {},
        }];
    const matches = rawMatches
      .filter(match => match && match.match_scene_id)
      .map(match => ({
        ...match,
        recommendation_id: Number(match.recommendation_id),
        match_scene_id: String(match.match_scene_id),
        confidence: Number(match.confidence || 0),
      }))
      .sort((a, b) => {
        if (a.confidence !== b.confidence) return b.confidence - a.confidence;
        return Number(b.recommendation_id || 0) - Number(a.recommendation_id || 0);
      });

    container.innerHTML = '<div class="ss-loading">Loading scene details...</div>';

    let sourceScene;
    let activeMatchEntries = [];
    try {
      sourceScene = await RecommendationsAPI.getSceneDetail(sourceSceneId);
      activeMatchEntries = await Promise.all(
        matches.map(async (match) => {
          try {
            const scene = await RecommendationsAPI.getSceneDetail(match.match_scene_id);
            return { match, scene, selected: false };
          } catch (_) {
            return { match, scene: null, selected: false };
          }
        })
      );
    } catch (e) {
      container.innerHTML = '<div class="ss-error-state"><p>Failed to load scenes: ' + e.message + '</p></div>';
      return;
    }

    const sourceSummary = details.source_summary || details.scene_a_summary || {};

    function formatDuration(seconds) {
      if (!seconds) return 'N/A';
      const m = Math.floor(seconds / 60);
      const s = Math.round(seconds % 60);
      return m + ':' + String(s).padStart(2, '0');
    }

    function formatFileSize(bytes) {
      if (!bytes) return 'N/A';
      const gb = bytes / (1024 * 1024 * 1024);
      if (gb >= 1) return gb.toFixed(1) + ' GB';
      return (bytes / (1024 * 1024)).toFixed(0) + ' MB';
    }

    function renderLocalEntityLink(entityType, entityId, entityName) {
      if (!entityName) return '';
      if (!entityId) return escapeHtml(entityName);
      return '<a class="ss-detail-entity-link" href="/' + entityType + '/' + encodeURIComponent(String(entityId)) + '" target="_blank" rel="noopener">' + escapeHtml(entityName) + '</a>';
    }

    function renderLocalPerformerLinks(performers) {
      if (!performers || !performers.length) return '';
      return performers.map(function(p) {
        return renderLocalEntityLink('performers', p?.id, p?.name);
      }).filter(Boolean).join(', ');
    }

    function renderSummaryFallback(summary, id) {
      const title = summary?.title || `Scene ${id}`;
      const studio = summary?.studio ? '<li><strong>Studio:</strong> ' + escapeHtml(summary.studio) + '</li>' : '';
      const performers = summary?.performers?.length
        ? '<li><strong>Performers:</strong> ' + escapeHtml(summary.performers.join(', ')) + '</li>'
        : '';
      const duration = summary?.duration ? '<li><strong>Duration:</strong> ' + formatDuration(summary.duration) + '</li>' : '';
      return {
        title,
        metaHtml: studio + performers + duration,
      };
    }

    function renderSceneCard(scene, id, options = {}) {
      const {
        summaryFallback = null,
        matchMeta = null,
        showCheckbox = false,
        checkboxChecked = false,
        isSource = false,
        actionButtonsHtml = '',
      } = options;
      const file = scene?.files?.[0];
      const resolution = file ? file.width + 'x' + file.height : 'N/A';
      const screenshotUrl = relativeUrl(scene?.paths?.screenshot);
      const previewUrl = relativeUrl(scene?.paths?.preview);
      const studioLink = renderLocalEntityLink('studios', scene?.studio?.id, scene?.studio?.name);
      const performerLinks = renderLocalPerformerLinks(scene?.performers || []);
      const fallback = renderSummaryFallback(summaryFallback, id);
      const title = scene?.title || fallback.title || 'Unknown Scene';
      const metaHtml = [
        studioLink ? '<li><strong>Studio:</strong> ' + studioLink + '</li>' : '',
        performerLinks ? '<li><strong>Performers:</strong> ' + performerLinks + '</li>' : '',
        scene?.date ? '<li><strong>Date:</strong> ' + scene.date + '</li>' : '',
        (!scene?.date && summaryFallback?.date) ? '<li><strong>Date:</strong> ' + escapeHtml(summaryFallback.date) + '</li>' : '',
        '<li><strong>Duration:</strong> ' + formatDuration(file?.duration || summaryFallback?.duration) + '</li>',
        file ? '<li><strong>File:</strong> ' + resolution + ' &middot; ' + (file.video_codec || 'N/A') + ' &middot; ' + formatFileSize(file.size) + '</li>' : '',
      ].filter(Boolean).join('') || fallback.metaHtml || '<li><strong>Scene ID:</strong> ' + escapeHtml(id) + '</li>';
      const confidenceHtml = matchMeta
        ? '<div class="ss-dup-match-badges">' +
            '<span class="ss-signal-badge" style="color:' + (matchMeta.confidence >= 80 ? '#28a745' : matchMeta.confidence >= 60 ? '#ffc107' : '#6c757d') + '">' +
              Math.round(matchMeta.confidence) + '% confidence' +
            '</span>' +
            ((matchMeta.reasoning && matchMeta.reasoning[0]) ? '<span class="ss-signal-badge">' + escapeHtml(matchMeta.reasoning[0]) + '</span>' : '') +
          '</div>'
        : '';
      const checkboxHtml = showCheckbox
        ? '<label class="ss-dup-match-checkbox">' +
            '<input type="checkbox" class="ss-dup-match-select" data-rec-id="' + escapeHtml(String(matchMeta?.recommendation_id || '')) + '" data-scene-id="' + escapeHtml(String(id)) + '"' + (checkboxChecked ? ' checked' : '') + ' />' +
            '<span>Select for merge</span>' +
          '</label>'
        : '';
      const sourceNote = '';

      return '<div class="ss-dup-scene-card' + (isSource ? ' ss-dup-source-card' : '') + '" data-id="' + id + '">' +
        '<div class="ss-dup-scene-thumb">' +
          (screenshotUrl ? '<img src="' + screenshotUrl + '" alt="Scene ' + id + '" loading="lazy" onerror="this.style.display=\'none\'" />' : '<div class="ss-no-image">No Screenshot</div>') +
          (previewUrl ? '<video class="ss-dup-scene-preview" muted loop preload="none" data-src="' + previewUrl + '"></video>' : '') +
        '</div>' +
        '<div class="ss-dup-scene-header-row">' +
          '<div class="ss-dup-scene-title-block">' +
            '<h4><a href="/scenes/' + id + '" target="_blank" rel="noopener">' + escapeHtml(title) + '</a></h4>' +
            sourceNote +
          '</div>' +
          checkboxHtml +
        '</div>' +
        confidenceHtml +
        '<ul class="ss-dup-scene-meta">' + metaHtml + '</ul>' +
        actionButtonsHtml +
      '</div>';
    }

    function getSceneTitle(scene, summaryFallback, fallbackId) {
      return scene?.title || summaryFallback?.title || ('Scene ' + fallbackId);
    }

    function getSceneTitles() {
      const sceneTitles = {};
      sceneTitles[sourceSceneId] = getSceneTitle(sourceScene, sourceSummary, sourceSceneId);
      activeMatchEntries.forEach(function(entry) {
        sceneTitles[entry.match.match_scene_id] = getSceneTitle(entry.scene, entry.match.match_summary, entry.match.match_scene_id);
      });
      return sceneTitles;
    }

    function getActiveMatches() {
      return activeMatchEntries.map(function(entry) { return entry.match; });
    }

    function getSelectedMatches() {
      if (activeMatchEntries.length <= 1) {
        return getActiveMatches();
      }
      return activeMatchEntries
        .filter(function(entry) { return entry.selected; })
        .map(function(entry) { return entry.match; });
    }

    function updateCurrentRecommendationState() {
      if (!currentState.selectedRec || currentState.selectedRec.id !== rec.id) return;
      currentState.selectedRec = {
        ...currentState.selectedRec,
        details: {
          ...currentState.selectedRec.details,
          duplicate_matches: activeMatchEntries.map(function(entry) { return entry.match; }),
          scene_b_id: activeMatchEntries[0]?.match?.match_scene_id || null,
          confidence: activeMatchEntries[0]?.match?.confidence || currentState.selectedRec.details?.confidence || 0,
          reasoning: activeMatchEntries[0]?.match?.reasoning || currentState.selectedRec.details?.reasoning || [],
          signal_breakdown: activeMatchEntries[0]?.match?.signal_breakdown || currentState.selectedRec.details?.signal_breakdown || {},
          match_summary: activeMatchEntries[0]?.match?.match_summary || currentState.selectedRec.details?.match_summary || {},
        },
      };
    }

    function returnToRecommendationList() {
      removeFromListCache(rec.id);
      currentState.view = 'list';
      currentState.selectedRec = null;
      renderCurrentView(document.getElementById('ss-recommendations') || container);
    }

    function ensureSuccessful(result, fallbackMessage) {
      if (!result || result.success !== true) {
        throw new Error(fallbackMessage);
      }
    }

    function disableAllDupActions() {
      container.querySelectorAll('.ss-dup-action-btn, #ss-dismiss-btn, .ss-dup-match-select').forEach(function(el) {
        el.disabled = true;
      });
    }

    function enableAllDupActions() {
      container.querySelectorAll('.ss-dup-action-btn, #ss-dismiss-btn, .ss-dup-match-select').forEach(function(el) {
        el.disabled = false;
      });
      const keepMergeBtn = container.querySelector('#ss-dup-keep-merge-btn');
      if (keepMergeBtn) {
        keepMergeBtn.disabled = activeMatchEntries.length > 1 && getSelectedMatches().length === 0;
      }
    }

    function attachPreviewHover() {
      container.querySelectorAll('.ss-dup-scene-thumb').forEach(function(thumb) {
        var video = thumb.querySelector('.ss-dup-scene-preview');
        if (!video) return;
        var img = thumb.querySelector('img');

        thumb.addEventListener('mouseenter', function() {
          if (!video.src && video.dataset.src) {
            video.src = video.dataset.src;
          }
          if (img) img.style.opacity = '0';
          video.style.opacity = '1';
          video.play().catch(function() {});
        });

        thumb.addEventListener('mouseleave', function() {
          video.pause();
          if (img) img.style.opacity = '1';
          video.style.opacity = '0';
          setTimeout(function() { video.currentTime = 0; }, 200);
        });
      });
    }

    function findMatchEntryByRecommendationId(recId) {
      return activeMatchEntries.find(function(entry) {
        return String(entry.match.recommendation_id) === String(recId);
      });
    }

    async function handleDeleteMatch(entry, buttonEl) {
      const sceneTitles = getSceneTitles();
      const sceneTitle = sceneTitles[entry.match.match_scene_id] || ('Scene ' + entry.match.match_scene_id);
      const titleBlock = '<div style="margin:8px 0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:600">"' + escapeHtml(sceneTitle) + '"</div>';
      showConfirmModal(
        '',
        async function() {
          try {
            disableAllDupActions();
            buttonEl.textContent = 'Deleting...';

            const deleteResult = await RecommendationsAPI.deleteDuplicateSceneMatch(
              sourceSceneId,
              entry.match.match_scene_id,
              entry.match.recommendation_id,
              true,
            );
            ensureSuccessful(deleteResult, 'Delete failed');

            activeMatchEntries = activeMatchEntries.filter(function(candidate) {
              return String(candidate.match.recommendation_id) !== String(entry.match.recommendation_id);
            });
            updateCurrentRecommendationState();

            if (!activeMatchEntries.length) {
              showToast('Matched scene and its file permanently deleted. No duplicate matches remain for this source.', 'info');
              returnToRecommendationList();
              return;
            }

            renderDetail();
            showToast('Matched scene and its file permanently deleted.', 'info');
          } catch (e) {
            buttonEl.textContent = 'Failed: ' + e.message;
            buttonEl.classList.add('ss-btn-error');
            enableAllDupActions();
          }
        },
        {
          showDontAsk: true,
          storageKey: 'delete-dup-scene-match',
          htmlBody: 'Permanently delete matched scene and its file?' + titleBlock + 'This cannot be undone.',
        }
      );
    }

    async function handleMergeIntoMatch(entry, buttonEl) {
      const sceneTitles = getSceneTitles();
      const otherEntries = activeMatchEntries.filter(function(candidate) {
        return String(candidate.match.recommendation_id) !== String(entry.match.recommendation_id);
      });

      showConfirmModal(
        'Keep matched scene "' + sceneTitles[entry.match.match_scene_id] + '" and merge source scene "' + sceneTitles[sourceSceneId] + '" into it? Any other matched scenes in this review will be deleted.',
        async function() {
          try {
            disableAllDupActions();
            buttonEl.textContent = 'Merging...';

            const mergeResult = await RecommendationsAPI.mergeSourceIntoDuplicateSceneMatch(
              sourceSceneId,
              entry.match.match_scene_id,
              entry.match.recommendation_id,
              otherEntries.map(function(otherEntry) {
                return {
                  recommendation_id: otherEntry.match.recommendation_id,
                  scene_id: otherEntry.match.match_scene_id,
                };
              }),
            );
            ensureSuccessful(mergeResult, 'Merge failed');
            const deleteWarnings = Array.isArray(mergeResult.delete_failures)
              ? mergeResult.delete_failures
              : [];
            if (deleteWarnings.length) {
              buttonEl.textContent = 'Merged';
              buttonEl.classList.add('ss-btn-success');
              showToast(
                'Source merged, but some other matches could not be deleted: ' +
                  deleteWarnings.map(function(item) {
                    const label = sceneTitles[item.scene_id] || ('Scene ' + item.scene_id);
                    return label + ': ' + item.error;
                  }).join('; '),
                'warning',
                6000,
              );
              setTimeout(returnToRecommendationList, 1500);
              return;
            }

            showSuccessAndReturn(buttonEl, 'Merged!');
          } catch (e) {
            buttonEl.textContent = 'Failed: ' + e.message;
            buttonEl.classList.add('ss-btn-error');
            enableAllDupActions();
          }
        }
      );
    }

    function renderDetail() {
      const activeMatches = getActiveMatches();
      const multipleMatches = activeMatchEntries.length > 1;
      const topMatch = activeMatches[0] || null;
      const topConfidence = topMatch ? Number(topMatch.confidence || 0) : getDuplicateSceneConfidencePercent(rec);
      const confColor = topConfidence >= 80 ? '#28a745' : topConfidence >= 60 ? '#ffc107' : '#6c757d';
      const topReasoning = Array.isArray(topMatch?.reasoning) && topMatch.reasoning.length
        ? topMatch.reasoning
        : (details.reasoning || []);

      const matchCardsHtml = activeMatchEntries.map(function(entry) {
        return renderSceneCard(entry.scene, entry.match.match_scene_id, {
          summaryFallback: entry.match.match_summary || {},
          matchMeta: entry.match,
          showCheckbox: multipleMatches,
          checkboxChecked: !!entry.selected,
          actionButtonsHtml:
            '<div class="ss-dup-card-actions">' +
              '<button class="ss-btn ss-btn-primary ss-dup-match-keep-btn ss-dup-action-btn" data-rec-id="' + escapeHtml(String(entry.match.recommendation_id)) + '">Keep Match + Merge Source</button>' +
              '<button class="ss-btn ss-btn-danger ss-dup-match-delete-btn ss-dup-action-btn" data-rec-id="' + escapeHtml(String(entry.match.recommendation_id)) + '">Delete Match</button>' +
            '</div>',
        });
      }).join('');

      container.innerHTML =
        '<div class="ss-detail-dup-scenes">' +
          '<h2>Duplicate Scenes</h2>' +
          '<div class="ss-dup-confidence" style="color: ' + confColor + '">' +
            Math.round(topConfidence) + '% confidence' + (topReasoning[0] ? ' &mdash; ' + escapeHtml(topReasoning[0]) : '') +
          '</div>' +
          '<div class="ss-dup-group-note">' + activeMatchEntries.length + ' possible duplicate' + (activeMatchEntries.length !== 1 ? 's' : '') + ' for this source scene</div>' +
          '<div class="ss-dup-signals">' +
            topReasoning.slice(1).map(function(r) { return '<span class="ss-signal-badge">' + escapeHtml(r) + '</span>'; }).join('') +
          '</div>' +
          '<div class="ss-dup-scenes-grid">' +
            '<div class="ss-dup-source-column">' +
              '<div class="ss-dup-source-title">Source scene</div>' +
              renderSceneCard(sourceScene, sourceSceneId, {
                summaryFallback: sourceSummary,
                isSource: true,
              }) +
              '<div class="ss-dup-card-actions">' +
                '<button class="ss-btn ss-btn-primary ss-dup-keep-merge-btn ss-dup-action-btn" id="ss-dup-keep-merge-btn">' + (multipleMatches ? 'Keep + Merge Selected' : 'Keep + Merge') + '</button>' +
                '<button class="ss-btn ss-btn-danger ss-dup-delete-btn ss-dup-action-btn" id="ss-dup-delete-btn">Delete Source</button>' +
              '</div>' +
            '</div>' +
            '<div class="ss-dup-matches-column">' +
              '<div class="ss-dup-matches-title">Possible Matches</div>' +
              '<div class="ss-dup-matches-list">' + matchCardsHtml + '</div>' +
            '</div>' +
          '</div>' +
          '<div class="ss-detail-actions ss-detail-actions-center">' +
            '<button class="ss-btn ss-btn-secondary" id="ss-dismiss-btn">Not Duplicates</button>' +
          '</div>' +
        '</div>';

      const keepMergeBtn = container.querySelector('#ss-dup-keep-merge-btn');
      const deleteBtn = container.querySelector('#ss-dup-delete-btn');
      const dismissBtn = container.querySelector('#ss-dismiss-btn');
      const matchCheckboxes = Array.from(container.querySelectorAll('.ss-dup-match-select'));
      const sceneTitles = getSceneTitles();

      matchCheckboxes.forEach(function(cb) {
        cb.addEventListener('change', function() {
          const entry = findMatchEntryByRecommendationId(cb.dataset.recId);
          if (!entry) return;
          entry.selected = cb.checked;
          if (keepMergeBtn) {
            keepMergeBtn.disabled = activeMatchEntries.length > 1 && getSelectedMatches().length === 0;
          }
        });
      });

      if (keepMergeBtn) {
        keepMergeBtn.disabled = multipleMatches && getSelectedMatches().length === 0;
        keepMergeBtn.addEventListener('click', function() {
          const selectedMatches = getSelectedMatches();
          const selectedSceneIds = selectedMatches.map(function(match) { return match.match_scene_id; });
          const selectedRecIds = selectedMatches.map(function(match) { return match.recommendation_id; });
          const unselectedRecIds = activeMatches
            .filter(function(match) { return selectedRecIds.indexOf(match.recommendation_id) === -1; })
            .map(function(match) { return match.recommendation_id; });
          if (selectedSceneIds.length === 0) return;

          showConfirmModal(
            'Keep "' + sceneTitles[sourceSceneId] + '" and merge ' + selectedSceneIds.length + ' selected scene' + (selectedSceneIds.length !== 1 ? 's' : '') + ' into it? All unselected matches will be marked not duplicates.',
            async function() {
              try {
                disableAllDupActions();
                keepMergeBtn.textContent = 'Merging...';
                await RecommendationsAPI.mergeDuplicateSceneGroup(
                  sourceSceneId,
                  selectedSceneIds,
                  selectedRecIds,
                  unselectedRecIds,
                );
                showSuccessAndReturn(keepMergeBtn, 'Merged!');
              } catch (e) {
                keepMergeBtn.textContent = 'Failed: ' + e.message;
                keepMergeBtn.classList.add('ss-btn-error');
                enableAllDupActions();
              }
            }
          );
        });
      }

      if (deleteBtn) {
        deleteBtn.addEventListener('click', function() {
          const sourceTitle = sceneTitles[sourceSceneId] || ('Scene ' + sourceSceneId);
          const sourceTitleBlock = '<div style="margin:8px 0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:600">"' + escapeHtml(sourceTitle) + '"</div>';
          showConfirmModal(
            '',
            async function() {
              try {
                disableAllDupActions();
                deleteBtn.textContent = 'Deleting...';
                await RecommendationsAPI.deleteDuplicateSceneGroup(
                  sourceSceneId,
                  activeMatches.map(function(match) { return match.recommendation_id; }),
                  true,
                );
                showSuccessAndReturn(deleteBtn, 'Deleted!');
              } catch (e) {
                deleteBtn.textContent = 'Failed: ' + e.message;
                deleteBtn.classList.add('ss-btn-error');
                enableAllDupActions();
              }
            },
            {
              showDontAsk: true,
              storageKey: 'delete-dup-scene',
              htmlBody: 'Permanently delete source scene and its file?' + sourceTitleBlock + 'All current duplicate matches for this source will be closed. This cannot be undone.',
            }
          );
        });
      }

      container.querySelectorAll('.ss-dup-match-delete-btn').forEach(function(buttonEl) {
        buttonEl.addEventListener('click', function() {
          const entry = findMatchEntryByRecommendationId(buttonEl.dataset.recId);
          if (!entry) return;
          handleDeleteMatch(entry, buttonEl);
        });
      });

      container.querySelectorAll('.ss-dup-match-keep-btn').forEach(function(buttonEl) {
        buttonEl.addEventListener('click', function() {
          const entry = findMatchEntryByRecommendationId(buttonEl.dataset.recId);
          if (!entry) return;
          handleMergeIntoMatch(entry, buttonEl);
        });
      });

      attachPreviewHover();

      dismissBtn.addEventListener('click', async function() {
        try {
          dismissBtn.disabled = true;
          dismissBtn.textContent = 'Dismissing...';
          await RecommendationsAPI.dismissDuplicateSceneGroup(
            activeMatches.map(function(match) { return match.recommendation_id; }),
            'Marked not duplicates',
          );
          returnToRecommendationList();
        } catch (e) {
          dismissBtn.textContent = 'Failed: ' + e.message;
          dismissBtn.disabled = false;
        }
      });
    }

    renderDetail();
  }

  // ==================== Confirmation Modal ====================

  function showConfirmModal(message, onConfirm, options = {}) {
    const { showDontAsk = false, storageKey = null, htmlBody = null } = options;

    // Check "don't ask again"
    if (storageKey && localStorage.getItem(`ss-skip-confirm-${storageKey}`) === '1') {
      onConfirm();
      return;
    }

    // Use raw DOM to avoid Stash CSS interference
    const overlay = document.createElement('div');
    overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.6);display:flex;align-items:center;justify-content:center;z-index:10000;';

    const modal = document.createElement('div');
    modal.style.cssText = 'background:#2a2a2a;border:1px solid #444;border-radius:10px;padding:1.5rem;max-width:420px;width:auto;min-width:300px;box-shadow:0 8px 32px rgba(0,0,0,0.4);';

    const body = document.createElement('div');
    body.style.cssText = 'font-size:0.95rem;line-height:1.5;margin-bottom:1rem;color:#fff;';
    if (htmlBody) {
      body.innerHTML = htmlBody;
    } else {
      body.textContent = message;
    }
    modal.appendChild(body);

    let dontAskCheckbox = null;
    if (showDontAsk) {
      const label = document.createElement('label');
      label.style.cssText = 'display:flex;align-items:center;gap:6px;font-size:0.85rem;color:#888;margin-bottom:1rem;cursor:pointer;';
      dontAskCheckbox = document.createElement('input');
      dontAskCheckbox.type = 'checkbox';
      label.appendChild(dontAskCheckbox);
      label.appendChild(document.createTextNode("Don't ask again"));
      modal.appendChild(label);
    }

    const actions = document.createElement('div');
    actions.style.cssText = 'display:flex;gap:0.75rem;justify-content:flex-end;';

    const confirmBtn = document.createElement('button');
    confirmBtn.className = 'ss-btn ss-btn-danger';
    confirmBtn.style.cssText = 'padding:8px 18px;border-radius:6px;font-size:0.9rem;';
    confirmBtn.textContent = 'Confirm';

    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'ss-btn ss-btn-secondary';
    cancelBtn.style.cssText = 'padding:8px 18px;border-radius:6px;font-size:0.9rem;';
    cancelBtn.textContent = 'Cancel';

    actions.appendChild(confirmBtn);
    actions.appendChild(cancelBtn);
    modal.appendChild(actions);
    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    confirmBtn.addEventListener('click', () => {
      if (storageKey && dontAskCheckbox && dontAskCheckbox.checked) {
        localStorage.setItem(`ss-skip-confirm-${storageKey}`, '1');
      }
      closeOverlay();
      onConfirm();
    });

    cancelBtn.addEventListener('click', () => {
      closeOverlay();
    });

    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) closeOverlay();
    });

    const escHandler = (e) => {
      if (e.key === 'Escape') closeOverlay();
    };

    function closeOverlay() {
      document.removeEventListener('keydown', escHandler);
      overlay.remove();
    }

    document.addEventListener('keydown', escHandler);
  }

  // ==================== Help System ====================

  const HELP_FINGERPRINTS = `
    <div class="ss-help-section">
      <h4>What is the Identification Database?</h4>
      <p>The identification database contains face embeddings for known performers across multiple stash-box sources. It powers performer identification and face-based duplicate detection.</p>
    </div>
    <div class="ss-help-section">
      <h4>What are Fingerprints?</h4>
      <p>Scene fingerprints are face recognition data extracted from your local video files. Each fingerprint records which performers' faces were detected in a scene. Generate them from the Operations tab.</p>
    </div>
    <div class="ss-help-section">
      <h4>What Does "Need Refresh" Mean?</h4>
      <p>The face recognition database was updated with improved data (better face alignment, more performers). Scenes with outdated fingerprints should be regenerated for improved accuracy.</p>
    </div>
  `;

  // HELP_ACTION_RUNNER removed - actions now live on Operations tab

  const HELP_REC_TYPES = `
    <div class="ss-help-section">
      <h4>How Recommendations Work</h4>
      <p>Running an analysis creates recommendations. Each recommendation can be:</p>
      <ul class="ss-help-list">
        <li><strong>Acted on</strong> &mdash; merge performers, delete files, or apply upstream updates</li>
        <li><strong>Dismissed</strong> &mdash; hide the recommendation (view later from the Dismissed tab)</li>
        <li><strong>Left pending</strong> &mdash; come back to it later</li>
      </ul>
    </div>
    <div class="ss-help-section">
      <h4>Status Counts</h4>
      <p>Each type shows pending (needs review), resolved (action taken), and dismissed (hidden) counts. Click "View All" to browse recommendations of that type.</p>
    </div>
  `;

  const HELP_DUP_PERFORMER = `
    <div class="ss-help-section">
      <h4>Suggested Keeper</h4>
      <p>The performer with the most content (scenes, images, galleries) is suggested as the keeper. You can override this by selecting a different performer.</p>
    </div>
    <div class="ss-help-section">
      <h4>What Merging Does</h4>
      <p>Merging moves all scenes, images, and galleries from the other performer(s) to the keeper, then deletes the duplicates. This cannot be undone.</p>
    </div>
  `;

  const HELP_DUP_SCENE_FILES = `
    <div class="ss-help-section">
      <h4>Quality Scoring</h4>
      <p>Files are scored based on resolution (primary factor), bitrate, and codec. The highest-scoring file is marked as "Best Quality" and pre-selected.</p>
    </div>
    <div class="ss-help-tip">File deletion is permanent and irreversible. Make sure you've selected the right file to keep before proceeding.</div>
  `;

  const HELP_UPSTREAM_DETAIL = `
    <div class="ss-help-section">
      <h4>3-Column Comparison</h4>
      <p><strong>Local</strong> &mdash; your current value in Stash<br>
      <strong>Upstream</strong> &mdash; the current value on StashDB<br>
      <strong>Result</strong> &mdash; the value that will be written to Stash when you apply</p>
    </div>
    <div class="ss-help-section">
      <h4>Snapshots</h4>
      <p>A snapshot records what StashDB looked like when you last synced. The 3-way diff uses this to distinguish "you changed it locally" from "StashDB changed it upstream".</p>
    </div>
    <div class="ss-help-section">
      <h4>Dismiss Options</h4>
      <p><strong>Dismiss</strong> &mdash; hides this recommendation until the next analysis finds new changes.<br>
      <strong>Permanent dismiss</strong> &mdash; ignores this performer until you un-dismiss from the Dismissed tab.</p>
    </div>
  `;

  function createInfoIcon(onClick) {
    const btn = document.createElement('button');
    btn.className = 'ss-info-btn';
    btn.title = 'Help';
    btn.setAttribute('aria-label', 'Help');
    btn.textContent = 'i';
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      onClick();
    });
    return btn;
  }

  function showHelpModal(title, contentHtml) {
    const overlay = document.createElement('div');
    overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.6);display:flex;align-items:center;justify-content:center;z-index:10001;';

    const modal = document.createElement('div');
    modal.style.cssText = 'background:#2a2a2a;border:1px solid #444;border-radius:10px;max-width:560px;width:90%;max-height:80vh;display:flex;flex-direction:column;box-shadow:0 8px 32px rgba(0,0,0,0.4);';

    const header = document.createElement('div');
    header.style.cssText = 'display:flex;align-items:center;justify-content:space-between;padding:1rem 1.25rem;border-bottom:1px solid #444;';
    const titleEl = document.createElement('h3');
    titleEl.style.cssText = 'margin:0;font-size:1.1rem;font-weight:600;color:#fff;';
    titleEl.textContent = title;
    const closeBtn = document.createElement('button');
    closeBtn.style.cssText = 'background:none;border:none;font-size:1.5rem;color:#888;cursor:pointer;padding:0;line-height:1;';
    closeBtn.setAttribute('aria-label', 'Close');
    closeBtn.textContent = '\u00d7';
    closeBtn.addEventListener('click', () => closeOverlay());
    header.appendChild(titleEl);
    header.appendChild(closeBtn);

    const body = document.createElement('div');
    body.style.cssText = 'padding:1.25rem;overflow-y:auto;flex:1;';
    body.innerHTML = contentHtml;

    modal.appendChild(header);
    modal.appendChild(body);
    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) closeOverlay();
    });

    const escHandler = (e) => {
      if (e.key === 'Escape') closeOverlay();
    };

    function closeOverlay() {
      document.removeEventListener('keydown', escHandler);
      overlay.remove();
    }

    document.addEventListener('keydown', escHandler);
  }


  // ==================== Upstream Performer Validation ====================

  async function validatePerformerMerge(performerId, proposedName, proposedDisambig, proposedAliases) {
    const errors = [];
    let nameConflict = null;

    // 1. Name uniqueness - query all performers with this name
    if (proposedName) {
      try {
        const nameCheck = await SS.stashQuery(`
          query FindPerformersByName($name: String!) {
            findPerformers(performer_filter: { name: { value: $name, modifier: EQUALS } }) {
              performers { id name disambiguation image_path scene_count }
            }
          }
        `, { name: proposedName });
        const conflicts = (nameCheck?.findPerformers?.performers || [])
          .filter(p => p.id !== performerId);
        if (conflicts.length > 0) {
          const candidate = conflicts[0];
          // Skip conflict dialog when disambiguations differ (distinct people).
          // Allow conflict dialog when both share the same disambiguation
          // (likely duplicates) or when neither is disambiguated.
          const candidateDisambig = (candidate.disambiguation || '').trim().toLowerCase();
          const proposedDisambigVal = (proposedDisambig || '').trim().toLowerCase();
          const hasDisambig = !!(candidateDisambig || proposedDisambigVal);
          const sameDisambig = candidateDisambig === proposedDisambigVal;
          const isDistinctPerson = hasDisambig && !sameDisambig;
          if (!isDistinctPerson) {
            nameConflict = candidate;
          }
        }
      } catch (e) {
        console.warn('[Stash Sense] Name uniqueness check failed:', e);
      }
    }

    // 2. Alias can't match performer's own name (filtered in diff engine now, but keep as safety net)
    // Note: intentionally mutates proposedAliases in-place to clean up the caller's field data
    if (proposedName && proposedAliases) {
      const nameLower = proposedName.toLowerCase();
      const filtered = proposedAliases.filter(a => a.toLowerCase() !== nameLower);
      if (filtered.length < proposedAliases.length) {
        proposedAliases.length = 0;
        proposedAliases.push(...filtered);
      }
    }

    // 3. No duplicate aliases
    if (proposedAliases) {
      const seen = new Set();
      for (const alias of proposedAliases) {
        const lower = alias.toLowerCase();
        if (seen.has(lower)) {
          errors.push(`Duplicate alias: "${alias}"`);
        }
        seen.add(lower);
      }
    }

    return { errors, nameConflict };
  }


  /**
   * Mark a button as successful and navigate back to the recommendation list
   * after a short delay. Replaces the repeated setTimeout + navigate pattern.
   */
  function showSuccessAndReturn(buttonEl, successText) {
    buttonEl.textContent = successText;
    buttonEl.classList.add('ss-btn-success');
    setTimeout(() => {
      // The action that got us here (accept/merge/resolve/etc.) already
      // succeeded server-side -- drop this one recommendation from the
      // cached list instead of invalidating it, so returning to the list
      // is instant and doesn't reorder/reload everything else.
      removeFromListCache(currentState.selectedRec?.id);
      currentState.view = 'list';
      currentState.selectedRec = null;
      renderCurrentView(document.getElementById('ss-recommendations'));
    }, 1500);
  }

  function showToast(message, type = 'info', durationMs = 3200) {
    const text = String(message || '').trim();
    if (!text) return;

    let container = document.getElementById('ss-toast-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'ss-toast-container';
      container.className = 'ss-toast-container';
      document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = `ss-toast ss-toast-${type}`;
    toast.textContent = text;
    container.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add('show'));

    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => {
        toast.remove();
        if (!container.childElementCount) container.remove();
      }, 220);
    }, durationMs);
  }


  function showNameConflictDialog(performerId, conflictingPerformer, onMerge, onCancel, currentPerformer, onUpdateOnly) {
    const overlay = document.createElement('div');
    overlay.className = 'ss-modal-overlay';
    overlay.style.zIndex = '10001';

    const c = conflictingPerformer;
    const cur = currentPerformer || {};

    overlay.innerHTML = `
      <div class="ss-modal" style="max-width: 600px;">
        <div class="ss-modal-header">
          <h3>Name Conflict</h3>
          <button class="ss-modal-close" data-action="close">&times;</button>
        </div>
        <div class="ss-modal-body" style="padding: 16px;">
          <p style="margin: 0 0 12px; color: var(--ss-text-muted);">
            A performer with this name already exists. You can update fields without merging,
            merge the duplicate into this performer (scenes will be reassigned and the duplicate deleted),
            or skip this conflict.
          </p>
          <div class="ss-conflict-comparison">
            <div class="ss-conflict-card">
              <div class="ss-conflict-card-label">This Performer</div>
              <img src="${escapeHtml(relativeUrl(cur.image_path) || '')}" class="ss-conflict-thumb" onerror="this.style.display='none'" />
              <div class="ss-conflict-name">${escapeHtml(cur.name || 'Unknown')}</div>
              ${cur.disambiguation ? `<div class="ss-conflict-disambig">${escapeHtml(cur.disambiguation)}</div>` : ''}
              <div class="ss-conflict-meta">ID: ${escapeHtml(String(performerId))}</div>
              <a href="/performers/${performerId}" target="_blank" class="ss-conflict-link">View in Stash</a>
            </div>
            <div class="ss-conflict-card">
              <div class="ss-conflict-card-label">Conflicting Performer</div>
              <img src="${escapeHtml(relativeUrl(c.image_path) || '')}" class="ss-conflict-thumb" onerror="this.style.display='none'" />
              <div class="ss-conflict-name">${escapeHtml(c.name)}</div>
              ${c.disambiguation ? `<div class="ss-conflict-disambig">${escapeHtml(c.disambiguation)}</div>` : ''}
              <div class="ss-conflict-meta">ID: ${escapeHtml(c.id)} &middot; ${c.scene_count || 0} scenes</div>
              <a href="/performers/${c.id}" target="_blank" class="ss-conflict-link">View in Stash</a>
            </div>
          </div>
          <div style="display: flex; gap: 8px; justify-content: flex-end; margin-top: 16px;">
            <button class="ss-btn" data-action="skip">Skip</button>
            <button class="ss-btn" data-action="merge">Merge &amp; Continue</button>
            <button class="ss-btn ss-btn-primary" data-action="update-only">Update Fields Only</button>
          </div>
        </div>
      </div>
    `;

    function closeDialog() {
      overlay.remove();
      document.removeEventListener('keydown', escHandler);
    }

    overlay.addEventListener('click', (e) => {
      const action = e.target.dataset?.action || e.target.closest('[data-action]')?.dataset?.action;
      if (action === 'close' || action === 'skip') {
        closeDialog();
        onCancel();
      } else if (action === 'merge') {
        closeDialog();
        onMerge(c.id);
      } else if (action === 'update-only') {
        closeDialog();
        if (onUpdateOnly) onUpdateOnly();
      } else if (e.target === overlay) {
        closeDialog();
        onCancel();
      }
    });

    document.body.appendChild(overlay);

    function escHandler(e) {
      if (e.key === 'Escape') {
        closeDialog();
        onCancel();
      }
    }
    document.addEventListener('keydown', escHandler);
  }

  // ==================== Upstream Performer Detail ====================

  async function renderUpstreamPerformerDetail(container, rec) {
    const details = rec.details;
    const rawChanges = details.changes || [];
    const performerId = details.performer_id;
    const upstreamStatus = details.upstream_status || 'active';
    const relink = details.relink || null;
    const needsRelink = upstreamStatus === 'merged'
      && Array.isArray(relink?.stash_ids_after_relink)
      && relink.stash_ids_after_relink.length > 0;

    // Check if performer still exists in Stash (may have been deleted)
    try {
      const performer = await SS.getPerformer(performerId);
      if (!performer) {
        try {
          await RecommendationsAPI.resolve(rec.id, 'auto_resolved', { note: 'Performer was deleted from Stash' });
        } catch (_) {}
        removeFromListCache(rec.id);
        currentState.view = 'list';
        currentState.selectedRec = null;
        renderCurrentView(document.getElementById('ss-recommendations'));
        return;
      }
    } catch (_) {
      // If query fails, proceed with rendering — the update will fail with a clear error
    }

    const changes = filterRealChanges(rawChanges);

    // If all changes were filtered out, auto-resolve only for normal upstream updates.
    // For merged/deleted upstream performers we keep the rec visible for explicit user action.
    if (changes.length === 0 && upstreamStatus === 'active') {
      try {
        await RecommendationsAPI.resolve(rec.id, 'accepted_no_changes', { note: 'All differences were cosmetic' });
      } catch (_) {}
      removeFromListCache(rec.id);
      currentState.view = 'list';
      currentState.selectedRec = null;
      renderCurrentView(document.getElementById('ss-recommendations'));
      return;
    }

    // Load normalize setting
    let normalizeEnum = true;
    try {
      const val = await RecommendationsAPI.getUserSetting('normalize_enum_display');
      normalizeEnum = val !== false;
    } catch (_) {}

    // Display value helper - applies enum normalization if enabled
    // cup_size (e.g. "DD") and country (ISO code e.g. "US") must stay ALL_CAPS
    const _noNormalizeFields = new Set(['cup_size', 'country']);
    function displayValue(val, fieldName) {
      const formatted = formatFieldValue(val);
      if (!normalizeEnum || (fieldName && _noNormalizeFields.has(fieldName))) return formatted;
      return normalizeEnumValue(formatted);
    }

    // Smart default: prefer upstream (stash-box is source of truth)
    function smartDefault(localVal, upstreamVal) {
      const upstreamEmpty = isEmptyLikeValue(upstreamVal);
      if (!upstreamEmpty) return 'upstream';
      return 'local';
    }

    // Build the header with gear icon
    const wrapper = document.createElement('div');
    wrapper.className = 'ss-detail-upstream-performer';

    // Header
    const headerDiv = document.createElement('div');
    headerDiv.className = 'ss-upstream-header';
    headerDiv.innerHTML = `
      <img src="${relativeUrl(details.performer_image_path) || ''}" alt="${details.performer_name || ''}" onerror="this.style.display='none'" />
      <div style="flex:1;">
        <h2 style="margin: 0 0 4px 0;">
          <a href="/performers/${performerId}" target="_blank">${details.performer_name || 'Unknown'}</a>
        </h2>
        <a href="${details.endpoint.replace(/\/graphql$/, '')}/performers/${details.stash_box_id}" target="_blank" class="ss-upstream-endpoint-badge">${details.endpoint_name || 'Upstream'}</a>
      </div>
    `;
    headerDiv.appendChild(createInfoIcon(() => showHelpModal('Upstream Performer Changes', HELP_UPSTREAM_DETAIL)));
    wrapper.appendChild(headerDiv);

    if (upstreamStatus === 'merged' && relink) {
      const mergeNotice = document.createElement('div');
      mergeNotice.className = 'ss-scene-local-context';
      mergeNotice.innerHTML = `
        <span class="ss-scene-local-context-label">Merged Upstream:</span>
        performer ID changed from <code>${escapeHtml(relink.old_stashbox_id || details.stash_box_id || '')}</code>
        to <code>${escapeHtml(relink.new_stashbox_id || '')}</code>. Applying will relink this local performer.
      `;
      wrapper.appendChild(mergeNotice);
    } else if (upstreamStatus === 'deleted') {
      const deletedNotice = document.createElement('div');
      deletedNotice.className = 'ss-scene-local-context';
      deletedNotice.innerHTML = `
        <span class="ss-scene-local-context-label">Deleted Upstream:</span>
        this upstream performer no longer exists. Review this local performer and either resolve or dismiss.
      `;
      wrapper.appendChild(deletedNotice);
    }

    // Quick select buttons
    const quickActions = document.createElement('div');
    quickActions.className = 'ss-upstream-quick-actions';
    const keepAllBtn = document.createElement('button');
    keepAllBtn.textContent = 'Keep All Local';
    const acceptAllBtn = document.createElement('button');
    acceptAllBtn.textContent = 'Accept All Upstream';
    quickActions.appendChild(keepAllBtn);
    quickActions.appendChild(acceptAllBtn);
    wrapper.appendChild(quickActions);

    // Build field rows
    changes.forEach((change, idx) => {
      const mergeType = change.merge_type || 'simple';
      const fieldRow = document.createElement('div');
      fieldRow.className = 'ss-upstream-field-row';
      fieldRow.dataset.fieldIndex = idx;
      fieldRow.dataset.fieldKey = change.field;
      fieldRow.dataset.mergeType = mergeType;

      // Field label
      const label = document.createElement('div');
      label.className = 'ss-upstream-field-label';
      label.textContent = change.field_label || change.field;
      fieldRow.appendChild(label);

      if (mergeType === 'alias_list') {
        // Alias list: 2-column sub-layout (items on left, result summary on right)
        renderAliasListField(fieldRow, change, idx);
      } else {
        // Simple, name, text: 3-column layout
        renderCompareField(fieldRow, change, idx, mergeType, displayValue, smartDefault);
      }

      wrapper.appendChild(fieldRow);
    });

    // Validation errors
    const errorDiv = document.createElement('div');
    errorDiv.id = 'ss-upstream-validation-errors';
    errorDiv.className = 'ss-upstream-validation-error';
    errorDiv.style.display = 'none';
    wrapper.appendChild(errorDiv);

    // Action bar
    const actionBar = document.createElement('div');
    actionBar.className = 'ss-upstream-action-bar';

    const applyBtn = document.createElement('button');
    applyBtn.className = 'ss-btn ss-upstream-apply-btn';
    applyBtn.id = 'ss-upstream-apply';
    applyBtn.textContent = upstreamStatus === 'deleted' ? 'Mark Reviewed' : 'Apply Selected Changes';

    const dismissDropdown = document.createElement('div');
    dismissDropdown.style.cssText = 'position:relative;';
    const dismissToggle = document.createElement('button');
    dismissToggle.className = 'ss-btn ss-upstream-dismiss-btn';
    dismissToggle.textContent = 'Dismiss';
    const dismissMenu = document.createElement('div');
    dismissMenu.style.cssText = 'display:none;position:absolute;bottom:100%;left:0;background:#2a2a2a;border:1px solid #444;border-radius:6px;padding:4px 0;min-width:220px;z-index:10;';

    const dismissOptions = [
      { label: 'Dismiss this update', permanent: false },
      { label: 'Never show for this performer', permanent: true },
    ];

    dismissOptions.forEach(opt => {
      const optBtn = document.createElement('button');
      optBtn.style.cssText = 'display:block;width:100%;text-align:left;padding:8px 12px;background:none;border:none;color:#fff;cursor:pointer;font-size:0.85rem;';
      optBtn.textContent = opt.label;
      optBtn.addEventListener('mouseenter', () => { optBtn.style.background = 'rgba(255,255,255,0.05)'; });
      optBtn.addEventListener('mouseleave', () => { optBtn.style.background = 'none'; });
      optBtn.addEventListener('click', async () => {
        dismissMenu.style.display = 'none';
        dismissToggle.disabled = true;
        dismissToggle.textContent = 'Dismissing...';
        try {
          await RecommendationsAPI.dismissUpstream(rec.id, 'User dismissed', opt.permanent);
          removeFromListCache(rec.id);
          currentState.view = 'list';
          currentState.selectedRec = null;
          renderCurrentView(document.getElementById('ss-recommendations'));
        } catch (e) {
          dismissToggle.textContent = `Failed: ${e.message}`;
          dismissToggle.disabled = false;
        }
      });
      dismissMenu.appendChild(optBtn);
    });

    dismissToggle.addEventListener('click', () => {
      dismissMenu.style.display = dismissMenu.style.display === 'none' ? 'block' : 'none';
    });

    document.addEventListener('click', function closeDismissMenu(e) {
      if (!dismissToggle.contains(e.target) && !dismissMenu.contains(e.target)) {
        dismissMenu.style.display = 'none';
      }
      if (!document.contains(container)) {
        document.removeEventListener('click', closeDismissMenu);
      }
    });

    dismissDropdown.appendChild(dismissToggle);
    dismissDropdown.appendChild(dismissMenu);
    actionBar.appendChild(applyBtn);
    actionBar.appendChild(dismissDropdown);
    wrapper.appendChild(actionBar);

    container.innerHTML = '';
    container.appendChild(wrapper);

    // === Quick select wiring ===
    keepAllBtn.addEventListener('click', () => {
      wrapper.querySelectorAll('.ss-upstream-field-row').forEach(row => {
        const mt = row.dataset.mergeType;
        if (mt === 'readonly') return;
        if (mt === 'alias_list') {
          // Check all local/both items, uncheck upstream-only
          row.querySelectorAll('.ss-upstream-alias-item').forEach(item => {
            const cb = item.querySelector('input[type="checkbox"]');
            cb.checked = item.classList.contains('local-only') || item.classList.contains('both');
          });
          updateAliasResultSummary(row);
        } else {
          const localCb = row.querySelector('.ss-upstream-cb-local');
          const upstreamCb = row.querySelector('.ss-upstream-cb-upstream');
          const resultInput = row.querySelector('.ss-upstream-result-input, .ss-upstream-textarea');
          if (localCb && upstreamCb && resultInput) {
            localCb.checked = true;
            upstreamCb.checked = false;
            const change = changes[parseInt(row.dataset.fieldIndex)];
            resultInput.value = formatFieldValue(change.local_value) === '(empty)' ? '' : formatFieldValue(change.local_value);
          }
        }
      });
    });

    acceptAllBtn.addEventListener('click', () => {
      wrapper.querySelectorAll('.ss-upstream-field-row').forEach(row => {
        const mt = row.dataset.mergeType;
        if (mt === 'readonly') return;
        if (mt === 'alias_list') {
          // Check all items (merge all)
          row.querySelectorAll('.ss-upstream-alias-item input[type="checkbox"]').forEach(cb => { cb.checked = true; });
          updateAliasResultSummary(row);
        } else {
          const localCb = row.querySelector('.ss-upstream-cb-local');
          const upstreamCb = row.querySelector('.ss-upstream-cb-upstream');
          const resultInput = row.querySelector('.ss-upstream-result-input, .ss-upstream-textarea');
          if (localCb && upstreamCb && resultInput) {
            localCb.checked = false;
            upstreamCb.checked = true;
            const change = changes[parseInt(row.dataset.fieldIndex)];
            resultInput.value = formatFieldValue(change.upstream_value) === '(empty)' ? '' : formatFieldValue(change.upstream_value);
          }
          // For name type: also check "add old name as alias" when switching to upstream
          const aliasOpt = row.querySelector('.ss-upstream-name-alias-cb');
          if (aliasOpt) aliasOpt.checked = true;
        }
      });
    });

    // === Apply handler ===
    applyBtn.addEventListener('click', async () => {
      errorDiv.style.display = 'none';
      errorDiv.innerHTML = '';
      const fields = {};

      wrapper.querySelectorAll('.ss-upstream-field-row').forEach(row => {
        const fieldKey = row.dataset.fieldKey;
        const mergeType = row.dataset.mergeType;
        const fieldIndex = parseInt(row.dataset.fieldIndex);
        const change = changes[fieldIndex];

        if (mergeType === 'readonly') {
          return;
        }

        if (mergeType === 'alias_list') {
          const checkedAliases = [];
          row.querySelectorAll('.ss-upstream-alias-item input[type="checkbox"]:checked').forEach(cb => {
            checkedAliases.push(cb.value);
          });
          fields[fieldKey] = checkedAliases;
        } else {
          const resultInput = row.querySelector('.ss-upstream-result-input, .ss-upstream-textarea');
          if (!resultInput) return;

          const resultVal = resultInput.value.trim();
          const localStr = formatFieldValue(change.local_value) === '(empty)' ? '' : String(change.local_value || '');

          // Skip if result equals local (no change)
          if (resultVal === localStr) {
            // But check if name type has alias add
            if (mergeType === 'name') {
              const aliasCb = row.querySelector('.ss-upstream-name-alias-cb');
              if (aliasCb && aliasCb.checked) {
                const upstreamCb = row.querySelector('.ss-upstream-cb-upstream');
                if (upstreamCb && !upstreamCb.checked) {
                  // Keeping local name, add upstream as alias
                  fields['_alias_add'] = fields['_alias_add'] || [];
                  fields['_alias_add'].push(String(change.upstream_value || ''));
                }
              }
            }
            return;
          }

          // Result differs from local -> apply change
          fields[fieldKey] = resultVal;

          // Handle name merge alias addition
          if (mergeType === 'name') {
            const aliasCb = row.querySelector('.ss-upstream-name-alias-cb');
            if (aliasCb && aliasCb.checked) {
              const upstreamCb = row.querySelector('.ss-upstream-cb-upstream');
              if (upstreamCb && upstreamCb.checked) {
                // Accepting upstream name, demote local to alias
                fields['_alias_add'] = fields['_alias_add'] || [];
                fields['_alias_add'].push(String(change.local_value || ''));
              }
            }
          }
        }
      });

      if (needsRelink) {
        fields.stash_ids = relink.stash_ids_after_relink;
      }

      // Check if any changes were selected
      const hasChanges = Object.keys(fields).length > 0;
      if (!hasChanges) {
        try {
          applyBtn.disabled = true;
          applyBtn.textContent = upstreamStatus === 'deleted' ? 'Reviewing...' : 'Resolving...';
          await RecommendationsAPI.resolve(rec.id, 'accepted_no_changes', {});
          showSuccessAndReturn(applyBtn, upstreamStatus === 'deleted' ? 'Reviewed' : 'Done!');
        } catch (e) {
          applyBtn.textContent = `Failed: ${e.message}`;
          applyBtn.classList.add('ss-btn-error');
          applyBtn.disabled = false;
        }
        return;
      }

      const hasEditableChanges = Object.keys(fields).some(k => k !== 'stash_ids');

      // Relink-only flow (merged upstream performer, no field edits selected)
      if (!hasEditableChanges && needsRelink) {
        try {
          applyBtn.disabled = true;
          applyBtn.textContent = 'Relinking...';
          await RecommendationsAPI.updatePerformer(performerId, fields);
          await RecommendationsAPI.resolve(rec.id, 'relinked', { stash_ids: fields.stash_ids });
          showSuccessAndReturn(applyBtn, 'Relinked!');
        } catch (e) {
          errorDiv.innerHTML = `<div>${escapeHtml(e.message)}</div>`;
          errorDiv.style.display = 'block';
          applyBtn.textContent = 'Apply Selected Changes';
          applyBtn.classList.remove('ss-btn-error');
          applyBtn.disabled = false;
        }
        return;
      }

      // Run validation before applying
      applyBtn.disabled = true;
      applyBtn.textContent = 'Validating...';

      const proposedName = fields.name || details.performer_name;
      // Get disambiguation: from accepted fields first, then from changes, then from stored details
      let proposedDisambig = fields.disambiguation || null;
      if (!proposedDisambig) {
        const disambigChange = changes.find(c => c.field === 'disambiguation');
        if (disambigChange) proposedDisambig = disambigChange.local_value || null;
      }
      if (!proposedDisambig) {
        proposedDisambig = details.performer_disambiguation || null;
      }
      const proposedAliases = fields.aliases || fields._alias_add || [];

      const { errors: validationErrors, nameConflict } = await validatePerformerMerge(
        performerId, proposedName, proposedDisambig, proposedAliases
      );

      if (validationErrors.length > 0) {
        errorDiv.innerHTML = validationErrors.map(e => `<div>${escapeHtml(e)}</div>`).join('');
        errorDiv.style.display = 'block';
        applyBtn.textContent = 'Apply Selected Changes';
        applyBtn.disabled = false;
        return;
      }

      // Handle name conflict with a dialog instead of blocking
      if (nameConflict) {
        applyBtn.textContent = 'Apply Selected Changes';
        applyBtn.disabled = false;
        showNameConflictDialog(performerId, nameConflict,
          async (conflictId) => {
            // User chose "Merge & Continue"
            applyBtn.disabled = true;
            applyBtn.textContent = 'Merging & Applying...';
            errorDiv.style.display = 'none';
            try {
              await RecommendationsAPI.mergePerformers(performerId, [conflictId]);
            } catch (mergeErr) {
              const msg = String(mergeErr.message || mergeErr);
              // If the conflicting performer was already deleted, the merge is unnecessary — proceed
              if (msg.includes('not found') || msg.includes('does not exist') || msg.includes('404')) {
                console.warn('[Stash Sense] Conflicting performer already deleted, skipping merge');
              } else {
                errorDiv.innerHTML = `<div>Merge failed: ${escapeHtml(msg)}</div>`;
                errorDiv.style.display = 'block';
                applyBtn.textContent = 'Apply Selected Changes';
                applyBtn.disabled = false;
                return;
              }
            }
            try {
              await RecommendationsAPI.updatePerformer(performerId, fields);
              await RecommendationsAPI.resolve(rec.id, 'applied', { fields, auto_merged: conflictId });
              showSuccessAndReturn(applyBtn, 'Merged & Applied!');
            } catch (updateErr) {
              errorDiv.innerHTML = `<div>${escapeHtml(updateErr.message)}</div>`;
              errorDiv.style.display = 'block';
              applyBtn.textContent = 'Apply Selected Changes';
              applyBtn.disabled = false;
            }
          },
          () => { /* User chose skip */ },
          { name: details.performer_name, image_path: relativeUrl(details.performer_image_path), disambiguation: proposedDisambig },
          async () => {
            // User chose "Update Fields Only" — apply without merging.
            // Strip name, disambiguation (could erase what distinguishes this performer),
            // and _alias_add entries that conflict with existing performer names.
            const safeFields = Object.assign({}, fields);
            delete safeFields.name;
            delete safeFields.disambiguation;
            if (safeFields._alias_add) {
              const strippedAliases = [];
              // Filter aliases that match any existing performer name
              const remaining = [];
              for (const alias of safeFields._alias_add) {
                try {
                  const check = await SS.stashQuery(`
                    query FindPerformersByName($name: String!) {
                      findPerformers(performer_filter: { name: { value: $name, modifier: EQUALS } }) {
                        performers { id name }
                      }
                    }
                  `, { name: alias });
                  const matches = (check?.findPerformers?.performers || [])
                    .filter(p => p.id !== performerId);
                  if (matches.length > 0) {
                    strippedAliases.push(alias);
                  } else {
                    remaining.push(alias);
                  }
                } catch (e) {
                  remaining.push(alias); // keep on query failure
                }
              }
              if (strippedAliases.length > 0) {
                console.warn('[Stash Sense] Stripped aliases matching existing performers:', strippedAliases);
                const note = document.createElement('div');
                note.style.cssText = 'color: #f59e0b; font-size: 0.8rem; margin-top: 4px;';
                note.textContent = `Skipped alias${strippedAliases.length > 1 ? 'es' : ''} matching existing performer${strippedAliases.length > 1 ? 's' : ''}: ${strippedAliases.join(', ')}`;
                errorDiv.appendChild(note);
                errorDiv.style.display = 'block';
              }
              safeFields._alias_add = remaining;
              if (safeFields._alias_add.length === 0) delete safeFields._alias_add;
            }

            // If no fields remain after stripping, just resolve without calling update
            const hasFields = Object.keys(safeFields).length > 0;
            if (!hasFields) {
              try {
                await RecommendationsAPI.resolve(rec.id, 'applied', { skipped_name: true, no_other_fields: true });
                applyBtn.disabled = true;
                showSuccessAndReturn(applyBtn, 'Resolved (name skipped)');
              } catch (resolveErr) {
                errorDiv.innerHTML = `<div>${escapeHtml(resolveErr.message)}</div>`;
                errorDiv.style.display = 'block';
                applyBtn.textContent = 'Apply Selected Changes';
                applyBtn.disabled = false;
              }
              return;
            }

            applyBtn.disabled = true;
            applyBtn.textContent = 'Applying...';
            errorDiv.style.display = 'none';
            try {
              await RecommendationsAPI.updatePerformer(performerId, safeFields);
              await RecommendationsAPI.resolve(rec.id, 'applied', { fields: safeFields, skipped_name: true });
              showSuccessAndReturn(applyBtn, 'Applied!');
            } catch (updateErr) {
              errorDiv.innerHTML = `<div>${escapeHtml(updateErr.message)}</div>`;
              errorDiv.style.display = 'block';
              applyBtn.textContent = 'Apply Selected Changes';
              applyBtn.disabled = false;
            }
          }
        );
        return;
      }

      try {
        applyBtn.textContent = 'Applying...';
        const result = await RecommendationsAPI.updatePerformer(performerId, fields);
        await RecommendationsAPI.resolve(rec.id, 'applied', { fields });

        showSuccessAndReturn(applyBtn, result?.auto_merged ? 'Merged & Applied!' : 'Applied!');
      } catch (e) {
        let errorMsg = e.message;
        if (errorMsg.includes('different disambiguation') || errorMsg.includes('cannot be auto-merged')) {
          errorMsg = 'Name conflict: a performer with this name has a different disambiguation — they are different people. Use "Update Fields Only" to apply other changes without the name.';
        } else if (errorMsg.includes('duplicate') || errorMsg.includes('alias')) {
          errorMsg = `Alias conflict: ${errorMsg}. Try removing duplicate aliases.`;
        }
        errorDiv.innerHTML = `<div>${escapeHtml(errorMsg)}</div>`;
        errorDiv.style.display = 'block';
        applyBtn.textContent = 'Apply Selected Changes';
        applyBtn.classList.remove('ss-btn-error');
        applyBtn.disabled = false;
      }
    });
  }

  /**
   * Upstream Tag Detail View
   * Simpler than performer — only name, description, aliases. No image, no compound fields.
   */
  async function renderUpstreamTagDetail(container, rec) {
    const details = rec.details;
    const rawChanges = details.changes || [];
    const tagId = details.tag_id;

    const changes = filterRealChanges(rawChanges);

    // If all changes were filtered out, auto-resolve and go back
    if (changes.length === 0) {
      try {
        await RecommendationsAPI.resolve(rec.id, 'accepted_no_changes', { note: 'All differences were cosmetic' });
      } catch (_) {}
      removeFromListCache(rec.id);
      currentState.view = 'list';
      currentState.selectedRec = null;
      renderCurrentView(document.getElementById('ss-recommendations'));
      return;
    }

    // Display value helper
    function displayValue(val) {
      return formatFieldValue(val);
    }

    // Smart default: prefer upstream (stash-box is source of truth)
    function smartDefault(localVal, upstreamVal) {
      const upstreamEmpty = isEmptyLikeValue(upstreamVal);
      if (!upstreamEmpty) return 'upstream';
      return 'local';
    }

    const wrapper = document.createElement('div');
    wrapper.className = 'ss-detail-upstream-performer'; // reuse performer styles

    // Header (tag icon instead of image)
    const headerDiv = document.createElement('div');
    headerDiv.className = 'ss-upstream-header';
    headerDiv.innerHTML = `
      <div class="ss-rec-tag-icon" style="width:48px;height:48px;display:flex;align-items:center;justify-content:center;border-radius:8px;background:rgba(255,255,255,0.05);flex-shrink:0;">
        <svg viewBox="0 0 24 24" width="28" height="28" fill="currentColor"><path d="M21.41 11.58l-9-9C12.05 2.22 11.55 2 11 2H4c-1.1 0-2 .9-2 2v7c0 .55.22 1.05.59 1.42l9 9c.36.36.86.58 1.41.58.55 0 1.05-.22 1.41-.59l7-7c.37-.36.59-.86.59-1.41 0-.55-.23-1.06-.59-1.42zM5.5 7C4.67 7 4 6.33 4 5.5S4.67 4 5.5 4 7 4.67 7 5.5 6.33 7 5.5 7z"/></svg>
      </div>
      <div style="flex:1;">
        <h2 style="margin: 0 0 4px 0;">
          <a href="/tags/${tagId}" target="_blank">${details.tag_name || 'Unknown'}</a>
        </h2>
        <a href="${details.endpoint.replace(/\/graphql$/, '')}/tags/${details.stash_box_id}" target="_blank" class="ss-upstream-endpoint-badge">${details.endpoint_name || 'Upstream'}</a>
      </div>
    `;
    wrapper.appendChild(headerDiv);

    // Quick select buttons
    const quickActions = document.createElement('div');
    quickActions.className = 'ss-upstream-quick-actions';
    const keepAllBtn = document.createElement('button');
    keepAllBtn.textContent = 'Keep All Local';
    const acceptAllBtn = document.createElement('button');
    acceptAllBtn.textContent = 'Accept All Upstream';
    quickActions.appendChild(keepAllBtn);
    quickActions.appendChild(acceptAllBtn);
    wrapper.appendChild(quickActions);

    // Build field rows
    changes.forEach((change, idx) => {
      const mergeType = change.merge_type || 'simple';
      const fieldRow = document.createElement('div');
      fieldRow.className = 'ss-upstream-field-row';
      fieldRow.dataset.fieldIndex = idx;
      fieldRow.dataset.fieldKey = change.field;
      fieldRow.dataset.mergeType = mergeType;

      const label = document.createElement('div');
      label.className = 'ss-upstream-field-label';
      label.textContent = change.field_label || change.field;
      fieldRow.appendChild(label);

      if (mergeType === 'alias_list') {
        renderAliasListField(fieldRow, change, idx);
      } else {
        renderCompareField(fieldRow, change, idx, mergeType, displayValue, smartDefault);
      }

      wrapper.appendChild(fieldRow);
    });

    // Validation errors
    const errorDiv = document.createElement('div');
    errorDiv.id = 'ss-upstream-validation-errors';
    errorDiv.className = 'ss-upstream-validation-error';
    errorDiv.style.display = 'none';
    wrapper.appendChild(errorDiv);

    // Action bar
    const actionBar = document.createElement('div');
    actionBar.className = 'ss-upstream-action-bar';

    const applyBtn = document.createElement('button');
    applyBtn.className = 'ss-btn ss-upstream-apply-btn';
    applyBtn.textContent = 'Apply Selected Changes';

    // Dismiss dropdown (reuse performer pattern)
    const dismissDropdown = document.createElement('div');
    dismissDropdown.style.cssText = 'position:relative;';
    const dismissToggle = document.createElement('button');
    dismissToggle.className = 'ss-btn ss-upstream-dismiss-btn';
    dismissToggle.textContent = 'Dismiss';
    const dismissMenu = document.createElement('div');
    dismissMenu.style.cssText = 'display:none;position:absolute;bottom:100%;left:0;background:#2a2a2a;border:1px solid #444;border-radius:6px;padding:4px 0;min-width:220px;z-index:10;';

    const dismissOptions = [
      { label: 'Dismiss this update', permanent: false },
      { label: 'Never show for this tag', permanent: true },
    ];

    dismissOptions.forEach(opt => {
      const optBtn = document.createElement('button');
      optBtn.style.cssText = 'display:block;width:100%;text-align:left;padding:8px 12px;background:none;border:none;color:#fff;cursor:pointer;font-size:0.85rem;';
      optBtn.textContent = opt.label;
      optBtn.addEventListener('mouseenter', () => { optBtn.style.background = 'rgba(255,255,255,0.05)'; });
      optBtn.addEventListener('mouseleave', () => { optBtn.style.background = 'none'; });
      optBtn.addEventListener('click', async () => {
        dismissMenu.style.display = 'none';
        dismissToggle.disabled = true;
        dismissToggle.textContent = 'Dismissing...';
        try {
          await RecommendationsAPI.dismissUpstream(rec.id, 'User dismissed', opt.permanent);
          removeFromListCache(rec.id);
          currentState.view = 'list';
          currentState.selectedRec = null;
          renderCurrentView(document.getElementById('ss-recommendations'));
        } catch (e) {
          dismissToggle.textContent = `Failed: ${e.message}`;
          dismissToggle.disabled = false;
        }
      });
      dismissMenu.appendChild(optBtn);
    });

    dismissToggle.addEventListener('click', () => {
      dismissMenu.style.display = dismissMenu.style.display === 'none' ? 'block' : 'none';
    });

    document.addEventListener('click', function closeDismissMenu(e) {
      if (!dismissToggle.contains(e.target) && !dismissMenu.contains(e.target)) {
        dismissMenu.style.display = 'none';
      }
      if (!document.contains(container)) {
        document.removeEventListener('click', closeDismissMenu);
      }
    });

    dismissDropdown.appendChild(dismissToggle);
    dismissDropdown.appendChild(dismissMenu);
    actionBar.appendChild(applyBtn);
    actionBar.appendChild(dismissDropdown);
    wrapper.appendChild(actionBar);

    container.innerHTML = '';
    container.appendChild(wrapper);

    // === Quick select wiring ===
    keepAllBtn.addEventListener('click', () => {
      wrapper.querySelectorAll('.ss-upstream-field-row').forEach(row => {
        const mt = row.dataset.mergeType;
        if (mt === 'alias_list') {
          row.querySelectorAll('.ss-upstream-alias-item').forEach(item => {
            const cb = item.querySelector('input[type="checkbox"]');
            cb.checked = item.classList.contains('local-only') || item.classList.contains('both');
          });
          updateAliasResultSummary(row);
        } else {
          const localCb = row.querySelector('.ss-upstream-cb-local');
          const upstreamCb = row.querySelector('.ss-upstream-cb-upstream');
          const resultInput = row.querySelector('.ss-upstream-result-input, .ss-upstream-textarea');
          if (localCb && upstreamCb && resultInput) {
            localCb.checked = true;
            upstreamCb.checked = false;
            const change = changes[parseInt(row.dataset.fieldIndex)];
            resultInput.value = formatFieldValue(change.local_value) === '(empty)' ? '' : formatFieldValue(change.local_value);
          }
        }
      });
    });

    acceptAllBtn.addEventListener('click', () => {
      wrapper.querySelectorAll('.ss-upstream-field-row').forEach(row => {
        const mt = row.dataset.mergeType;
        if (mt === 'alias_list') {
          row.querySelectorAll('.ss-upstream-alias-item input[type="checkbox"]').forEach(cb => { cb.checked = true; });
          updateAliasResultSummary(row);
        } else {
          const localCb = row.querySelector('.ss-upstream-cb-local');
          const upstreamCb = row.querySelector('.ss-upstream-cb-upstream');
          const resultInput = row.querySelector('.ss-upstream-result-input, .ss-upstream-textarea');
          if (localCb && upstreamCb && resultInput) {
            localCb.checked = false;
            upstreamCb.checked = true;
            const change = changes[parseInt(row.dataset.fieldIndex)];
            resultInput.value = formatFieldValue(change.upstream_value) === '(empty)' ? '' : formatFieldValue(change.upstream_value);
          }
          const aliasOpt = row.querySelector('.ss-upstream-name-alias-cb');
          if (aliasOpt) aliasOpt.checked = true;
        }
      });
    });

    // === Apply handler ===
    applyBtn.addEventListener('click', async () => {
      errorDiv.style.display = 'none';
      errorDiv.innerHTML = '';
      const fields = {};

      wrapper.querySelectorAll('.ss-upstream-field-row').forEach(row => {
        const fieldKey = row.dataset.fieldKey;
        const mergeType = row.dataset.mergeType;
        const fieldIndex = parseInt(row.dataset.fieldIndex);
        const change = changes[fieldIndex];

        if (mergeType === 'alias_list') {
          const checkedAliases = [];
          row.querySelectorAll('.ss-upstream-alias-item input[type="checkbox"]:checked').forEach(cb => {
            checkedAliases.push(cb.value);
          });
          fields[fieldKey] = checkedAliases;
        } else {
          const resultInput = row.querySelector('.ss-upstream-result-input, .ss-upstream-textarea');
          if (!resultInput) return;

          const resultVal = resultInput.value.trim();
          const localStr = formatFieldValue(change.local_value) === '(empty)' ? '' : String(change.local_value || '');

          // Skip if result equals local (no change)
          if (resultVal === localStr) {
            if (mergeType === 'name') {
              const aliasCb = row.querySelector('.ss-upstream-name-alias-cb');
              if (aliasCb && aliasCb.checked) {
                const upstreamCb = row.querySelector('.ss-upstream-cb-upstream');
                if (upstreamCb && !upstreamCb.checked) {
                  fields['_alias_add'] = fields['_alias_add'] || [];
                  fields['_alias_add'].push(String(change.upstream_value || ''));
                }
              }
            }
            return;
          }

          fields[fieldKey] = resultVal;

          if (mergeType === 'name') {
            const aliasCb = row.querySelector('.ss-upstream-name-alias-cb');
            if (aliasCb && aliasCb.checked) {
              const upstreamCb = row.querySelector('.ss-upstream-cb-upstream');
              if (upstreamCb && upstreamCb.checked) {
                fields['_alias_add'] = fields['_alias_add'] || [];
                fields['_alias_add'].push(String(change.local_value || ''));
              }
            }
          }
        }
      });

      const hasChanges = Object.keys(fields).length > 0;
      if (!hasChanges) {
        try {
          applyBtn.disabled = true;
          applyBtn.textContent = 'Resolving...';
          await RecommendationsAPI.resolve(rec.id, 'accepted_no_changes', {});
          showSuccessAndReturn(applyBtn, 'Done!');
        } catch (e) {
          applyBtn.textContent = `Failed: ${e.message}`;
          applyBtn.classList.add('ss-btn-error');
          applyBtn.disabled = false;
        }
        return;
      }

      // No name uniqueness validation needed for tags — just apply
      try {
        applyBtn.disabled = true;
        applyBtn.textContent = 'Applying...';
        await RecommendationsAPI.updateTag(tagId, fields);
        await RecommendationsAPI.resolve(rec.id, 'applied', { fields });

        showSuccessAndReturn(applyBtn, 'Applied!');
      } catch (e) {
        errorDiv.innerHTML = `<div>${escapeHtml(e.message)}</div>`;
        errorDiv.style.display = 'block';
        applyBtn.textContent = 'Apply Selected Changes';
        applyBtn.disabled = false;
      }
    });
  }

  /**
   * Upstream Studio Detail View
   * Similar to tag — name, url, parent_studio. No image, no compound fields.
   * Parent studio shows human-readable name with StashBox UUID stored for resolution.
   */
  async function renderUpstreamStudioDetail(container, rec) {
    const details = rec.details;
    const rawChanges = details.changes || [];
    const studioId = details.studio_id;

    const changes = filterRealChanges(rawChanges);

    // If all changes were filtered out, auto-resolve and go back
    if (changes.length === 0) {
      try {
        await RecommendationsAPI.resolve(rec.id, 'accepted_no_changes', { note: 'All differences were cosmetic' });
      } catch (_) {}
      removeFromListCache(rec.id);
      currentState.view = 'list';
      currentState.selectedRec = null;
      renderCurrentView(document.getElementById('ss-recommendations'));
      return;
    }

    // Display value helper
    function displayValue(val) {
      return formatFieldValue(val);
    }

    // Smart default: prefer upstream (stash-box is source of truth)
    function smartDefault(localVal, upstreamVal) {
      const upstreamEmpty = isEmptyLikeValue(upstreamVal);
      if (!upstreamEmpty) return 'upstream';
      return 'local';
    }

    const wrapper = document.createElement('div');
    wrapper.className = 'ss-detail-upstream-performer'; // reuse performer styles

    // Header (studio icon instead of image)
    const headerDiv = document.createElement('div');
    headerDiv.className = 'ss-upstream-header';
    headerDiv.innerHTML = `
      <div class="ss-rec-tag-icon" style="width:48px;height:48px;display:flex;align-items:center;justify-content:center;border-radius:8px;background:rgba(255,255,255,0.05);flex-shrink:0;">
        <svg viewBox="0 0 24 24" width="28" height="28" fill="currentColor"><path d="M12 7V3H2v18h20V7H12zM6 19H4v-2h2v2zm0-4H4v-2h2v2zm0-4H4V9h2v2zm0-4H4V5h2v2zm4 12H8v-2h2v2zm0-4H8v-2h2v2zm0-4H8V9h2v2zm0-4H8V5h2v2zm10 12h-8v-2h2v-2h-2v-2h2v-2h-2V9h8v10zm-2-8h-2v2h2v-2zm0 4h-2v2h2v-2z"/></svg>
      </div>
      <div style="flex:1;">
        <h2 style="margin: 0 0 4px 0;">
          <a href="/studios/${studioId}" target="_blank">${details.studio_name || 'Unknown'}</a>
        </h2>
        <a href="${details.endpoint.replace(/\/graphql$/, '')}/studios/${details.stash_box_id}" target="_blank" class="ss-upstream-endpoint-badge">${details.endpoint_name || 'Upstream'}</a>
      </div>
    `;
    wrapper.appendChild(headerDiv);

    // Quick select buttons
    const quickActions = document.createElement('div');
    quickActions.className = 'ss-upstream-quick-actions';
    const keepAllBtn = document.createElement('button');
    keepAllBtn.textContent = 'Keep All Local';
    const acceptAllBtn = document.createElement('button');
    acceptAllBtn.textContent = 'Accept All Upstream';
    quickActions.appendChild(keepAllBtn);
    quickActions.appendChild(acceptAllBtn);
    wrapper.appendChild(quickActions);

    // Build field rows
    changes.forEach((change, idx) => {
      const mergeType = change.merge_type || 'simple';
      const fieldRow = document.createElement('div');
      fieldRow.className = 'ss-upstream-field-row';
      fieldRow.dataset.fieldIndex = idx;
      fieldRow.dataset.fieldKey = change.field;
      fieldRow.dataset.mergeType = mergeType;

      const label = document.createElement('div');
      label.className = 'ss-upstream-field-label';
      label.textContent = change.field_label || change.field;
      fieldRow.appendChild(label);

      if (mergeType === 'alias_list') {
        renderAliasListField(fieldRow, change, idx);
      } else {
        renderCompareField(fieldRow, change, idx, mergeType, displayValue, smartDefault);
      }

      wrapper.appendChild(fieldRow);
    });

    // Validation errors
    const errorDiv = document.createElement('div');
    errorDiv.id = 'ss-upstream-validation-errors';
    errorDiv.className = 'ss-upstream-validation-error';
    errorDiv.style.display = 'none';
    wrapper.appendChild(errorDiv);

    // Action bar
    const actionBar = document.createElement('div');
    actionBar.className = 'ss-upstream-action-bar';

    const applyBtn = document.createElement('button');
    applyBtn.className = 'ss-btn ss-upstream-apply-btn';
    applyBtn.textContent = 'Apply Selected Changes';

    // Dismiss dropdown (reuse tag pattern)
    const dismissDropdown = document.createElement('div');
    dismissDropdown.style.cssText = 'position:relative;';
    const dismissToggle = document.createElement('button');
    dismissToggle.className = 'ss-btn ss-upstream-dismiss-btn';
    dismissToggle.textContent = 'Dismiss';
    const dismissMenu = document.createElement('div');
    dismissMenu.style.cssText = 'display:none;position:absolute;bottom:100%;left:0;background:#2a2a2a;border:1px solid #444;border-radius:6px;padding:4px 0;min-width:220px;z-index:10;';

    const dismissOptions = [
      { label: 'Dismiss this update', permanent: false },
      { label: 'Never show for this studio', permanent: true },
    ];
    dismissOptions.forEach(opt => {
      const optBtn = document.createElement('button');
      optBtn.textContent = opt.label;
      optBtn.style.cssText = 'display:block;width:100%;text-align:left;padding:8px 16px;background:none;border:none;color:#fff;cursor:pointer;font-size:13px;';
      optBtn.addEventListener('mouseenter', () => { optBtn.style.background = 'rgba(255,255,255,0.05)'; });
      optBtn.addEventListener('mouseleave', () => { optBtn.style.background = 'none'; });
      optBtn.addEventListener('click', async () => {
        dismissToggle.disabled = true;
        dismissToggle.textContent = 'Dismissing...';
        try {
          await RecommendationsAPI.dismissUpstream(rec.id, null, opt.permanent);
          removeFromListCache(rec.id);
          currentState.view = 'list';
          currentState.selectedRec = null;
          renderCurrentView(document.getElementById('ss-recommendations'));
        } catch (e) {
          errorDiv.innerHTML = `<div>${escapeHtml(e.message)}</div>`;
          errorDiv.style.display = 'block';
          dismissToggle.disabled = false;
          dismissToggle.textContent = 'Dismiss';
        }
      });
      dismissMenu.appendChild(optBtn);
    });
    dismissDropdown.appendChild(dismissToggle);
    dismissDropdown.appendChild(dismissMenu);

    actionBar.appendChild(applyBtn);
    actionBar.appendChild(dismissDropdown);
    wrapper.appendChild(actionBar);

    container.innerHTML = '';
    container.appendChild(wrapper);

    // === Event handlers ===

    dismissToggle.addEventListener('click', () => {
      dismissMenu.style.display = dismissMenu.style.display === 'none' ? 'block' : 'none';
    });

    // Close dismiss menu when clicking outside (match tag pattern)
    document.addEventListener('click', function closeDismissMenu(e) {
      if (!dismissToggle.contains(e.target) && !dismissMenu.contains(e.target)) {
        dismissMenu.style.display = 'none';
      }
      if (!document.contains(container)) {
        document.removeEventListener('click', closeDismissMenu);
      }
    });

    keepAllBtn.addEventListener('click', () => {
      wrapper.querySelectorAll('.ss-upstream-field-row').forEach(row => {
        const mt = row.dataset.mergeType;
        if (mt === 'alias_list') {
          row.querySelectorAll('.ss-upstream-alias-item').forEach(item => {
            const cb = item.querySelector('input[type="checkbox"]');
            cb.checked = item.classList.contains('local-only') || item.classList.contains('both');
          });
          updateAliasResultSummary(row);
        } else {
          const localCb = row.querySelector('.ss-upstream-cb-local');
          const upstreamCb = row.querySelector('.ss-upstream-cb-upstream');
          if (localCb) { localCb.checked = true; localCb.dispatchEvent(new Event('change')); }
          if (upstreamCb) upstreamCb.checked = false;
        }
      });
    });

    acceptAllBtn.addEventListener('click', () => {
      wrapper.querySelectorAll('.ss-upstream-field-row').forEach(row => {
        const mt = row.dataset.mergeType;
        if (mt === 'alias_list') {
          row.querySelectorAll('.ss-upstream-alias-item input[type="checkbox"]').forEach(cb => { cb.checked = true; });
          updateAliasResultSummary(row);
        } else {
          const localCb = row.querySelector('.ss-upstream-cb-local');
          const upstreamCb = row.querySelector('.ss-upstream-cb-upstream');
          if (upstreamCb) { upstreamCb.checked = true; upstreamCb.dispatchEvent(new Event('change')); }
          if (localCb) localCb.checked = false;
        }
      });
    });

    // === Apply handler ===
    applyBtn.addEventListener('click', async () => {
      errorDiv.style.display = 'none';
      errorDiv.innerHTML = '';
      const fields = {};

      wrapper.querySelectorAll('.ss-upstream-field-row').forEach(row => {
        const fieldKey = row.dataset.fieldKey;
        const mergeType = row.dataset.mergeType;
        const fieldIndex = parseInt(row.dataset.fieldIndex);
        const change = changes[fieldIndex];

        if (mergeType === 'alias_list') {
          const checkedAliases = [];
          row.querySelectorAll('.ss-upstream-alias-item input[type="checkbox"]:checked').forEach(cb => {
            checkedAliases.push(cb.value);
          });
          fields[fieldKey] = checkedAliases;
        } else {
          const resultInput = row.querySelector('.ss-upstream-result-input, .ss-upstream-textarea');
          if (!resultInput) return;

          // Use raw value if available (e.g. parent_studio stores UUID, displays name)
          const resultVal = (resultInput.dataset.rawValue !== undefined ? resultInput.dataset.rawValue : resultInput.value).trim();
          const localStr = formatFieldValue(change.local_value) === '(empty)' ? '' : String(change.local_value || '');

          // Skip if result equals local (no change)
          if (resultVal === localStr) return;

          fields[fieldKey] = resultVal;
        }
      });

      const hasChanges = Object.keys(fields).length > 0;
      if (!hasChanges) {
        try {
          applyBtn.disabled = true;
          applyBtn.textContent = 'Resolving...';
          await RecommendationsAPI.resolve(rec.id, 'accepted_no_changes', {});
          showSuccessAndReturn(applyBtn, 'Done!');
        } catch (e) {
          applyBtn.textContent = `Failed: ${e.message}`;
          applyBtn.classList.add('ss-btn-error');
          applyBtn.disabled = false;
        }
        return;
      }

      try {
        applyBtn.disabled = true;
        applyBtn.textContent = 'Applying...';
        await RecommendationsAPI.updateStudio(studioId, fields, details.endpoint);
        await RecommendationsAPI.resolve(rec.id, 'applied', { fields });

        showSuccessAndReturn(applyBtn, 'Applied!');
      } catch (e) {
        errorDiv.innerHTML = `<div>${escapeHtml(e.message)}</div>`;
        errorDiv.style.display = 'block';
        applyBtn.textContent = 'Apply Selected Changes';
        applyBtn.disabled = false;
      }
    });
  }

  // ==================== Upstream Scene Detail ====================

  /**
   * Build the "Local Scene" / "Stash-Box Match" preview comparison, matching
   * the layout used by the Scene Stash-Box Tagger, so users can visually
   * confirm the match is correct before applying field changes.
   */
  function buildUpstreamSceneComparisonEl(opts) {
    const { sceneId, localTitle, localScene, endpoint, endpointName, stashboxId, upstream } = opts;

    const localScreenshotUrl = relativeUrl(localScene?.paths?.screenshot);
    const localPreviewUrl = relativeUrl(localScene?.paths?.preview);
    const localStudio = localScene?.studio || null;
    const localPerformers = localScene?.performers || [];
    const localDate = localScene?.date || '';
    const localFiles = localScene?.files || [];
    const localFilenames = localFiles
      .map(f => String(f?.path || '').split(/[\\/]/).pop())
      .filter(Boolean);
    const localDuration = localFiles[0]?.duration;

    const stashboxBaseHref = String(endpoint || '').replace(/\/graphql$/, '');
    const stashboxSceneHref = `${stashboxBaseHref}/scenes/${stashboxId}`;
    const upstreamStudio = upstream?.studio || null;
    const upstreamPerformers = upstream?.performers || [];
    const upstreamDate = upstream?.date || '';
    const upstreamTitle = upstream?.title || '';
    const upstreamCoverUrl = upstream?.cover_url || null;
    const upstreamDuration = upstream?.duration;

    const el = document.createElement('div');
    el.className = 'ss-fp-detail-comparison ss-scene-tagger-preview';
    el.innerHTML = `
      <div class="ss-fp-detail-side">
        <h4>Local Scene</h4>
        <div class="ss-dup-scene-thumb ss-us-local-thumb">
          ${localScreenshotUrl
            ? `<img src="${escapeHtml(localScreenshotUrl)}" alt="Local scene screenshot" loading="lazy" onerror="this.style.display='none'" />`
            : '<div class="ss-no-image">No Screenshot</div>'
          }
          ${localPreviewUrl
            ? `<video class="ss-dup-scene-preview ss-us-scene-preview" muted loop preload="none" data-src="${escapeHtml(localPreviewUrl)}"></video>`
            : ''
          }
        </div>
        <div class="ss-fp-scene-title">
          <a href="/scenes/${encodeURIComponent(String(sceneId))}" target="_blank" rel="noopener">${escapeHtmlBreakable(localTitle || 'Unknown')}</a>
        </div>
        ${localFilenames.length ? `<div class="ss-fp-field"><strong>File:</strong> ${localFilenames.map(f => escapeHtmlBreakable(f)).join(', ')}</div>` : ''}
        ${localStudio ? `<div class="ss-fp-field"><strong>Studio:</strong> ${escapeHtml(localStudio.name)}</div>` : ''}
        ${localPerformers.length ? `<div class="ss-fp-field"><strong>Performers:</strong> ${localPerformers.map(p => escapeHtml(p.name)).join(', ')}</div>` : ''}
        ${localDate ? `<div class="ss-fp-field"><strong>Date:</strong> ${escapeHtml(localDate)}</div>` : ''}
        ${localDuration ? `<div class="ss-fp-field"><strong>Duration:</strong> ${formatDuration(localDuration)}</div>` : ''}
      </div>
      <div class="ss-fp-detail-divider"></div>
      <div class="ss-fp-detail-side">
        <h4>Stash-Box Match</h4>
        <div class="ss-dup-scene-thumb ss-us-cover-thumb">
          ${upstreamCoverUrl
            ? `<img src="${escapeHtml(upstreamCoverUrl)}" alt="Stash-Box scene cover" loading="lazy" onerror="this.style.display='none'" />`
            : '<div class="ss-no-image">No Cover</div>'
          }
        </div>
        <div class="ss-fp-scene-title">
          <a href="${escapeHtml(stashboxSceneHref)}" target="_blank" rel="noopener">${escapeHtmlBreakable(upstreamTitle || 'Unknown')}</a>
        </div>
        ${upstreamStudio ? `<div class="ss-fp-field"><strong>Studio:</strong> ${escapeHtml(upstreamStudio.name)}</div>` : ''}
        ${upstreamPerformers.length ? `<div class="ss-fp-field"><strong>Performers:</strong> ${upstreamPerformers.map(p => escapeHtml(p.name || '')).join(', ')}</div>` : ''}
        ${upstreamDate ? `<div class="ss-fp-field"><strong>Date:</strong> ${escapeHtml(upstreamDate)}</div>` : ''}
        ${upstreamDuration ? `<div class="ss-fp-field"><strong>Duration:</strong> ${formatDuration(upstreamDuration)}</div>` : ''}
        <div class="ss-fp-field">
          <strong>Endpoint:</strong>
          <a class="ss-detail-entity-link" href="${escapeHtml(stashboxSceneHref)}" target="_blank" rel="noopener">${escapeHtml(endpointName || endpoint)}</a>
        </div>
      </div>
    `;

    // Local preview video on hover (same behavior as duplicate scenes / tagger).
    const localThumb = el.querySelector('.ss-us-local-thumb');
    if (localThumb) {
      const video = localThumb.querySelector('.ss-us-scene-preview');
      const img = localThumb.querySelector('img');
      if (video) {
        localThumb.addEventListener('mouseenter', () => {
          if (!video.src && video.dataset.src) {
            video.src = video.dataset.src;
          }
          if (img) img.style.opacity = '0';
          video.style.opacity = '1';
          video.play().catch(() => {});
        });

        localThumb.addEventListener('mouseleave', () => {
          video.pause();
          if (img) img.style.opacity = '1';
          video.style.opacity = '0';
          setTimeout(() => { video.currentTime = 0; }, 200);
        });
      }
    }

    return el;
  }

  async function renderUpstreamSceneDetail(container, rec) {
    const details = rec.details;
    const simpleChanges = filterRealChanges(details.changes || []);
    const studioChange = details.studio_change;
    const rawPerformerChanges = details.performer_changes || { added: [], removed: [], alias_changed: [] };
    const performerChanges = { ...rawPerformerChanges, alias_changed: [] };
    const tagChanges = details.tag_changes || { added: [], removed: [] };
    const sceneId = details.scene_id;
    const endpoint = details.endpoint;

    const hasSimple = simpleChanges.length > 0;
    const hasStudio = studioChange !== null && studioChange !== undefined;
    const hasPerformers = performerChanges.added.length > 0 || performerChanges.removed.length > 0;
    const hasTags = tagChanges.added.length > 0 || tagChanges.removed.length > 0;

    async function findLinkedByStashId(entityType, stashboxId) {
      const key = `${endpoint}|${stashboxId}`;
      const cachedId = entityCache.get(key);
      if (cachedId) return { id: cachedId, name: null, from: 'cache' };
      try {
        const response = await RecommendationsAPI.findLinkedEntity(entityType, endpoint, stashboxId);
        const linked = response?.result;
        if (linked && linked.id != null) {
          entityCache.set(key, linked.id);
          return { id: linked.id, name: linked.name || null, from: 'stash_id' };
        }
      } catch (e) {
        // Keep UI usable even if linked lookup fails.
      }
      return null;
    }

    if (!hasSimple && !hasStudio && !hasPerformers && !hasTags) {
      try {
        await RecommendationsAPI.resolve(rec.id, 'accepted_no_changes', { note: 'All differences were cosmetic' });
      } catch (_) {}
      removeFromListCache(rec.id);
      currentState.view = 'list';
      currentState.selectedRec = null;
      renderCurrentView(document.getElementById('ss-recommendations'));
      return;
    }

    function displayValue(val) { return formatFieldValue(val); }
    function smartDefault(localVal, upstreamVal) {
      const upstreamEmpty = isEmptyLikeValue(upstreamVal);
      return !upstreamEmpty ? 'upstream' : 'local';
    }

    const wrapper = document.createElement('div');
    wrapper.className = 'ss-detail-upstream-performer'; // reuse styles

    // Header
    const headerDiv = document.createElement('div');
    headerDiv.className = 'ss-upstream-header';
    headerDiv.innerHTML = `
      <div class="ss-rec-tag-icon" style="width:48px;height:48px;display:flex;align-items:center;justify-content:center;border-radius:8px;background:rgba(255,255,255,0.05);flex-shrink:0;">
        <svg viewBox="0 0 24 24" width="28" height="28" fill="currentColor"><path d="M18 4l2 4h-3l-2-4h-2l2 4h-3l-2-4H8l2 4H7L5 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V4h-4z"/></svg>
      </div>
      <div style="flex:1;">
        <h2 style="margin: 0 0 4px 0;">
          <a href="/scenes/${sceneId}" target="_blank">${escapeHtml(details.scene_name || 'Unknown Scene')}</a>
        </h2>
        <a href="${details.endpoint.replace(/\/graphql$/, '')}/scenes/${details.stash_box_id}" target="_blank" class="ss-upstream-endpoint-badge">${escapeHtml(details.endpoint_name || 'Upstream')}</a>
      </div>
    `;
    wrapper.appendChild(headerDiv);

    // Local scene / Stash-box match preview (tagger-style comparison), so
    // users can visually confirm this is actually the right scene even when
    // there's little else to go on besides a stashdb ID.
    let localScenePreview = null;
    let upstreamScenePreview = null;
    const previewResults = await Promise.allSettled([
      RecommendationsAPI.getSceneDetail(sceneId),
      RecommendationsAPI.getUpstreamScenePreview(endpoint, details.stash_box_id),
    ]);
    if (previewResults[0].status === 'fulfilled') localScenePreview = previewResults[0].value;
    if (previewResults[1].status === 'fulfilled') upstreamScenePreview = previewResults[1].value;
    wrapper.appendChild(buildUpstreamSceneComparisonEl({
      sceneId,
      localTitle: details.scene_name,
      localScene: localScenePreview,
      endpoint,
      endpointName: details.endpoint_name,
      stashboxId: details.stash_box_id,
      upstream: upstreamScenePreview,
    }));

    // Local context summary
    const localPerformers = details.current_performers || [];
    const localTags = details.current_tags || [];
    const localStudio = details.current_studio;
    const contextParts = [];
    if (localPerformers.length > 0) {
      contextParts.push(localPerformers.map(p => escapeHtml(p.name)).join(', '));
    }
    if (localTags.length > 0) {
      contextParts.push('Tags: ' + localTags.map(t => escapeHtml(t.name)).join(', '));
    }
    if (localStudio) {
      contextParts.push('Studio: ' + escapeHtml(localStudio.name));
    }
    if (contextParts.length > 0) {
      const contextDiv = document.createElement('div');
      contextDiv.className = 'ss-scene-local-context';
      contextDiv.innerHTML = `<span class="ss-scene-local-context-label">Local:</span> ${contextParts.join(' &middot; ')}`;
      wrapper.appendChild(contextDiv);
    }

    // Scene cover update toggle -- ticked by default. Only offered when
    // there's actually an upstream cover to set; unticking keeps the local
    // cover untouched.
    const upstreamCoverUrl = upstreamScenePreview?.cover_url || null;
    let updateCoverCheckbox = null;
    if (upstreamCoverUrl) {
      const coverToggleLabel = document.createElement('label');
      coverToggleLabel.className = 'ss-scene-cover-toggle';
      coverToggleLabel.style.cssText = 'display:flex;align-items:center;gap:6px;margin:6px 0;';
      updateCoverCheckbox = document.createElement('input');
      updateCoverCheckbox.type = 'checkbox';
      updateCoverCheckbox.checked = true;
      updateCoverCheckbox.className = 'ss-scene-cover-cb';
      coverToggleLabel.appendChild(updateCoverCheckbox);
      coverToggleLabel.appendChild(document.createTextNode('Update scene cover'));
      wrapper.appendChild(coverToggleLabel);
    }

    // ===== Simple field diffs =====
    if (hasSimple) {
      const simpleSection = document.createElement('div');
      simpleSection.className = 'ss-scene-section';

      const quickActions = document.createElement('div');
      quickActions.className = 'ss-upstream-quick-actions';
      const keepAllBtn = document.createElement('button');
      keepAllBtn.className = 'ss-btn ss-btn-sm ss-btn-secondary';
      keepAllBtn.textContent = 'Keep All Local';
      const acceptAllBtn = document.createElement('button');
      acceptAllBtn.className = 'ss-btn ss-btn-sm ss-btn-primary';
      acceptAllBtn.textContent = 'Accept All Upstream';
      quickActions.appendChild(keepAllBtn);
      quickActions.appendChild(acceptAllBtn);
      simpleSection.appendChild(quickActions);

      simpleChanges.forEach((change, idx) => {
        const mergeType = change.merge_type || 'simple';
        const fieldRow = document.createElement('div');
        fieldRow.className = 'ss-upstream-field-row';
        fieldRow.dataset.fieldIndex = idx;
        fieldRow.dataset.fieldKey = change.field;
        fieldRow.dataset.mergeType = mergeType;

        const label = document.createElement('div');
        label.className = 'ss-upstream-field-label';
        label.textContent = change.field_label || change.field;
        fieldRow.appendChild(label);

        if (mergeType === 'alias_list') {
          renderAliasListField(fieldRow, change, idx);
        } else {
          renderCompareField(fieldRow, change, idx, mergeType, displayValue, smartDefault);
        }

        simpleSection.appendChild(fieldRow);
      });

      wrapper.appendChild(simpleSection);

      // Quick select wiring
      keepAllBtn.addEventListener('click', () => {
        simpleSection.querySelectorAll('.ss-upstream-field-row').forEach(row => {
          const mt = row.dataset.mergeType;
          if (mt === 'alias_list') {
            row.querySelectorAll('.ss-upstream-alias-item').forEach(item => {
              const cb = item.querySelector('input[type="checkbox"]');
              cb.checked = item.classList.contains('local-only') || item.classList.contains('both');
            });
            updateAliasResultSummary(row);
          } else {
            const localCb = row.querySelector('.ss-upstream-cb-local');
            const upstreamCb = row.querySelector('.ss-upstream-cb-upstream');
            const resultInput = row.querySelector('.ss-upstream-result-input, .ss-upstream-textarea');
            if (localCb && upstreamCb && resultInput) {
              localCb.checked = true;
              upstreamCb.checked = false;
              const change = simpleChanges[parseInt(row.dataset.fieldIndex)];
              resultInput.value = formatFieldValue(change.local_value) === '(empty)' ? '' : formatFieldValue(change.local_value);
            }
          }
        });
      });

      acceptAllBtn.addEventListener('click', () => {
        simpleSection.querySelectorAll('.ss-upstream-field-row').forEach(row => {
          const mt = row.dataset.mergeType;
          if (mt === 'alias_list') {
            row.querySelectorAll('.ss-upstream-alias-item').forEach(item => {
              const cb = item.querySelector('input[type="checkbox"]');
              cb.checked = true;
            });
            updateAliasResultSummary(row);
          } else {
            const localCb = row.querySelector('.ss-upstream-cb-local');
            const upstreamCb = row.querySelector('.ss-upstream-cb-upstream');
            const resultInput = row.querySelector('.ss-upstream-result-input, .ss-upstream-textarea');
            if (upstreamCb) { upstreamCb.checked = true; }
            if (localCb) { localCb.checked = false; }
            if (resultInput) {
              const change = simpleChanges[parseInt(row.dataset.fieldIndex)];
              resultInput.value = formatFieldValue(change.upstream_value) === '(empty)' ? '' : formatFieldValue(change.upstream_value);
            }
          }
        });
      });
    }

    // ===== Studio change section =====
    if (hasStudio) {
      const studioSection = document.createElement('div');
      studioSection.className = 'ss-scene-section';
      const studioHeader = document.createElement('h3');
      studioHeader.className = 'ss-scene-section-title';
      studioHeader.textContent = 'Studio';
      studioSection.appendChild(studioHeader);

      const studioRow = document.createElement('div');
      studioRow.className = 'ss-scene-entity-row';
      const localStudio = studioChange.local || details.current_studio;
      const upstreamStudio = studioChange.upstream;
      studioRow.innerHTML = `
        <div class="ss-scene-entity-compare">
          <div class="ss-scene-entity-local">
            <span class="ss-upstream-value-label">Local</span>
            <span>${localStudio ? escapeHtml(localStudio.name) : '(none)'}</span>
          </div>
          <span style="color:#666;">\u2192</span>
          <div class="ss-scene-entity-upstream">
            <span class="ss-upstream-value-label">Upstream</span>
            <span>${upstreamStudio ? escapeHtml(upstreamStudio.name) : '(none)'}</span>
          </div>
        </div>
      `;

      // Check if upstream studio exists in entity cache
      const studioCheckbox = document.createElement('input');
      studioCheckbox.type = 'checkbox';
      studioCheckbox.className = 'ss-scene-studio-cb';
      studioCheckbox.checked = true;
      studioCheckbox.dataset.stashboxId = upstreamStudio ? upstreamStudio.id : '';

      const studioLabel = document.createElement('label');
      studioLabel.style.cssText = 'display:flex;align-items:center;gap:6px;margin-top:6px;';
      studioLabel.appendChild(studioCheckbox);
      studioLabel.appendChild(document.createTextNode('Apply studio change'));
      studioRow.appendChild(studioLabel);

      // Link/Create studio (shown when studio doesn't exist locally)
      if (upstreamStudio) {
        const cacheKey = `${endpoint}|${upstreamStudio.id}`;
        const cachedId = entityCache.get(cacheKey);
        const linkedByStashId = !upstreamStudio.local_match && !cachedId
          ? await findLinkedByStashId('studio', upstreamStudio.id)
          : null;

        if (upstreamStudio.local_match) {
          entityCache.set(cacheKey, upstreamStudio.local_match.id);
          const note = document.createElement('span');
          note.className = 'ss-scene-entity-linked';
          note.textContent = 'Linked';
          studioRow.appendChild(note);
        } else if (linkedByStashId?.id) {
          studioCheckbox.checked = true;
          const note = document.createElement('span');
          note.className = 'ss-scene-entity-linked';
          note.textContent = linkedByStashId.name ? `Linked: ${linkedByStashId.name}` : 'Linked';
          studioRow.appendChild(note);
        } else if (cachedId) {
          const note = document.createElement('span');
          note.className = 'ss-scene-entity-created';
          note.textContent = `Created (ID: ${cachedId})`;
          studioRow.appendChild(note);
        } else {
          const dropdown = createEntitySearchDropdown('studio', endpoint, upstreamStudio.id, (localId, localName) => {
            entityCache.set(cacheKey, localId);
            const linked = document.createElement('span');
            linked.className = 'ss-scene-entity-linked';
            linked.textContent = `Linked: ${localName}`;
            dropdown.replaceWith(linked);
            const btn = studioRow.querySelector('.ss-scene-create-btn');
            if (btn) btn.remove();
          }, upstreamStudio.name);
          studioRow.appendChild(dropdown);

          const createBtn = document.createElement('button');
          createBtn.className = 'ss-btn ss-btn-sm ss-scene-create-btn';
          createBtn.textContent = 'Create Studio';
          createBtn.dataset.entityType = 'studio';
          createBtn.dataset.stashboxId = upstreamStudio.id;
          createBtn.addEventListener('click', async () => {
            createBtn.disabled = true;
            createBtn.textContent = 'Creating...';
            try {
              const result = await RecommendationsAPI.createStudio(
                { name: upstreamStudio.name },
                endpoint,
                upstreamStudio.id
              );
              entityCache.set(cacheKey, result.studio.id);
              createBtn.textContent = `Created (ID: ${result.studio.id})`;
              createBtn.classList.add('ss-btn-success');
              const dd = studioRow.querySelector('.ss-entity-search-dropdown');
              if (dd) dd.remove();
            } catch (e) {
              createBtn.textContent = `Failed: ${e.message}`;
              createBtn.disabled = false;
            }
          });
          studioRow.appendChild(createBtn);
        }
      }

      studioSection.appendChild(studioRow);
      wrapper.appendChild(studioSection);
    }

    // ===== Performer changes section =====
    if (hasPerformers) {
      const perfSection = document.createElement('div');
      perfSection.className = 'ss-scene-section';
      const perfHeader = document.createElement('h3');
      perfHeader.className = 'ss-scene-section-title';
      perfHeader.textContent = 'Performers';
      perfSection.appendChild(perfHeader);

      // Added performers
      if (performerChanges.added.length > 0) {
        const addedDiv = document.createElement('div');
        addedDiv.className = 'ss-scene-subsection';
        const addedLabel = document.createElement('div');
        addedLabel.className = 'ss-scene-subsection-label ss-scene-added';
        addedLabel.textContent = `+ ${performerChanges.added.length} to add`;
        addedDiv.appendChild(addedLabel);

        for (const perf of performerChanges.added) {
          const row = document.createElement('div');
          row.className = 'ss-scene-entity-item';
          row.dataset.stashboxId = perf.id;
          row.dataset.entityType = 'performer';

          const cb = document.createElement('input');
          cb.type = 'checkbox';
          cb.className = 'ss-scene-perf-add-cb';
          cb.dataset.stashboxId = perf.id;
          cb.dataset.name = perf.name || perf.id;

          const nameSpan = document.createElement('span');
          nameSpan.className = 'ss-scene-entity-name';
          nameSpan.textContent = perf.name || perf.id;
          if (perf.as) {
            nameSpan.textContent += ` (as "${perf.as}")`;
          }

          row.appendChild(cb);
          row.appendChild(nameSpan);

          const cacheKey = `${endpoint}|${perf.id}`;
          const cachedId = entityCache.get(cacheKey);
          const linkedByStashId = !perf.local_match && !cachedId
            ? await findLinkedByStashId('performer', perf.id)
            : null;

          // Auto-match: local entity found by name during analysis
          if (perf.local_match) {
            entityCache.set(cacheKey, perf.local_match.id);
            cb.checked = true;
            const note = document.createElement('span');
            note.className = 'ss-scene-entity-linked';
            note.textContent = 'Linked';
            row.appendChild(note);
          } else if (linkedByStashId?.id) {
            cb.checked = true;
            const note = document.createElement('span');
            note.className = 'ss-scene-entity-linked';
            note.textContent = linkedByStashId.name ? `Linked: ${linkedByStashId.name}` : 'Linked';
            row.appendChild(note);
          } else if (cachedId) {
            cb.checked = true;
            const note = document.createElement('span');
            note.className = 'ss-scene-entity-created';
            note.textContent = `Created (ID: ${cachedId})`;
            row.appendChild(note);
          } else {
            cb.checked = false;
            // Search dropdown for manual linking
            const dropdown = createEntitySearchDropdown('performer', endpoint, perf.id, (localId, localName) => {
              entityCache.set(cacheKey, localId);
              cb.checked = true;
              // Replace dropdown and create button with linked indicator
              const linked = document.createElement('span');
              linked.className = 'ss-scene-entity-linked';
              linked.textContent = `Linked: ${localName}`;
              dropdown.replaceWith(linked);
              const btn = row.querySelector('.ss-scene-create-btn');
              if (btn) btn.remove();
            }, perf.name);
            row.appendChild(dropdown);

            const createBtn = document.createElement('button');
            createBtn.className = 'ss-btn ss-btn-sm ss-scene-create-btn';
            createBtn.textContent = 'Create';
            createBtn.addEventListener('click', async () => {
              createBtn.disabled = true;
              createBtn.textContent = 'Creating...';
              try {
                const performerPayload = { name: perf.name };
                if (Array.isArray(perf.aliases) && perf.aliases.length) {
                  performerPayload.aliases = perf.aliases;
                }
                if (perf.gender) {
                  performerPayload.gender = perf.gender;
                }
                const result = await RecommendationsAPI.createPerformer(
                  performerPayload,
                  endpoint,
                  perf.id
                );
                entityCache.set(cacheKey, result.performer.id);
                cb.checked = true;
                createBtn.textContent = `Created (ID: ${result.performer.id})`;
                createBtn.classList.add('ss-btn-success');
                // Remove the search dropdown
                const dd = row.querySelector('.ss-entity-search-dropdown');
                if (dd) dd.remove();
              } catch (e) {
                createBtn.textContent = `Failed`;
                createBtn.disabled = false;
              }
            });
            row.appendChild(createBtn);
          }

          addedDiv.appendChild(row);
        }
        perfSection.appendChild(addedDiv);
      }

      // Removed performers
      if (performerChanges.removed.length > 0) {
        const removedDiv = document.createElement('div');
        removedDiv.className = 'ss-scene-subsection';
        const removedLabel = document.createElement('div');
        removedLabel.className = 'ss-scene-subsection-label ss-scene-removed';
        removedLabel.textContent = `- ${performerChanges.removed.length} removed upstream`;
        removedDiv.appendChild(removedLabel);

        performerChanges.removed.forEach(perf => {
          const row = document.createElement('div');
          row.className = 'ss-scene-entity-item ss-scene-entity-removed';
          row.innerHTML = `<span class="ss-scene-entity-name">${escapeHtml(perf.name || perf.id)}</span>`;
          removedDiv.appendChild(row);
        });
        perfSection.appendChild(removedDiv);
      }

      // Alias changes
      if (performerChanges.alias_changed.length > 0) {
        const aliasDiv = document.createElement('div');
        aliasDiv.className = 'ss-scene-subsection';
        const aliasLabel = document.createElement('div');
        aliasLabel.className = 'ss-scene-subsection-label';
        aliasLabel.textContent = `Alias changes`;
        aliasDiv.appendChild(aliasLabel);

        performerChanges.alias_changed.forEach(perf => {
          const row = document.createElement('div');
          row.className = 'ss-scene-entity-item';
          row.innerHTML = `
            <span class="ss-scene-entity-name">${escapeHtml(perf.name)}</span>
            <span style="color:#888;">"${escapeHtml(perf.local_alias || '')}" \u2192 "${escapeHtml(perf.upstream_alias || '')}"</span>
          `;
          aliasDiv.appendChild(row);
        });
        perfSection.appendChild(aliasDiv);
      }

      wrapper.appendChild(perfSection);
    }

    // ===== Tag changes section =====
    if (hasTags) {
      const tagSection = document.createElement('div');
      tagSection.className = 'ss-scene-section';
      const tagHeader = document.createElement('h3');
      tagHeader.className = 'ss-scene-section-title';
      tagHeader.textContent = 'Tags';
      tagSection.appendChild(tagHeader);

      // Added tags
      if (tagChanges.added.length > 0) {
        const addedDiv = document.createElement('div');
        addedDiv.className = 'ss-scene-subsection';
        const addedLabel = document.createElement('div');
        addedLabel.className = 'ss-scene-subsection-label ss-scene-added';
        addedLabel.textContent = `+ ${tagChanges.added.length} to add`;
        addedDiv.appendChild(addedLabel);

        const tagList = document.createElement('div');
        tagList.className = 'ss-scene-tag-list';

        for (const tag of tagChanges.added) {
          const chip = document.createElement('div');
          chip.className = 'ss-scene-tag-chip';
          chip.dataset.stashboxId = tag.id;

          const cacheKey = `${endpoint}|${tag.id}`;
          const cachedId = entityCache.get(cacheKey);
          const linkedByStashId = !tag.local_match && !cachedId
            ? await findLinkedByStashId('tag', tag.id)
            : null;

          const cb = document.createElement('input');
          cb.type = 'checkbox';
          cb.className = 'ss-scene-tag-add-cb';
          cb.dataset.stashboxId = tag.id;
          cb.dataset.name = tag.name;

          const nameSpan = document.createElement('span');
          nameSpan.textContent = tag.name;

          chip.appendChild(cb);
          chip.appendChild(nameSpan);

          if (tag.local_match) {
            entityCache.set(cacheKey, tag.local_match.id);
            cb.checked = true;
            const note = document.createElement('span');
            note.className = 'ss-scene-entity-linked';
            note.textContent = 'Linked';
            chip.appendChild(note);
          } else if (linkedByStashId?.id) {
            cb.checked = true;
            const note = document.createElement('span');
            note.className = 'ss-scene-entity-linked';
            note.textContent = linkedByStashId.name ? `Linked: ${linkedByStashId.name}` : 'Linked';
            chip.appendChild(note);
          } else if (cachedId) {
            cb.checked = true;
            const note = document.createElement('span');
            note.className = 'ss-scene-entity-created';
            note.textContent = '(created)';
            chip.appendChild(note);
          } else {
            cb.checked = false;
            const dropdown = createEntitySearchDropdown('tag', endpoint, tag.id, (localId, localName) => {
              entityCache.set(cacheKey, localId);
              cb.checked = true;
              const linked = document.createElement('span');
              linked.className = 'ss-scene-entity-linked';
              linked.textContent = 'Linked';
              dropdown.replaceWith(linked);
              const btn = chip.querySelector('.ss-scene-create-btn');
              if (btn) btn.remove();
            }, tag.name);
            chip.appendChild(dropdown);

            const createBtn = document.createElement('button');
            createBtn.className = 'ss-btn ss-btn-sm ss-scene-create-btn';
            createBtn.textContent = 'Create';
            createBtn.addEventListener('click', async () => {
              createBtn.disabled = true;
              createBtn.textContent = '...';
              try {
                const result = await RecommendationsAPI.createTag(
                  { name: tag.name },
                  endpoint,
                  tag.id
                );
                entityCache.set(cacheKey, result.tag.id);
                cb.checked = true;
                createBtn.textContent = 'Created';
                createBtn.classList.add('ss-btn-success');
                const dd = chip.querySelector('.ss-entity-search-dropdown');
                if (dd) dd.remove();
              } catch (e) {
                createBtn.textContent = 'Fail';
                createBtn.disabled = false;
              }
            });
            chip.appendChild(createBtn);
          }

          tagList.appendChild(chip);
        }
        addedDiv.appendChild(tagList);
        tagSection.appendChild(addedDiv);
      }

      // Removed tags
      if (tagChanges.removed.length > 0) {
        const removedDiv = document.createElement('div');
        removedDiv.className = 'ss-scene-subsection';
        const removedLabel = document.createElement('div');
        removedLabel.className = 'ss-scene-subsection-label ss-scene-removed';
        removedLabel.textContent = `- ${tagChanges.removed.length} removed upstream`;
        removedDiv.appendChild(removedLabel);

        const tagList = document.createElement('div');
        tagList.className = 'ss-scene-tag-list';
        tagChanges.removed.forEach(tag => {
          const chip = document.createElement('div');
          chip.className = 'ss-scene-tag-chip ss-scene-tag-removed';
          chip.textContent = tag.name;
          tagList.appendChild(chip);
        });
        removedDiv.appendChild(tagList);
        tagSection.appendChild(removedDiv);
      }

      wrapper.appendChild(tagSection);
    }

    // ===== Validation errors =====
    const errorDiv = document.createElement('div');
    errorDiv.id = 'ss-upstream-validation-errors';
    errorDiv.className = 'ss-upstream-validation-error';
    errorDiv.style.display = 'none';
    wrapper.appendChild(errorDiv);

    // ===== Action bar =====
    const actionBar = document.createElement('div');
    actionBar.className = 'ss-upstream-action-bar';

    const applyBtn = document.createElement('button');
    applyBtn.className = 'ss-btn ss-btn-primary ss-upstream-apply-btn';
    applyBtn.textContent = 'Apply Changes';

    // Dismiss dropdown
    const dismissDropdown = document.createElement('div');
    dismissDropdown.style.cssText = 'position:relative;';
    const dismissToggle = document.createElement('button');
    dismissToggle.className = 'ss-btn ss-btn-danger ss-upstream-dismiss-btn';
    dismissToggle.textContent = 'Dismiss';
    const dismissMenu = document.createElement('div');
    dismissMenu.style.cssText = 'display:none;position:absolute;bottom:100%;left:0;background:#2a2a2a;border:1px solid #444;border-radius:6px;padding:4px 0;min-width:220px;z-index:10;';

    [
      { label: 'Dismiss this update', permanent: false },
      { label: 'Never show for this scene', permanent: true },
    ].forEach(opt => {
      const optBtn = document.createElement('button');
      optBtn.textContent = opt.label;
      optBtn.style.cssText = 'display:block;width:100%;text-align:left;padding:8px 16px;background:none;border:none;color:#fff;cursor:pointer;font-size:13px;';
      optBtn.addEventListener('mouseenter', () => { optBtn.style.background = 'rgba(255,255,255,0.05)'; });
      optBtn.addEventListener('mouseleave', () => { optBtn.style.background = 'none'; });
      optBtn.addEventListener('click', async () => {
        dismissToggle.disabled = true;
        dismissToggle.textContent = 'Dismissing...';
        try {
          await RecommendationsAPI.dismissUpstream(rec.id, null, opt.permanent);
          removeFromListCache(rec.id);
          currentState.view = 'list';
          currentState.selectedRec = null;
          renderCurrentView(document.getElementById('ss-recommendations'));
        } catch (e) {
          errorDiv.innerHTML = `<div>${escapeHtml(e.message)}</div>`;
          errorDiv.style.display = 'block';
          dismissToggle.disabled = false;
          dismissToggle.textContent = 'Dismiss';
        }
      });
      dismissMenu.appendChild(optBtn);
    });
    dismissDropdown.appendChild(dismissToggle);
    dismissDropdown.appendChild(dismissMenu);

    actionBar.appendChild(applyBtn);
    actionBar.appendChild(dismissDropdown);
    wrapper.appendChild(actionBar);

    container.innerHTML = '';
    container.appendChild(wrapper);

    // === Dismiss menu toggle ===
    dismissToggle.addEventListener('click', () => {
      dismissMenu.style.display = dismissMenu.style.display === 'none' ? 'block' : 'none';
    });
    document.addEventListener('click', function closeDismissMenu(e) {
      if (!dismissToggle.contains(e.target) && !dismissMenu.contains(e.target)) {
        dismissMenu.style.display = 'none';
      }
      if (!document.contains(container)) {
        document.removeEventListener('click', closeDismissMenu);
      }
    });

    // === Apply handler ===
    applyBtn.addEventListener('click', async () => {
      errorDiv.style.display = 'none';
      errorDiv.innerHTML = '';

      // 1. Collect simple field changes
      const fields = {};
      wrapper.querySelectorAll('.ss-upstream-field-row').forEach(row => {
        const fieldKey = row.dataset.fieldKey;
        const mergeType = row.dataset.mergeType;
        const fieldIndex = parseInt(row.dataset.fieldIndex);
        const change = simpleChanges[fieldIndex];
        if (!change) return;

        if (mergeType === 'alias_list') {
          const checkedAliases = [];
          row.querySelectorAll('.ss-upstream-alias-item input[type="checkbox"]:checked').forEach(cb => {
            checkedAliases.push(cb.value);
          });
          fields[fieldKey] = checkedAliases;
        } else {
          const resultInput = row.querySelector('.ss-upstream-result-input, .ss-upstream-textarea');
          if (!resultInput) return;

          const resultVal = resultInput.value.trim();
          const localStr = formatFieldValue(change.local_value) === '(empty)' ? '' : String(change.local_value || '');
          if (resultVal === localStr) return;

          fields[fieldKey] = resultVal;
        }
      });

      if (updateCoverCheckbox && updateCoverCheckbox.checked && upstreamCoverUrl) {
        fields['cover_image'] = upstreamCoverUrl;
      }

      // 2. Validate checked entities are created, collect performer IDs
      const uncreated = [];
      let performerIds = null;
      if (hasPerformers) {
        const addedPerfs = [];
        wrapper.querySelectorAll('.ss-scene-perf-add-cb:checked').forEach(cb => {
          const stashboxId = cb.dataset.stashboxId;
          const cacheKey = `${endpoint}|${stashboxId}`;
          const localId = entityCache.get(cacheKey);
          if (localId) {
            addedPerfs.push(localId);
          } else {
            uncreated.push(`Performer: ${cb.dataset.name || stashboxId}`);
          }
        });
        if (addedPerfs.length > 0) {
          // Merge with current scene performers (SceneUpdateInput replaces all)
          const currentIds = details.current_performer_ids || [];
          performerIds = [...new Set([...currentIds, ...addedPerfs])];
        }
      }

      // 3. Collect tag IDs
      let tagIds = null;
      if (hasTags) {
        const addedTags = [];
        wrapper.querySelectorAll('.ss-scene-tag-add-cb:checked').forEach(cb => {
          const stashboxId = cb.dataset.stashboxId;
          const cacheKey = `${endpoint}|${stashboxId}`;
          const localId = entityCache.get(cacheKey);
          if (localId) {
            addedTags.push(localId);
          } else {
            uncreated.push(`Tag: ${cb.dataset.name || stashboxId}`);
          }
        });
        if (addedTags.length > 0) {
          // Merge with current scene tags (SceneUpdateInput replaces all)
          const currentIds = details.current_tag_ids || [];
          tagIds = [...new Set([...currentIds, ...addedTags])];
        }
      }

      // 4. Collect studio ID
      let studioId = null;
      if (hasStudio) {
        const studioCb = wrapper.querySelector('.ss-scene-studio-cb:checked');
        if (studioCb && studioChange.upstream) {
          const cacheKey = `${endpoint}|${studioChange.upstream.id}`;
          const localId = entityCache.get(cacheKey);
          if (localId) {
            studioId = localId;
          } else {
            uncreated.push(`Studio: ${studioChange.upstream.name || studioChange.upstream.id}`);
          }
        }
      }

      // Block apply if checked entities haven't been linked or created yet
      if (uncreated.length > 0) {
        errorDiv.innerHTML = `<strong>Please link or create these entities first:</strong><br>${uncreated.map(e => escapeHtml(e)).join('<br>')}`;
        errorDiv.style.display = 'block';
        return;
      }

      const hasAnyChanges = Object.keys(fields).length > 0 || performerIds || tagIds || studioId;

      if (!hasAnyChanges) {
        try {
          applyBtn.disabled = true;
          applyBtn.textContent = 'Resolving...';
          await RecommendationsAPI.resolve(rec.id, 'accepted_no_changes', {});
          showSuccessAndReturn(applyBtn, 'Done!');
        } catch (e) {
          const msg = String(e?.message || e || '');
          const stale = /recommendation not found|recommendation removed because referenced scene no longer exists/i.test(msg);
          if (stale) {
            showToast('Recommendation removed because source/target scene no longer exists.', 'warning');
            removeFromListCache(rec.id);
            currentState.view = 'list';
            currentState.selectedRec = null;
            renderCurrentView(document.getElementById('ss-recommendations') || container);
            return;
          }
          applyBtn.textContent = `Failed: ${msg}`;
          applyBtn.disabled = false;
        }
        return;
      }

      try {
        applyBtn.disabled = true;
        applyBtn.textContent = 'Applying...';
        await RecommendationsAPI.updateScene(sceneId, fields, performerIds, tagIds, studioId);
        await RecommendationsAPI.resolve(rec.id, 'applied', { fields });

        showSuccessAndReturn(applyBtn, 'Applied!');
      } catch (e) {
        const msg = String(e?.message || e || '');
        const stale = /scene .*not found|removed stale upstream scene recommendation|recommendation removed because referenced scene no longer exists|recommendation not found/i.test(msg);
        if (stale) {
          showToast('Recommendation removed because source/target scene no longer exists.', 'warning');
          removeFromListCache(rec.id);
          currentState.view = 'list';
          currentState.selectedRec = null;
          renderCurrentView(document.getElementById('ss-recommendations') || container);
          return;
        }
        errorDiv.innerHTML = `<div>${escapeHtml(msg)}</div>`;
        errorDiv.style.display = 'block';
        applyBtn.textContent = 'Apply Changes';
        applyBtn.disabled = false;
      }
    });
  }

  /**
   * Render a 3-column compare field row (simple, name, text merge types).
   * Layout: [x] Local value | [ ] Upstream value | Result: [editable input]
   */
  function renderCompareField(fieldRow, change, idx, mergeType, displayValue, smartDefault) {
    const compareRow = document.createElement('div');
    compareRow.className = 'ss-upstream-compare-row';

    const defaultChoice = smartDefault(change.local_value, change.upstream_value);

    // Local cell
    const localCell = document.createElement('div');
    localCell.className = 'ss-upstream-value-cell local';
    const localLabel = document.createElement('div');
    localLabel.className = 'ss-upstream-value-label';
    localLabel.textContent = 'Local';
    const localCheckLabel = document.createElement('label');
    const localCb = document.createElement('input');
    localCb.type = 'checkbox';
    localCb.className = 'ss-upstream-cb-local';
    localCb.checked = defaultChoice === 'local';
    localCheckLabel.appendChild(localCb);
    localCheckLabel.appendChild(document.createTextNode(' ' + displayValue(change.local_display || change.local_value, change.field)));
    localCell.appendChild(localLabel);
    localCell.appendChild(localCheckLabel);

    // Upstream cell
    const upstreamCell = document.createElement('div');
    upstreamCell.className = 'ss-upstream-value-cell upstream';
    const upstreamLabel = document.createElement('div');
    upstreamLabel.className = 'ss-upstream-value-label';
    upstreamLabel.textContent = 'Upstream';
    const upstreamCheckLabel = document.createElement('label');
    const upstreamCb = document.createElement('input');
    upstreamCb.type = 'checkbox';
    upstreamCb.className = 'ss-upstream-cb-upstream';
    upstreamCb.checked = defaultChoice === 'upstream';
    upstreamCheckLabel.appendChild(upstreamCb);
    upstreamCheckLabel.appendChild(document.createTextNode(' ' + displayValue(change.upstream_display || change.upstream_value, change.field)));
    upstreamCell.appendChild(upstreamLabel);
    upstreamCell.appendChild(upstreamCheckLabel);

    // Result cell
    const resultCell = document.createElement('div');
    resultCell.className = 'ss-upstream-value-cell result';
    const resultLabel = document.createElement('div');
    resultLabel.className = 'ss-upstream-value-label';
    resultLabel.textContent = 'Result';

    let resultInput;
    if (mergeType === 'text') {
      resultInput = document.createElement('textarea');
      resultInput.className = 'ss-upstream-textarea';
    } else {
      resultInput = document.createElement('input');
      resultInput.type = 'text';
      resultInput.className = 'ss-upstream-result-input';
    }

    // Set initial result value based on smart default
    // For fields with display names (e.g. parent_studio), show the name but store the raw value
    const hasDisplay = change.local_display || change.upstream_display;
    const defaultVal = defaultChoice === 'upstream' ? change.upstream_value : change.local_value;
    const defaultDisplay = defaultChoice === 'upstream'
      ? (change.upstream_display || change.upstream_value)
      : (change.local_display || change.local_value);
    const rawDefaultStr = formatFieldValue(defaultDisplay) === '(empty)' ? '' : String(defaultDisplay || '');
    // Apply the same normalization as the local/upstream labels so the result preview matches
    const normalizedDefault = rawDefaultStr && typeof displayValue === 'function'
      ? displayValue(defaultDisplay, change.field)
      : rawDefaultStr;
    resultInput.value = normalizedDefault;
    if (hasDisplay) {
      resultInput.dataset.rawValue = defaultVal != null ? String(defaultVal) : '';
      resultInput.readOnly = true;
      resultInput.title = defaultVal != null ? String(defaultVal) : '';
    }

    resultCell.appendChild(resultLabel);
    resultCell.appendChild(resultInput);

    compareRow.appendChild(localCell);
    compareRow.appendChild(upstreamCell);
    compareRow.appendChild(resultCell);
    fieldRow.appendChild(compareRow);

    // Paired checkbox logic: clicking local unchecks upstream (and vice versa)
    localCb.addEventListener('change', () => {
      if (localCb.checked) {
        upstreamCb.checked = false;
        const srcDisplay = hasDisplay ? (change.local_display || change.local_value) : change.local_value;
        const rawStr = formatFieldValue(srcDisplay) === '(empty)' ? '' : String(srcDisplay || '');
        resultInput.value = rawStr && typeof displayValue === 'function' ? displayValue(srcDisplay, change.field) : rawStr;
        if (hasDisplay) resultInput.dataset.rawValue = change.local_value != null ? String(change.local_value) : '';
      }
    });

    upstreamCb.addEventListener('change', () => {
      if (upstreamCb.checked) {
        localCb.checked = false;
        const srcDisplay = hasDisplay ? (change.upstream_display || change.upstream_value) : change.upstream_value;
        const rawStr = formatFieldValue(srcDisplay) === '(empty)' ? '' : String(srcDisplay || '');
        resultInput.value = rawStr && typeof displayValue === 'function' ? displayValue(srcDisplay, change.field) : rawStr;
        if (hasDisplay) resultInput.dataset.rawValue = change.upstream_value != null ? String(change.upstream_value) : '';
      }
    });

    // Editing result directly unchecks both checkboxes
    resultInput.addEventListener('input', () => {
      localCb.checked = false;
      upstreamCb.checked = false;
    });

    // Name merge type: add alias checkbox only for real entity name fields.
    // Scene titles are marked with merge_type=name for comparison behavior,
    // but scenes do not support aliases.
    if (mergeType === 'name' && change.field === 'name') {
      const aliasOption = document.createElement('div');
      aliasOption.className = 'ss-upstream-name-alias-option';
      const aliasLabel = document.createElement('label');
      const aliasCb = document.createElement('input');
      aliasCb.type = 'checkbox';
      aliasCb.className = 'ss-upstream-name-alias-cb';
      aliasCb.checked = true;
      aliasLabel.appendChild(aliasCb);
      aliasLabel.appendChild(document.createTextNode(' Add old name as alias when switching'));
      aliasOption.appendChild(aliasLabel);
      fieldRow.appendChild(aliasOption);
    }

    if (mergeType === 'readonly') {
      localCb.disabled = true;
      upstreamCb.disabled = true;
      resultInput.readOnly = true;
    }
  }

  /**
   * Render alias list field: vertical stacked items with per-item checkbox,
   * all items checked by default (merge behavior), with result summary.
   */
  function renderAliasListField(fieldRow, change, idx) {
    const localAliases = change.local_value || [];
    const upstreamAliases = change.upstream_value || [];
    const allAliases = buildAliasList(localAliases, upstreamAliases);

    const aliasContainer = document.createElement('div');
    aliasContainer.className = 'ss-upstream-alias-list-container';
    aliasContainer.dataset.fieldIndex = idx;
    aliasContainer.dataset.mergeType = 'alias_list';
    aliasContainer.dataset.fieldKey = change.field;

    // Two-column sub-layout
    const subLayout = document.createElement('div');
    subLayout.style.cssText = 'display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;align-items:start;';

    // Left: item checkboxes
    const itemsCol = document.createElement('div');
    const aliasList = document.createElement('div');
    aliasList.className = 'ss-upstream-alias-list';

    allAliases.forEach((a, ai) => {
      const item = document.createElement('label');
      item.className = `ss-upstream-alias-item ${a.source}`;
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.name = `field_${idx}_alias_${ai}`;
      cb.value = a.value;
      cb.checked = true; // All checked by default (merge behavior)
      item.appendChild(cb);
      const span = document.createElement('span');
      span.textContent = a.value;
      item.appendChild(span);
      const tag = document.createElement('span');
      tag.className = 'ss-upstream-alias-tag';
      tag.textContent = a.source === 'both' ? 'both' : a.source === 'local-only' ? 'local' : 'upstream';
      item.appendChild(tag);
      aliasList.appendChild(item);

      cb.addEventListener('change', () => updateAliasResultSummary(fieldRow));
    });

    itemsCol.appendChild(aliasList);

    // Add custom alias/URL button
    const isUrlField = change.field === 'urls';
    const addBtn = document.createElement('button');
    addBtn.className = 'ss-btn ss-btn-secondary';
    addBtn.style.cssText = 'margin-top:0.5rem;padding:4px 10px;font-size:0.8rem;';
    addBtn.textContent = isUrlField ? '+ Add custom URL' : '+ Add custom alias';
    addBtn.addEventListener('click', () => {
      const newAlias = prompt(isUrlField ? 'Enter new URL:' : 'Enter new alias:');
      if (!newAlias || !newAlias.trim()) return;
      const existingCount = aliasList.querySelectorAll('.ss-upstream-alias-item').length;
      const item = document.createElement('label');
      item.className = 'ss-upstream-alias-item both';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.name = `field_${idx}_alias_${existingCount}`;
      cb.value = newAlias.trim();
      cb.checked = true;
      item.appendChild(cb);
      const span = document.createElement('span');
      span.textContent = newAlias.trim();
      item.appendChild(span);
      const tag = document.createElement('span');
      tag.className = 'ss-upstream-alias-tag';
      tag.textContent = 'custom';
      item.appendChild(tag);
      aliasList.appendChild(item);
      cb.addEventListener('change', () => updateAliasResultSummary(fieldRow));
      updateAliasResultSummary(fieldRow);
    });
    itemsCol.appendChild(addBtn);

    // Right: result summary
    const resultCol = document.createElement('div');
    resultCol.className = 'ss-upstream-alias-result';
    resultCol.innerHTML = '<div class="ss-upstream-value-label">Result</div><div class="ss-upstream-alias-result-content"></div>';

    subLayout.appendChild(itemsCol);
    subLayout.appendChild(resultCol);
    aliasContainer.appendChild(subLayout);
    fieldRow.appendChild(aliasContainer);

    // Initial result summary
    updateAliasResultSummary(fieldRow);
  }

  function updateAliasResultSummary(fieldRow) {
    const resultContent = fieldRow.querySelector('.ss-upstream-alias-result-content');
    if (!resultContent) return;
    const checkedAliases = [];
    fieldRow.querySelectorAll('.ss-upstream-alias-item input[type="checkbox"]:checked').forEach(cb => {
      checkedAliases.push(cb.value);
    });
    if (checkedAliases.length === 0) {
      resultContent.innerHTML = '<span class="ss-upstream-alias-result-count">0</span> items';
    } else {
      resultContent.innerHTML = `<span class="ss-upstream-alias-result-count">${checkedAliases.length}</span> items<ul style="margin:0.25rem 0 0 1.25rem;padding:0;list-style:disc;">${checkedAliases.map(a => `<li style="font-size:0.8rem;color:#fff;margin-bottom:2px;">${escapeHtml(a)}</li>`).join('')}</ul>`;
    }
  }

  // ==================== Scene Stash-Box Tagger Detail ====================

  function formatDuration(seconds) {
    if (!seconds && seconds !== 0) return 'N/A';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  }

  async function renderFingerprintMatchDetail(container, rec) {
    const d = rec.details;
    const isPending = rec.status === 'pending';
    const localSceneId = String(d.local_scene_id || '');
    const localSceneHref = `/scenes/${localSceneId}`;
    const stashboxBaseHref = String(d.endpoint || '').replace(/\/graphql$/, '');
    const stashboxSceneHref = `${stashboxBaseHref}/scenes/${d.stashbox_scene_id}`;

    function buildEndpointEntityHref(entityType, entityId) {
      if (!stashboxBaseHref || !entityId) return '';
      return `${stashboxBaseHref}/${entityType}/${encodeURIComponent(String(entityId))}`;
    }

    function renderDetailLink(href, label) {
      if (!label) return '';
      if (!href) return escapeHtml(label);
      return `<a class="ss-detail-entity-link" href="${escapeHtml(href)}" target="_blank" rel="noopener">${escapeHtml(label)}</a>`;
    }

    function renderLocalEntityLink(entityType, entityId, entityName) {
      if (!entityName) return '';
      if (!entityId) return escapeHtml(entityName);
      return renderDetailLink(`/${entityType}/${encodeURIComponent(String(entityId))}`, entityName);
    }

    function renderLocalPerformerLinks(performers) {
      return (performers || [])
        .map(p => renderLocalEntityLink('performers', p?.id, p?.name))
        .filter(Boolean)
        .join(', ');
    }

    function renderUpstreamPerformerLinks(performers) {
      return (performers || [])
        .map(p => renderDetailLink(buildEndpointEntityHref('performers', p?.id), p?.name))
        .filter(Boolean)
        .join(', ');
    }

    container.innerHTML = '<div class="ss-loading">Loading scene details...</div>';

    let localScene = null;
    try {
      if (localSceneId) {
        localScene = await RecommendationsAPI.getSceneDetail(localSceneId);
      }
    } catch (_) {}

    const localScreenshotUrl = relativeUrl(localScene?.paths?.screenshot);
    const localPreviewUrl = relativeUrl(localScene?.paths?.preview);
    const localStudio = localScene?.studio || null;
    const localPerformers = localScene?.performers || [];
    const localStudioLink = renderLocalEntityLink('studios', localStudio?.id, localStudio?.name);
    const localPerformerLinks = renderLocalPerformerLinks(localPerformers);
    const localDate = localScene?.date || '';
    const stashboxCoverUrl = d.stashbox_cover_url;
    const upstreamStudioLink = d.stashbox_studio
      ? renderDetailLink(buildEndpointEntityHref('studios', d.stashbox_studio_id), d.stashbox_studio)
      : '';
    const upstreamPerformers = Array.isArray(d.stashbox_performer_links)
      ? d.stashbox_performer_links
      : (d.stashbox_performers || []).map((name) => ({ id: null, name }));
    const upstreamPerformerLinks = renderUpstreamPerformerLinks(upstreamPerformers);

    container.innerHTML = `
      <div class="ss-fp-detail">
        <div class="ss-fp-detail-header">
          <h2>Scene Stash-Box Tagger</h2>
          <span class="ss-badge ${d.high_confidence ? 'ss-badge-success' : 'ss-badge-warning'}">
            ${d.high_confidence ? 'High Confidence' : 'Review Recommended'}
          </span>
        </div>

        <div class="ss-fp-detail-comparison">
          <div class="ss-fp-detail-side">
            <h4>Local Scene</h4>
            <div class="ss-dup-scene-thumb ss-fp-local-thumb">
              ${localScreenshotUrl
                ? `<img src="${localScreenshotUrl}" alt="Local scene screenshot" loading="lazy" onerror="this.style.display='none'" />`
                : '<div class="ss-no-image">No Screenshot</div>'
              }
              ${localPreviewUrl
                ? `<video class="ss-dup-scene-preview ss-fp-scene-preview" muted loop preload="none" data-src="${localPreviewUrl}"></video>`
                : ''
              }
            </div>
            <div class="ss-fp-scene-title">
              <a href="${escapeHtml(localSceneHref)}" target="_blank" rel="noopener">${escapeHtmlBreakable(d.local_scene_title || 'Unknown')}</a>
            </div>
            ${localStudioLink ? `<div class="ss-fp-field"><strong>Studio:</strong> ${localStudioLink}</div>` : ''}
            ${localPerformerLinks ? `<div class="ss-fp-field"><strong>Performers:</strong> ${localPerformerLinks}</div>` : ''}
            ${localDate ? `<div class="ss-fp-field"><strong>Date:</strong> ${escapeHtml(localDate)}</div>` : ''}
            <div class="ss-fp-field"><strong>Duration:</strong> ${d.duration_local ? formatDuration(d.duration_local) : 'N/A'}</div>
            <div class="ss-fp-field"><strong>Fingerprints:</strong> ${d.total_local_fingerprints}</div>
          </div>
          <div class="ss-fp-detail-divider"></div>
          <div class="ss-fp-detail-side">
            <h4>Stash-Box Match</h4>
            <div class="ss-dup-scene-thumb ss-fp-cover-thumb">
              ${stashboxCoverUrl
                ? `<img src="${escapeHtml(stashboxCoverUrl)}" alt="Stash-Box scene cover" loading="lazy" onerror="this.style.display='none'" />`
                : '<div class="ss-no-image">No Cover</div>'
              }
            </div>
            <div class="ss-fp-scene-title">
              <a href="${escapeHtml(stashboxSceneHref)}" target="_blank" rel="noopener">${escapeHtmlBreakable(d.stashbox_scene_title || 'Unknown')}</a>
            </div>
            ${upstreamStudioLink ? `<div class="ss-fp-field"><strong>Studio:</strong> ${upstreamStudioLink}</div>` : ''}
            ${upstreamPerformerLinks ? `<div class="ss-fp-field"><strong>Performers:</strong> ${upstreamPerformerLinks}</div>` : ''}
            ${d.stashbox_date ? `<div class="ss-fp-field"><strong>Date:</strong> ${d.stashbox_date}</div>` : ''}
            <div class="ss-fp-field"><strong>Duration:</strong> ${d.duration_remote ? formatDuration(d.duration_remote) : 'N/A'}</div>
            <div class="ss-fp-field">
              <strong>Endpoint:</strong>
              <a class="ss-detail-entity-link" href="${escapeHtml(stashboxSceneHref)}" target="_blank" rel="noopener">${escapeHtml(d.endpoint_name || d.endpoint)}</a>
            </div>
          </div>
        </div>

        <div class="ss-fp-detail-fingerprints">
          <h4>Fingerprint Comparison</h4>
          <div class="ss-fp-table-wrap">
            <table class="ss-fp-table">
              <thead>
                <tr>
                  <th>Algorithm</th>
                  <th>Hash</th>
                  <th>Duration</th>
                  <th>Submissions</th>
                </tr>
              </thead>
              <tbody>
                ${(d.matching_fingerprints || []).map(fp => `
                  <tr>
                    <td><span class="ss-badge ss-badge-${fp.algorithm === 'PHASH' ? 'warning' : 'success'}">${fp.algorithm}</span></td>
                    <td class="ss-fp-hash">${fp.hash}</td>
                    <td>${fp.duration ? formatDuration(fp.duration) : 'N/A'}</td>
                    <td>${fp.submissions || 0}</td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
          <div class="ss-fp-summary-line">
            <strong>${d.match_count}/${d.total_local_fingerprints}</strong> fingerprints match
            (${d.match_percentage}%)
            ${d.duration_agreement ? '&mdash; duration agrees' : '&mdash; <span class="ss-text-warning">duration mismatch</span>'}
          </div>
        </div>

        ${isPending ? `
          <div class="ss-fp-detail-actions">
            <button id="ss-fp-accept-btn" class="ss-btn ss-btn-primary">Accept Match</button>
            <button id="ss-fp-dismiss-btn" class="ss-btn ss-btn-secondary">Dismiss</button>
          </div>
        ` : `
          <div class="ss-fp-detail-status">
            Status: <strong>${rec.status}</strong>
            ${rec.resolution_action ? ` (${rec.resolution_action})` : ''}
          </div>
        `}
      </div>
    `;

    // Local preview video on hover (same behavior as duplicate scenes)
    const localThumb = container.querySelector('.ss-fp-local-thumb');
    if (localThumb) {
      const video = localThumb.querySelector('.ss-fp-scene-preview');
      const img = localThumb.querySelector('img');
      if (video) {
        localThumb.addEventListener('mouseenter', () => {
          if (!video.src && video.dataset.src) {
            video.src = video.dataset.src;
          }
          if (img) img.style.opacity = '0';
          video.style.opacity = '1';
          video.play().catch(() => {});
        });

        localThumb.addEventListener('mouseleave', () => {
          video.pause();
          if (img) img.style.opacity = '1';
          video.style.opacity = '0';
          setTimeout(() => { video.currentTime = 0; }, 200);
        });
      }
    }

    if (!isPending) return;

    // Accept button
    const acceptBtn = container.querySelector('#ss-fp-accept-btn');
    acceptBtn.addEventListener('click', async () => {
      acceptBtn.disabled = true;
      acceptBtn.textContent = 'Accepting...';
      try {
        await RecommendationsAPI.acceptFingerprintMatch(
          rec.id, d.local_scene_id, d.endpoint, d.stashbox_scene_id
        );
        showSuccessAndReturn(acceptBtn, 'Accepted!');
      } catch (e) {
        acceptBtn.textContent = `Failed: ${e.message}`;
        acceptBtn.classList.add('ss-btn-error');
        acceptBtn.disabled = false;
      }
    });

    // Dismiss button
    const dismissBtn = container.querySelector('#ss-fp-dismiss-btn');
    dismissBtn.addEventListener('click', async () => {
      dismissBtn.disabled = true;
      dismissBtn.textContent = 'Dismissing...';
      try {
        await RecommendationsAPI.dismiss(rec.id);
        showSuccessAndReturn(dismissBtn, 'Dismissed!');
      } catch (e) {
        dismissBtn.textContent = `Failed: ${e.message}`;
        dismissBtn.classList.add('ss-btn-error');
        dismissBtn.disabled = false;
      }
    });
  }

  async function renderSceneFaceMatchDetail(container, rec) {
    const d = rec.details || {};
    const isPending = rec.status === 'pending';
    const sceneId = String(d.scene_id || '');
    const sceneHref = `/scenes/${sceneId}`;
    const persons = d.persons || [];

    container.innerHTML = '<div class="ss-loading">Loading scene details...</div>';

    let scene = null;
    try {
      if (sceneId) scene = await RecommendationsAPI.getSceneDetail(sceneId);
    } catch (_) {}

    // sceneStreams (multiple negotiated variants -- direct file, HLS,
    // WEBM/lower-res transcodes, same set Stash's own native player
    // uses) lets the browser fall back past a codec it can't natively
    // decode (e.g. HEVC), unlike a single <video src> pointed straight
    // at paths.stream -- confirmed live: a scene playing fine in Stash's
    // own page threw "No video with supported format and MIME type
    // found" here because only the raw direct-file URL was offered, no
    // fallback source. Falls back to the single paths.stream source only
    // if sceneStreams is empty/missing (e.g. an older Stash without that
    // field) -- same behavior as before this fix in that case.
    const sceneStreams = Array.isArray(scene?.sceneStreams) ? scene.sceneStreams : [];
    const streamUrl = relativeUrl(scene?.paths?.stream);
    const videoSourcesHtml = sceneStreams.length
      ? sceneStreams.map(s => {
          const url = relativeUrl(s.url);
          if (!url) return '';
          const typeAttr = s.mime_type ? ` type="${escapeHtml(s.mime_type)}"` : '';
          return `<source src="${escapeHtml(url)}"${typeAttr} />`;
        }).join('')
      : (streamUrl ? `<source src="${escapeHtml(streamUrl)}" />` : '');
    const sceneTitle = scene?.title || d.scene_title || `Scene ${sceneId}`;
    const posterUrl = relativeUrl(scene?.paths?.screenshot);

    // Same link-building convention the manual scene-identify result view
    // uses (stash-sense.js's _matchLinksHtml/_stashboxPerformerUrl/
    // _localPerformerUrl): catalogue sources link to their own profile/site,
    // a local-index match always gets a "View local performer" link (plus a
    // stashdb.org link too if it's *also* linked to a real StashDB id --
    // two independently-verifiable signals), everything else gets a single
    // "View on <endpoint>" stashbox link.
    function sceneFaceMatchLinksHtml(c) {
      const endpoint = c.endpoint || 'stashdb.org';
      if (c.source) {
        const href = c.profile_url || c.catalogue_url;
        if (!href) return '';
        const label = c.profile_url ? 'View profile' : `View on ${c.source}`;
        return `<a href="${escapeHtml(href)}" target="_blank" rel="noopener" class="ss-link">${escapeHtml(label)}</a>`;
      }
      if (!c.local_performer_id) {
        if (!c.stashdb_id) return '';
        const stashboxUrl = `https://${endpoint}/performers/${c.stashdb_id}`;
        return `<a href="${escapeHtml(stashboxUrl)}" target="_blank" rel="noopener" class="ss-link">View on ${escapeHtml(endpoint)}</a>`;
      }
      const localLink = `<a href="/performers/${encodeURIComponent(c.local_performer_id)}" target="_blank" rel="noopener" class="ss-link ss-link-local">View local performer</a>`;
      const hasStashDbLink = c.stashdb_id && c.stashdb_id !== c.local_performer_id;
      if (!hasStashDbLink) return localLink;
      const stashDbUrl = `https://stashdb.org/performers/${c.stashdb_id}`;
      return `<a href="${escapeHtml(stashDbUrl)}" target="_blank" rel="noopener" class="ss-link">View on stashdb.org</a> ${localLink}`;
    }

    function renderCandidate(c) {
      const isCandidatePending = c.status === 'pending';
      const jumpButtons = (c.top_timestamps_sec || []).slice(0, 4).map(t => (
        `<button type="button" class="ss-btn ss-btn-tiny ss-sfm-jump-btn" data-time="${t}">${formatDuration(t)}</button>`
      )).join('');
      const linksHtml = sceneFaceMatchLinksHtml(c);
      // Don't pre-select a weak match: 5 or fewer frames, or under 10%
      // confidence, needs a deliberate look before it gets added to a scene.
      const meetsPreselectThreshold = (c.frame_count || 0) > 5 && (c.confidence || 0) >= 0.10;
      const preselect = c.is_best_match && meetsPreselectThreshold;
      return `
        <div class="ss-sfm-candidate${isCandidatePending ? '' : ' ss-sfm-candidate-inactive'}" data-rec-id="${c.recommendation_id}">
          <label class="ss-sfm-candidate-select">
            <input type="checkbox" class="ss-sfm-candidate-cb" data-rec-id="${c.recommendation_id}"
              ${preselect ? 'checked' : ''} ${isCandidatePending ? '' : 'disabled'} />
            <div class="ss-sfm-candidate-thumb">
              ${c.image_url
                ? `<img src="${escapeHtml(c.image_url)}" alt="${escapeHtml(c.name || '')}" loading="lazy" onerror="this.style.display='none'" />`
                : '<div class="ss-no-image">No Image</div>'
              }
            </div>
          </label>
          <div class="ss-sfm-candidate-name">${escapeHtmlBreakable(c.name || 'Unknown')}</div>
          <div class="ss-sfm-candidate-meta">
            ${Math.round((c.confidence || 0) * 100)}% match
            ${!isCandidatePending ? ` &middot; <span class="ss-sfm-candidate-status">${escapeHtml(c.status)}</span>` : ''}
          </div>
          ${linksHtml ? `<div class="ss-sfm-candidate-links">${linksHtml}</div>` : ''}
          ${jumpButtons ? `<div class="ss-sfm-jump-row">${jumpButtons}</div>` : ''}
          ${isPending && isCandidatePending
            ? `<button type="button" class="ss-btn ss-btn-tiny ss-btn-secondary ss-sfm-dismiss-one-btn" data-rec-id="${c.recommendation_id}">Dismiss</button>`
            : ''
          }
        </div>
      `;
    }

    function renderPersonColumn(person) {
      const frameCount = person.frame_count || 0;
      return `
        <div class="ss-sfm-person-column">
          <div class="ss-sfm-person-header">
            Person ${(person.person_id ?? 0) + 1}
            <span class="ss-sfm-person-frames">(${frameCount} frame${frameCount === 1 ? '' : 's'})</span>
          </div>
          <div class="ss-sfm-person-candidates">
            ${(person.candidates || []).map(renderCandidate).join('')}
          </div>
        </div>
      `;
    }

    container.innerHTML = `
      <div class="ss-sfm-detail">
        <div class="ss-sfm-detail-header">
          <h2>${escapeHtmlBreakable(sceneTitle)}</h2>
          <a class="ss-detail-entity-link" href="${escapeHtml(sceneHref)}" target="_blank" rel="noopener">Open in Stash</a>
        </div>

        ${videoSourcesHtml
          ? `<video class="ss-sfm-video" controls preload="metadata"${posterUrl ? ` poster="${escapeHtml(posterUrl)}"` : ''}>${videoSourcesHtml}</video>`
          : '<div class="ss-no-image ss-sfm-no-video">No video preview available</div>'
        }

        <div class="ss-sfm-persons">
          ${persons.map(renderPersonColumn).join('')}
        </div>

        ${isPending ? `
          <div class="ss-sfm-detail-actions">
            <button id="ss-sfm-accept-btn" class="ss-btn ss-btn-primary">Accept Selected</button>
            <button id="ss-sfm-reject-all-btn" class="ss-btn ss-btn-secondary">Reject All</button>
          </div>
        ` : `
          <div class="ss-fp-detail-status">
            Status: <strong>${rec.status}</strong>
          </div>
        `}
      </div>
    `;

    const videoEl = container.querySelector('.ss-sfm-video');
    container.querySelectorAll('.ss-sfm-jump-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        if (!videoEl) return;
        const t = parseFloat(btn.dataset.time);
        if (!Number.isNaN(t)) {
          videoEl.currentTime = t;
          videoEl.play().catch(() => {});
        }
      });
    });

    if (!isPending) return;

    container.querySelectorAll('.ss-sfm-dismiss-one-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        const recId = parseInt(btn.dataset.recId, 10);
        btn.disabled = true;
        try {
          await RecommendationsAPI.dismissSceneFaceMatch(recId);
          const card = btn.closest('.ss-sfm-candidate');
          if (card) {
            card.classList.add('ss-sfm-candidate-inactive');
            const cb = card.querySelector('.ss-sfm-candidate-cb');
            if (cb) { cb.checked = false; cb.disabled = true; }
            btn.remove();
          }
          showToast('Candidate dismissed.', 'info');
        } catch (e) {
          showToast(`Failed to dismiss: ${e.message}`, 'error');
          btn.disabled = false;
        }
      });
    });

    const acceptBtn = container.querySelector('#ss-sfm-accept-btn');
    acceptBtn.addEventListener('click', async () => {
      const selectedIds = Array.from(container.querySelectorAll('.ss-sfm-candidate-cb:checked'))
        .map(cb => parseInt(cb.dataset.recId, 10));
      if (selectedIds.length === 0) {
        showToast('Select at least one performer to accept.', 'warning');
        return;
      }
      acceptBtn.disabled = true;
      acceptBtn.textContent = 'Accepting...';
      try {
        await RecommendationsAPI.acceptSceneFaceMatches(sceneId, selectedIds);
        showSuccessAndReturn(acceptBtn, 'Accepted!');
      } catch (e) {
        acceptBtn.textContent = `Failed: ${e.message}`;
        acceptBtn.classList.add('ss-btn-error');
        acceptBtn.disabled = false;
      }
    });

    const rejectBtn = container.querySelector('#ss-sfm-reject-all-btn');
    rejectBtn.addEventListener('click', () => {
      showConfirmModal(
        'Reject all identified performers for this scene? It moves to the Dismissed tab, but may reappear if a future scan finds new matches.',
        async () => {
          rejectBtn.disabled = true;
          rejectBtn.textContent = 'Rejecting...';
          try {
            await RecommendationsAPI.rejectAllSceneFaceMatches(sceneId);
            showSuccessAndReturn(rejectBtn, 'Rejected!');
          } catch (e) {
            rejectBtn.textContent = `Failed: ${e.message}`;
            rejectBtn.classList.add('ss-btn-error');
            rejectBtn.disabled = false;
          }
        },
      );
    });
  }

  /**
   * Create a search dropdown for linking a local entity to a stash-box ID.
   * @param {string} entityType - "performer", "tag", or "studio"
   * @param {string} endpoint - stash-box endpoint URL
   * @param {string} stashboxId - stash-box entity UUID
   * @param {function} onMatch - callback(localId, localName) when linked
   * @returns {HTMLElement} the dropdown container element
   */
  function createEntitySearchDropdown(entityType, endpoint, stashboxId, onMatch, initialSearch) {
    const container = document.createElement('div');
    container.className = 'ss-entity-search-dropdown';
    let resolved = false;

    const trigger = document.createElement('button');
    trigger.className = 'ss-btn ss-btn-sm ss-entity-link-btn';
    trigger.textContent = 'Link';
    container.appendChild(trigger);

    const panel = document.createElement('div');
    panel.className = 'ss-entity-search-panel';
    panel.style.display = 'none';

    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'ss-entity-search-input';
    input.placeholder = `Search ${entityType}s...`;
    panel.appendChild(input);

    const resultsList = document.createElement('div');
    resultsList.className = 'ss-entity-search-results';
    panel.appendChild(resultsList);

    container.appendChild(panel);

    function resolveMatch(localId, localName) {
      if (resolved) return;
      resolved = true;
      onMatch(localId, localName);
    }

    function normalizeEndpoint(value) {
      if (!value) return '';
      let v = String(value).trim().toLowerCase().replace(/\/+$/, '');
      if (v.startsWith('https://')) v = v.slice('https://'.length);
      else if (v.startsWith('http://')) v = v.slice('http://'.length);
      if (v.endsWith('/graphql')) v = v.slice(0, -'/graphql'.length);
      return v;
    }

    function hasExactStashLink(result) {
      const targetEndpoint = normalizeEndpoint(endpoint);
      const targetStashId = String(stashboxId || '');
      return (result.stash_ids || []).some(sid =>
        normalizeEndpoint(sid.endpoint) === targetEndpoint &&
        String(sid.stash_id || '') === targetStashId
      );
    }

    function matchesNeedle(result, needleRaw) {
      const needle = String(needleRaw || '').trim().toLowerCase();
      if (!needle) return false;
      const name = String(result.name || '').trim().toLowerCase();
      if (name === needle) return true;
      const aliases = Array.isArray(result.aliases) ? result.aliases : [];
      return aliases.some(a => String(a || '').trim().toLowerCase() === needle);
    }

    // Pre-fill query but keep explicit "Link" button visible.
    if (initialSearch) {
      input.value = initialSearch;
      // Run the same lookup server-side on load so pre-existing links are
      // auto-selected even when not present in in-browser cache.
      setTimeout(async () => {
        if (resolved) return;
        const q = String(initialSearch || '').trim();
        if (q.length < 2) return;
        try {
          const resp = await RecommendationsAPI.searchEntities(entityType, q, endpoint);
          const results = resp.results || [];
          if (!results.length) return;

          const exactLinked = results.find(r => matchesNeedle(r, q) && hasExactStashLink(r));
          if (exactLinked) {
            resolveMatch(exactLinked.id, exactLinked.name);
            return;
          }

          const nonLinkedMatches = results.filter(r => !r.linked && matchesNeedle(r, q));
          if (nonLinkedMatches.length === 1) {
            const r = nonLinkedMatches[0];
            try {
              await RecommendationsAPI.linkEntity(entityType, r.id, endpoint, stashboxId);
              resolveMatch(r.id, r.name);
            } catch (_) {
              // Keep manual link flow available on failure.
            }
          }
        } catch (_) {
          // Keep manual link flow available on failure.
        }
      }, 0);
    }

    let debounceTimer = null;

    trigger.addEventListener('click', (e) => {
      e.stopPropagation();
      const isOpen = panel.style.display !== 'none';
      panel.style.display = isOpen ? 'none' : '';
      if (!isOpen) {
        input.focus();
        // If we already have a pre-filled query (e.g. performer name), run search immediately.
        const q = input.value.trim();
        if (q.length >= 2 && !resultsList.innerHTML.trim()) {
          input.dispatchEvent(new Event('input'));
        }
      }
    });

    // Close when clicking outside
    document.addEventListener('click', (e) => {
      if (!container.contains(e.target)) {
        panel.style.display = 'none';
      }
    });

    input.addEventListener('input', () => {
      clearTimeout(debounceTimer);
      const q = input.value.trim();
      if (q.length < 2) {
        resultsList.innerHTML = '';
        return;
      }
      debounceTimer = setTimeout(async () => {
        resultsList.innerHTML = '<div class="ss-entity-search-loading">Searching...</div>';
        try {
          const resp = await RecommendationsAPI.searchEntities(entityType, q, endpoint);
          const results = resp.results || [];
          if (results.length === 0) {
            resultsList.innerHTML = '<div class="ss-entity-search-empty">No results</div>';
            return;
          }
          resultsList.innerHTML = '';
          results.forEach(r => {
            const item = document.createElement('div');
            item.className = 'ss-entity-search-result-item';
            if (r.linked) item.classList.add('ss-entity-already-linked');

            let label = escapeHtml(r.name);
            if (r.disambiguation) label += ` <span class="ss-entity-search-disambig">(${escapeHtml(r.disambiguation)})</span>`;
            if (r.linked) label += ' <span class="ss-entity-search-linked-badge">linked</span>';

            item.innerHTML = label;
            item.addEventListener('click', async () => {
              item.textContent = 'Linking...';
              try {
                await RecommendationsAPI.linkEntity(entityType, r.id, endpoint, stashboxId);
                panel.style.display = 'none';
                resolveMatch(r.id, r.name);
              } catch (err) {
                item.textContent = `Failed: ${err.message}`;
              }
            });
            resultsList.appendChild(item);
          });
        } catch (err) {
          resultsList.innerHTML = `<div class="ss-entity-search-empty">Error: ${escapeHtml(err.message)}</div>`;
        }
      }, 300);
    });

    return container;
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(String(str)));
    return div.innerHTML;
  }

  /**
   * Escape HTML and insert <wbr> break opportunities after underscores,
   * dots, commas, and dashes, so long titles/filenames with no spaces
   * (e.g. "some_long_file-name_2024.mp4") still wrap instead of
   * overflowing their container.
   */
  function escapeHtmlBreakable(str) {
    return escapeHtml(str).replace(/([_.,-])/g, '$1<wbr>');
  }

  /**
   * Convert an absolute URL to a relative path. Stash's API returns image URLs
   * with the internal origin (e.g. http://10.0.0.4:6969/scene/1/screenshot).
   * When the user accesses Stash via a reverse proxy on a different domain,
   * these absolute URLs break (mixed content, unreachable host). Stripping to
   * a relative path lets the browser resolve against the current origin.
   */
  function relativeUrl(url) {
    if (!url) return url;
    try { return new URL(url).pathname; }
    catch (e) { return url; }
  }

  function formatFieldValue(val) {
    if (val === null || val === undefined || isNullishString(val)) return '(empty)';
    if (Array.isArray(val)) {
      const normalized = val
        .map(v => String(v).trim())
        .filter(v => v && v.toLowerCase() !== 'null');
      return normalized.join(', ') || '(empty)';
    }
    const text = String(val);
    return text || '(empty)';
  }

  function buildAliasList(localAliases, upstreamAliases) {
    const localArr = (localAliases || []).map(String);
    const upstreamArr = (upstreamAliases || []).map(String);
    // Case-insensitive lookup maps with trailing slash normalization (lowercase -> original value)
    const normalize = s => s.toLowerCase().replace(/\/+$/, '');
    const localLower = new Map(localArr.map(a => [normalize(a), a]));
    const upstreamLower = new Map(upstreamArr.map(a => [normalize(a), a]));
    // Merge keys (deduplicated by lowercase)
    const allKeys = new Set([...localLower.keys(), ...upstreamLower.keys()]);
    const result = [];
    for (const key of allKeys) {
      const inLocal = localLower.has(key);
      const inUpstream = upstreamLower.has(key);
      // Prefer local's casing when both exist
      const value = inLocal ? localLower.get(key) : upstreamLower.get(key);
      let source;
      if (inLocal && inUpstream) {
        source = 'both';
      } else if (inLocal) {
        source = 'local-only';
      } else {
        source = 'upstream-only';
      }
      result.push({ value, source });
    }
    return result;
  }

  // ==================== View Router ====================

  function renderCurrentView(container) {
    const content = container.querySelector('.ss-dashboard-content') || container;
    switch (currentState.view) {
      case 'dashboard':
        renderDashboard(container, content);
        break;
      case 'list':
        renderList(content);
        break;
      case 'detail':
        renderDetail(content);
        break;
      default:
        renderDashboard(container, content);
    }
  }

  // ==================== Plugin Page Injection ====================

  function injectPluginPage() {
    // Check if we're on the plugin page
    const route = SS.getRoute();
    if (route.type !== 'plugin') return;

    // Check if already injected
    if (document.getElementById('ss-recommendations')) {
      console.log('[Stash Sense] Dashboard already injected');
      return;
    }

    // Try multiple selectors for Stash version compatibility
    const containerSelectors = [
      '.PluginRoutes',           // Plugin routes container
      '#root > div > div.main',  // Main content area
      '.main',                   // Fallback main
      '#root > div > div',       // Generic fallback
    ];

    let mainContainer = null;
    for (const selector of containerSelectors) {
      const el = document.querySelector(selector);
      if (el) {
        mainContainer = el;
        console.log(`[Stash Sense] Found container: ${selector}`);
        break;
      }
    }

    if (!mainContainer) {
      // Create floating container as last resort
      console.warn('[Stash Sense] No container found, creating overlay');
      mainContainer = document.createElement('div');
      mainContainer.className = 'ss-floating-overlay';
      mainContainer.style.cssText = 'position: fixed; top: 60px; left: 0; right: 0; bottom: 0; overflow-y: auto; background: var(--bs-body-bg, #1a1a1a); z-index: 100;';
      document.body.appendChild(mainContainer);
    }

    // Create and inject our dashboard
    const container = createDashboardContainer();

    // If the container has existing content, replace it
    // For plugin routes, we should be able to append
    if (mainContainer.classList.contains('PluginRoutes')) {
      mainContainer.innerHTML = '';
    }
    mainContainer.appendChild(container);

    // Reset state
    currentState = {
      view: 'dashboard',
      type: null,
      status: 'pending',
      page: 0,
      selectedRec: null,
      counts: null,
      listCache: null,
    };

    renderCurrentView(container);
  }

  // ==================== Initialization ====================

  function cleanup() {
    // Remove the main dashboard container
    const dashboard = document.getElementById('ss-recommendations');
    if (dashboard) dashboard.remove();

    // Remove floating overlay fallback container (if created as last-resort)
    document.querySelectorAll('.ss-floating-overlay').forEach(el => el.remove());

    // Remove any modal overlays appended to document.body
    document.querySelectorAll('.ss-modal-overlay').forEach(el => el.remove());

    // Reset state
    currentState = {
      view: 'dashboard',
      type: null,
      status: 'pending',
      page: 0,
      selectedRec: null,
      counts: null,
      listCache: null,
    };

    entityCache.clear();
  }

  function init() {
    // Try to inject if we're already on the plugin page
    setTimeout(injectPluginPage, 300);

    // Watch for navigation to plugin page
    SS.onNavigate((route) => {
      if (route.type === 'plugin') {
        setTimeout(injectPluginPage, 300);
      }
    });

    // Clean up when leaving plugin page
    SS.onLeavePlugin(cleanup);

    console.log(`[${SS.PLUGIN_NAME}] Recommendations module loaded`);
  }

  // Refresh type-card counts on the dashboard without a full re-render.
  // Called by the settings module when switching back to the recommendations
  // tab -- this is "returning to the main recommendations tab" from
  // elsewhere in the plugin, same category of navigation as the in-page
  // dashboard Back button, so it invalidates the cached list page too (see
  // that Back handler's comment) rather than just refreshing card counts.
  async function refreshCounts() {
    invalidateListCache();
    try {
      const countsResult = await RecommendationsAPI.getCounts();
      if (!countsResult) return;
      currentState.counts = countsResult;

      // Update total pending badge
      const badge = document.querySelector('#ss-recommendations .ss-section-header .ss-count-badge');
      if (badge) badge.textContent = countsResult.total_pending ?? '';

      // Update per-type count numbers on each card
      const typeCards = document.querySelectorAll('#ss-recommendations .ss-type-card[data-type]');
      typeCards.forEach(card => {
        const type = card.dataset.type;
        const typeCounts = countsResult.counts?.[type] || {};
        const pendingEl = card.querySelector('.ss-count-pending .ss-count-number');
        const resolvedEl = card.querySelector('.ss-count-resolved .ss-count-number');
        const dismissedEl = card.querySelector('.ss-count-dismissed .ss-count-number');
        if (pendingEl) pendingEl.textContent = typeCounts.pending ?? 0;
        if (resolvedEl) resolvedEl.textContent = typeCounts.resolved ?? 0;
        if (dismissedEl) dismissedEl.textContent = typeCounts.dismissed ?? 0;
      });
    } catch (e) {
      // Ignore errors — this is a best-effort refresh
    }
  }

  // Export for testing/debugging
  window.StashSenseRecommendations = {
    API: RecommendationsAPI,
    getState: () => currentState,
    refreshCounts,
    init,
  };

  // Initialize
  init();
})();
