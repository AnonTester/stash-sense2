/**
 * Stash Sense Main Entry Point
 *
 * Loads core module and feature modules:
 * - Face recognition (scene page integration)
 * - Recommendations dashboard (plugin page)
 */
(function() {
  'use strict';

  // Wait for core module
  function waitForCore(callback, attempts = 0) {
    if (window.StashSense) {
      callback();
    } else if (attempts < 50) {
      setTimeout(() => waitForCore(callback, attempts + 1), 100);
    } else {
      console.error('[Stash Sense] Core module failed to load');
    }
  }

  waitForCore(() => {
    const SS = window.StashSense;

    // Shared between button-creation templates and updateButtonStatus() so
    // the "outdated sidecar" indicator (distinct from plain connected/
    // disconnected) is consistent everywhere a button gets its initial
    // class/title, not just on the periodic health-check re-render.
    function statusIconClass(status) {
      const versionInfo = SS.getSidecarVersionInfo();
      const pluginInfo = SS.getPluginVersionInfo();
      if (status === true && versionInfo && versionInfo.outdated) return 'ss-outdated';
      if (status === true && pluginInfo && pluginInfo.tooOld) return 'ss-outdated';
      // Non-required "a newer release exists" -- distinct, lower-alarm styling
      // from the red required-update ss-outdated class (mirrors the existing
      // .ss-db-stat-local vs .ss-db-stat-warning distinction elsewhere in this
      // plugin: FYI vs action-needed).
      if (status === true && ((versionInfo && versionInfo.updateAvailable) || (pluginInfo && pluginInfo.updateAvailable))) {
        return 'ss-update-available';
      }
      if (status === true) return 'ss-connected';
      if (status === false) return 'ss-disconnected';
      return '';
    }
    function statusTitle(defaultTitle) {
      const status = SS.getSidecarStatus();
      const versionInfo = SS.getSidecarVersionInfo();
      const pluginInfo = SS.getPluginVersionInfo();
      if (status === true && versionInfo && versionInfo.outdated) {
        return `${SS.PLUGIN_NAME}: sidecar v${versionInfo.current} is older than required `
          + `(v${versionInfo.required}+). Update the sidecar container to restore full functionality.`;
      }
      if (status === true && pluginInfo && pluginInfo.tooOld) {
        return `${SS.PLUGIN_NAME}: this plugin (v${pluginInfo.current}) is older than the connected `
          + `sidecar expects (v${pluginInfo.minRequired}+). Update via Settings > Plugins > Available Plugins.`;
      }
      if (status === true && versionInfo && versionInfo.updateAvailable) {
        return `${SS.PLUGIN_NAME}: a newer sidecar release is available (v${versionInfo.latestVersion}).`;
      }
      if (status === true && pluginInfo && pluginInfo.updateAvailable) {
        return `${SS.PLUGIN_NAME}: a newer plugin release is available (v${pluginInfo.latestVersion}). `
          + `Update via Settings > Plugins > Available Plugins.`;
      }
      if (status === false) return `${SS.PLUGIN_NAME}: Not connected`;
      return defaultTitle;
    }

    // Convert an absolute URL to a relative path. Stash's GraphQL API
    // returns image URLs with whatever origin it was queried through --
    // for /stash/search-performers that's the sidecar's own STASH_URL,
    // not necessarily the address the browser uses to reach Stash (a
    // reverse proxy or a different LAN hostname/port), so the raw
    // absolute URL can point somewhere the browser can't reach. Stripping
    // to a relative path lets it resolve against the current page's own
    // origin instead. Same fix already applied in
    // stash-sense-recommendations.js's own relativeUrl().
    function relativeUrl(url) {
      if (!url) return url;
      try { return new URL(url).pathname; }
      catch (e) { return url; }
    }

    // Poll /health while an identify request is in flight so the loading
    // modal can show real feedback during the multi-second lazy face
    // recognition model load instead of sitting on "Connecting to Stash
    // Sense..." with no indication anything is happening. Returns a stop
    // function; safe to call even if the identify request finishes before
    // the first poll response comes back.
    function pollModelLoading(sidecarUrl, onProgress) {
      let stopped = false;
      let shownLoadingMessage = false;

      const poll = async () => {
        if (stopped) return;
        try {
          const health = await SS.runPluginOperation('health', { sidecar_url: sidecarUrl });
          if (stopped) return;
          if (health && health.face_recognition_loading) {
            shownLoadingMessage = true;
            onProgress?.('Loading face recognition models (first use after idle)...');
          } else if (shownLoadingMessage) {
            shownLoadingMessage = false;
            onProgress?.('Identifying performers...');
          }
        } catch (e) {
          // Ignore poll errors -- this is best-effort UI feedback, not
          // load-bearing for the actual identify request.
        }
      };

      poll();
      const interval = setInterval(poll, 700);
      return () => {
        stopped = true;
        clearInterval(interval);
      };
    }

    // ==================== Face Recognition Module ====================

    const FaceRecognition = {
      // Convert distance to confidence percentage
      distanceToConfidence(distance) {
        const clamped = Math.max(0, Math.min(1, distance));
        return Math.round((1 - clamped) * 100);
      },

      // Get scene's existing performer StashDB IDs
      async getScenePerformerStashDBIds(sceneId) {
        const query = `
          query GetScenePerformers($id: ID!) {
            findScene(id: $id) {
              performers {
                id
                name
                stash_ids { endpoint stash_id }
              }
            }
          }
        `;
        try {
          const data = await SS.stashQuery(query, { id: sceneId });
          const performers = data?.findScene?.performers || [];
          return performers;
        } catch (e) {
          console.error('Failed to get scene performers:', e);
          return [];
        }
      },

      // Call the face recognition API via Python backend. extraOptions can
      // carry use_sprite/skip_frame_extraction/use_cache for the "Identify
      // full video" flow (see handleIdentifyFullVideo) -- omitted entirely
      // for a plain call, matching prior behavior exactly.
      async identifyScene(sceneId, onProgress, extraOptions = {}) {
        const settings = await SS.getSettings();
        onProgress?.(`Connecting to ${SS.PLUGIN_NAME}...`);

        const stopPolling = pollModelLoading(settings.sidecarUrl, onProgress);
        try {
          // Get existing performer StashDB IDs for tagged-performer awareness
          const scenePerformers = await this.getScenePerformerStashDBIds(sceneId);
          const stashdbIds = [];
          for (const p of scenePerformers) {
            for (const sid of (p.stash_ids || [])) {
              if (sid.endpoint === 'https://stashdb.org/graphql') {
                stashdbIds.push(sid.stash_id);
              }
            }
          }

          const result = await SS.runPluginOperation('identify_scene', {
            scene_id: sceneId,
            sidecar_url: settings.sidecarUrl,
            top_k: settings.maxResults,
            scene_performer_stashdb_ids: stashdbIds,
            // Omitted params (num_frames, max_distance, min_face_size) default from sidecar face_config.py
            ...extraOptions,
          });

          if (result.error) {
            throw new Error(result.error);
          }

          // Attach scene performers for UI rendering
          result._scenePerformers = scenePerformers;

          return result;
        } finally {
          stopPolling();
        }
      },

      // Call the face recognition API for a captured video frame or crop
      // (base64 JPEG) -- used by "Identify current frame" and "Select to
      // identify". Mirrors identifyImage but hits the generic /identify
      // endpoint (no Stash entity involved).
      async identifyFrame(imageBase64, onProgress) {
        const settings = await SS.getSettings();
        onProgress?.(`Connecting to ${SS.PLUGIN_NAME}...`);

        const stopPolling = pollModelLoading(settings.sidecarUrl, onProgress);
        try {
          const result = await SS.runPluginOperation('identify_frame', {
            image_base64: imageBase64,
            sidecar_url: settings.sidecarUrl,
          });

          if (result.error) {
            throw new Error(result.error);
          }

          return result;
        } finally {
          stopPolling();
        }
      },

      // Add performer to scene
      async addPerformerToScene(sceneId, performerId) {
        const getQuery = `
          query GetScene($id: ID!) {
            findScene(id: $id) {
              performers { id }
            }
          }
        `;

        const updateQuery = `
          mutation UpdateScene($id: ID!, $performer_ids: [ID!]) {
            sceneUpdate(input: { id: $id, performer_ids: $performer_ids }) {
              id
            }
          }
        `;

        try {
          const getResult = await SS.stashQuery(getQuery, { id: sceneId });
          const currentPerformers = getResult?.findScene?.performers || [];
          const currentIds = currentPerformers.map(p => p.id);

          if (!currentIds.includes(performerId)) {
            currentIds.push(performerId);
          }

          await SS.stashQuery(updateQuery, { id: sceneId, performer_ids: currentIds });
          return true;
        } catch (e) {
          console.error('Failed to add performer:', e);
          return false;
        }
      },

      // Call the face recognition API for a single image
      async identifyImage(imageId, onProgress) {
        const settings = await SS.getSettings();
        onProgress?.(`Connecting to ${SS.PLUGIN_NAME}...`);

        const stopPolling = pollModelLoading(settings.sidecarUrl, onProgress);
        try {
          const result = await SS.runPluginOperation('identify_image', {
            image_id: imageId,
            sidecar_url: settings.sidecarUrl,
          });

          if (result.error) {
            throw new Error(result.error);
          }

          return result;
        } finally {
          stopPolling();
        }
      },

      // Add performer to image
      async addPerformerToImage(imageId, performerId) {
        const getQuery = `
          query GetImage($id: ID!) {
            findImage(id: $id) {
              performers { id }
            }
          }
        `;

        const updateQuery = `
          mutation UpdateImage($id: ID!, $performer_ids: [ID!]) {
            imageUpdate(input: { id: $id, performer_ids: $performer_ids }) {
              id
            }
          }
        `;

        try {
          const getResult = await SS.stashQuery(getQuery, { id: imageId });
          const currentPerformers = getResult?.findImage?.performers || [];
          const currentIds = currentPerformers.map(p => p.id);

          if (!currentIds.includes(performerId)) {
            currentIds.push(performerId);
          }

          await SS.stashQuery(updateQuery, { id: imageId, performer_ids: currentIds });
          return true;
        } catch (e) {
          console.error('Failed to add performer to image:', e);
          return false;
        }
      },

      // Create the results modal
      createModal() {
        const existing = document.getElementById('ss-modal');
        if (existing) existing.remove();

        // Use inline styles to prevent Stash's Bootstrap CSS from
        // turning this into a drawer/sheet layout
        const modal = document.createElement('div');
        modal.id = 'ss-modal';
        modal.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;z-index:10000;display:flex;align-items:center;justify-content:center;';

        const backdrop = document.createElement('div');
        backdrop.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.7);';
        modal.appendChild(backdrop);

        const content = document.createElement('div');
        content.className = 'ss-modal-content';
        content.style.cssText = 'position:relative;background:var(--bs-body-bg, #1a1a1a);border-radius:8px;width:90%;max-width:700px;max-height:85vh;overflow:hidden;display:flex;flex-direction:column;box-shadow:0 8px 32px rgba(0,0,0,0.5);';

        const header = document.createElement('div');
        header.style.cssText = 'display:flex;align-items:center;justify-content:space-between;padding:16px 20px;border-bottom:1px solid var(--bs-border-color, #333);';
        const title = document.createElement('h3');
        title.style.cssText = 'margin:0;font-size:18px;font-weight:600;color:var(--bs-body-color, #fff);';
        title.textContent = `${SS.PLUGIN_NAME} Results`;
        const closeBtn = document.createElement('button');
        closeBtn.className = 'ss-modal-close';
        closeBtn.style.cssText = 'background:none;border:none;font-size:24px;color:var(--bs-secondary-color, #888);cursor:pointer;padding:0;line-height:1;';
        closeBtn.setAttribute('aria-label', 'Close');
        closeBtn.innerHTML = '&times;';
        header.appendChild(title);
        header.appendChild(closeBtn);
        content.appendChild(header);

        const body = document.createElement('div');
        body.className = 'ss-modal-body';
        body.style.cssText = 'padding:20px;overflow-y:auto;flex:1;';
        body.innerHTML = `
          <div class="ss-loading">
            <div class="ss-spinner"></div>
            <p class="ss-loading-text">Connecting to ${SS.PLUGIN_NAME}...</p>
            <p class="ss-loading-detail"></p>
          </div>
          <div class="ss-results" style="display: none;"></div>
          <div class="ss-error" style="display: none;"></div>
        `;
        content.appendChild(body);
        modal.appendChild(content);

        document.body.appendChild(modal);

        const escHandler = (e) => {
          if (e.key === 'Escape') closeModal();
        };
        const closeModal = () => {
          document.removeEventListener('keydown', escHandler);
          modal.remove();
        };
        modal._close = closeModal;
        closeBtn.addEventListener('click', closeModal);
        backdrop.addEventListener('click', closeModal);
        document.addEventListener('keydown', escHandler);

        return modal;
      },

      // Find the current entity edit form's Save button, if one is open.
      // Stash shares this exact class/label across scene, image, and
      // gallery edit tabs (also used by e.g. the Delete button, hence the
      // text match). Its disabled state IS the form's dirty flag --
      // Stash enables Save the moment there's an unsaved change and
      // disables it again once saved/reverted. No *visible* Save button
      // means there's no open edit form to worry about.
      //
      // offsetParent !== null is required, not optional: Stash's tabs
      // (Details/Edit/Markers/...) use react-bootstrap Tab.Pane, which
      // stays mounted in the DOM after switching away from it -- only its
      // `active`/`show` classes toggle (display: none otherwise).
      // Confirmed live: after ever visiting a scene's Edit tab, its Save
      // button remains findable (and its own performer_ids field
      // selectable) via querySelector from any other tab of that same
      // page for the rest of the page's lifetime, unless this check is
      // here -- which silently staged an add into that hidden, inactive
      // form instead of mutating directly while the user was looking at
      // the Details tab, with zero visible feedback.
      _findSaveButton() {
        return Array.from(document.querySelectorAll('.edit-button'))
          .find((b) => /^save$/i.test((b.textContent || '').trim()) && b.offsetParent !== null) || null;
      },

      // Close the results modal after an add-performer action completes.
      //
      // Whether this reloads the page depends on how the performer was
      // actually added, decided by the caller and passed in as `staged`:
      //
      // - staged: false -- no edit tab was open, so the caller mutated the
      //   scene/image/gallery directly via GraphQL. That's already saved
      //   server-side and there's no open form with unsaved edits to lose,
      //   so reloading is both safe and necessary -- it's the only way the
      //   Details-tab view (cover image, performer list) picks up the
      //   change.
      // - staged: true -- an edit tab was open, so the caller staged the
      //   performer into the form's own PerformerSelect field instead of
      //   mutating directly (see _selectPerformerInPendingForm). Reloading
      //   here would discard that pending edit -- and any other unsaved
      //   edits on the form -- so this just closes the modal and leaves
      //   the user to review/Save (or remove it again) themselves.
      //
      // This used to instead mutate directly in both cases and try to
      // reconcile the open form afterwards (first by typing the
      // performer's name into react-select and pressing Enter -- unsafe,
      // since the mutation already excluded them from their own
      // suggestion list, so Enter could confirm an unrelated same-named
      // performer instead; then by showing a "refresh manually" banner).
      // Both were symptoms of the same root problem: mutating while a form
      // is open racing against that form's own eventual Save, which
      // resubmits its whole performer_ids list and silently reverts
      // anything added out-of-band. Staging into the form itself removes
      // the race entirely.
      async _finishMutation(modal, { staged = false } = {}) {
        if (staged) {
          if (modal && typeof modal._close === 'function') modal._close();
          return;
        }
        window.location.reload();
      },

      _setNativeInputValue(input, value) {
        const proto = Object.getPrototypeOf(input);
        const desc = Object.getOwnPropertyDescriptor(proto, 'value')
          || Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');
        if (desc?.set) {
          desc.set.call(input, value);
        } else {
          input.value = value;
        }
        input.dispatchEvent(new Event('input', { bubbles: true }));
      },

      // Stage a performer into the currently-open Scene/Image/Gallery edit
      // form's own Performers field, instead of mutating the entity
      // directly. Used whenever an edit tab is open (see _findSaveButton)
      // -- see _finishMutation's comment for why a direct mutation is
      // unsafe in that case.
      //
      // Stash's Scene/Image/Gallery edit panels all render this field via
      // renderField("performer_ids", ...), which wraps it in
      // `<Form.Group data-field="performer_ids">` -- a reliable,
      // version-stable anchor (confirmed against Stash's own source), so
      // this doesn't need the old heuristic multi-field scoring approach.
      //
      // Query preference: the performer's StashBox UUID if we have one
      // (Stash's PerformerSelect runs an exact server-side stash_id match
      // for UUID-shaped input, so this can't land on a same-named/aliased
      // performer), falling back to their name otherwise. Either way, the
      // option actually clicked is verified against the performer's real
      // local Stash ID first -- parsed from the option's own
      // `/performers/{id}` link -- and never just trusted from typing a
      // name and hitting Enter. That's what protects against alias /
      // disambiguation collisions (confirmed live: this happened with a
      // single-name performer, the exact failure mode this guards
      // against).
      //
      // Returns false (and leaves the field's typed text cleared) if no
      // query produces a verified match -- callers must NOT fall back to
      // a direct mutation in that case, since it could still be reverted
      // by the form's own next Save.
      async _selectPerformerInPendingForm(performerId, { name, stashdbId } = {}) {
        const container = document.querySelector('[data-field="performer_ids"]');
        if (!container) return false;
        const input = container.querySelector('input');
        if (!input) return false;

        const queries = [stashdbId, name].filter(Boolean);
        for (const query of queries) {
          if (await this._tryVerifiedSelect(container, input, query, performerId)) {
            return true;
          }
        }
        this._setNativeInputValue(input, '');
        return false;
      },

      async _tryVerifiedSelect(container, input, query, performerId) {
        input.focus();
        this._setNativeInputValue(input, query);

        // Options load async (debounced, plus a GraphQL round trip) --
        // poll for them rather than a fixed sleep.
        const deadline = Date.now() + 4000;
        let match = null;
        while (Date.now() < deadline && !match) {
          await new Promise((r) => setTimeout(r, 120));
          const options = container.querySelectorAll('.react-select__option');
          for (const opt of options) {
            const link = opt.querySelector('a[href^="/performers/"]');
            const idMatch = link?.getAttribute('href')?.match(/\/performers\/(\d+)/);
            if (idMatch && idMatch[1] === String(performerId)) {
              match = opt;
              break;
            }
          }
        }

        if (!match) {
          this._setNativeInputValue(input, '');
          return false;
        }

        const opts = { bubbles: true, cancelable: true, button: 0 };
        match.dispatchEvent(new MouseEvent('mousedown', opts));
        match.dispatchEvent(new MouseEvent('mouseup', opts));
        match.dispatchEvent(new MouseEvent('click', opts));
        await new Promise((r) => setTimeout(r, 150));
        return true;
      },

      _withTimeout(promise, timeoutMs, timeoutMessage) {
        return Promise.race([
          promise,
          new Promise((_, reject) => {
            setTimeout(() => reject(new Error(timeoutMessage)), timeoutMs);
          }),
        ]);
      },

      updateLoading(modal, message, detail = '') {
        const loadingText = modal.querySelector('.ss-loading-text');
        const loadingDetail = modal.querySelector('.ss-loading-detail');
        if (loadingText) loadingText.textContent = message;
        if (loadingDetail) loadingDetail.textContent = detail;
      },

      async renderResults(modal, results, sceneId, options = {}) {
        const loading = modal.querySelector('.ss-loading');
        const resultsDiv = modal.querySelector('.ss-results');
        const errorDiv = modal.querySelector('.ss-error');

        loading.style.display = 'none';

        // options.onReidentify, if given, wires up a "Re-identify" action --
        // shown whenever results came from storage (options.fromStorage) or
        // after any fresh identify, so it's always available to get an
        // up-to-date answer (e.g. the performer database updated, or a
        // performer was just tagged on this scene since the stored result
        // was computed).
        const reidentifyBtnHtml = options.onReidentify
          ? '<button class="ss-btn ss-btn-secondary ss-reidentify-btn">Re-identify</button>'
          : '';
        const wireReidentifyBtn = (container) => {
          const btn = container.querySelector('.ss-reidentify-btn');
          if (btn) btn.addEventListener('click', () => options.onReidentify());
        };

        if (!results.persons || results.persons.length === 0) {
          errorDiv.innerHTML = `
            <div class="ss-error-icon">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="48" height="48" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z"/>
              </svg>
            </div>
            <p class="ss-error-title">No faces detected</p>
            <p class="ss-error-hint">
              This could mean the scene doesn't have clear face shots, or the sprite sheet quality is too low.
            </p>
            ${reidentifyBtnHtml}
          `;
          errorDiv.style.display = 'block';
          wireReidentifyBtn(errorDiv);
          return;
        }

        // Build set of StashDB IDs already tagged on this scene. Reuse
        // results._scenePerformers when a fresh identifyScene() call
        // already fetched it (avoids a duplicate query); a stored result
        // (fingerprint_get_scene_result, the common "instant" path -- see
        // handleIdentifyFullVideo) never has that attached at all, so
        // without this live fallback every performer looked already
        // untagged regardless of the scene's actual current state --
        // confirmed live: "Add to Scene" showing for a performer already
        // on the scene. Fetching fresh here also self-heals staleness in
        // the *stored* already_tagged flag itself (computed once, at
        // identify time, never refreshed just from viewing the result).
        const taggedStashDBIds = new Set();
        const scenePerformers = results._scenePerformers || await this.getScenePerformerStashDBIds(sceneId);
        const scenePerformerLocalIds = new Set();
        for (const p of scenePerformers) {
          scenePerformerLocalIds.add(p.id);
          for (const sid of (p.stash_ids || [])) {
            if (sid.endpoint === 'https://stashdb.org/graphql') {
              taggedStashDBIds.add(sid.stash_id);
            }
          }
        }

        // Separate persons with multi-frame clusters from singletons
        const multiFrame = results.persons.filter(p => p.frame_count > 1 && p.best_match);
        const singleFrame = results.persons.filter(p => p.frame_count <= 1 && p.best_match);
        const unknown = results.persons.filter(p => !p.best_match);

        const clusterCount = multiFrame.length;
        const totalPersons = results.persons.filter(p => p.best_match).length;

        resultsDiv.innerHTML = `
          <p class="ss-summary">
            Analyzed <strong>${results.frames_analyzed}</strong> frames,
            detected <strong>${results.faces_detected}</strong> faces,
            found <strong>${clusterCount}</strong> distinct person(s)${singleFrame.length ? ` + ${singleFrame.length} single-frame detection(s)` : ''}.
            ${options.fromStorage ? '<span class="ss-stored-badge" title="Shown from this scene\'s last identification, not re-run just now">from last identification</span>' : ''}
          </p>
          ${reidentifyBtnHtml}
          <div class="ss-persons"></div>
          ${singleFrame.length ? '<div class="ss-singletons"></div>' : ''}
        `;
        wireReidentifyBtn(resultsDiv);

        const personsDiv = resultsDiv.querySelector('.ss-persons');

        // Render multi-frame persons (high confidence clusters)
        for (const person of multiFrame) {
          try {
            const personDiv = await this._renderPerson(person, sceneId, taggedStashDBIds, scenePerformerLocalIds);
            personsDiv.appendChild(personDiv);
          } catch (renderErr) {
            console.error('[Stash Sense] Failed to render person:', renderErr);
            const fallback = document.createElement('div');
            fallback.className = 'ss-person ss-render-error';
            fallback.textContent = 'Failed to load person data';
            personsDiv.appendChild(fallback);
          }
        }

        // Render single-frame detections collapsed by default
        if (singleFrame.length) {
          const singletonsDiv = resultsDiv.querySelector('.ss-singletons');
          const details = document.createElement('details');
          details.className = 'ss-singleton-section';
          details.innerHTML = `
            <summary class="ss-singleton-header">Single-frame detections (${singleFrame.length})</summary>
          `;
          const innerDiv = document.createElement('div');
          innerDiv.className = 'ss-singleton-list';
          for (const person of singleFrame) {
            try {
              const personDiv = await this._renderPerson(person, sceneId, taggedStashDBIds, scenePerformerLocalIds);
              innerDiv.appendChild(personDiv);
            } catch (renderErr) {
              console.error('[Stash Sense] Failed to render person:', renderErr);
              const fallback = document.createElement('div');
              fallback.className = 'ss-person ss-render-error';
              fallback.textContent = 'Failed to load person data';
              innerDiv.appendChild(fallback);
            }
          }
          details.appendChild(innerDiv);
          singletonsDiv.appendChild(details);
        }

        // Add click handlers for "Add to Scene" buttons
        resultsDiv.querySelectorAll('.ss-btn-add').forEach(btn => {
          btn.addEventListener('click', async (e) => {
            const performerId = btn.dataset.performerId;
            const performerName = btn.dataset.performerName;
            const stashdbId = btn.dataset.stashdbId;
            const targetSceneId = btn.dataset.sceneId;
            btn.disabled = true;
            btn.textContent = 'Adding...';

            // An open edit tab's own Save resubmits its whole
            // performer_ids list -- mutating the scene directly here
            // would just get silently reverted by that. Stage into the
            // form instead when one is open; only mutate directly
            // otherwise. See _finishMutation's comment for the full
            // reasoning.
            const staged = !!this._findSaveButton();
            const success = staged
              ? await this._selectPerformerInPendingForm(performerId, { name: performerName, stashdbId })
              : await this.addPerformerToScene(targetSceneId, performerId);

            if (success) {
              btn.textContent = staged ? 'Added to form' : 'Added!';
              btn.classList.add('ss-btn-success');
              await this._finishMutation(modal, { staged });
            } else {
              btn.textContent = staged ? 'Could not add automatically' : 'Failed';
              btn.classList.add('ss-btn-error');
              btn.disabled = false;
            }
          });
        });

        // "Add to Stash + Scene" handlers
        resultsDiv.querySelectorAll('.ss-btn-create').forEach(btn => {
          btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const { endpoint, stashdbId, sceneId: targetSceneId, source, name, country, imageUrl, catalogueUrl, profileUrl } = btn.dataset;
            btn.disabled = true;
            btn.textContent = 'Creating...';

            const staged = !!this._findSaveButton();

            try {
              const settings = await SS.getSettings();
              // Catalogue (non-stash-box) matches have no external API to
              // re-fetch full details from -- everything the backend needs
              // is already on the button's data attributes (see
              // _catalogueDataAttrs), carried straight from the match.
              const result = await this._withTimeout(
                source
                  ? SS.runPluginOperation('create_performer_from_catalogue', {
                      source, name, country: country || undefined,
                      image_url: imageUrl || undefined, catalogue_url: catalogueUrl || undefined,
                      profile_url: profileUrl || undefined,
                      ...(staged ? {} : { scene_id: targetSceneId }),
                      sidecar_url: settings.sidecarUrl,
                    })
                  : SS.runPluginOperation('create_performer_from_stashbox', {
                      endpoint,
                      stashdb_id: stashdbId,
                      // Omitted (not just empty) when staged -- the backend
                      // only skips its own scene-assignment step when this key
                      // is entirely absent/falsy.
                      ...(staged ? {} : { scene_id: targetSceneId }),
                      sidecar_url: settings.sidecarUrl,
                    }),
                45000,
                'Create performer operation timed out',
              );

              if (result.error) throw new Error(result.error);

              const success = staged
                ? await this._selectPerformerInPendingForm(result.performer_id, { name: result.name, stashdbId })
                : true;

              if (success) {
                btn.textContent = staged ? 'Created — added to form' : 'Added!';
                btn.classList.add('ss-btn-success');
                await this._finishMutation(modal, { staged });
              } else {
                // Performer was created in the library either way -- just
                // couldn't be verified-selected into the open form.
                btn.textContent = 'Created — add manually';
                btn.classList.add('ss-btn-error');
              }
            } catch (err) {
              // Fallback: if the plugin call timed out but performer creation
              // actually succeeded, complete UI flow anyway. Catalogue
              // creations have no stashbox id to search back by, so skip
              // this recovery for them (less likely to time out anyway --
              // no external API round-trip involved).
              if (!source && (err?.message || '').toLowerCase().includes('timed out')) {
                try {
                  const graphqlUrl = this._stashboxGraphqlUrl(endpoint);
                  const localPerformer = await SS.findPerformerByStashDBId(stashdbId, graphqlUrl);
                  if (localPerformer?.id) {
                    const success = staged
                      ? await this._selectPerformerInPendingForm(localPerformer.id, { name: localPerformer.name, stashdbId })
                      : await this.addPerformerToScene(targetSceneId, localPerformer.id);
                    if (success) {
                      btn.textContent = staged ? 'Created — added to form' : 'Added!';
                      btn.classList.add('ss-btn-success');
                      await this._finishMutation(modal, { staged });
                      return;
                    }
                    btn.textContent = 'Created — add manually';
                    btn.classList.add('ss-btn-error');
                    return;
                  }
                } catch (recoveryErr) {
                  console.warn('[Stash Sense] Timed out, recovery check failed:', recoveryErr);
                }
              }
              btn.textContent = 'Failed';
              btn.classList.add('ss-btn-error');
              btn.disabled = false;
              console.error('Failed to create performer:', err);
            }
          });
        });

        // "Add as..." handlers
        resultsDiv.querySelectorAll('.ss-btn-link-as').forEach(btn => {
          btn.addEventListener('click', (e) => {
            e.stopPropagation();
            this._openSearchPanel(btn);
          });
        });

        resultsDiv.style.display = 'block';
      },

      // Build stashbox performer URL from endpoint domain
      _stashboxPerformerUrl(endpoint, stashdbId) {
        const domain = endpoint || 'stashdb.org';
        return `https://${domain}/performers/${stashdbId}`;
      },

      // Get GraphQL endpoint URL from domain
      _stashboxGraphqlUrl(endpoint) {
        const domain = endpoint || 'stashdb.org';
        return `https://${domain}/graphql`;
      },

      // Local performers have no StashBox page to link to -- link to their
      // page on this Stash instance instead, so a match can still be
      // opened and verified.
      _localPerformerUrl(localPerformerId) {
        return `/performers/${localPerformerId}`;
      },

      // Data attributes carried on the "Add to Stash + ..." / "Add as..."
      // buttons for a catalogue (non-stash-box) match -- see
      // stashbox_router.py's create_performer_from_catalogue, which needs
      // this data directly since there's no external API it can re-fetch
      // full performer details from the way the stashbox flow does.
      // Returns '' for a stashbox/local match (nothing extra to carry).
      _catalogueDataAttrs(match) {
        if (!match.source) return '';
        const esc = (s) => SS.escapeHtml ? SS.escapeHtml(s || '') : (s || '');
        return `
                    data-source="${esc(match.source)}"
                    data-name="${esc(match.name)}"
                    data-country="${esc(match.country)}"
                    data-image-url="${esc(match.image_url)}"
                    data-catalogue-url="${esc(match.catalogue_url)}"
                    data-profile-url="${esc(match.profile_url)}"`;
      },

      // Build the "View on ..." links for a match. Catalogue (non-stash-box)
      // matches link to the actual external content site when the source
      // has one (e.g. onlyfans.com for seekfans), falling back to the
      // catalogue site's own profile page -- there's no stashbox page to
      // link to at all for these. Local matches always get a "View local
      // performer" link; if the local performer is *also* linked to a real
      // StashDB id (match.stashdb_id differs from match.local_performer_id
      // -- the sidecar falls back to the local id string when there's no
      // real link), show both, since that's two independent,
      // separately-verifiable signals for the same person. Non-local,
      // non-catalogue matches keep the existing single StashBox link.
      _matchLinksHtml(match, stashboxUrl, endpoint) {
        if (match.source) {
          const href = match.profile_url || match.catalogue_url;
          if (!href) return '';
          // Same "View on <domain>" style as the StashBox link below --
          // a real external profile_url (e.g. onlyfans.com for seekfans
          // matches) gets labeled by its own hostname rather than the
          // generic "View profile", so it reads the same way "View on
          // stashdb.org" does. Falls back to the source name for a
          // catalogue_url-only match (no external profile, just the
          // catalogue site's own page) or if the URL fails to parse.
          let label = `View on ${match.source}`;
          if (match.profile_url) {
            try {
              label = `View on ${new URL(match.profile_url).hostname.replace(/^www\./, '')}`;
            } catch (e) {
              label = 'View profile';
            }
          }
          return `<a href="${href}" target="_blank" rel="noopener" class="ss-link">${label}</a>`;
        }
        if (!match.local_performer_id) {
          return `<a href="${stashboxUrl}" target="_blank" rel="noopener" class="ss-link">View on ${endpoint}</a>`;
        }
        const localLink = `<a href="${this._localPerformerUrl(match.local_performer_id)}" target="_blank" rel="noopener" class="ss-link ss-link-local">View local performer</a>`;
        const hasStashDbLink = match.stashdb_id && match.stashdb_id !== match.local_performer_id;
        if (!hasStashDbLink) {
          return localLink;
        }
        const stashDbUrl = this._stashboxPerformerUrl('stashdb.org', match.stashdb_id);
        return `<a href="${stashDbUrl}" target="_blank" rel="noopener" class="ss-link">View on stashdb.org</a> ${localLink}`;
      },

      // Resolve a match to its local Stash performer, if any is already in
      // the library. Local-index matches (match.local_performer_id set)
      // already know their local id directly -- no GraphQL round-trip
      // needed, unlike a stashdb.org match which has to be cross-referenced
      // by its linked stash_id.
      async _resolveLibraryPerformer(match, graphqlUrl) {
        if (match.local_performer_id) {
          return { id: match.local_performer_id, name: match.name };
        }
        if (match.source) {
          // Catalogue match: match.stashdb_id is the internal database id,
          // not a real stash_ids-linkable uuid, and there's no stashbox
          // GraphQL endpoint to query against -- no way to cross-reference
          // against the library this way for these yet.
          return null;
        }
        return SS.findPerformerByStashDBId(match.stashdb_id, graphqlUrl);
      },

      async _renderPerson(person, sceneId, taggedStashDBIds, scenePerformerLocalIds) {
        const personDiv = document.createElement('div');
        personDiv.className = 'ss-person';

        if (!person.best_match) {
          personDiv.innerHTML = `
            <div class="ss-person-header">
              <span class="ss-person-label">Unknown Person</span>
              <span class="ss-person-frames">${person.frame_count} appearances</span>
            </div>
            <p class="ss-no-match">No match found in database</p>
          `;
          return personDiv;
        }

        const match = person.best_match;
        const confidence = this.distanceToConfidence(match.distance || (1 - match.confidence) || 0.5);
        const confidenceClass = SS.getConfidenceClass(confidence);
        const endpoint = match.endpoint || 'stashdb.org';
        const stashboxUrl = this._stashboxPerformerUrl(endpoint, match.stashdb_id);
        const graphqlUrl = this._stashboxGraphqlUrl(endpoint);

        // Check if already tagged (from API flag or local cross-reference)
        const isAlreadyTagged = match.already_tagged || taggedStashDBIds.has(match.stashdb_id);

        const localPerformer = await this._resolveLibraryPerformer(match, graphqlUrl);
        const isLocallyTagged = localPerformer && scenePerformerLocalIds.has(localPerformer.id);
        const showAlreadyTagged = isAlreadyTagged || isLocallyTagged;

        let actionsHtml;
        if (showAlreadyTagged) {
          actionsHtml = `<span class="ss-local-status ss-already-tagged">Already tagged on scene</span>`;
        } else if (localPerformer) {
          actionsHtml = `
            <button class="ss-btn ss-btn-add"
                    data-performer-id="${localPerformer.id}"
                    data-performer-name="${SS.escapeHtml ? SS.escapeHtml(localPerformer.name) : localPerformer.name}"
                    data-stashdb-id="${match.stashdb_id || ''}"
                    data-scene-id="${sceneId}">
              Add to Scene
            </button>
            <span class="ss-local-status">In library as: ${localPerformer.name}</span>`;
        } else {
          actionsHtml = `
            <button class="ss-btn ss-btn-create"
                    data-endpoint="${endpoint}"
                    data-stashdb-id="${match.stashdb_id}"
                    data-scene-id="${sceneId}"${this._catalogueDataAttrs(match)}>
              Add to Stash + Scene
            </button>
            <button class="ss-btn ss-btn-link-as"
                    data-endpoint="${endpoint}"
                    data-stashdb-id="${match.stashdb_id}"
                    data-scene-id="${sceneId}"${this._catalogueDataAttrs(match)}>
              Add as...
            </button>
            <span class="ss-local-status ss-not-in-library">Not in library</span>`;
        }

        personDiv.innerHTML = `
          <div class="ss-person-header">
            <span class="ss-person-label">Person ${person.person_id + 1}</span>
            <span class="ss-person-frames">${person.frame_count} appearances</span>
            ${showAlreadyTagged ? '<span class="ss-tagged-badge">Tagged</span>' : ''}
          </div>
          <div class="ss-match">
            <div class="ss-match-image">
              ${match.image_url ? `<img src="${match.image_url}" alt="${match.name}" loading="lazy" />` : '<div class="ss-no-image">No image</div>'}
            </div>
            <div class="ss-match-info">
              <h4>${match.name}</h4>
              <div class="ss-confidence ${confidenceClass}">${confidence}% match</div>
              ${match.country ? `<div class="ss-country">${match.country}</div>` : ''}
              <div class="ss-links">
                ${this._matchLinksHtml(match, stashboxUrl, endpoint)}
              </div>
              <div class="ss-actions">
                ${actionsHtml}
              </div>
            </div>
          </div>
        `;

        // Build alt matches section with action buttons
        if (person.all_matches && person.all_matches.length > 1) {
          const details = document.createElement('details');
          details.className = 'ss-other-matches';
          details.innerHTML = `<summary>Other possible matches (${person.all_matches.length - 1})</summary>`;

          const ul = document.createElement('ul');
          for (const m of person.all_matches.slice(1)) {
            const altConf = this.distanceToConfidence(m.distance || (1 - m.confidence) || 0.5);
            const altConfClass = SS.getConfidenceClass(altConf);
            const altEndpoint = m.endpoint || 'stashdb.org';
            const altStashboxUrl = this._stashboxPerformerUrl(altEndpoint, m.stashdb_id);
            const altGraphqlUrl = this._stashboxGraphqlUrl(altEndpoint);
            const altTagged = m.already_tagged || taggedStashDBIds.has(m.stashdb_id);

            const altLocalPerformer = await this._resolveLibraryPerformer(m, altGraphqlUrl);
            const altIsLocallyTagged = altLocalPerformer && scenePerformerLocalIds.has(altLocalPerformer.id);
            const altShowAlreadyTagged = altTagged || altIsLocallyTagged;

            let altActionsHtml;
            if (altShowAlreadyTagged) {
              altActionsHtml = `<span class="ss-local-status ss-already-tagged">Already tagged</span>`;
            } else if (altLocalPerformer) {
              altActionsHtml = `
                <button class="ss-btn ss-btn-add ss-btn-sm"
                        data-performer-id="${altLocalPerformer.id}"
                        data-performer-name="${SS.escapeHtml ? SS.escapeHtml(altLocalPerformer.name) : altLocalPerformer.name}"
                        data-stashdb-id="${m.stashdb_id || ''}"
                        data-scene-id="${sceneId}">
                  Add to Scene
                </button>
                <span class="ss-local-status">In library as: ${altLocalPerformer.name}</span>`;
            } else {
              altActionsHtml = `
                <button class="ss-btn ss-btn-create ss-btn-sm"
                        data-endpoint="${altEndpoint}"
                        data-stashdb-id="${m.stashdb_id}"
                        data-scene-id="${sceneId}"${this._catalogueDataAttrs(m)}>
                  Add to Stash + Scene
                </button>
                <button class="ss-btn ss-btn-link-as ss-btn-sm"
                        data-endpoint="${altEndpoint}"
                        data-stashdb-id="${m.stashdb_id}"
                        data-scene-id="${sceneId}"${this._catalogueDataAttrs(m)}>
                  Add as...
                </button>
                <span class="ss-local-status ss-not-in-library">Not in library</span>`;
            }

            const li = document.createElement('li');
            li.className = 'ss-alt-match-item';
            li.innerHTML = `
              <div class="ss-match">
                <div class="ss-match-image">
                  ${m.image_url ? `<img src="${m.image_url}" alt="${m.name}" loading="lazy" />` : '<div class="ss-no-image">No image</div>'}
                </div>
                <div class="ss-match-info">
                  <h4>${m.name}</h4>
                  <div class="ss-confidence ${altConfClass}">${altConf}% match</div>
                  ${altShowAlreadyTagged ? '<span class="ss-tagged-badge ss-tagged-badge-sm">Tagged</span>' : ''}
                  ${m.country ? `<div class="ss-country">${m.country}</div>` : ''}
                  <div class="ss-links">
                    ${this._matchLinksHtml(m, altStashboxUrl, altEndpoint)}
                  </div>
                  <div class="ss-actions ss-alt-match-actions">
                    ${altActionsHtml}
                  </div>
                </div>
              </div>
            `;
            ul.appendChild(li);
          }
          details.appendChild(ul);
          personDiv.appendChild(details);
        }

        return personDiv;
      },

      _cleanupSearchPanel(panel) {
        if (panel._cleanup) panel._cleanup();
        panel.remove();
      },

      _openSearchPanel(triggerBtn) {
        // Close any existing panel
        const existing = document.querySelector('.ss-search-panel');
        if (existing) {
          const wasSameTrigger = existing._triggerBtn === triggerBtn;
          this._cleanupSearchPanel(existing);
          if (wasSameTrigger) return; // Toggle off
        }

        const panel = document.createElement('div');
        panel.className = 'ss-search-panel';
        panel._triggerBtn = triggerBtn;
        const endpoint = triggerBtn.dataset.endpoint;
        const stashdbId = triggerBtn.dataset.stashdbId;
        const sceneId = triggerBtn.dataset.sceneId;
        const imageId = triggerBtn.dataset.imageId;
        const graphqlUrl = this._stashboxGraphqlUrl(endpoint);
        // Catalogue (non-stash-box) matches have no real stashbox id to
        // link -- triggerBtn.dataset.stashdbId is just the internal
        // database id for these, so offering to "link" it would write
        // garbage. Omit the checkbox entirely rather than show a control
        // that does the wrong thing.
        const isCatalogue = !!triggerBtn.dataset.source;

        panel.innerHTML = `
          <input type="text" class="ss-search-input" placeholder="Search performers in library..." />
          ${isCatalogue ? '' : `
          <label class="ss-update-meta-label">
            <input type="checkbox" class="ss-update-meta-checkbox" checked />
            Link StashBox ID to performer
          </label>`}
          <div class="ss-search-results"></div>
        `;

        // Insert after the parent actions div
        const actionsDiv = triggerBtn.closest('.ss-actions') || triggerBtn.closest('.ss-alt-match-actions');
        if (actionsDiv && actionsDiv.parentElement) {
          actionsDiv.parentElement.insertBefore(panel, actionsDiv.nextSibling);
        } else {
          triggerBtn.parentElement.appendChild(panel);
        }

        const input = panel.querySelector('.ss-search-input');
        const resultsDiv2 = panel.querySelector('.ss-search-results');
        const updateMetaCheckbox = panel.querySelector('.ss-update-meta-checkbox');
        const self = this;

        // Real cover photos and a verifiable "View on stashdb.org" link
        // per candidate, instead of a same-looking text row per result --
        // a same-name/alias collision is exactly what silently linked the
        // wrong performer once already (see the sidecar's
        // PerformerIdentityAmbiguous). See stash-sense-core.js's
        // renderPerformerCandidateCards.
        async function linkTo(performerId, performerName) {
          const updateMeta = updateMetaCheckbox ? updateMetaCheckbox.checked : false;
          const staged = !!self._findSaveButton();

          resultsDiv2.innerHTML = '<div class="ss-search-loading">Linking...</div>';

          try {
            const stashIds = updateMeta ? [{ endpoint: graphqlUrl, stash_id: stashdbId }] : [];
            const settings = await SS.getSettings();
            const linkResult = await SS.runPluginOperation('link_performer_stashbox', {
              // Omitted (not just empty) when staged -- see the
              // "Add to Stash + Scene" handler above for why.
              ...(staged ? {} : (sceneId ? { scene_id: sceneId } : { image_id: imageId })),
              performer_id: performerId,
              stash_ids: stashIds,
              update_metadata: updateMeta,
              sidecar_url: settings.sidecarUrl,
            });

            if (linkResult.error) throw new Error(linkResult.error);

            // A UUID search only has something to find once the
            // stash_id link above has actually been written.
            const success = staged
              ? await self._selectPerformerInPendingForm(performerId, {
                  name: performerName,
                  stashdbId: updateMeta ? stashdbId : undefined,
                })
              : true;

            if (panel._cleanup) panel._cleanup();
            panel.remove();
            triggerBtn.style.display = 'none';
            // Hide the create button next to it
            const createBtn = triggerBtn.closest('.ss-actions, .ss-alt-match-actions')?.querySelector('.ss-btn-create');
            if (createBtn) createBtn.style.display = 'none';
            // Update status text
            const notInLib = triggerBtn.closest('.ss-actions, .ss-alt-match-actions')?.querySelector('.ss-not-in-library');
            if (notInLib) {
              notInLib.textContent = success
                ? `Added as: ${performerName}`
                : `Linked as: ${performerName} (add to form manually)`;
              notInLib.classList.remove('ss-not-in-library');
            }
            // Note: staged here reflects whether an edit tab was
            // open (so a reload would risk losing unsaved edits),
            // not whether the verified-select itself succeeded --
            // a failed select still leaves an open form's other
            // pending edits in place, which a reload would lose.
            const modal = triggerBtn.closest('#ss-modal');
            await self._finishMutation(modal, { staged });
          } catch (err) {
            resultsDiv2.innerHTML = `<div class="ss-search-error">Failed: ${SS.escapeHtml(err.message)}</div>`;
            console.error('Failed to link performer:', err);
          }
        }

        let debounceTimer;

        input.addEventListener('input', () => {
          clearTimeout(debounceTimer);
          debounceTimer = setTimeout(async () => {
            const query = input.value.trim();
            if (query.length < 2) {
              resultsDiv2.innerHTML = '';
              return;
            }
            resultsDiv2.innerHTML = '<div class="ss-search-loading">Searching...</div>';

            try {
              const settings = await SS.getSettings();
              const result = await SS.runPluginOperation('search_performers', {
                query,
                sidecar_url: settings.sidecarUrl,
              });

              if (result.error) throw new Error(result.error);

              const performers = result.performers || result || [];
              if (performers.length === 0) {
                resultsDiv2.innerHTML = '<div class="ss-search-empty">No performers found</div>';
                return;
              }

              const byId = new Map(performers.map(p => [String(p.id), p]));
              SS.renderPerformerCandidateCards(resultsDiv2, performers, {
                onSelectExisting: (performerId) => {
                  const p = byId.get(String(performerId));
                  linkTo(performerId, p ? p.name : performerId);
                },
              });
            } catch (err) {
              resultsDiv2.innerHTML = `<div class="ss-search-error">Search failed: ${SS.escapeHtml(err.message)}</div>`;
            }
          }, 300);
        });

        input.focus();

        // Close on Escape
        const escHandler = (e) => {
          if (e.key === 'Escape') {
            this._cleanupSearchPanel(panel);
          }
        };
        document.addEventListener('keydown', escHandler);

        // Store cleanup function for use when panel is removed externally
        panel._cleanup = () => {
          clearTimeout(debounceTimer);
          document.removeEventListener('keydown', escHandler);
        };
      },

      showError(modal, message) {
        const loading = modal.querySelector('.ss-loading');
        const errorDiv = modal.querySelector('.ss-error');

        loading.style.display = 'none';

        let title = 'Analysis Failed';
        let hint = `Check plugin settings and ensure ${SS.PLUGIN_NAME} is running.`;

        if (message.includes('Connection') || message.includes('connect')) {
          title = 'Connection Failed';
          hint = `Could not connect to ${SS.PLUGIN_NAME}. Make sure the sidecar container is running.`;
        } else if (message.includes('timeout') || message.includes('Timeout')) {
          title = 'Request Timed Out';
          hint = 'Scene analysis took too long.';
        }

        errorDiv.innerHTML = `
          <div class="ss-error-icon">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="48" height="48" fill="currentColor">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
            </svg>
          </div>
          <p class="ss-error-title">${SS.escapeHtml(title)}</p>
          <p class="ss-error-message">${SS.escapeHtml(message)}</p>
          <p class="ss-error-hint">${SS.escapeHtml(hint)}</p>
        `;
        errorDiv.style.display = 'block';
      },

      // Locate the scene page's actual <video> element. No existing plugin
      // code touches Stash's real player (only unrelated hover-preview
      // clips elsewhere), so this tries a few likely selectors in order --
      // the bare 'video' fallback is safe on a scene page since the main
      // player is the only <video> present there.
      getVideoElement() {
        const selectors = ['.video-js video', '#VideoJsPlayer video', '.scene-player video', 'video'];
        for (const sel of selectors) {
          const el = document.querySelector(sel);
          if (el) return el;
        }
        return null;
      },

      // Draw a frame (or just cropRect, in native source-pixel coordinates)
      // from a <video> or <img> source to an offscreen canvas and return
      // base64 JPEG (no data: URL prefix) for the /identify image_base64
      // field. sourceWidth/sourceHeight are the source's own native pixel
      // dimensions (video.videoWidth/Height, or an image's natural size) --
      // passed explicitly since an <img> source doesn't have videoWidth.
      captureVideoFrameBase64(source, sourceWidth, sourceHeight, cropRect) {
        const canvas = document.createElement('canvas');
        let sx = 0, sy = 0, sw = sourceWidth, sh = sourceHeight;
        if (cropRect) {
          sx = Math.max(0, Math.round(cropRect.x));
          sy = Math.max(0, Math.round(cropRect.y));
          sw = Math.max(1, Math.round(cropRect.width));
          sh = Math.max(1, Math.round(cropRect.height));
        }
        canvas.width = sw;
        canvas.height = sh;
        canvas.getContext('2d').drawImage(source, sx, sy, sw, sh, 0, 0, sw, sh);
        return canvas.toDataURL('image/jpeg', 0.92).split(',')[1];
      },

      // Resolve something drawable for the scene player: the <video>
      // element itself once it has decoded a frame (videoWidth > 0), or --
      // before playback has started, when only the poster/cover image is
      // showing -- an <img> loaded from that same poster URL instead.
      // Returns { source, width, height, isVideo } or null if neither is
      // available (e.g. no video element, or no poster set either).
      async getFrameSource(video) {
        if (video.videoWidth > 0) {
          return { source: video, width: video.videoWidth, height: video.videoHeight, isVideo: true };
        }
        if (!video.poster) return null;
        const img = new Image();
        img.src = video.poster;
        await new Promise((resolve, reject) => {
          if (img.complete && img.naturalWidth > 0) return resolve();
          img.onload = () => resolve();
          img.onerror = () => reject(new Error('Failed to load cover image'));
        }).catch(() => null);
        if (!img.naturalWidth) return null;
        return { source: img, width: img.naturalWidth, height: img.naturalHeight, isVideo: false };
      },

      async handleIdentifyCurrentFrame() {
        const route = SS.getRoute();
        if (route.type !== 'scene') return;
        const sceneId = route.id;

        const video = this.getVideoElement();
        const frameSource = video ? await this.getFrameSource(video) : null;
        if (!frameSource) {
          alert('Could not find a video frame or cover image to identify.');
          return;
        }

        const imageBase64 = this.captureVideoFrameBase64(frameSource.source, frameSource.width, frameSource.height);
        const modal = this.createModal();
        try {
          this.updateLoading(modal, 'Analyzing current frame...', 'Detecting faces');
          const results = await this.identifyFrame(imageBase64, (stage) => {
            this.updateLoading(modal, stage);
          });
          this.updateLoading(modal, 'Processing results...');
          await this.renderFrameResults(modal, results, sceneId);
        } catch (error) {
          console.error(`[${SS.PLUGIN_NAME}] Frame analysis failed:`, error);
          this.showError(modal, error.message);
        }
      },

      async handleSelectToIdentify() {
        const route = SS.getRoute();
        if (route.type !== 'scene') return;
        const sceneId = route.id;

        const video = this.getVideoElement();
        const frameSource = video ? await this.getFrameSource(video) : null;
        if (!frameSource) {
          alert('Could not find a video frame or cover image to identify.');
          return;
        }

        const wasPaused = video.paused;
        if (frameSource.isVideo) video.pause();

        const rect = video.getBoundingClientRect();
        const overlay = SS.createElement('div', {
          className: 'ss-select-overlay',
          attrs: { style: `position:fixed;top:${rect.top}px;left:${rect.left}px;width:${rect.width}px;height:${rect.height}px;z-index:9999;cursor:crosshair;background:rgba(0,0,0,0.15);` },
        });
        const box = SS.createElement('div', { className: 'ss-select-box' });
        const toolbar = SS.createElement('div', { className: 'ss-select-toolbar' });
        const confirmBtn = SS.createElement('button', { className: 'ss-btn ss-btn-primary ss-btn-sm', textContent: 'Identify Selection' });
        const cancelBtn = SS.createElement('button', { className: 'ss-btn ss-btn-sm', textContent: 'Cancel' });
        toolbar.appendChild(confirmBtn);
        toolbar.appendChild(cancelBtn);
        overlay.appendChild(box);
        overlay.appendChild(toolbar);
        document.body.appendChild(overlay);

        let startX = 0, startY = 0, selRect = null, dragging = false;

        const cleanup = () => {
          overlay.remove();
          if (frameSource.isVideo && !wasPaused) video.play().catch(() => {});
        };

        // The toolbar is a child of overlay (positioned on top of the
        // video), so a mousedown/mousemove/mouseup on its buttons would
        // otherwise bubble up into overlay's own drag-tracking listeners
        // below -- restarting a zero-size drag and immediately re-hiding
        // the toolbar mid-click, which silently ate clicks on "Identify
        // Selection"/"Cancel" (confirmed live). Stop it at the source.
        toolbar.addEventListener('mousedown', (e) => e.stopPropagation());
        toolbar.addEventListener('mousemove', (e) => e.stopPropagation());
        toolbar.addEventListener('mouseup', (e) => e.stopPropagation());

        overlay.addEventListener('mousedown', (e) => {
          dragging = true;
          startX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
          startY = Math.max(0, Math.min(rect.height, e.clientY - rect.top));
          Object.assign(box.style, { left: `${startX}px`, top: `${startY}px`, width: '0px', height: '0px', display: 'block' });
          toolbar.style.display = 'none';
        });
        overlay.addEventListener('mousemove', (e) => {
          if (!dragging) return;
          const curX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
          const curY = Math.max(0, Math.min(rect.height, e.clientY - rect.top));
          const x = Math.min(startX, curX);
          const y = Math.min(startY, curY);
          const w = Math.abs(curX - startX);
          const h = Math.abs(curY - startY);
          Object.assign(box.style, { left: `${x}px`, top: `${y}px`, width: `${w}px`, height: `${h}px` });
          selRect = { x, y, w, h };
        });
        overlay.addEventListener('mouseup', () => {
          dragging = false;
          if (!selRect || selRect.w < 10 || selRect.h < 10) return;
          toolbar.style.left = `${Math.min(selRect.x, Math.max(0, rect.width - 180))}px`;
          toolbar.style.top = `${Math.max(0, selRect.y - 36)}px`;
          toolbar.style.display = 'flex';
        });

        cancelBtn.addEventListener('click', cleanup);

        confirmBtn.addEventListener('click', async () => {
          if (!selRect || selRect.w < 10 || selRect.h < 10) return;
          // Scale from displayed CSS pixels to native source pixel coordinates
          const scaleX = frameSource.width / rect.width;
          const scaleY = frameSource.height / rect.height;
          const cropRect = {
            x: selRect.x * scaleX,
            y: selRect.y * scaleY,
            width: selRect.w * scaleX,
            height: selRect.h * scaleY,
          };
          const imageBase64 = this.captureVideoFrameBase64(frameSource.source, frameSource.width, frameSource.height, cropRect);
          cleanup();

          const modal = this.createModal();
          try {
            this.updateLoading(modal, 'Analyzing selection...', 'Detecting faces');
            const results = await this.identifyFrame(imageBase64, (stage) => {
              this.updateLoading(modal, stage);
            });
            this.updateLoading(modal, 'Processing results...');
            await this.renderFrameResults(modal, results, sceneId);
          } catch (error) {
            console.error(`[${SS.PLUGIN_NAME}] Selection analysis failed:`, error);
            this.showError(modal, error.message);
          }
        });
      },

      // ---- "Identify full video" (fingerprint data + sprite) ----

      // Coarse, best-effort step lists shown while /identify/scene runs,
      // matched against the sidecar's _set_stage() calls via polling
      // (identify_scene_progress). Not every stage a call could report is
      // listed for every flow -- unlisted/unmatched stages are just
      // ignored by updateStepList, so this stays robust to the backend
      // skipping a step (e.g. sprite already cached).
      FULL_VIDEO_STEPS: {
        fingerprinting: [
          ['extracting_frames', 'Extracting frames from video'],
          ['analyzing_frames', 'Analyzing frames'],
          ['analyzing_screenshot', 'Analyzing scene screenshot'],
          ['analyzing_sprite', 'Analyzing sprite thumbnails'],
          ['matching_performers', 'Matching against performer database'],
          ['saving_fingerprint', 'Saving fingerprint'],
        ],
        cached: [
          ['cache_check', 'Loading cached video data'],
          ['analyzing_sprite', 'Analyzing sprite thumbnails'],
          ['matching_performers', 'Matching against performer database'],
          ['saving_fingerprint', 'Saving fingerprint'],
        ],
        spriteOnly: [
          ['analyzing_sprite', 'Analyzing sprite thumbnails'],
          ['matching_performers', 'Matching against performer database'],
        ],
      },

      createStepList(steps) {
        const container = SS.createElement('div', { className: 'ss-step-list' });
        for (const [key, label] of steps) {
          const item = SS.createElement('div', {
            className: 'ss-step-item ss-step-pending',
            innerHTML: `<span class="ss-step-marker"></span><span class="ss-step-label">${SS.escapeHtml(label)}</span>`,
          });
          item.dataset.stage = key;
          container.appendChild(item);
        }
        return container;
      },

      updateStepList(container, currentStage) {
        const items = Array.from(container.querySelectorAll('.ss-step-item'));
        const idx = items.findIndex((el) => el.dataset.stage === currentStage);
        if (idx === -1) return;
        items.forEach((el, i) => {
          el.classList.remove('ss-step-pending', 'ss-step-active', 'ss-step-done');
          el.classList.add(i < idx ? 'ss-step-done' : i === idx ? 'ss-step-active' : 'ss-step-pending');
        });
      },

      // Poll the sidecar for which stage this scene's in-flight
      // /identify/scene call is currently on, updating the step list.
      // Best-effort/ephemeral -- poll errors are silently ignored.
      pollSceneIdentifyProgress(sceneId, sidecarUrl, stepListEl) {
        let stopped = false;
        const poll = async () => {
          if (stopped) return;
          try {
            const progress = await SS.runPluginOperation('identify_scene_progress', {
              scene_id: sceneId, sidecar_url: sidecarUrl,
            });
            if (!stopped && progress && progress.stage) {
              this.updateStepList(stepListEl, progress.stage);
            }
          } catch (e) {
            // best-effort UI feedback only
          }
        };
        poll();
        const interval = setInterval(poll, 700);
        return () => { stopped = true; clearInterval(interval); };
      },

      async handleIdentifyFullVideo() {
        const route = SS.getRoute();
        if (route.type !== 'scene') return;
        const sceneId = route.id;

        const modal = this.createModal();
        this.updateLoading(modal, 'Checking for existing identification...');

        // Re-identify always goes through the same "fingerprinting" flow as
        // a first-time full analysis (fresh detect+embed+match, hits the
        // video-frame cache if nothing's changed since last time) -- wired
        // into every results view (stored or fresh) via renderResults'
        // onReidentify, not just the initial prompt.
        const reidentify = async () => {
          const settings = await SS.getSettings();
          await this._runFullVideoIdentify(
            modal, sceneId, settings,
            { use_sprite: true, skip_frame_extraction: false },
            this.FULL_VIDEO_STEPS.cached,
            { onReidentify: reidentify },
          );
        };

        try {
          const settings = await SS.getSettings();

          // Face Identification's stored result (if this scene already has
          // one) renders instantly with no fresh /identify/scene call at
          // all -- see recommendations_router.py's
          // /fingerprints/scene/{id}/result.
          const stored = await SS.runPluginOperation('fingerprint_get_scene_result', {
            scene_id: sceneId, sidecar_url: settings.sidecarUrl,
          });
          if (stored && stored.error) {
            throw new Error(stored.error);
          }
          if (stored && stored.available) {
            await this.renderResults(modal, stored.result, sceneId, {
              fromStorage: true, onReidentify: reidentify,
            });
            return;
          }

          // Nothing stored yet -- same fingerprint-prompt flow as before.
          this._showFingerprintPrompt(modal, async (fingerprintNow) => {
            try {
              if (fingerprintNow) {
                await this._runFullVideoIdentify(
                  modal, sceneId, settings,
                  { use_sprite: true, skip_frame_extraction: false },
                  this.FULL_VIDEO_STEPS.fingerprinting,
                  { onReidentify: reidentify },
                );
              } else {
                await this._runFullVideoIdentify(
                  modal, sceneId, settings,
                  { use_sprite: true, skip_frame_extraction: true },
                  this.FULL_VIDEO_STEPS.spriteOnly,
                );
              }
            } catch (error) {
              console.error(`[${SS.PLUGIN_NAME}] Full video analysis failed:`, error);
              this.showError(modal, error.message);
            }
          });
        } catch (error) {
          console.error(`[${SS.PLUGIN_NAME}] Full video analysis failed:`, error);
          this.showError(modal, error.message);
        }
      },

      _showFingerprintPrompt(modal, onChoice) {
        const loading = modal.querySelector('.ss-loading');
        loading.innerHTML = `
          <p class="ss-loading-text">This scene hasn't been fingerprinted yet</p>
          <p class="ss-loading-detail">
            Fingerprinting analyzes full video frames for the most accurate results, but can take
            a while for long scenes. You can skip it and identify from the sprite thumbnails only
            instead.
          </p>
          <div class="ss-fp-prompt-actions">
            <button class="ss-btn ss-btn-primary ss-fp-prompt-yes">Fingerprint now</button>
            <button class="ss-btn ss-fp-prompt-no">Skip (sprite only)</button>
          </div>
        `;
        loading.querySelector('.ss-fp-prompt-yes').addEventListener('click', () => onChoice(true));
        loading.querySelector('.ss-fp-prompt-no').addEventListener('click', () => onChoice(false));
      },

      async _runFullVideoIdentify(modal, sceneId, settings, flags, steps, renderOptions = {}) {
        const loading = modal.querySelector('.ss-loading');
        const resultsDiv = modal.querySelector('.ss-results');
        const errorDiv = modal.querySelector('.ss-error');
        // A Re-identify click starts this from an already-rendered results/
        // error view (both currently visible, .ss-loading hidden) -- switch
        // back to the loading view explicitly rather than relying on
        // renderResults' own display toggles, which only ever show it, not hide it.
        resultsDiv.style.display = 'none';
        errorDiv.style.display = 'none';
        loading.style.display = '';
        loading.innerHTML = `
          <div class="ss-spinner"></div>
          <p class="ss-loading-text">Identifying performers...</p>
          <p class="ss-loading-detail"></p>
        `;
        loading.appendChild(this.createStepList(steps));
        const stepListEl = loading.querySelector('.ss-step-list');

        const stopStepPolling = this.pollSceneIdentifyProgress(sceneId, settings.sidecarUrl, stepListEl);
        try {
          const results = await this.identifyScene(sceneId, (stage) => {
            this.updateLoading(modal, stage);
          }, flags);
          await this.renderResults(modal, results, sceneId, renderOptions);
        } finally {
          stopStepPolling();
        }
      },

      async handleIdentifyImage() {
        const route = SS.getRoute();
        if (route.type !== 'image') return;

        const imageId = route.id;
        const modal = this.createModal();

        try {
          this.updateLoading(modal, 'Analyzing image...', 'Detecting faces');

          const results = await this.identifyImage(imageId, (stage) => {
            this.updateLoading(modal, stage);
          });

          this.updateLoading(modal, 'Processing results...');
          await this.renderImageResults(modal, results, imageId);
        } catch (error) {
          console.error(`[${SS.PLUGIN_NAME}] Image analysis failed:`, error);
          this.showError(modal, error.message);
        }
      },

      async renderImageResults(modal, results, imageId) {
        const loading = modal.querySelector('.ss-loading');
        const resultsDiv = modal.querySelector('.ss-results');
        const errorDiv = modal.querySelector('.ss-error');

        loading.style.display = 'none';

        if (!results.faces || results.faces.length === 0) {
          errorDiv.innerHTML = `
            <div class="ss-error-icon">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="48" height="48" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z"/>
              </svg>
            </div>
            <p class="ss-error-title">No faces detected</p>
            <p class="ss-error-hint">The image may not contain clear face shots.</p>
          `;
          errorDiv.style.display = 'block';
          return;
        }

        // Only faces with at least one database match get their own card --
        // a card just saying "no match" for every unmatched face detected
        // in a busy image was pure noise. If literally nothing in the image
        // matched anything, say so once instead of per empty card.
        const matchedFaces = results.faces.filter(f => f.matches && f.matches.length > 0);

        resultsDiv.innerHTML = `
          <p class="ss-summary">
            Detected <strong>${results.face_count}</strong> face(s) in image.
          </p>
          ${matchedFaces.length === 0 ? '<p class="ss-no-match">No matches found in database</p>' : ''}
          <div class="ss-persons"></div>
        `;

        const personsDiv = resultsDiv.querySelector('.ss-persons');

        for (let i = 0; i < matchedFaces.length; i++) {
          const face = matchedFaces[i];
          const personDiv = document.createElement('div');
          personDiv.className = 'ss-person';

          {
            const match = face.matches[0];
            const confidence = this.distanceToConfidence(match.distance);
            const confidenceClass = SS.getConfidenceClass(confidence);
            const imgEndpoint = match.endpoint || 'stashdb.org';
            const imgStashboxUrl = this._stashboxPerformerUrl(imgEndpoint, match.stashdb_id);
            const imgGraphqlUrl = this._stashboxGraphqlUrl(imgEndpoint);
            const localPerformer = await this._resolveLibraryPerformer(match, imgGraphqlUrl);

            personDiv.innerHTML = `
              <div class="ss-person-header">
                <span class="ss-person-label">Face ${i + 1}</span>
              </div>
              <div class="ss-match">
                <div class="ss-match-image">
                  ${match.image_url ? `<img src="${match.image_url}" alt="${match.name}" loading="lazy" />` : '<div class="ss-no-image">No image</div>'}
                </div>
                <div class="ss-match-info">
                  <h4>${match.name}</h4>
                  <div class="ss-confidence ${confidenceClass}">${confidence}% match</div>
                  ${match.country ? `<div class="ss-country">${match.country}</div>` : ''}
                  <div class="ss-links">
                    ${this._matchLinksHtml(match, imgStashboxUrl, imgEndpoint)}
                  </div>
                  <div class="ss-actions">
                    ${localPerformer
                      ? `<button class="ss-btn ss-btn-add"
                                 data-performer-id="${localPerformer.id}"
                                 data-performer-name="${SS.escapeHtml ? SS.escapeHtml(localPerformer.name) : localPerformer.name}"
                                 data-stashdb-id="${match.stashdb_id || ''}"
                                 data-image-id="${imageId}">
                           Add to Image
                         </button>
                         <span class="ss-local-status">In library as: ${localPerformer.name}</span>`
                      : `<button class="ss-btn ss-btn-create"
                                 data-endpoint="${imgEndpoint}"
                                 data-stashdb-id="${match.stashdb_id}"
                                 data-image-id="${imageId}"${this._catalogueDataAttrs(match)}>
                           Add to Stash + Image
                         </button>
                         <button class="ss-btn ss-btn-link-as"
                                 data-endpoint="${imgEndpoint}"
                                 data-stashdb-id="${match.stashdb_id}"
                                 data-image-id="${imageId}"${this._catalogueDataAttrs(match)}>
                           Add as...
                         </button>
                         <span class="ss-local-status ss-not-in-library">Not in library</span>`
                    }
                  </div>
                </div>
              </div>
            `;

            // Build alt matches with endpoint-aware links
            if (face.matches.length > 1) {
              const details = document.createElement('details');
              details.className = 'ss-other-matches';
              details.innerHTML = `<summary>Other possible matches (${face.matches.length - 1})</summary>`;
              const ul = document.createElement('ul');
              for (const m of face.matches.slice(1)) {
                const altConf = this.distanceToConfidence(m.distance);
                const altConfClass = SS.getConfidenceClass(altConf);
                const altEp = m.endpoint || 'stashdb.org';
                const altUrl = this._stashboxPerformerUrl(altEp, m.stashdb_id);
                const altGraphqlUrl = this._stashboxGraphqlUrl(altEp);
                const altLocalPerformer = await this._resolveLibraryPerformer(m, altGraphqlUrl);
                const li = document.createElement('li');
                li.className = 'ss-alt-match-item';
                li.innerHTML = `
                  <div class="ss-match">
                    <div class="ss-match-image">
                      ${m.image_url ? `<img src="${m.image_url}" alt="${m.name}" loading="lazy" />` : '<div class="ss-no-image">No image</div>'}
                    </div>
                    <div class="ss-match-info">
                      <h4>${m.name}</h4>
                      <div class="ss-confidence ${altConfClass}">${altConf}% match</div>
                      ${m.country ? `<div class="ss-country">${m.country}</div>` : ''}
                      <div class="ss-links">
                        ${this._matchLinksHtml(m, altUrl, altEp)}
                      </div>
                      <div class="ss-actions ss-alt-match-actions">
                        ${altLocalPerformer
                          ? `<button class="ss-btn ss-btn-add ss-btn-sm"
                                     data-performer-id="${altLocalPerformer.id}"
                                     data-performer-name="${SS.escapeHtml ? SS.escapeHtml(altLocalPerformer.name) : altLocalPerformer.name}"
                                     data-stashdb-id="${m.stashdb_id || ''}"
                                     data-image-id="${imageId}">
                               Add to Image
                             </button>
                             <span class="ss-local-status">In library as: ${altLocalPerformer.name}</span>`
                          : `<button class="ss-btn ss-btn-create ss-btn-sm"
                                     data-endpoint="${altEp}"
                                     data-stashdb-id="${m.stashdb_id}"
                                     data-image-id="${imageId}"${this._catalogueDataAttrs(m)}>
                               Add to Stash + Image
                             </button>
                             <button class="ss-btn ss-btn-link-as ss-btn-sm"
                                     data-endpoint="${altEp}"
                                     data-stashdb-id="${m.stashdb_id}"
                                     data-image-id="${imageId}"${this._catalogueDataAttrs(m)}>
                               Add as...
                             </button>
                             <span class="ss-local-status ss-not-in-library">Not in library</span>`
                        }
                      </div>
                    </div>
                  </div>
                `;
                ul.appendChild(li);
              }
              details.appendChild(ul);
              personDiv.appendChild(details);
            }
          }

          personsDiv.appendChild(personDiv);
        }

        // Add click handlers for "Add to Image" buttons
        resultsDiv.querySelectorAll('.ss-btn-add').forEach(btn => {
          btn.addEventListener('click', async (e) => {
            const performerId = btn.dataset.performerId;
            const performerName = btn.dataset.performerName;
            const stashdbId = btn.dataset.stashdbId;
            const targetImageId = btn.dataset.imageId;
            btn.disabled = true;
            btn.textContent = 'Adding...';

            const staged = !!this._findSaveButton();
            const success = staged
              ? await this._selectPerformerInPendingForm(performerId, { name: performerName, stashdbId })
              : await this.addPerformerToImage(targetImageId, performerId);

            if (success) {
              btn.textContent = staged ? 'Added to form' : 'Added!';
              btn.classList.add('ss-btn-success');
              await this._finishMutation(modal, { staged });
            } else {
              btn.textContent = staged ? 'Could not add automatically' : 'Failed';
              btn.classList.add('ss-btn-error');
              btn.disabled = false;
            }
          });
        });

        // "Add to Stash + Image" handlers
        resultsDiv.querySelectorAll('.ss-btn-create').forEach(btn => {
          btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const { endpoint, stashdbId, imageId: targetImageId, source, name, country, imageUrl, catalogueUrl, profileUrl } = btn.dataset;
            btn.disabled = true;
            btn.textContent = 'Creating...';

            const staged = !!this._findSaveButton();

            try {
              const settings = await SS.getSettings();
              // Catalogue (non-stash-box) matches have no external API to
              // re-fetch full details from -- everything the backend needs
              // is already on the button's data attributes (see
              // _catalogueDataAttrs), carried straight from the match.
              const result = await this._withTimeout(
                source
                  ? SS.runPluginOperation('create_performer_from_catalogue', {
                      source, name, country: country || undefined,
                      image_url: imageUrl || undefined, catalogue_url: catalogueUrl || undefined,
                      profile_url: profileUrl || undefined,
                      ...(staged ? {} : { image_id: targetImageId }),
                      sidecar_url: settings.sidecarUrl,
                    })
                  : SS.runPluginOperation('create_performer_from_stashbox', {
                      endpoint,
                      stashdb_id: stashdbId,
                      // Omitted (not just empty) when staged -- the backend
                      // only skips its own image-assignment step when this key
                      // is entirely absent/falsy.
                      ...(staged ? {} : { image_id: targetImageId }),
                      sidecar_url: settings.sidecarUrl,
                    }),
                45000,
                'Create performer operation timed out',
              );

              if (result.error) throw new Error(result.error);

              const success = staged
                ? await this._selectPerformerInPendingForm(result.performer_id, { name: result.name, stashdbId })
                : true;

              if (success) {
                btn.textContent = staged ? 'Created — added to form' : 'Added!';
                btn.classList.add('ss-btn-success');
                await this._finishMutation(modal, { staged });
              } else {
                // Performer was created in the library either way -- just
                // couldn't be verified-selected into the open form.
                btn.textContent = 'Created — add manually';
                btn.classList.add('ss-btn-error');
              }
            } catch (err) {
              // Fallback: if the plugin call timed out but performer creation
              // actually succeeded, complete UI flow anyway. Catalogue
              // creations have no stashbox id to search back by, so skip
              // this recovery for them (less likely to time out anyway --
              // no external API round-trip involved).
              if (!source && (err?.message || '').toLowerCase().includes('timed out')) {
                try {
                  const graphqlUrl = this._stashboxGraphqlUrl(endpoint);
                  const localPerformer = await SS.findPerformerByStashDBId(stashdbId, graphqlUrl);
                  if (localPerformer?.id) {
                    const success = staged
                      ? await this._selectPerformerInPendingForm(localPerformer.id, { name: localPerformer.name, stashdbId })
                      : await this.addPerformerToImage(targetImageId, localPerformer.id);
                    if (success) {
                      btn.textContent = staged ? 'Created — added to form' : 'Added!';
                      btn.classList.add('ss-btn-success');
                      await this._finishMutation(modal, { staged });
                      return;
                    }
                    btn.textContent = 'Created — add manually';
                    btn.classList.add('ss-btn-error');
                    return;
                  }
                } catch (recoveryErr) {
                  console.warn('[Stash Sense] Timed out, recovery check failed:', recoveryErr);
                }
              }
              btn.textContent = 'Failed';
              btn.classList.add('ss-btn-error');
              btn.disabled = false;
              console.error('Failed to create performer:', err);
            }
          });
        });

        // "Add as..." handlers
        resultsDiv.querySelectorAll('.ss-btn-link-as').forEach(btn => {
          btn.addEventListener('click', (e) => {
            e.stopPropagation();
            this._openSearchPanel(btn);
          });
        });

        resultsDiv.style.display = 'block';
      },

      // Same shape as renderImageResults (both consume the generic
      // /identify {faces: [{box, matches}]} response) but wired to the
      // scene, not an image -- "Add to Scene" via addPerformerToScene /
      // scene-form staging. Used by "Identify current frame" and "Select
      // to identify".
      async renderFrameResults(modal, results, sceneId) {
        const loading = modal.querySelector('.ss-loading');
        const resultsDiv = modal.querySelector('.ss-results');
        const errorDiv = modal.querySelector('.ss-error');

        loading.style.display = 'none';

        if (!results.faces || results.faces.length === 0) {
          errorDiv.innerHTML = `
            <div class="ss-error-icon">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="48" height="48" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z"/>
              </svg>
            </div>
            <p class="ss-error-title">No faces detected</p>
            <p class="ss-error-hint">The captured frame may not contain clear face shots.</p>
          `;
          errorDiv.style.display = 'block';
          return;
        }

        // Build set of StashDB IDs (and local performer ids) already
        // tagged on this scene -- fetched live every render, same fix as
        // renderResults' own scenePerformers lookup (see its comment):
        // this endpoint's response never carries an already_tagged flag on
        // face.matches at all, so without this every performer looked
        // untagged regardless of the scene's actual current state --
        // confirmed live: "Add to Scene" showing for a performer already
        // on the scene (scene 5525, performer already added as Jenny).
        const taggedStashDBIds = new Set();
        const scenePerformerLocalIds = new Set();
        for (const p of await this.getScenePerformerStashDBIds(sceneId)) {
          scenePerformerLocalIds.add(p.id);
          for (const sid of (p.stash_ids || [])) {
            if (sid.endpoint === 'https://stashdb.org/graphql') {
              taggedStashDBIds.add(sid.stash_id);
            }
          }
        }

        // Only faces with at least one database match get their own card --
        // see renderImageResults' identical comment for why.
        const matchedFaces = results.faces.filter(f => f.matches && f.matches.length > 0);

        resultsDiv.innerHTML = `
          <p class="ss-summary">
            Detected <strong>${results.face_count}</strong> face(s) in frame.
          </p>
          ${matchedFaces.length === 0 ? '<p class="ss-no-match">No matches found in database</p>' : ''}
          <div class="ss-persons"></div>
        `;

        const personsDiv = resultsDiv.querySelector('.ss-persons');

        for (let i = 0; i < matchedFaces.length; i++) {
          const face = matchedFaces[i];
          const personDiv = document.createElement('div');
          personDiv.className = 'ss-person';

          {
            const match = face.matches[0];
            const confidence = this.distanceToConfidence(match.distance);
            const confidenceClass = SS.getConfidenceClass(confidence);
            const imgEndpoint = match.endpoint || 'stashdb.org';
            const imgStashboxUrl = this._stashboxPerformerUrl(imgEndpoint, match.stashdb_id);
            const imgGraphqlUrl = this._stashboxGraphqlUrl(imgEndpoint);
            const localPerformer = await this._resolveLibraryPerformer(match, imgGraphqlUrl);

            // Check if already tagged (from API flag or local cross-reference)
            // -- same logic as renderResults' _renderPerson, see the
            // taggedStashDBIds/scenePerformerLocalIds fetch above.
            const isAlreadyTagged = match.already_tagged || taggedStashDBIds.has(match.stashdb_id);
            const isLocallyTagged = localPerformer && scenePerformerLocalIds.has(localPerformer.id);
            const showAlreadyTagged = isAlreadyTagged || isLocallyTagged;

            personDiv.innerHTML = `
              <div class="ss-person-header">
                <span class="ss-person-label">Face ${i + 1}</span>
              </div>
              <div class="ss-match">
                <div class="ss-match-image">
                  ${match.image_url ? `<img src="${match.image_url}" alt="${match.name}" loading="lazy" />` : '<div class="ss-no-image">No image</div>'}
                </div>
                <div class="ss-match-info">
                  <h4>${match.name}</h4>
                  <div class="ss-confidence ${confidenceClass}">${confidence}% match</div>
                  ${match.country ? `<div class="ss-country">${match.country}</div>` : ''}
                  <div class="ss-links">
                    ${this._matchLinksHtml(match, imgStashboxUrl, imgEndpoint)}
                  </div>
                  <div class="ss-actions">
                    ${showAlreadyTagged
                      ? `<span class="ss-local-status ss-already-tagged">Already tagged on scene</span>`
                      : localPerformer
                      ? `<button class="ss-btn ss-btn-add"
                                 data-performer-id="${localPerformer.id}"
                                 data-performer-name="${SS.escapeHtml ? SS.escapeHtml(localPerformer.name) : localPerformer.name}"
                                 data-stashdb-id="${match.stashdb_id || ''}"
                                 data-scene-id="${sceneId}">
                           Add to Scene
                         </button>
                         <span class="ss-local-status">In library as: ${localPerformer.name}</span>`
                      : `<button class="ss-btn ss-btn-create"
                                 data-endpoint="${imgEndpoint}"
                                 data-stashdb-id="${match.stashdb_id}"
                                 data-scene-id="${sceneId}"${this._catalogueDataAttrs(match)}>
                           Add to Stash + Scene
                         </button>
                         <button class="ss-btn ss-btn-link-as"
                                 data-endpoint="${imgEndpoint}"
                                 data-stashdb-id="${match.stashdb_id}"
                                 data-scene-id="${sceneId}"${this._catalogueDataAttrs(match)}>
                           Add as...
                         </button>
                         <span class="ss-local-status ss-not-in-library">Not in library</span>`
                    }
                  </div>
                </div>
              </div>
            `;

            // Build alt matches with endpoint-aware links
            if (face.matches.length > 1) {
              const details = document.createElement('details');
              details.className = 'ss-other-matches';
              details.innerHTML = `<summary>Other possible matches (${face.matches.length - 1})</summary>`;
              const ul = document.createElement('ul');
              for (const m of face.matches.slice(1)) {
                const altConf = this.distanceToConfidence(m.distance);
                const altConfClass = SS.getConfidenceClass(altConf);
                const altEp = m.endpoint || 'stashdb.org';
                const altUrl = this._stashboxPerformerUrl(altEp, m.stashdb_id);
                const altGraphqlUrl = this._stashboxGraphqlUrl(altEp);
                const altLocalPerformer = await this._resolveLibraryPerformer(m, altGraphqlUrl);
                const altIsAlreadyTagged = m.already_tagged || taggedStashDBIds.has(m.stashdb_id);
                const altIsLocallyTagged = altLocalPerformer && scenePerformerLocalIds.has(altLocalPerformer.id);
                const altShowAlreadyTagged = altIsAlreadyTagged || altIsLocallyTagged;
                const li = document.createElement('li');
                li.className = 'ss-alt-match-item';
                li.innerHTML = `
                  <div class="ss-match">
                    <div class="ss-match-image">
                      ${m.image_url ? `<img src="${m.image_url}" alt="${m.name}" loading="lazy" />` : '<div class="ss-no-image">No image</div>'}
                    </div>
                    <div class="ss-match-info">
                      <h4>${m.name}</h4>
                      <div class="ss-confidence ${altConfClass}">${altConf}% match</div>
                      ${m.country ? `<div class="ss-country">${m.country}</div>` : ''}
                      <div class="ss-links">
                        ${this._matchLinksHtml(m, altUrl, altEp)}
                      </div>
                      <div class="ss-actions ss-alt-match-actions">
                        ${altShowAlreadyTagged
                          ? `<span class="ss-local-status ss-already-tagged">Already tagged on scene</span>`
                          : altLocalPerformer
                          ? `<button class="ss-btn ss-btn-add ss-btn-sm"
                                     data-performer-id="${altLocalPerformer.id}"
                                     data-performer-name="${SS.escapeHtml ? SS.escapeHtml(altLocalPerformer.name) : altLocalPerformer.name}"
                                     data-stashdb-id="${m.stashdb_id || ''}"
                                     data-scene-id="${sceneId}">
                               Add to Scene
                             </button>
                             <span class="ss-local-status">In library as: ${altLocalPerformer.name}</span>`
                          : `<button class="ss-btn ss-btn-create ss-btn-sm"
                                     data-endpoint="${altEp}"
                                     data-stashdb-id="${m.stashdb_id}"
                                     data-scene-id="${sceneId}"${this._catalogueDataAttrs(m)}>
                               Add to Stash + Scene
                             </button>
                             <button class="ss-btn ss-btn-link-as ss-btn-sm"
                                     data-endpoint="${altEp}"
                                     data-stashdb-id="${m.stashdb_id}"
                                     data-scene-id="${sceneId}"${this._catalogueDataAttrs(m)}>
                               Add as...
                             </button>
                             <span class="ss-local-status ss-not-in-library">Not in library</span>`
                        }
                      </div>
                    </div>
                  </div>
                `;
                ul.appendChild(li);
              }
              details.appendChild(ul);
              personDiv.appendChild(details);
            }
          }

          personsDiv.appendChild(personDiv);
        }

        // Add click handlers for "Add to Scene" buttons
        resultsDiv.querySelectorAll('.ss-btn-add').forEach(btn => {
          btn.addEventListener('click', async (e) => {
            const performerId = btn.dataset.performerId;
            const performerName = btn.dataset.performerName;
            const stashdbId = btn.dataset.stashdbId;
            const targetSceneId = btn.dataset.sceneId;
            btn.disabled = true;
            btn.textContent = 'Adding...';

            const staged = !!this._findSaveButton();
            const success = staged
              ? await this._selectPerformerInPendingForm(performerId, { name: performerName, stashdbId })
              : await this.addPerformerToScene(targetSceneId, performerId);

            if (success) {
              btn.textContent = staged ? 'Added to form' : 'Added!';
              btn.classList.add('ss-btn-success');
              await this._finishMutation(modal, { staged });
            } else {
              btn.textContent = staged ? 'Could not add automatically' : 'Failed';
              btn.classList.add('ss-btn-error');
              btn.disabled = false;
            }
          });
        });

        // "Add to Stash + Scene" handlers
        resultsDiv.querySelectorAll('.ss-btn-create').forEach(btn => {
          btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const { endpoint, stashdbId, sceneId: targetSceneId, source, name, country, imageUrl, catalogueUrl, profileUrl } = btn.dataset;
            btn.disabled = true;
            btn.textContent = 'Creating...';

            const staged = !!this._findSaveButton();

            try {
              const settings = await SS.getSettings();
              // Catalogue (non-stash-box) matches have no external API to
              // re-fetch full details from -- everything the backend needs
              // is already on the button's data attributes (see
              // _catalogueDataAttrs), carried straight from the match.
              const result = await this._withTimeout(
                source
                  ? SS.runPluginOperation('create_performer_from_catalogue', {
                      source, name, country: country || undefined,
                      image_url: imageUrl || undefined, catalogue_url: catalogueUrl || undefined,
                      profile_url: profileUrl || undefined,
                      ...(staged ? {} : { scene_id: targetSceneId }),
                      sidecar_url: settings.sidecarUrl,
                    })
                  : SS.runPluginOperation('create_performer_from_stashbox', {
                      endpoint,
                      stashdb_id: stashdbId,
                      // Omitted (not just empty) when staged -- the backend
                      // only skips its own scene-assignment step when this key
                      // is entirely absent/falsy.
                      ...(staged ? {} : { scene_id: targetSceneId }),
                      sidecar_url: settings.sidecarUrl,
                    }),
                45000,
                'Create performer operation timed out',
              );

              if (result.error) throw new Error(result.error);

              const success = staged
                ? await this._selectPerformerInPendingForm(result.performer_id, { name: result.name, stashdbId })
                : true;

              if (success) {
                btn.textContent = staged ? 'Created — added to form' : 'Added!';
                btn.classList.add('ss-btn-success');
                await this._finishMutation(modal, { staged });
              } else {
                // Performer was created in the library either way -- just
                // couldn't be verified-selected into the open form.
                btn.textContent = 'Created — add manually';
                btn.classList.add('ss-btn-error');
              }
            } catch (err) {
              // Fallback: if the plugin call timed out but performer creation
              // actually succeeded, complete UI flow anyway. Catalogue
              // creations have no stashbox id to search back by, so skip
              // this recovery for them (less likely to time out anyway --
              // no external API round-trip involved).
              if (!source && (err?.message || '').toLowerCase().includes('timed out')) {
                try {
                  const graphqlUrl = this._stashboxGraphqlUrl(endpoint);
                  const localPerformer = await SS.findPerformerByStashDBId(stashdbId, graphqlUrl);
                  if (localPerformer?.id) {
                    const success = staged
                      ? await this._selectPerformerInPendingForm(localPerformer.id, { name: localPerformer.name, stashdbId })
                      : await this.addPerformerToScene(targetSceneId, localPerformer.id);
                    if (success) {
                      btn.textContent = staged ? 'Created — added to form' : 'Added!';
                      btn.classList.add('ss-btn-success');
                      await this._finishMutation(modal, { staged });
                      return;
                    }
                    btn.textContent = 'Created — add manually';
                    btn.classList.add('ss-btn-error');
                    return;
                  }
                } catch (recoveryErr) {
                  console.warn('[Stash Sense] Timed out, recovery check failed:', recoveryErr);
                }
              }
              btn.textContent = 'Failed';
              btn.classList.add('ss-btn-error');
              btn.disabled = false;
              console.error('Failed to create performer:', err);
            }
          });
        });

        // "Add as..." handlers
        resultsDiv.querySelectorAll('.ss-btn-link-as').forEach(btn => {
          btn.addEventListener('click', (e) => {
            e.stopPropagation();
            this._openSearchPanel(btn);
          });
        });

        resultsDiv.style.display = 'block';
      },

      // Scene toolbar entry point: a dropdown (not a plain button) offering
      // "Select to identify" / "Identify current frame" / "Identify full
      // video". Follows the same hand-rolled toggle+menu+outside-click-
      // dismiss pattern already used repeatedly in
      // stash-sense-recommendations.js, rather than a UI library. The
      // toggle keeps the .ss-identify-btn class so the existing
      // `document.querySelector('.ss-identify-btn')` idempotency guards in
      // injectButton/updateButtonStatus keep working unchanged.
      createButton() {
        const status = SS.getSidecarStatus();
        const wrapper = SS.createElement('div', {
          className: 'ss-identify-dropdown',
          attrs: { style: 'position:relative;display:inline-block;' },
        });

        const toggle = SS.createElement('button', {
          className: 'ss-identify-btn btn btn-secondary',
          attrs: {
            title: statusTitle('Identify performers using face recognition'),
            'data-default-title': 'Identify performers using face recognition',
            'data-ss-plugin': SS.PLUGIN_ID,
          },
          innerHTML: `
            <span class="ss-btn-icon ${statusIconClass(status)}">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z"/>
              </svg>
            </span>
            <span class="ss-dropdown-caret">▾</span>
          `,
        });

        const menu = SS.createElement('div', { className: 'ss-identify-menu' });
        const items = [
          { label: 'Select to identify', handler: () => this.handleSelectToIdentify() },
          { label: 'Identify current frame', handler: () => this.handleIdentifyCurrentFrame() },
          { label: 'Identify full video', handler: () => this.handleIdentifyFullVideo() },
        ];
        for (const item of items) {
          const itemBtn = SS.createElement('button', {
            className: 'ss-identify-menu-item',
            textContent: item.label,
          });
          itemBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            menu.classList.remove('ss-open');
            item.handler();
          });
          menu.appendChild(itemBtn);
        }

        toggle.addEventListener('click', (e) => {
          e.stopPropagation();
          const opening = !menu.classList.contains('ss-open');
          if (opening) {
            // Flip the menu above the toggle when it wouldn't fit below in
            // the viewport, same as Stash's own hamburger menu. The menu
            // stays measurable even while closed (visibility:hidden, not
            // display:none -- see CSS) so offsetHeight is accurate here.
            const toggleRect = toggle.getBoundingClientRect();
            const spaceBelow = window.innerHeight - toggleRect.bottom;
            menu.classList.toggle('ss-menu-up', spaceBelow < menu.offsetHeight);
          }
          menu.classList.toggle('ss-open');
        });
        document.addEventListener('click', function closeMenu(e) {
          if (!wrapper.contains(e.target)) {
            menu.classList.remove('ss-open');
          }
          if (!document.contains(wrapper)) {
            document.removeEventListener('click', closeMenu);
          }
        });

        wrapper.appendChild(toggle);
        wrapper.appendChild(menu);
        return wrapper;
      },

      updateButtonStatus(connected) {
        const versionInfo = SS.getSidecarVersionInfo();
        const pluginInfo = SS.getPluginVersionInfo();
        const outdated = connected === true
          && ((versionInfo && versionInfo.outdated) || (pluginInfo && pluginInfo.tooOld));
        const updateAvailable = connected === true && !outdated
          && ((versionInfo && versionInfo.updateAvailable) || (pluginInfo && pluginInfo.updateAvailable));
        // Scoped to this plugin's own buttons (data-ss-plugin) -- v1 and v2
        // share the .ss-identify-btn class name, so an unscoped selector
        // would also flip v1's connection dot based on v2's sidecar status.
        document.querySelectorAll(`.ss-identify-btn[data-ss-plugin="${SS.PLUGIN_ID}"]`).forEach(btn => {
          const icon = btn.querySelector('.ss-btn-icon');
          if (!icon) return;
          icon.classList.remove('ss-connected', 'ss-disconnected', 'ss-outdated', 'ss-update-available');
          if (outdated) {
            // Connected, but either side is below the other's required
            // floor -- distinct from a plain "disconnected" so it's clear
            // the fix is a version mismatch, not connectivity.
            icon.classList.add('ss-outdated');
            btn.title = statusTitle(btn.title);
          } else if (updateAvailable) {
            // Connected and both sides meet their required floors, but a
            // newer (non-required) release exists for one of them --
            // lower-alarm styling than ss-outdated, FYI rather than
            // action-needed.
            icon.classList.add('ss-update-available');
            btn.title = statusTitle(btn.title);
          } else if (connected === true) {
            icon.classList.add('ss-connected');
            if (btn.dataset.defaultTitle) btn.title = btn.dataset.defaultTitle;
          } else if (connected === false) {
            icon.classList.add('ss-disconnected');
            btn.title = statusTitle(btn.title);
          }
        });
      },

      injectSceneButton() {
        const route = SS.getRoute();
        if (route.type !== 'scene') return;
        if (document.querySelector('.ss-identify-btn')) return;

        const buttonContainers = [
          '.scene-toolbar .btn-group',
          '.detail-header .ml-auto .btn-group',
          '.scene-header .btn-group',
          '.detail-header-buttons',
          '.scene-operations',
          '.ml-auto.btn-group',
        ];

        for (const selector of buttonContainers) {
          const container = document.querySelector(selector);
          if (container) {
            container.appendChild(this.createButton());
            console.log(`[${SS.PLUGIN_NAME}] Button injected into ${selector}`);
            return;
          }
        }

        // Fallback: floating button
        const floatingBtn = this.createButton();
        floatingBtn.classList.add('ss-floating-btn');
        document.body.appendChild(floatingBtn);
      },

      createImageButton() {
        const status = SS.getSidecarStatus();
        const btn = SS.createElement('button', {
          className: 'ss-identify-btn btn btn-secondary',
          attrs: {
            title: statusTitle('Identify performers using face recognition'),
            'data-default-title': 'Identify performers using face recognition',
            'data-ss-plugin': SS.PLUGIN_ID,
          },
          innerHTML: `
            <span class="ss-btn-icon ${statusIconClass(status)}">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z"/>
              </svg>
            </span>
          `,
        });
        btn.addEventListener('click', () => this.handleIdentifyImage());
        return btn;
      },

      injectImageButton() {
        const route = SS.getRoute();
        if (route.type !== 'image') return;
        if (document.querySelector('.ss-identify-btn')) return;

        const buttonContainers = [
          '.image-toolbar .btn-group',
          '.detail-header .ml-auto .btn-group',
          '.image-header .btn-group',
          '.detail-header-buttons',
          '.ml-auto.btn-group',
        ];

        for (const selector of buttonContainers) {
          const container = document.querySelector(selector);
          if (container) {
            container.appendChild(this.createImageButton());
            console.log(`[${SS.PLUGIN_NAME}] Image button injected into ${selector}`);
            return;
          }
        }

        // Fallback: floating button
        const floatingBtn = this.createImageButton();
        floatingBtn.classList.add('ss-floating-btn');
        document.body.appendChild(floatingBtn);
      },

      // Call the gallery identification API
      async identifyGallery(galleryId, onProgress) {
        const settings = await SS.getSettings();
        onProgress?.(`Connecting to ${SS.PLUGIN_NAME}...`);

        const stopPolling = pollModelLoading(settings.sidecarUrl, onProgress);
        try {
          const result = await SS.runPluginOperation('identify_gallery', {
            gallery_id: galleryId,
            sidecar_url: settings.sidecarUrl,
          });

          if (result.error) {
            throw new Error(result.error);
          }

          return result;
        } finally {
          stopPolling();
        }
      },

      // Add performer to gallery
      async addPerformerToGallery(galleryId, performerId) {
        const getQuery = `
          query GetGallery($id: ID!) {
            findGallery(id: $id) {
              performers { id }
            }
          }
        `;

        const updateQuery = `
          mutation UpdateGallery($id: ID!, $performer_ids: [ID!]) {
            galleryUpdate(input: { id: $id, performer_ids: $performer_ids }) {
              id
            }
          }
        `;

        try {
          const getResult = await SS.stashQuery(getQuery, { id: galleryId });
          const currentPerformers = getResult?.findGallery?.performers || [];
          const currentIds = currentPerformers.map(p => p.id);

          if (!currentIds.includes(performerId)) {
            currentIds.push(performerId);
          }

          await SS.stashQuery(updateQuery, { id: galleryId, performer_ids: currentIds });
          return true;
        } catch (e) {
          console.error('Failed to add performer to gallery:', e);
          return false;
        }
      },

      async handleIdentifyGallery() {
        const route = SS.getRoute();
        if (route.type !== 'gallery') return;

        const galleryId = route.id;
        const modal = this.createModal();

        try {
          this.updateLoading(modal, 'Identifying performers in gallery...', 'This may take a while for large galleries');

          const results = await this.identifyGallery(galleryId, (stage) => {
            this.updateLoading(modal, stage);
          });

          this.updateLoading(modal, 'Processing results...');
          await this.renderGalleryResults(modal, results, galleryId);
        } catch (error) {
          console.error(`[${SS.PLUGIN_NAME}] Gallery analysis failed:`, error);
          this.showError(modal, error.message);
        }
      },

      async renderGalleryResults(modal, results, galleryId) {
        const loading = modal.querySelector('.ss-loading');
        const resultsDiv = modal.querySelector('.ss-results');
        const errorDiv = modal.querySelector('.ss-error');

        loading.style.display = 'none';

        if (!results.performers || results.performers.length === 0) {
          errorDiv.innerHTML = `
            <div class="ss-error-icon">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="48" height="48" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z"/>
              </svg>
            </div>
            <p class="ss-error-title">No performers identified</p>
            <p class="ss-error-hint">
              Processed ${results.images_processed || 0}/${results.total_images || 0} images
              but no confident matches were found.
            </p>
          `;
          errorDiv.style.display = 'block';
          return;
        }

        resultsDiv.innerHTML = `
          <p class="ss-summary">
            Processed <strong>${results.images_processed}</strong>/${results.total_images} images,
            detected <strong>${results.faces_detected}</strong> faces,
            identified <strong>${results.performers.length}</strong> performer(s).
          </p>
          <div class="ss-gallery-actions-bar">
            <button class="ss-btn ss-btn-primary ss-accept-all-btn">Accept All</button>
          </div>
          <div class="ss-persons"></div>
        `;

        const personsDiv = resultsDiv.querySelector('.ss-persons');

        for (const performer of results.performers) {
          const personDiv = document.createElement('div');
          personDiv.className = 'ss-person';

          const confidence = this.distanceToConfidence(performer.best_distance);
          const confidenceClass = SS.getConfidenceClass(confidence);

          const galEndpoint = performer.endpoint || 'stashdb.org';
          const galStashboxUrl = this._stashboxPerformerUrl(galEndpoint, performer.performer_id);
          const galGraphqlUrl = this._stashboxGraphqlUrl(galEndpoint);
          const localPerformer = await this._resolveLibraryPerformer(
            { local_performer_id: performer.local_performer_id, stashdb_id: performer.performer_id, name: performer.name },
            galGraphqlUrl,
          );

          personDiv.innerHTML = `
            <div class="ss-person-header">
              <span class="ss-person-label">${performer.name}</span>
              <span class="ss-person-frames">Found in ${performer.image_count}/${results.total_images} images</span>
            </div>
            <div class="ss-match">
              <div class="ss-match-image">
                ${performer.image_url ? `<img src="${performer.image_url}" alt="${performer.name}" loading="lazy" />` : '<div class="ss-no-image">No image</div>'}
              </div>
              <div class="ss-match-info">
                <div class="ss-confidence ${confidenceClass}">${confidence}% match</div>
                ${performer.country ? `<div class="ss-country">${performer.country}</div>` : ''}
                <div class="ss-links">
                  ${this._matchLinksHtml(
                    { local_performer_id: performer.local_performer_id, stashdb_id: performer.performer_id },
                    galStashboxUrl, galEndpoint,
                  )}
                </div>
                ${localPerformer ? `
                  <div class="ss-gallery-performer-actions" data-performer-id="${localPerformer.id}" data-stashdb-id="${performer.performer_id}">
                    <div class="ss-gallery-tag-toggle">
                      <label class="ss-toggle-label">
                        <input type="checkbox" class="ss-tag-images-toggle" />
                        <span>Also tag individual images</span>
                      </label>
                    </div>
                    <div class="ss-actions">
                      <button class="ss-btn ss-btn-add ss-gallery-accept-btn"
                              data-performer-id="${localPerformer.id}"
                              data-performer-name="${SS.escapeHtml ? SS.escapeHtml(localPerformer.name) : localPerformer.name}"
                              data-stashdb-id="${performer.performer_id || ''}"
                              data-gallery-id="${galleryId}"
                              data-image-ids='${JSON.stringify(performer.image_ids)}'>
                        Add to Gallery
                      </button>
                      <span class="ss-local-status">In library as: ${localPerformer.name}</span>
                    </div>
                  </div>
                ` : `
                  <div class="ss-actions">
                    <span class="ss-local-status ss-not-in-library">Not in library</span>
                  </div>
                `}
              </div>
            </div>
          `;

          personsDiv.appendChild(personDiv);
        }

        let bulkAcceptInProgress = false;

        const runGalleryAccept = async (btn, deferFinish = false) => {
          const performerId = btn.dataset.performerId;
          const performerName = btn.dataset.performerName;
          const stashdbId = btn.dataset.stashdbId;
          const targetGalleryId = btn.dataset.galleryId;
          let imageIds;
          try {
            imageIds = JSON.parse(btn.dataset.imageIds);
          } catch (_) {
            imageIds = [];
          }
          const tagImages = btn.closest('.ss-gallery-performer-actions')
            ?.querySelector('.ss-tag-images-toggle')?.checked || false;

          btn.disabled = true;
          btn.textContent = 'Adding...';

          // Only the gallery-level add needs the staged/direct branch --
          // no Image edit form can be open while viewing gallery results,
          // so per-image tagging below always mutates directly.
          const staged = !!this._findSaveButton();
          let success = staged
            ? await this._selectPerformerInPendingForm(performerId, { name: performerName, stashdbId })
            : await this.addPerformerToGallery(targetGalleryId, performerId);

          if (success && tagImages) {
            btn.textContent = 'Tagging images...';
            for (const imgId of imageIds) {
              await this.addPerformerToImage(imgId, performerId);
            }
          }

          if (success) {
            btn.textContent = tagImages
              ? `${staged ? 'Added to form' : 'Added to gallery'} + ${imageIds.length} images`
              : (staged ? 'Added to form' : 'Added to gallery!');
            btn.classList.add('ss-btn-success');
            if (!deferFinish && !bulkAcceptInProgress) {
              await this._finishMutation(modal, { staged });
            }
          } else {
            btn.textContent = staged ? 'Could not add automatically' : 'Failed';
            btn.classList.add('ss-btn-error');
            btn.disabled = false;
          }

          return success;
        };

        // Click handlers for individual accept buttons
        resultsDiv.querySelectorAll('.ss-gallery-accept-btn').forEach(btn => {
          btn.addEventListener('click', async () => {
            await runGalleryAccept(btn, false);
          });
        });

        // Accept All handler
        resultsDiv.querySelector('.ss-accept-all-btn')?.addEventListener('click', async (e) => {
          const acceptAllBtn = e.target;
          acceptAllBtn.disabled = true;
          acceptAllBtn.textContent = 'Accepting...';
          bulkAcceptInProgress = true;

          const staged = !!this._findSaveButton();
          let successCount = 0;
          const buttons = Array.from(resultsDiv.querySelectorAll('.ss-gallery-accept-btn:not(:disabled)'));
          for (const btn of buttons) {
            const ok = await runGalleryAccept(btn, true);
            if (ok) successCount += 1;
            // Small delay between operations
            await new Promise(r => setTimeout(r, 200));
          }

          bulkAcceptInProgress = false;
          if (successCount > 0) {
            acceptAllBtn.textContent = `Accepted ${successCount}`;
            acceptAllBtn.classList.add('ss-btn-success');
            await this._finishMutation(modal, { staged });
          } else {
            acceptAllBtn.textContent = 'No changes applied';
            acceptAllBtn.classList.add('ss-btn-error');
            acceptAllBtn.disabled = false;
          }
        });

        resultsDiv.style.display = 'block';
      },

      createGalleryButton() {
        const status = SS.getSidecarStatus();
        const btn = SS.createElement('button', {
          className: 'ss-identify-btn btn btn-secondary',
          attrs: {
            title: statusTitle('Identify all performers in this gallery'),
            'data-default-title': 'Identify all performers in this gallery',
            'data-ss-plugin': SS.PLUGIN_ID,
          },
          innerHTML: `
            <span class="ss-btn-icon ${statusIconClass(status)}">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z"/>
              </svg>
            </span>
          `,
        });
        btn.addEventListener('click', () => this.handleIdentifyGallery());
        return btn;
      },

      injectGalleryButton() {
        const route = SS.getRoute();
        if (route.type !== 'gallery') return;
        if (document.querySelector('.ss-identify-btn')) return;

        const buttonContainers = [
          '.gallery-toolbar .btn-group',
          '.detail-header .ml-auto .btn-group',
          '.gallery-header .btn-group',
          '.detail-header-buttons',
          '.ml-auto.btn-group',
        ];

        for (const selector of buttonContainers) {
          const container = document.querySelector(selector);
          if (container) {
            container.appendChild(this.createGalleryButton());
            console.log(`[${SS.PLUGIN_NAME}] Gallery button injected into ${selector}`);
            return;
          }
        }

        // Fallback: floating button
        const floatingBtn = this.createGalleryButton();
        floatingBtn.classList.add('ss-floating-btn');
        document.body.appendChild(floatingBtn);
      },
    };

    // ==================== Initialization ====================

    // Wait for a DOM element to appear, retrying with increasing delays
    function waitForElement(selector, callback, maxAttempts = 20) {
      let attempts = 0;
      function check() {
        const el = document.querySelector(selector);
        if (el) {
          callback(el);
        } else if (attempts < maxAttempts) {
          attempts++;
          setTimeout(check, 250);
        }
      }
      setTimeout(check, 300);
    }

    // Inject button into the appropriate toolbar for the current page type.
    // Idempotency check is scoped to this plugin's own buttons
    // (data-ss-plugin) -- v1 and v2 share the .ss-identify-btn class name,
    // so an unscoped selector meant "whichever plugin's JS ran first wins,
    // the other silently sees a button that isn't its own and never
    // injects" when both are installed side by side.
    const OWN_BUTTON_SELECTOR = `.ss-identify-btn[data-ss-plugin="${SS.PLUGIN_ID}"]`;
    function injectButton(route) {
      if (document.querySelector(OWN_BUTTON_SELECTOR)) return;

      const toolbarMap = {
        scene:   { selector: '.scene-toolbar-group:last-child',   create: () => FaceRecognition.createButton() },
        image:   { selector: '.image-toolbar-group:last-child',   create: () => FaceRecognition.createImageButton() },
        gallery: { selector: '.gallery-toolbar-group:last-child', create: () => FaceRecognition.createGalleryButton() },
      };

      const config = toolbarMap[route.type];
      if (!config) return;

      waitForElement(config.selector, (container) => {
        if (document.querySelector(OWN_BUTTON_SELECTOR)) return; // re-check after wait
        container.appendChild(config.create());
        console.log(`[${SS.PLUGIN_NAME}] Button injected into ${config.selector}`);
      });
    }

    // ==================== Version Mismatch Modals ====================
    //
    // Two *required* (not just "newer exists") mismatches, each shown as
    // its own auto-popup modal (never just a passive badge) per explicit
    // request: this plugin below the connected sidecar's own
    // min_plugin_version, and the connected sidecar below this plugin's
    // MIN_SIDECAR_VERSION. The two are NOT symmetric in severity:
    //
    // - Plugin too old: dismissable prompt with a CTA to update via
    //   Settings > Plugins > Available Plugins. The plugin can't fix
    //   this itself, but continuing to use an old plugin against a
    //   newer sidecar is "may render/behave oddly," not "may corrupt
    //   data" -- the user can reasonably choose to proceed.
    // - Sidecar too old: non-dismissable, BLOCKS the entire interface
    //   (full-viewport overlay, no close button, backdrop doesn't
    //   dismiss). There is nothing the browser side can do to fix an
    //   outdated Docker container, and letting the user keep taking
    //   actions (merges, identify-and-add, database operations) against
    //   a sidecar that predates fields/behavior this plugin build
    //   assumes risks real data loss/corruption, not just a degraded
    //   UI -- so this one doesn't offer a way through, it just waits
    //   for the next health poll to see the sidecar has been updated.
    const VersionMismatch = {
      _blockEl: null,
      _promptEl: null,
      _dismissedKey: null, // "<current>|<minRequired>" the prompt was last dismissed for

      evaluate() {
        const sidecarInfo = SS.getSidecarVersionInfo();
        const pluginInfo = SS.getPluginVersionInfo();

        if (sidecarInfo && sidecarInfo.outdated) {
          this._showBlock(sidecarInfo);
        } else {
          this._hideBlock();
        }

        if (pluginInfo && pluginInfo.tooOld) {
          const key = `${pluginInfo.current}|${pluginInfo.minRequired}`;
          if (key !== this._dismissedKey) this._showPrompt(pluginInfo);
        } else {
          this._hidePrompt();
          this._dismissedKey = null;
        }
      },

      _backdropStyle: 'position:fixed;top:0;left:0;width:100%;height:100%;z-index:2147483000;'
        + 'display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,0.75);',
      _boxStyle: 'position:relative;background:var(--bs-body-bg, #1a1a1a);border-radius:8px;'
        + 'width:90%;max-width:520px;max-height:85vh;overflow:auto;padding:24px;'
        + 'box-shadow:0 8px 32px rgba(0,0,0,0.6);color:var(--bs-body-color, #fff);',

      _renderChangelog(entries) {
        if (!entries || !entries.length) return '';
        const items = entries.map(e => `
          <div style="margin-bottom:10px;">
            <div style="font-weight:600;font-size:0.85rem;">v${e.version}${e.date ? ` <span style="font-weight:400;opacity:0.6;">(${e.date})</span>` : ''}</div>
            <ul style="margin:4px 0 0 0;padding-left:18px;font-size:0.85rem;opacity:0.85;">
              ${e.bullets.map(b => `<li>${SS.escapeHtml ? SS.escapeHtml(b) : b}</li>`).join('')}
            </ul>
          </div>
        `).join('');
        return `<div style="margin-top:14px;max-height:220px;overflow-y:auto;border-top:1px solid var(--bs-border-color, #333);padding-top:12px;">${items}</div>`;
      },

      _showBlock(info) {
        if (this._blockEl) return; // already showing -- avoid re-render/flicker on every poll
        const el = document.createElement('div');
        el.id = 'ss-version-block';
        el.style.cssText = this._backdropStyle;
        el.innerHTML = `
          <div style="${this._boxStyle}">
            <h3 style="margin:0 0 10px 0;font-size:18px;">${SS.PLUGIN_NAME}: Sidecar update required</h3>
            <p style="margin:0 0 8px 0;font-size:0.9rem;">
              The connected sidecar is running <strong>v${info.current}</strong>, but this plugin
              (v${SS.PLUGIN_VERSION}) requires at least <strong>v${info.required}</strong>.
            </p>
            <p style="margin:0;font-size:0.9rem;opacity:0.85;">
              Continuing against an older sidecar risks acting on assumptions it doesn't actually meet
              (missing fields, different behavior) -- the interface is disabled until the sidecar container
              is rebuilt/updated to a newer version. This will resolve automatically once that happens
              (checked every 60s).
            </p>
            ${this._renderChangelog(info.changelog)}
          </div>
        `;
        document.body.appendChild(el);
        this._blockEl = el;
      },

      _hideBlock() {
        if (this._blockEl) {
          this._blockEl.remove();
          this._blockEl = null;
        }
      },

      _showPrompt(info) {
        this._hidePrompt(); // replace any stale one (e.g. minRequired changed) rather than stacking
        const el = document.createElement('div');
        el.id = 'ss-version-prompt';
        el.style.cssText = this._backdropStyle;
        el.innerHTML = `
          <div style="${this._boxStyle}">
            <div style="display:flex;align-items:baseline;justify-content:space-between;gap:12px;">
              <h3 style="margin:0 0 10px 0;font-size:18px;">${SS.PLUGIN_NAME}: Plugin update recommended</h3>
              <button class="ss-version-prompt-close" style="background:none;border:none;font-size:22px;color:var(--bs-secondary-color, #888);cursor:pointer;padding:0;line-height:1;" aria-label="Close">&times;</button>
            </div>
            <p style="margin:0 0 8px 0;font-size:0.9rem;">
              This plugin (v${info.current}) is older than the connected sidecar expects
              (v${info.minRequired}+). Some features may be missing or behave unexpectedly until it's
              updated.
            </p>
            <p style="margin:0 0 14px 0;font-size:0.85rem;opacity:0.85;">
              Update from <strong>Settings &gt; Plugins &gt; Available Plugins</strong> in Stash.
            </p>
            ${this._renderChangelog(info.changelog)}
          </div>
        `;
        el.querySelector('.ss-version-prompt-close').addEventListener('click', () => {
          this._dismissedKey = `${info.current}|${info.minRequired}`;
          this._hidePrompt();
        });
        document.body.appendChild(el);
        this._promptEl = el;
      },

      _hidePrompt() {
        if (this._promptEl) {
          this._promptEl.remove();
          this._promptEl = null;
        }
      },
    };

    async function init() {
      console.log(`[${SS.PLUGIN_NAME}] Initializing...`);

      // Check sidecar health
      const health = await SS.checkHealth();
      VersionMismatch.evaluate();
      if (health) {
        console.log(`[${SS.PLUGIN_NAME}] Sidecar connected: ${health.performer_count} performers`);
      } else {
        console.warn(`[${SS.PLUGIN_NAME}] Sidecar not available`);
      }

      // Initialize navigation watcher
      SS.initNavigationWatcher();

      // Inject button for current page
      injectButton(SS.getRoute());

      // Watch for navigation
      SS.onNavigate((route) => {
        injectButton(route);
      });

      // Periodic health check — guard against duplicate intervals
      if (window._ssHealthCheckInterval) {
        clearInterval(window._ssHealthCheckInterval);
      }
      // Tracks outdated/updateAvailable together (badgeKey) so a button
      // re-render also fires on a pure "update became available" change,
      // not just the required-outdated flip the old lastOutdated-only
      // check covered.
      function badgeKey() {
        const info = SS.getSidecarVersionInfo();
        const pluginInfo = SS.getPluginVersionInfo();
        const outdated = (info && info.outdated) || (pluginInfo && pluginInfo.tooOld);
        const updateAvailable = !outdated
          && ((info && info.updateAvailable) || (pluginInfo && pluginInfo.updateAvailable));
        return `${outdated}|${updateAvailable}`;
      }
      let lastBadgeKey = SS.getSidecarStatus() ? badgeKey() : 'false|false';
      window._ssHealthCheckInterval = setInterval(async () => {
        try {
          const health = await Promise.race([
            SS.checkHealth(),
            new Promise((_, reject) => setTimeout(() => reject(new Error('timeout')), 10000)),
          ]);
          const newStatus = health ? true : false;
          const newBadgeKey = newStatus ? badgeKey() : 'false|false';
          if (newStatus !== SS.getSidecarStatus() || newBadgeKey !== lastBadgeKey) {
            SS.setSidecarStatus(newStatus);
            lastBadgeKey = newBadgeKey;
            FaceRecognition.updateButtonStatus(newStatus);
          }
          VersionMismatch.evaluate();
        } catch (_) {
          if (SS.getSidecarStatus()) {
            SS.setSidecarStatus(false);
            FaceRecognition.updateButtonStatus(false);
          }
          VersionMismatch.evaluate();
        }
      }, 60000);

      console.log(`[${SS.PLUGIN_NAME}] Initialized`);
    }

    // Start
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init);
    } else {
      init();
    }
  });
})();
