/**
 * Stash Sense Core Module
 * Shared utilities, settings, and API client
 */
(function() {
  'use strict';

  // Plugin configuration
  //
  // PLUGIN_ID must match the directory name this plugin is actually
  // installed under in Stash (Stash derives a local plugin's id from its
  // folder name, not from any field in the .yml manifest -- see
  // stash-plugin-repo's own docs for the same rule on the distributed
  // path). It's used both to look up this plugin's own saved settings
  // (configuration.plugins[PLUGIN_ID] via GraphQL) and to route
  // runPluginOperation calls to the right backend script. Getting this
  // wrong doesn't error -- it silently reads/drives *whichever other
  // plugin* happens to be installed under the stale id instead, which is
  // exactly what happened here: this stayed 'stash-sense' (v1's id) while
  // v2 was installed side-by-side under 'stash-sense2', so v2's UI was
  // unknowingly reading v1's sidecar URL and invoking v1's backend the
  // entire time. If you install this under a different local folder name,
  // change this constant to match.
  const PLUGIN_ID = 'stash-sense2';
  const PLUGIN_NAME = 'Stash Sense 2';
  const PLUGIN_VERSION = '0.23.0';

  // Lowest sidecar version this plugin JS actually works against -- bump
  // this alongside PLUGIN_VERSION whenever a JS change starts depending on
  // a sidecar-side API/field that didn't exist before (e.g. the
  // source/catalogue_url/profile_url match fields added in sidecar 0.13.3,
  // or the create-performer-from-catalogue endpoint in 0.13.3). Compared
  // against the sidecar's own reported /health version in checkHealth()
  // below so a plugin update that outran its sidecar container shows a
  // clear "sidecar needs updating" signal instead of silently missing
  // fields or hitting 404s on endpoints that don't exist yet.
  //
  // Bumped to 0.14.12 for Scene Face Matches: the recommendations UI now
  // depends on the scene_face_match job type existing in JOB_REGISTRY, its
  // three new /recommendations/actions/*-scene-face-matches* endpoints, and
  // PerformerMatchResponse.top_timestamps_sec -- none of which exist on an
  // older sidecar.
  const MIN_SIDECAR_VERSION = '0.24.0';

  // Default settings
  const DEFAULTS = {
    sidecarUrl: 'http://localhost:5000',
    minConfidence: 50,
    maxResults: 5,
  };

  // Cached state
  let cachedSettings = null;
  let sidecarStatus = null; // null = unknown, true = connected, false = disconnected
  // null = unknown, else { current, required, outdated, latestVersion,
  // updateAvailable, changelog } -- `outdated`/`required` is the existing
  // "this plugin needs a newer sidecar" (blocking) check; `latestVersion`/
  // `updateAvailable` is the newer, non-blocking "a newer sidecar exists"
  // notice; `changelog` is what's new since `current`, from the sidecar's
  // own changelog.txt (see release_info.py server-side).
  let sidecarVersionInfo = null;
  // null = unknown, else { current, minRequired, tooOld, latestVersion,
  // updateAvailable, changelog } -- the reverse direction: `tooOld` is
  // this plugin build being below the *connected sidecar's own*
  // min_plugin_version (blocking -- the sidecar may have shipped
  // fields/endpoints this plugin doesn't know about yet).
  let pluginVersionInfo = null;

  // ==================== Version Comparison ====================

  /**
   * Compares two "X.Y.Z" version strings. Returns negative if a < b, 0 if
   * equal, positive if a > b. Deliberately simple (numeric dot-segments
   * only) -- matches this project's own plain-semver versioning, not a
   * full semver spec (no pre-release/build metadata to worry about).
   */
  function compareVersions(a, b) {
    const pa = String(a || '0').split('.').map(n => parseInt(n, 10) || 0);
    const pb = String(b || '0').split('.').map(n => parseInt(n, 10) || 0);
    for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
      const diff = (pa[i] || 0) - (pb[i] || 0);
      if (diff !== 0) return diff;
    }
    return 0;
  }

  // ==================== Settings ====================

  async function getSettings() {
    if (cachedSettings) return cachedSettings;

    try {
      const response = await fetch('/graphql', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: `query Configuration { configuration { plugins } }`,
        }),
      });
      const result = await response.json();
      const pluginConfig = result?.data?.configuration?.plugins?.[PLUGIN_ID];

      cachedSettings = {
        sidecarUrl: (pluginConfig?.sidecarUrl || DEFAULTS.sidecarUrl).replace(/\/$/, ''),
        minConfidence: parseInt(pluginConfig?.minConfidence || DEFAULTS.minConfidence, 10),
        maxResults: parseInt(pluginConfig?.maxResults || DEFAULTS.maxResults, 10),
      };

      console.log(`[${PLUGIN_NAME}] Settings loaded:`, cachedSettings);
      return cachedSettings;
    } catch (e) {
      console.error(`[${PLUGIN_NAME}] Failed to load settings:`, e);
      return DEFAULTS;
    }
  }

  function clearSettingsCache() {
    cachedSettings = null;
  }

  // ==================== Sidecar API Client ====================

  /**
   * Make a request to the sidecar API
   */
  async function sidecarFetch(endpoint, options = {}) {
    const settings = await getSettings();
    const url = `${settings.sidecarUrl}${endpoint}`;

    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Sidecar API error: ${response.status} - ${error}`);
    }

    return response.json();
  }

  /**
   * Run a plugin operation via Stash's GraphQL API (for operations that need Stash access)
   */
  async function runPluginOperation(mode, args = {}) {
    const query = `
      mutation RunPluginOperation($plugin_id: ID!, $args: Map!) {
        runPluginOperation(plugin_id: $plugin_id, args: $args)
      }
    `;

    const response = await fetch('/graphql', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        variables: {
          plugin_id: PLUGIN_ID,
          args: { mode, ...args },
        },
      }),
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => 'Unknown error');
      console.error(`[${PLUGIN_NAME}] runPluginOperation HTTP error: ${response.status} - ${errorText}`);
      throw new Error(`Plugin operation failed: HTTP ${response.status}`);
    }

    const result = await response.json();

    if (result.errors) {
      throw new Error(result.errors[0]?.message || 'GraphQL error');
    }

    const output = result?.data?.runPluginOperation;
    if (typeof output === 'string') {
      try {
        const parsed = JSON.parse(output);
        return parsed.output || parsed;
      } catch {
        return { error: output };
      }
    }

    return output?.output || output || {};
  }

  /**
   * Check sidecar health (via Python backend to bypass CSP)
   */
  async function checkHealth() {
    try {
      const settings = await getSettings();
      const result = await runPluginOperation('health', {
        sidecar_url: settings.sidecarUrl,
        plugin_version: PLUGIN_VERSION,
      });
      if (result.error) {
        sidecarStatus = false;
        sidecarVersionInfo = null;
        pluginVersionInfo = null;
        return null;
      }
      sidecarStatus = true;
      if (result.version) {
        const outdated = compareVersions(result.version, MIN_SIDECAR_VERSION) < 0;
        const latestVersion = result.latest_sidecar_version || null;
        const updateAvailable = !!latestVersion && compareVersions(latestVersion, result.version) > 0;
        sidecarVersionInfo = {
          current: result.version, required: MIN_SIDECAR_VERSION, outdated,
          latestVersion, updateAvailable, changelog: result.sidecar_changelog || [],
        };
        if (outdated) {
          console.warn(
            `[${PLUGIN_NAME}] Sidecar is running v${result.version}, but this plugin (v${PLUGIN_VERSION}) `
            + `needs at least v${MIN_SIDECAR_VERSION}. Some features may be missing or broken until the `
            + `sidecar container is rebuilt/updated to a newer version.`
          );
        }
      } else {
        sidecarVersionInfo = null; // older sidecar that predates /health reporting a version at all
      }

      // Reverse direction: does the *connected sidecar* require a newer
      // plugin than this one? Only known once a sidecar with
      // min_plugin_version support has actually responded -- an older
      // sidecar predating this field leaves pluginVersionInfo null, same
      // "unknown, not false" treatment as sidecarVersionInfo above.
      if (result.min_plugin_version) {
        const tooOld = compareVersions(PLUGIN_VERSION, result.min_plugin_version) < 0;
        const latestVersion = result.latest_plugin_version || null;
        const updateAvailable = !!latestVersion && compareVersions(latestVersion, PLUGIN_VERSION) > 0;
        pluginVersionInfo = {
          current: PLUGIN_VERSION, minRequired: result.min_plugin_version, tooOld,
          latestVersion, updateAvailable, changelog: result.plugin_changelog || [],
        };
        if (tooOld) {
          console.warn(
            `[${PLUGIN_NAME}] This plugin (v${PLUGIN_VERSION}) is older than the connected sidecar `
            + `requires (v${result.min_plugin_version}+). Update the plugin from Settings > Plugins > `
            + `Available Plugins to avoid broken or missing functionality.`
          );
        }
      } else {
        pluginVersionInfo = null;
      }

      return result;
    } catch (e) {
      sidecarStatus = false;
      sidecarVersionInfo = null;
      pluginVersionInfo = null;
      return null;
    }
  }

  function getSidecarStatus() {
    return sidecarStatus;
  }

  function setSidecarStatus(status) {
    sidecarStatus = status;
  }

  function getPluginVersionInfo() {
    return pluginVersionInfo;
  }

  /** Returns { current, required, outdated } from the last checkHealth() call, or null if unknown/unavailable. */
  function getSidecarVersionInfo() {
    return sidecarVersionInfo;
  }

  // ==================== Stash GraphQL Helpers ====================

  /**
   * Execute a GraphQL query against Stash
   */
  async function stashQuery(query, variables = {}) {
    let response;
    try {
      response = await fetch('/graphql', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, variables }),
      });
    } catch (e) {
      console.error(`[${PLUGIN_NAME}] stashQuery fetch failed:`, e);
      return null;
    }
    if (!response.ok) {
      console.error(`[${PLUGIN_NAME}] stashQuery HTTP error: ${response.status}`);
      return null;
    }
    const result = await response.json();
    if (result.errors) {
      throw new Error(result.errors[0]?.message || 'GraphQL error');
    }
    return result.data;
  }

  /**
   * Look up a performer by StashDB ID
   */
  async function findPerformerByStashDBId(stashdbId, endpoint = 'https://stashdb.org/graphql') {
    const query = `
      query FindByStashDBId($stashdb_id: String!, $endpoint: String!) {
        findPerformers(performer_filter: {
          stash_id_endpoint: {
            endpoint: $endpoint
            stash_id: $stashdb_id
            modifier: EQUALS
          }
        }) {
          performers {
            id
            name
            image_path
          }
        }
      }
    `;

    try {
      const data = await stashQuery(query, { stashdb_id: stashdbId, endpoint });
      const performers = data?.findPerformers?.performers || [];
      return performers.length > 0 ? performers[0] : null;
    } catch (e) {
      console.error('Failed to lookup performer:', e);
      return null;
    }
  }

  /**
   * Get performer details by ID
   */
  async function getPerformer(id) {
    const query = `
      query GetPerformer($id: ID!) {
        findPerformer(id: $id) {
          id
          name
          image_path
          scene_count
        }
      }
    `;
    const data = await stashQuery(query, { id });
    return data?.findPerformer;
  }

  /**
   * Get scene details by ID
   */
  async function getScene(id) {
    const query = `
      query GetScene($id: ID!) {
        findScene(id: $id) {
          id
          title
          date
          paths {
            screenshot
          }
          files {
            id
            path
            basename
            size
            duration
            video_codec
            width
            height
          }
          performers {
            id
            name
            image_path
          }
          studio {
            id
            name
          }
        }
      }
    `;
    const data = await stashQuery(query, { id });
    return data?.findScene;
  }

  /**
   * Get image details by ID
   */
  async function getImage(id) {
    const query = `
      query GetImage($id: ID!) {
        findImage(id: $id) {
          id
          title
          paths {
            image
            thumbnail
          }
          performers {
            id
            name
            image_path
          }
        }
      }
    `;
    const data = await stashQuery(query, { id });
    return data?.findImage;
  }

  /**
   * Get gallery details by ID
   */
  async function getGallery(id) {
    const query = `
      query GetGallery($id: ID!) {
        findGallery(id: $id) {
          id
          title
          image_count
          performers {
            id
            name
            image_path
          }
        }
      }
    `;
    const data = await stashQuery(query, { id });
    return data?.findGallery;
  }

  // ==================== URL Routing ====================

  /**
   * Get the current route info
   */
  function getRoute() {
    const path = window.location.pathname;

    // Scene page
    const sceneMatch = path.match(/\/scenes\/(\d+)/);
    if (sceneMatch) {
      return { type: 'scene', id: sceneMatch[1] };
    }

    // Image page
    const imageMatch = path.match(/\/images\/(\d+)/);
    if (imageMatch) {
      return { type: 'image', id: imageMatch[1] };
    }

    // Gallery page
    const galleryMatch = path.match(/\/galleries\/(\d+)/);
    if (galleryMatch) {
      return { type: 'gallery', id: galleryMatch[1] };
    }

    // Performer page
    const performerMatch = path.match(/\/performers\/(\d+)/);
    if (performerMatch) {
      return { type: 'performer', id: performerMatch[1] };
    }

    // Plugin page -- must be this plugin's own route specifically.
    // PLUGIN_ID-derived, not a hardcoded 'stash-sense' prefix: that
    // literal also matches '/plugins/stash-sense2' (a v1-vs-v2 route
    // collision when both are installed side-by-side -- see changelog
    // 0.14.1/0.14.2), and would equally mismatch if this plugin is ever
    // installed under yet another folder name.
    const pluginPrefix = '/plugins/' + PLUGIN_ID;
    if (path === pluginPrefix || path.startsWith(pluginPrefix + '/')) {
      const subpath = path.slice(pluginPrefix.length) || '/';
      return { type: 'plugin', subpath };
    }

    return { type: 'other', path };
  }

  // ==================== UI Utilities ====================

  /**
   * Format bytes to human readable
   */
  function formatSize(bytes) {
    if (bytes === 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return (bytes / Math.pow(1024, i)).toFixed(1) + ' ' + units[i];
  }

  /**
   * Format duration in seconds to HH:MM:SS
   */
  function formatDuration(seconds) {
    if (!seconds) return 'Unknown';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) {
      return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }
    return `${m}:${s.toString().padStart(2, '0')}`;
  }

  /**
   * Format a date string
   */
  function formatDate(dateStr) {
    if (!dateStr) return '';
    const date = new Date(dateStr);
    return date.toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  }

  /**
   * Create an element with classes and attributes
   */
  function createElement(tag, options = {}) {
    const el = document.createElement(tag);
    if (options.className) el.className = options.className;
    if (options.id) el.id = options.id;
    if (options.innerHTML) el.innerHTML = options.innerHTML;
    if (options.textContent) el.textContent = options.textContent;
    if (options.attrs) {
      for (const [key, value] of Object.entries(options.attrs)) {
        el.setAttribute(key, value);
      }
    }
    if (options.styles) Object.assign(el.style, options.styles);
    if (options.events) {
      for (const [event, handler] of Object.entries(options.events)) {
        el.addEventListener(event, handler);
      }
    }
    if (options.children) {
      for (const child of options.children) {
        el.appendChild(child);
      }
    }
    return el;
  }

  /**
   * Convert distance (0-2) to confidence percentage (0-100)
   */
  function distanceToConfidence(distance) {
    const clamped = Math.max(0, Math.min(1, distance));
    return Math.round((1 - clamped) * 100);
  }

  /**
   * Get confidence level class
   */
  function getConfidenceClass(confidence) {
    if (confidence >= 70) return 'high';
    if (confidence >= 50) return 'medium';
    return 'low';
  }

  /**
   * Escape a string for safe insertion into HTML
   */
  function escapeHtml(str) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(String(str)));
    return div.innerHTML;
  }

  // Strip an absolute sidecar-origin URL down to a same-origin-relative
  // path -- the sidecar's own STASH_URL isn't necessarily the address the
  // browser uses to reach Stash (reverse proxy, different LAN
  // hostname/port), so an absolute URL from it can point somewhere the
  // browser can't reach. Was duplicated identically in stash-sense.js and
  // stash-sense-recommendations.js; centralized here now that
  // renderPerformerCandidateCards (below) needs it too.
  function relativeUrl(url) {
    if (!url) return url;
    try { return new URL(url).pathname; }
    catch (e) { return url; }
  }

  /**
   * Rewrite an external performer match photo (StashDB/catalogue
   * `image_url`, e.g. on a match card or a Face Recommendations candidate)
   * to a source-specific *thumbnail* variant, instead of loading the
   * full-size original just to display it at card size -- meaningful
   * bandwidth savings across a list of many match cards. Falls back to the
   * original URL untouched for any source with no known rewrite (including
   * on a URL parse failure), so this always returns something displayable.
   *
   * Only for *display* -- never rewrite a URL that's about to be sent back
   * to the sidecar to actually download/import a performer photo (e.g.
   * _catalogueDataAttrs' data-image-url), which needs the real full-size
   * original.
   *
   * Known sources, add more here as needed:
   * - stashdb.org (`/images/<uuid>`) -- StashDB's own image endpoint
   *   accepts a `size` query param for a resized variant.
   * - media.seekfans.com -- no resizing of its own; proxied through
   *   seekfans.com's own Next.js image optimizer (the same one the site
   *   itself uses to serve these same photos at thumbnail size).
   *
   * Deliberately NOT rewritten: pornbox (cdn77-image.gtflixtv.com). Its
   * `image_url` is already a pre-sized CDN77 thumbnail as stored by
   * stash-sense2-data-gen's pornbox_discovery.py -- that module's own
   * docstring documents (against live testing) that changing the
   * signed URL's width/height query params is unreliable and mostly 403s,
   * so the stored URL is always left query-intact and used as-is. No
   * rewrite needed here since it's already thumbnail-sized; don't add one.
   */
  function thumbnailUrl(url) {
    if (!url) return url;
    try {
      const parsed = new URL(url);
      const host = parsed.hostname.replace(/^www\./, '');

      if (host === 'stashdb.org' && parsed.pathname.startsWith('/images/')) {
        parsed.searchParams.set('size', '600');
        return parsed.toString();
      }

      if (host === 'media.seekfans.com') {
        return `https://seekfans.com/_next/image?url=${encodeURIComponent(url)}&w=256&q=75`;
      }

      return url;
    } catch (e) {
      return url;
    }
  }

  /**
   * Render a set of local-performer candidates as visual cards (cover
   * image, name, alias preview, verifiable external links) for the user
   * to explicitly pick from or reject -- replaces matching by name/alias
   * text alone, which is exactly what let an alias collision silently
   * pick the wrong same-named performer in the past (a real incident: see
   * the sidecar's recommendations_router.py, PerformerIdentityAmbiguous).
   * A same-name text row can't be told apart; a real cover photo and a
   * verifiable "View on stashdb.org" link can.
   *
   * @param container - element to render into (innerHTML replaced)
   * @param candidates - [{id, name, image_path, alias_list, stash_ids, urls}, ...]
   *   (image_path is an absolute sidecar-origin URL, relativized here)
   * @param options.onSelectExisting(performerId) - called when a card's
   *   "Use this performer" is clicked
   * @param options.onCreateNew() - called when "create as new" is clicked;
   *   omit to hide that action (e.g. when creation isn't offered here)
   * @param options.localPerformerUrl(id) - builds the "View local
   *   performer" href, default `/performers/{id}`
   * @param options.stashboxPerformerUrl(domain, stashId) - builds a
   *   "View on {domain}" href, default `https://{domain}/performers/{id}`
   */
  function renderPerformerCandidateCards(container, candidates, options = {}) {
    const onSelectExisting = options.onSelectExisting;
    const onCreateNew = options.onCreateNew;
    const localPerformerUrl = options.localPerformerUrl || ((id) => `/performers/${id}`);
    const stashboxPerformerUrl = options.stashboxPerformerUrl
      || ((domain, stashId) => `https://${domain}/performers/${stashId}`);

    const cardHtml = (c) => {
      const aliasPreview = (c.alias_list || []).slice(0, 4).join(', ');
      const stashLinks = (c.stash_ids || []).map((sid) => {
        const domain = String(sid.endpoint || '').replace(/^https?:\/\//, '').replace(/\/graphql$/, '');
        const href = stashboxPerformerUrl(domain, sid.stash_id);
        return `<a href="${escapeHtml(href)}" target="_blank" rel="noopener" class="ss-link">View on ${escapeHtml(domain)}</a>`;
      }).join(' ');
      return `
        <div class="ss-candidate-card" data-performer-id="${escapeHtml(c.id)}">
          <div class="ss-candidate-card-image">
            ${c.image_path
              ? `<img src="${escapeHtml(relativeUrl(c.image_path))}" alt="" onerror="this.style.display='none'" />`
              : '<div class="ss-no-image">No Image</div>'
            }
          </div>
          <div class="ss-candidate-card-body">
            <div class="ss-candidate-card-name">${escapeHtml(c.name || 'Unknown')}</div>
            ${aliasPreview ? `<div class="ss-candidate-card-aliases">aka ${escapeHtml(aliasPreview)}</div>` : ''}
            <div class="ss-candidate-card-links">
              <a href="${escapeHtml(localPerformerUrl(c.id))}" target="_blank" rel="noopener" class="ss-link ss-link-local">View local performer</a>
              ${stashLinks}
            </div>
            <button type="button" class="ss-btn ss-btn-secondary ss-candidate-card-select">Use this performer</button>
          </div>
        </div>
      `;
    };

    container.innerHTML = `
      <div class="ss-candidate-cards">
        ${candidates.map(cardHtml).join('')}
      </div>
      ${onCreateNew ? '<button type="button" class="ss-btn ss-btn-secondary ss-candidate-create-new">None of these — create as new performer</button>' : ''}
    `;

    container.querySelectorAll('.ss-candidate-card').forEach((card) => {
      const btn = card.querySelector('.ss-candidate-card-select');
      btn.addEventListener('click', () => {
        if (onSelectExisting) onSelectExisting(card.dataset.performerId);
      });
    });
    const createBtn = container.querySelector('.ss-candidate-create-new');
    if (createBtn && onCreateNew) {
      createBtn.addEventListener('click', onCreateNew);
    }
  }

  // ==================== Tab URL State ====================

  function getTabFromUrl() {
    const params = new URLSearchParams(window.location.search);
    return params.get('tab') || 'recommendations';
  }

  function setTabInUrl(tabName) {
    const url = new URL(window.location.href);
    if (tabName === 'recommendations') {
      url.searchParams.delete('tab');
    } else {
      url.searchParams.set('tab', tabName);
    }
    history.replaceState(null, '', url.toString());
  }

  // ==================== SPA Navigation ====================

  const navigationCallbacks = [];
  const leavePluginCallbacks = [];

  function onNavigate(callback) {
    navigationCallbacks.push(callback);
  }

  function onLeavePlugin(callback) {
    leavePluginCallbacks.push(callback);
  }

  function initNavigationWatcher() {
    let lastUrl = window.location.href;
    let lastRouteType = getRoute().type;

    const observer = new MutationObserver(() => {
      if (window.location.href !== lastUrl) {
        lastUrl = window.location.href;
        const route = getRoute();

        // Fire leave-plugin callbacks when navigating away from plugin page
        if (lastRouteType === 'plugin' && route.type !== 'plugin') {
          for (const callback of leavePluginCallbacks) {
            try {
              callback();
            } catch (e) {
              console.error(`[${PLUGIN_NAME}] Leave plugin callback error:`, e);
            }
          }
        }
        lastRouteType = route.type;

        for (const callback of navigationCallbacks) {
          try {
            callback(route);
          } catch (e) {
            console.error(`[${PLUGIN_NAME}] Navigation callback error:`, e);
          }
        }
      }
    });

    observer.observe(document.body, { childList: true, subtree: true });
  }

  // ==================== Export ====================

  window.StashSense = {
    // Constants
    PLUGIN_ID,
    PLUGIN_NAME,
    PLUGIN_VERSION,
    MIN_SIDECAR_VERSION,
    DEFAULTS,

    // Settings
    getSettings,
    clearSettingsCache,

    // Sidecar API
    sidecarFetch,
    runPluginOperation,
    checkHealth,
    getSidecarStatus,
    setSidecarStatus,
    getSidecarVersionInfo,
    getPluginVersionInfo,
    compareVersions,

    // Stash GraphQL
    stashQuery,
    findPerformerByStashDBId,
    getPerformer,
    getScene,
    getImage,
    getGallery,

    // Routing
    getRoute,
    onNavigate,
    onLeavePlugin,
    initNavigationWatcher,
    getTabFromUrl,
    setTabInUrl,

    // Utilities
    formatSize,
    formatDuration,
    formatDate,
    createElement,
    distanceToConfidence,
    getConfidenceClass,
    escapeHtml,
    relativeUrl,
    thumbnailUrl,
    renderPerformerCandidateCards,
  };

  console.log(`[${PLUGIN_NAME}] Core module loaded`);
})();
