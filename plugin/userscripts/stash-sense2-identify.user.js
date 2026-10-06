// ==UserScript==
// @name         Stash Sense 2 - Identify anywhere
// @namespace    https://github.com/AnonTester/stash-sense2
// @version      0.5.3
// @description  Identify performers with your Stash Sense 2 sidecar from any web page: right-click an image, or draw an area (face + automatic margin) anywhere on a page. Also sends a drawn area to Google Translate (images).
// @match        *://*/*
// @noframes
// @run-at       document-idle
// @require      https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js
// @grant        GM_xmlhttpRequest
// @grant        GM_getValue
// @grant        GM_setValue
// @grant        GM_registerMenuCommand
// @grant        GM_openInTab
// @grant        GM_addValueChangeListener
// @grant        GM_removeValueChangeListener
// @grant        GM_deleteValue
// @grant        unsafeWindow
// @connect      *
// ==/UserScript==

/*
 * Stash Sense 2 - Identify anywhere (Violentmonkey / Tampermonkey userscript)
 *
 * Talks straight to the Stash Sense 2 sidecar's /identify endpoint (the same
 * one the Stash plugin's "Identify current frame" / "Select to identify" use)
 * and shows the matches in a modal like the plugin does. Unlike the plugin it
 * runs on any website, so there is no Stash entity to attach a result to:
 * results are read-only (links to the matched performer, no "Add to scene").
 *
 * How to use
 *   - Hold the trigger key (default: Ctrl) and right-click. A small menu
 *     appears with "Identify this image" and "Select area to identify...".
 *     A browser's own context menu cannot be extended from a userscript, so
 *     this script shows its own menu when the trigger key is held; a plain
 *     right-click still opens the normal browser menu. The trigger key can be
 *     changed (Ctrl / Alt / Shift / none = always) in the settings.
 *   - The same two actions plus "Settings" are in the userscript manager's
 *     toolbar menu for this script.
 *   - Select area: drag a box over the face, move/resize it until it fits,
 *     then "Use for identify" (or Enter). The area is cropped out of the
 *     image/video underneath it at the image's native resolution, and a
 *     margin is added around it (default 100% of the box size per side;
 *     where the image ends, neutral gray fills in) because the face detector
 *     needs context around the face and finds nothing when the face fills
 *     most of the picture. If no face is found it retries with wider and then
 *     narrower margins before giving up.
 *
 * Setup: open the settings (menu or toolbar) and enter the sidecar URL, e.g.
 * http://192.168.1.100:6961. It is stored by the userscript manager, not in
 * this file.
 *
 * Updates: install this file from the URL Stash serves it at (see README);
 * the manager re-checks that same URL and compares the @version above.
 */
(() => {
  'use strict';

  // The script also runs in the background tab that grabViaTab opens on an
  // image URL; there it only does one thing: draw that image document's image
  // and hand it back (grabInTab), then stay out of the way.
  // Declared before the early returns below, which use it (const is not hoisted).
  const TRANSLATE_JOB_KEY = 'ssvm_translate_job';
  const grabMatch = /^#ssvm-grab=([\w-]+)$/.exec(location.hash);
  if (grabMatch) {
    grabInTab(grabMatch[1]);
    return;
  }
  // On Google Translate itself the script only delivers a pending "translate
  // this area" job (see startTranslateFlow) and shows no menu of its own.
  if (location.hostname === 'translate.google.com') {
    deliverTranslateJob();
    return;
  }

  const NAME = 'Stash Sense 2';
  const CFG_KEY = 'ssvm_config';
  const DEFAULTS = { sidecarUrl: '', stashUrl: '', modifier: 'ctrl', topK: 5, marginPct: 100, translateTo: 'en' };
  const MIN_MEDIA_PX = 32;     // ignore icons/sprites smaller than this on screen
  const MAX_SIDE = 6000;       // longest side of the JPEG sent to the sidecar
  const JPEG_QUALITY = 0.92;
  const MIN_CROP_PX = 8;       // smallest crop (native pixels) worth sending
  const MAX_MARGIN_PCT = 300;

  // Runs in the background tab opened by grabViaTab. An image opened as its
  // own document is same-origin with itself, so drawing it never taints the
  // canvas, and the navigation carries the site's cookies (bot protection lets
  // it through, unlike a request made by the userscript manager). The result
  // goes back through the userscript manager's shared value storage, which
  // works across origins and windows (postMessage/opener don't, e.g. under
  // Cross-Origin-Opener-Policy).
  function grabInTab(token) {
    const key = `ssvm_grab_${token}`;
    const reply = (msg) => GM_setValue(key, msg);
    const run = async () => {
      if (!/^image\//i.test(document.contentType || '')) {
        const text = (document.body ? document.body.innerText : '').replace(/\s+/g, ' ').trim().slice(0, 140);
        reply({ error: `not an image document (${document.contentType}; title "${document.title}"; "${text}")` });
        return;
      }
      const img = document.images[0] || document.querySelector('img');
      if (!img) { reply({ error: 'no image element' }); return; }
      if (!img.complete) await new Promise((r) => { img.onload = r; img.onerror = r; });
      if (!img.naturalWidth) { reply({ error: 'image failed to load' }); return; }
      const c = document.createElement('canvas');
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      const ctx = c.getContext('2d');
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, c.width, c.height);
      ctx.drawImage(img, 0, 0);
      reply({ dataUrl: c.toDataURL('image/jpeg', 0.95) });
    };
    run().catch((e) => reply({ error: e.message }));
  }

  // Runs on translate.google.com: if a "translate this area" job is waiting
  // (stored a moment ago by startTranslateFlow on another page), put its image
  // into Google's own image-upload field, exactly as if a file had been chosen.
  function deliverTranslateJob() {
    let job = null;
    try { job = GM_getValue(TRANSLATE_JOB_KEY, null); } catch (e) { job = null; }
    if (!job || !job.dataUrl || Date.now() - job.ts > 120000) return;
    try { GM_deleteValue(TRANSLATE_JOB_KEY); } catch (e) { /* ignore */ }

    const toast = (text, isError) => {
      let el = document.getElementById('ssvm-toast');
      if (!el) {
        el = document.createElement('div');
        el.id = 'ssvm-toast';
        el.style.cssText = 'position:fixed;left:50%;bottom:24px;transform:translateX(-50%);z-index:2147483647;'
          + 'background:#202225;color:#fff;font:14px/1.4 sans-serif;padding:10px 16px;border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,.5);';
        document.documentElement.appendChild(el);
      }
      el.textContent = text;
      el.style.borderLeft = `4px solid ${isError ? '#dc3545' : '#0d6efd'}`;
      if (!isError) setTimeout(() => el.remove(), 6000);
    };

    const findInput = () => [...document.querySelectorAll('input[type=file]')]
      .find((i) => /image\//i.test(i.getAttribute('accept') || ''));
    const started = Date.now();
    toast('Stash Sense 2: uploading the selected area...');
    const tick = () => {
      const input = findInput();
      if (!input) {
        if (Date.now() - started > 20000) toast('Stash Sense 2: could not find the image upload field on this page.', true);
        else setTimeout(tick, 300);
        return;
      }
      const bin = atob(job.dataUrl.slice(job.dataUrl.indexOf(',') + 1));
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      const W = typeof unsafeWindow !== 'undefined' ? unsafeWindow : window;
      // Google's code runs in the page's own realm and ignores a File/FileList
      // made in the userscript sandbox in some browsers, so try the page's own
      // constructors first, the sandbox's as a fallback, then a synthetic drop.
      const attempts = [
        ['page objects', () => {
          const file = new W.File([new W.Uint8Array(bytes)], 'selected-area.png', { type: 'image/png' });
          const dt = new W.DataTransfer();
          dt.items.add(file);
          input.files = dt.files;
          input.dispatchEvent(new W.Event('change', { bubbles: true }));
        }],
        ['sandbox objects', () => {
          const file = new File([bytes], 'selected-area.png', { type: 'image/png' });
          const dt = new DataTransfer();
          dt.items.add(file);
          input.files = dt.files;
          input.dispatchEvent(new Event('change', { bubbles: true }));
        }],
      ];
      const errors = [];
      for (const [label, fn] of attempts) {
        try {
          fn();
          if (input.files && input.files.length) {
            toast(`Stash Sense 2: area sent to Google Translate (${label}).`);
            return;
          }
          errors.push(`${label}: field stayed empty`);
        } catch (e) {
          errors.push(`${label}: ${e.message}`);
        }
      }
      toast(`Stash Sense 2: upload failed (${errors.join('; ')}).`, true);
    };
    tick();
  }

  // ------------------------------------------------------------------ config

  function loadConfig() {
    let saved = {};
    try {
      saved = GM_getValue(CFG_KEY, {}) || {};
      if (typeof saved === 'string') saved = JSON.parse(saved);
    } catch (e) { saved = {}; }
    return { ...DEFAULTS, ...saved };
  }

  function saveConfig(cfg) {
    GM_setValue(CFG_KEY, cfg);
  }

  function normalizeUrl(u) {
    u = (u || '').trim().replace(/\/+$/, '');
    if (u && !/^https?:\/\//i.test(u)) u = 'http://' + u;
    return u;
  }

  // -------------------------------------------------------------- tiny utils

  function h(tag, props = {}, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (v === false || v == null) continue;
      if (k === 'class') el.className = v;
      else if (k === 'style') el.style.cssText = v;
      else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? '' : v);
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  function safeHref(u) {
    try {
      const p = new URL(u, location.href);
      return /^https?:$/.test(p.protocol) ? p.href : null;
    } catch (e) { return null; }
  }

  function link(href, label) {
    const safe = safeHref(href);
    if (!safe) return h('span', { class: 'link-disabled' }, label);
    return h('a', { class: 'link', href: safe, target: '_blank', rel: 'noopener noreferrer' }, label);
  }

  function gm(opts) {
    return new Promise((resolve, reject) => {
      GM_xmlhttpRequest({
        ...opts,
        onload: resolve,
        onerror: () => reject(new Error(`Connection failed: could not reach ${opts.url}`)),
        ontimeout: () => reject(new Error('Request timed out')),
        onabort: () => reject(new Error('Request aborted')),
      });
    });
  }

  // ------------------------------------------------------------------- shadow UI host

  const CSS = `
.ui{all:initial;display:block;font:14px/1.45 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:#fff;
  --bg:#1a1a1a;--card:#2a2a2a;--border:#444;--muted:#888;--primary:#0d6efd;--success:#198754;--danger:#dc3545}
.ui *,.ui *::before,.ui *::after{box-sizing:border-box}
:where(.ui button){font:inherit}
:where(.ui a){color:var(--primary)}

/* context menu */
.ctx{position:fixed;z-index:3;min-width:240px;background:#202225;border:1px solid var(--border);border-radius:8px;
  box-shadow:0 8px 24px rgba(0,0,0,.55);padding:4px}
.ctx .title{padding:6px 10px 4px;font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
.ctx button{display:flex;flex-direction:column;align-items:flex-start;width:100%;text-align:left;background:none;border:0;
  color:#fff;padding:7px 10px;border-radius:5px;cursor:pointer}
.ctx button small{color:#aaa;font-size:11px}
.ctx button:hover:not(:disabled){background:var(--primary)}
.ctx button:hover:not(:disabled) small{color:#dde}
.ctx button:disabled{color:#777;cursor:default}
.ctx button:disabled small{color:#666}
.ctx hr{border:0;border-top:1px solid var(--border);margin:4px 0}

/* modal (results + settings) */
.modal{position:fixed;inset:0;z-index:2;display:flex;align-items:center;justify-content:center}
.backdrop{position:absolute;inset:0;background:rgba(0,0,0,.7)}
.content{position:relative;background:var(--bg);border-radius:8px;width:90%;max-width:700px;max-height:85vh;overflow:hidden;
  display:flex;flex-direction:column;box-shadow:0 8px 32px rgba(0,0,0,.5)}
.header{display:flex;align-items:center;justify-content:space-between;padding:16px 20px;border-bottom:1px solid #333}
.header h3{margin:0;font-size:18px;font-weight:600;color:#fff}
.close{background:none;border:none;font-size:24px;color:var(--muted);cursor:pointer;padding:0;line-height:1}
.close:hover{color:#fff}
.body{padding:20px;overflow-y:auto;flex:1}

.loading{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:40px;gap:16px}
.spinner{width:40px;height:40px;border:3px solid #333;border-top-color:var(--primary);border-radius:50%;animation:spin 1s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.loading-text{margin:0;color:#fff;font-weight:500}
.loading-detail{margin:0;color:var(--muted);font-size:13px}

.summary{color:var(--muted);font-size:14px;margin:0 0 20px;padding-bottom:12px;border-bottom:1px solid #333}
.summary strong{color:#fff}
.preview{position:relative;display:inline-block;max-width:100%;margin:0 0 16px;border-radius:6px;overflow:hidden;background:#000;line-height:0}
.preview img{display:block;max-width:100%;max-height:170px}
.fbox{position:absolute;border:2px solid #ffc107;border-radius:2px}
.fbox.matched{border-color:#198754}
.fbox span{position:absolute;left:-2px;top:-18px;background:#198754;color:#fff;font:600 11px/16px sans-serif;padding:0 5px;border-radius:3px 3px 3px 0}
.no-match{color:var(--muted);font-style:italic;margin:0}

.person{background:var(--card);border-radius:8px;padding:16px;margin-bottom:16px}
.person:last-child{margin-bottom:0}
.person-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
.person-label{font-weight:600;color:#fff}
.match{display:flex;gap:16px}
.match-image{flex-shrink:0;width:100px;height:130px;border-radius:4px;overflow:hidden;background:var(--bg)}
.match-image.landscape{width:130px;height:100px}
.match-image img{width:100%;height:100%;object-fit:cover;object-position:top;display:block}
.no-image{width:100%;height:100%;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:12px}
.match-info{flex:1;min-width:0}
.match-info h4{margin:0 0 8px;font-size:16px;font-weight:600;color:#fff}
.disamb{color:var(--muted);font-weight:normal}
.orig-name{font-size:12px;font-style:italic;color:var(--muted);margin:-4px 0 6px}
.confidence{display:inline-block;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600;margin-bottom:8px}
.confidence.high{background:rgba(25,135,84,.2);color:#198754}
.confidence.medium{background:rgba(255,193,7,.2);color:#ffc107}
.confidence.low{background:rgba(220,53,69,.2);color:#dc3545}
.country{font-size:13px;color:var(--muted);margin-bottom:8px}
.links{display:flex;flex-wrap:wrap;gap:4px 10px;margin-bottom:4px}
.link{font-size:13px;color:var(--primary);text-decoration:none}
.link:hover{text-decoration:underline}
.link-disabled{font-size:13px;color:var(--muted);font-style:italic}
.badge{display:inline-block;margin-left:6px;padding:1px 6px;border-radius:4px;font-size:11px;background:#3a3a3a;color:#bbb;vertical-align:middle}
details.others{margin-top:12px;font-size:13px}
details.others summary{cursor:pointer;color:var(--muted);user-select:none}
details.others summary:hover{color:#fff}
details.others ul{margin:8px 0 0;padding:0;list-style:none}
.alt{display:block;padding:8px;border:1px solid var(--border);border-radius:6px;background:rgba(255,255,255,.02);margin-bottom:10px}
.alt .match-image{width:70px;height:90px}
.alt .match-image.landscape{width:90px;height:70px}
.alt .match-info h4{font-size:14px;margin-bottom:4px}

.error{text-align:center;padding:20px}
.error-icon{color:#666;margin-bottom:16px}
.error-title{color:#fff;font-size:18px;font-weight:600;margin:0 0 8px}
.error-message{color:var(--danger);font-family:monospace;font-size:13px;background:rgba(220,53,69,.1);padding:8px 12px;border-radius:4px;margin:12px 0;word-break:break-word}
.error-details{margin:12px 0;text-align:left}.error-details summary{cursor:pointer;color:var(--muted);font-size:12px;text-align:center}
.error-hint{color:var(--muted);font-size:13px;margin:0 0 12px;line-height:1.5}

.btn{padding:6px 12px;border-radius:4px;font-size:13px;font-weight:500;cursor:pointer;border:none;background:#6c757d;color:#fff;transition:filter .15s}
.btn:hover:not(:disabled){filter:brightness(1.15)}
.btn:disabled{opacity:.6;cursor:not-allowed}
.btn.primary{background:var(--primary)}

/* settings */
.field{margin-bottom:14px}
.field label{display:block;font-weight:600;margin-bottom:4px}
.field .hint{color:var(--muted);font-size:12px;margin-top:3px}
.field input,.field select{width:100%;padding:7px 9px;background:var(--card);color:#fff;border:1px solid var(--border);border-radius:4px;font:inherit}
.field input:focus,.field select:focus{outline:none;border-color:var(--primary)}
.actions{display:flex;gap:8px;align-items:center;justify-content:flex-end;margin-top:6px;flex-wrap:wrap}
.test-result{flex:1;font-size:12px;color:var(--muted);min-width:140px}
.test-result.ok{color:#2ea36b}
.test-result.bad{color:var(--danger)}

/* area selection */
.sel-layer{position:fixed;inset:0;z-index:1;cursor:crosshair;background:rgba(0,0,0,.28);touch-action:none;user-select:none;-webkit-user-select:none}
.sel-layer.has-box{background:transparent}
.sel-hint{position:absolute;top:12px;left:50%;transform:translateX(-50%);max-width:92vw;text-align:center;background:rgba(20,20,20,.94);
  padding:8px 14px;border-radius:6px;font-size:13px;pointer-events:none;box-shadow:0 2px 10px rgba(0,0,0,.4)}
.sel-box{position:absolute;border:2px solid var(--primary);box-shadow:0 0 0 9999px rgba(0,0,0,.5);cursor:move}
.sel-handle{position:absolute;width:12px;height:12px;background:#fff;border:2px solid var(--primary);border-radius:2px}
.h-nw{left:-7px;top:-7px;cursor:nwse-resize}.h-n{left:calc(50% - 6px);top:-7px;cursor:ns-resize}
.h-ne{right:-7px;top:-7px;cursor:nesw-resize}.h-e{right:-7px;top:calc(50% - 6px);cursor:ew-resize}
.h-se{right:-7px;bottom:-7px;cursor:nwse-resize}.h-s{left:calc(50% - 6px);bottom:-7px;cursor:ns-resize}
.h-sw{left:-7px;bottom:-7px;cursor:nesw-resize}.h-w{left:-7px;top:calc(50% - 6px);cursor:ew-resize}
.sel-toolbar{position:absolute;display:flex;gap:8px;cursor:default}
.sel-toolbar .btn{padding:7px 14px;font-size:14px;box-shadow:0 2px 8px rgba(0,0,0,.4)}
`;

  let host = null;
  let ui = null;

  // A modal <dialog> (page lightboxes) lives in the browser's top layer and
  // makes everything outside it inert, so UI appended to <html> would sit
  // behind it and be unclickable. Put the host inside the open modal (or the
  // fullscreen element) instead; otherwise on <html>.
  function hostParent() {
    let modal = null;
    try { modal = document.querySelector('dialog:modal'); } catch (e) { modal = document.querySelector('dialog[open]'); }
    if (modal) return modal;
    const fs = document.fullscreenElement;
    if (fs && !/^(VIDEO|IMG|CANVAS|IFRAME)$/.test(fs.tagName)) return fs;
    return document.documentElement;
  }

  function ensureUi() {
    if (host && host.isConnected) {
      const parent = hostParent();
      if (host.parentNode !== parent) parent.appendChild(host);
      return ui;
    }
    host = document.createElement('div');
    host.id = 'ssvm-host';
    host.style.cssText = 'display:block;position:fixed;top:0;left:0;width:0;height:0;margin:0;padding:0;border:0;z-index:2147483647;';
    const shadow = host.attachShadow({ mode: 'open' });
    // Constructable stylesheet first: a page CSP without 'unsafe-inline'
    // blocks a <style> element, but not this.
    try {
      const sheet = new CSSStyleSheet();
      sheet.replaceSync(CSS);
      shadow.adoptedStyleSheets = [sheet];
    } catch (e) {
      shadow.appendChild(h('style', {}, CSS));
    }
    ui = h('div', { class: 'ui' });
    shadow.appendChild(ui);
    // Our UI is inside the page's DOM tree now, so page handlers on ancestors
    // (e.g. a lightbox that closes on any click inside it) would see our
    // clicks. Our own listeners are deeper in the path and still run.
    for (const type of ['click', 'mousedown', 'mouseup', 'pointerdown', 'pointerup', 'dblclick', 'wheel']) {
      host.addEventListener(type, (e) => e.stopPropagation(), { passive: true });
    }
    hostParent().appendChild(host);
    return ui;
  }

  // Run fn(close) -- registers an Escape handler (capture, so the page's own
  // Escape handling, e.g. closing a lightbox, doesn't also fire) that is
  // removed again by the returned cleanup.
  function onEscape(fn) {
    const handler = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopImmediatePropagation();
        fn();
      }
    };
    window.addEventListener('keydown', handler, true);
    return () => window.removeEventListener('keydown', handler, true);
  }

  function modalShell(title, extraClass) {
    const root = ensureUi();
    root.querySelectorAll('.modal.' + extraClass).forEach((m) => m._close?.());
    const modal = h('div', { class: 'modal ' + extraClass });
    const backdrop = h('div', { class: 'backdrop' });
    const closeBtn = h('button', { class: 'close', 'aria-label': 'Close' }, '×');
    const body = h('div', { class: 'body' });
    const content = h('div', { class: 'content' },
      h('div', { class: 'header' }, h('h3', {}, title), closeBtn), body);
    modal.append(backdrop, content);
    root.appendChild(modal);
    const offEsc = onEscape(() => modal._close());
    const cleanups = [];
    modal._close = () => {
      offEsc();
      cleanups.forEach((fn) => { try { fn(); } catch (e) { /* ignore */ } });
      modal.remove();
      modal._onClose?.();
    };
    closeBtn.addEventListener('click', () => modal._close());
    backdrop.addEventListener('click', () => modal._close());
    return { modal, body, cleanups };
  }

  // ------------------------------------------------------------- media lookup

  // Elements under a viewport point, topmost first, including content inside
  // open shadow roots (web-component galleries).
  function deepElementsFromPoint(x, y) {
    const out = [];
    const seen = new Set();
    const visit = (rootNode) => {
      const els = rootNode.elementsFromPoint ? rootNode.elementsFromPoint(x, y) : [];
      for (const el of els) {
        if (el === host || seen.has(el)) continue;
        seen.add(el);
        if (el.shadowRoot) visit(el.shadowRoot);
        out.push(el);
      }
    };
    visit(document);
    return out;
  }

  function mediaOf(el) {
    if (!(el instanceof Element)) return null;
    const tag = el.tagName;
    if (tag === 'IMG' || tag === 'VIDEO' || tag === 'CANVAS') {
      const r = el.getBoundingClientRect();
      if (r.width < MIN_MEDIA_PX || r.height < MIN_MEDIA_PX) return null;
      const cs = getComputedStyle(el);
      if (cs.visibility === 'hidden' || cs.display === 'none') return null;
      if (tag === 'IMG') {
        if (!el.complete || !el.naturalWidth) return null;
        const url = el.currentSrc || el.src;
        return url ? { kind: 'img', el, url } : null;
      }
      if (tag === 'VIDEO') return el.videoWidth ? { kind: 'video', el } : null;
      return el.width && el.height ? { kind: 'canvas', el } : null;
    }
    const r = el.getBoundingClientRect();
    if (r.width < MIN_MEDIA_PX || r.height < MIN_MEDIA_PX) return null;
    const bg = getComputedStyle(el).backgroundImage;
    if (bg && bg !== 'none') {
      const m = bg.match(/url\((['"]?)(.*?)\1\)/);
      if (m && m[2]) return { kind: 'bg', el, url: m[2] };
    }
    return null;
  }

  function addCandidates(list, seen, els) {
    for (const el of els) {
      if (seen.has(el)) continue;
      seen.add(el);
      const m = mediaOf(el);
      if (m) list.push(m);
    }
  }

  // Candidates at a point, topmost first. elementsFromPoint skips
  // pointer-events:none elements, so img/video/canvas under such an overlay
  // are added from a plain scan afterwards.
  function mediaCandidatesAt(x, y) {
    const list = [];
    const seen = new Set();
    addCandidates(list, seen, deepElementsFromPoint(x, y));
    const scanned = [...document.querySelectorAll('img,video,canvas')].filter((el) => {
      const r = el.getBoundingClientRect();
      return x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;
    });
    addCandidates(list, seen, scanned);
    return list;
  }

  function mediaCandidatesIn(rect) {
    const list = [];
    const seen = new Set();
    const N = 5;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const x = rect.x + (rect.w * (i + 0.5)) / N;
        const y = rect.y + (rect.h * (j + 0.5)) / N;
        addCandidates(list, seen, deepElementsFromPoint(x, y));
      }
    }
    const scanned = [...document.querySelectorAll('img,video,canvas')].filter((el) => {
      const r = el.getBoundingClientRect();
      return r.right > rect.x && r.left < rect.x + rect.w && r.bottom > rect.y && r.top < rect.y + rect.h;
    });
    addCandidates(list, seen, scanned);
    return list;
  }

  // ---------------------------------------------------------------- geometry

  function intersect(a, b) {
    const x = Math.max(a.x, b.x);
    const y = Math.max(a.y, b.y);
    const r = Math.min(a.x + a.w, b.x + b.w);
    const btm = Math.min(a.y + a.h, b.y + b.h);
    return r > x && btm > y ? { x, y, w: r - x, h: btm - y } : null;
  }

  // Box of an element in viewport coordinates with border (and optionally
  // padding) removed. Accounts for CSS transform scale (zoomed lightboxes) by
  // comparing the rendered rect with the untransformed layout size.
  function elementBox(el, includePadding) {
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    const px = (v) => parseFloat(v) || 0;
    const sx = el.offsetWidth ? r.width / el.offsetWidth : 1;
    const sy = el.offsetHeight ? r.height / el.offsetHeight : 1;
    const l = px(cs.borderLeftWidth) + (includePadding ? 0 : px(cs.paddingLeft));
    const t = px(cs.borderTopWidth) + (includePadding ? 0 : px(cs.paddingTop));
    const rr = px(cs.borderRightWidth) + (includePadding ? 0 : px(cs.paddingRight));
    const b = px(cs.borderBottomWidth) + (includePadding ? 0 : px(cs.paddingBottom));
    return { x: r.left + l * sx, y: r.top + t * sy, w: r.width - (l + rr) * sx, h: r.height - (t + b) * sy, sx, sy };
  }

  function posOffset(token, free, scale) {
    if (!token) return free / 2;
    if (token.endsWith('%')) return (free * parseFloat(token)) / 100;
    return (parseFloat(token) || 0) * scale;
  }

  // Where the media's pixels actually land on screen (`draw`, may overflow its
  // element for object-fit: cover) and the part of that which is visible
  // (`clip`).
  function drawnRect(media, natW, natH) {
    const el = media.el;
    const cs = getComputedStyle(el);

    if (media.kind === 'bg') {
      const firstLayer = (v) => v.split(/,(?![^()]*\))/)[0].trim();
      const box = elementBox(el, true);
      const size = firstLayer(cs.backgroundSize);
      let dw;
      let dh;
      if (size === 'cover') {
        const s = Math.max(box.w / natW, box.h / natH);
        dw = natW * s; dh = natH * s;
      } else if (size === 'contain') {
        const s = Math.min(box.w / natW, box.h / natH);
        dw = natW * s; dh = natH * s;
      } else {
        const [wt, ht = 'auto'] = size.split(/\s+/);
        const conv = (t, whole, scale) => (t === 'auto' ? undefined : t.endsWith('%') ? (whole * parseFloat(t)) / 100 : parseFloat(t) * scale);
        dw = conv(wt, box.w, box.sx);
        dh = conv(ht, box.h, box.sy);
        if (dw === undefined && dh === undefined) { dw = natW * box.sx; dh = natH * box.sy; }
        else if (dw === undefined) dw = (dh * natW) / natH;
        else if (dh === undefined) dh = (dw * natH) / natW;
      }
      const [pxTok, pyTok] = firstLayer(cs.backgroundPosition).split(/\s+/);
      return {
        draw: { x: box.x + posOffset(pxTok, box.w - dw, box.sx), y: box.y + posOffset(pyTok, box.h - dh, box.sy), w: dw, h: dh },
        clip: box,
      };
    }

    const box = elementBox(el, false);
    const fit = cs.objectFit || 'fill';
    const contain = Math.min(box.w / natW, box.h / natH);
    const cover = Math.max(box.w / natW, box.h / natH);
    let dw;
    let dh;
    if (fit === 'contain') { dw = natW * contain; dh = natH * contain; }
    else if (fit === 'cover') { dw = natW * cover; dh = natH * cover; }
    else if (fit === 'none') { dw = natW * box.sx; dh = natH * box.sy; }
    else if (fit === 'scale-down') { const s = Math.min(box.sx, contain); dw = natW * s; dh = natH * s; }
    else { dw = box.w; dh = box.h; }
    const [pxTok, pyTok] = (cs.objectPosition || '50% 50%').split(/\s+/);
    return {
      draw: { x: box.x + posOffset(pxTok, box.w - dw, box.sx), y: box.y + posOffset(pyTok, box.h - dh, box.sy), w: dw, h: dh },
      clip: box,
    };
  }

  // -------------------------------------------------------------- image data

  async function gmBlob(url, headers, extra = {}) {
    const res = await gm({ method: 'GET', url, responseType: 'blob', timeout: 30000, headers, ...extra });
    if (res.status < 200 || res.status >= 300) {
      const mitigated = /cf-mitigated:\s*(\S+)/i.exec(res.responseHeaders || '');
      throw new Error(`HTTP ${res.status}${mitigated ? ` (cloudflare ${mitigated[1]})` : ''}`);
    }
    const blob = res.response;
    if (!blob || !blob.size) throw new Error('empty response');
    if (/html|json|text\//i.test(blob.type)) throw new Error(`got ${blob.type} instead of an image (bot protection?)`);
    return blob;
  }

  // Several ways to get the image bytes; hotlink/bot protection (Cloudflare
  // etc.) rejects some of them. All failures are reported, not just the last.
  // Not fetch(): a page CSP connect-src without data: blocks fetching data URLs.
  function dataUrlToBlob(dataUrl) {
    const comma = dataUrl.indexOf(',');
    const type = /^data:([^;,]+)/.exec(dataUrl)?.[1] || 'image/jpeg';
    const bin = atob(dataUrl.slice(comma + 1));
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Blob([bytes], { type });
  }

  // Last-resort download: open the image URL in a background tab (exactly what
  // "open image in new tab" does, which bot protection allows) and let this
  // script, running there, draw and return it (grabInTab).
  function grabViaTab(url) {
    return new Promise((resolve, reject) => {
      const token = Math.random().toString(36).slice(2);
      const key = `ssvm_grab_${token}`;
      let tab = null;
      let listener = null;
      let timer = null;
      const finish = (fn, value) => {
        clearTimeout(timer);
        try { GM_removeValueChangeListener(listener); } catch (e) { /* ignore */ }
        try { if (tab) tab.close(); } catch (e) { /* ignore */ }
        try { GM_deleteValue(key); } catch (e) { /* ignore */ }
        fn(value);
      };
      listener = GM_addValueChangeListener(key, (name, oldValue, newValue) => {
        if (!newValue) return;
        if (newValue.dataUrl) {
          try { finish(resolve, dataUrlToBlob(newValue.dataUrl)); } catch (e) { finish(reject, e); }
        } else {
          finish(reject, new Error(newValue.error || 'tab failed'));
        }
      });
      timer = setTimeout(() => finish(reject, new Error('timed out (the tab did not report back)')), 20000);
      tab = GM_openInTab(`${url.replace(/#.*$/, '')}#ssvm-grab=${token}`, { active: false, insert: true });
    });
  }

  async function fetchBlob(url) {
    if (/^data:/i.test(url)) return dataUrlToBlob(url);
    if (/^blob:/i.test(url)) return (await fetch(url)).blob();
    // Exactly what the browser sends for a normal <img> load; bot protection
    // (Cloudflare) can tell a script's XHR from that by these headers alone.
    const sameSite = new URL(url).hostname.split('.').slice(-2).join('.') === location.hostname.split('.').slice(-2).join('.');
    const imgHeaders = {
      Referer: location.href,
      Accept: 'image/avif,image/webp,image/png,image/svg+xml,image/*;q=0.8,*/*;q=0.5',
      'Sec-Fetch-Dest': 'image',
      'Sec-Fetch-Mode': 'no-cors',
      'Sec-Fetch-Site': sameSite ? 'same-site' : 'cross-site',
    };
    const attempts = [
      ['image-like request', () => gmBlob(url, imgHeaders)],
      ['page referer', () => gmBlob(url, { Referer: location.href })],
      ['page fetch', async () => {
        const r = await fetch(url, { credentials: 'include' });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.blob();
      }],
      ['background tab', () => grabViaTab(url)],
    ];
    // Every refused request is more traffic against a bot-protected site (and
    // can trip its rate limiting), so: remember per image host that only the
    // tab route works and start there, and after one Cloudflare refusal skip
    // the remaining direct attempts.
    const imgHost = new URL(url).hostname;
    let tabHosts = {};
    try { tabHosts = GM_getValue('ssvm_tab_hosts', {}) || {}; } catch (e) { tabHosts = {}; }
    const tabAttempt = attempts[attempts.length - 1];
    const ordered = tabHosts[imgHost] ? [tabAttempt, ...attempts.slice(0, -1)] : attempts;
    const errors = [];
    let skipDirect = false;
    for (const [label, fn] of ordered) {
      if (skipDirect && label !== tabAttempt[0]) continue;
      try {
        const blob = await fn();
        if (label === tabAttempt[0] && !tabHosts[imgHost]) {
          try { GM_setValue('ssvm_tab_hosts', { ...tabHosts, [imgHost]: true }); } catch (e) { /* ignore */ }
        }
        return blob;
      } catch (e) {
        errors.push(`${label}: ${e.message}`);
        if (/cloudflare/i.test(e.message)) skipDirect = true;
      }
    }
    throw new Error(errors.join('; '));
  }

  async function decodeBlob(blob) {
    try {
      const bmp = await createImageBitmap(blob);
      return { drawable: bmp, w: bmp.width, h: bmp.height };
    } catch (e) {
      const objUrl = URL.createObjectURL(blob);
      try {
        const img = new Image();
        img.src = objUrl;
        await img.decode();
        return { drawable: img, w: img.naturalWidth, h: img.naturalHeight };
      } finally {
        URL.revokeObjectURL(objUrl);
      }
    }
  }

  // Something drawImage() accepts plus its native pixel size. Images are
  // downloaded through the userscript manager (no CORS) and decoded fresh, so
  // the canvas is never tainted; video/canvas elements are drawn directly.
  async function getSource(media) {
    if (media.kind === 'video' || media.kind === 'canvas') {
      // Snapshot now: the video resumes playing / the canvas keeps changing
      // while retries and re-renders still need this exact frame.
      const w = media.kind === 'video' ? media.el.videoWidth : media.el.width;
      const h = media.kind === 'video' ? media.el.videoHeight : media.el.height;
      const snap = document.createElement('canvas');
      snap.width = w;
      snap.height = h;
      snap.getContext('2d').drawImage(media.el, 0, 0, w, h);
      return { drawable: snap, w, h };
    }
    try {
      return await decodeBlob(await fetchBlob(media.url));
    } catch (e) {
      // Drawing the page's own <img> only helps when it is same-origin/CORS
      // clean; otherwise the canvas is tainted and cannot be exported.
      if (media.kind === 'img' && media.el.complete && media.el.naturalWidth) {
        const direct = { drawable: media.el, w: media.el.naturalWidth, h: media.el.naturalHeight };
        if (isReadable(direct)) return direct;
      }
      // A CORS-enabled reload is made by the browser itself (with the site's
      // cookies, unlike the userscript manager's request), and works whenever
      // the image host sends CORS headers.
      let corsNote = 'crossorigin reload: ok';
      try {
        const cors = await loadCorsImage(media.url);
        if (isReadable(cors)) return cors;
        corsNote = 'crossorigin reload: not readable';
      } catch (e2) {
        corsNote = `crossorigin reload: ${e2.message}`;
      }
      const err = new Error(`Could not load the image (${e.message}; ${corsNote}).`);
      err.remoteUrl = /^https?:/i.test(media.url) ? media.url : null;
      throw err;
    }
  }

  function loadCorsImage(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve({ drawable: img, w: img.naturalWidth, h: img.naturalHeight });
      img.onerror = () => reject(new Error('blocked (no CORS headers?)'));
      img.src = url;
    });
  }

  function isReadable(src) {
    try {
      const c = document.createElement('canvas');
      c.width = c.height = 1;
      const ctx = c.getContext('2d');
      ctx.drawImage(src.drawable, 0, 0, 1, 1);
      ctx.getImageData(0, 0, 1, 1);
      return true;
    } catch (e) {
      return false;
    }
  }

  const SECURITY_MSG = 'Could not encode the image: the site blocks reading its pixels (cross-origin protected).';

  function renderJpeg(src, crop) {
    const sx = crop ? Math.round(crop.x) : 0;
    const sy = crop ? Math.round(crop.y) : 0;
    const sw = crop ? Math.max(1, Math.round(crop.w)) : src.w;
    const sh = crop ? Math.max(1, Math.round(crop.h)) : src.h;
    const scale = Math.min(1, MAX_SIDE / Math.max(sw, sh));
    const cw = Math.max(1, Math.round(sw * scale));
    const ch = Math.max(1, Math.round(sh * scale));
    return new Promise((resolve, reject) => {
      const canvas = document.createElement('canvas');
      canvas.width = cw;
      canvas.height = ch;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, cw, ch);
      ctx.drawImage(src.drawable, sx, sy, sw, sh, 0, 0, cw, ch);
      try {
        canvas.toBlob((blob) => (blob ? resolve({ blob, w: cw, h: ch }) : reject(new Error(SECURITY_MSG))), 'image/jpeg', JPEG_QUALITY);
      } catch (e) {
        reject(new Error(e.name === 'SecurityError' ? SECURITY_MSG : e.message));
      }
    });
  }

  // Crop `crop` (native pixels) out of src with `marginPct` of its size added
  // on every side. Margin beyond the image edge is filled with neutral gray
  // rather than clamped, so the face keeps the same share of the picture
  // wherever it is (a clamped crop at an image edge leaves the face too large
  // for the detector).
  function renderPadded(src, crop, marginPct, mime = 'image/jpeg') {
    const m = Math.max(0, marginPct) / 100;
    const wantX = crop.x - crop.w * m;
    const wantY = crop.y - crop.h * m;
    const wantW = crop.w * (1 + 2 * m);
    const wantH = crop.h * (1 + 2 * m);
    const scale = Math.min(1, MAX_SIDE / Math.max(wantW, wantH));
    const cw = Math.max(1, Math.round(wantW * scale));
    const ch = Math.max(1, Math.round(wantH * scale));
    const ix0 = Math.max(0, wantX);
    const iy0 = Math.max(0, wantY);
    const ix1 = Math.min(src.w, wantX + wantW);
    const iy1 = Math.min(src.h, wantY + wantH);
    return new Promise((resolve, reject) => {
      const canvas = document.createElement('canvas');
      canvas.width = cw;
      canvas.height = ch;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#808080';
      ctx.fillRect(0, 0, cw, ch);
      ctx.drawImage(src.drawable, ix0, iy0, ix1 - ix0, iy1 - iy0,
        (ix0 - wantX) * scale, (iy0 - wantY) * scale, (ix1 - ix0) * scale, (iy1 - iy0) * scale);
      try {
        canvas.toBlob((blob) => (blob ? resolve({ blob, w: cw, h: ch }) : reject(new Error(SECURITY_MSG))), mime, JPEG_QUALITY);
      } catch (e) {
        reject(new Error(e.name === 'SecurityError' ? SECURITY_MSG : e.message));
      }
    });
  }

  function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onload = () => resolve(String(fr.result).split(',')[1]);
      fr.onerror = () => reject(new Error('Could not read image data'));
      fr.readAsDataURL(blob);
    });
  }

  // prepare*() return { what, variants }: variants are tried in order until one
  // yields a detected face.
  async function prepareImage(media) {
    let src;
    try {
      src = await getSource(media);
    } catch (e) {
      // The browser can't read this image (bot/hotlink protection, CORS): as a
      // last resort let the sidecar download the URL itself.
      if (e.remoteUrl) {
        return { what: 'image', variants: [async () => ({ remote: e.remoteUrl })], remoteReason: e.message };
      }
      throw e;
    }
    // Pictures that are already a tight face crop (review-tool thumbnails,
    // headshots) make the detector find nothing -- the face fills the whole
    // frame -- so after the plain image come versions with gray margin added.
    const whole = { x: 0, y: 0, w: src.w, h: src.h };
    return {
      what: media.kind === 'video' ? 'video frame' : 'image',
      variants: [
        async () => renderJpeg(src, null),
        ...[50, 100, 200].map((m) => async () => ({ ...(await renderPadded(src, whole, m)), margin: m })),
      ],
    };
  }

  function visibleArea(media, rect) {
    const i = intersect(rect, elementBox(media.el, media.kind === 'bg'));
    return i ? i.w * i.h : 0;
  }

  // The part of the media under `rect`, as { src, crop } in native pixels.
  async function locateArea(rect) {
    const cands = mediaCandidatesIn(rect).map((m) => ({ m, area: visibleArea(m, rect) })).filter((c) => c.area > 0);
    if (!cands.length) throw new Error('No image or video found under the selected area.');
    // Real images/videos/canvases beat CSS backgrounds (page or container
    // backgrounds usually sit behind the photo being pointed at). Among those,
    // the one the box overlaps most; near-ties go to the topmost.
    const real = cands.filter((c) => c.m.kind !== 'bg');
    const pool = real.length ? real : cands;
    const best = Math.max(...pool.map((c) => c.area));
    const media = pool.find((c) => c.area >= best * 0.9).m;

    const src = await getSource(media);
    const { draw, clip } = drawnRect(media, src.w, src.h);
    const vis = intersect(intersect(rect, draw) || { x: 0, y: 0, w: 0, h: 0 }, clip);
    if (!vis) throw new Error('The selected area does not cover any part of the image.');

    const crop = {
      x: ((vis.x - draw.x) / draw.w) * src.w,
      y: ((vis.y - draw.y) / draw.h) * src.h,
      w: (vis.w / draw.w) * src.w,
      h: (vis.h / draw.h) * src.h,
    };
    if (crop.w < MIN_CROP_PX || crop.h < MIN_CROP_PX) {
      throw new Error('The selected area is too small at the image\'s native resolution.');
    }
    return { src, crop };
  }

  async function prepareArea(rect, cfg) {
    const { src, crop } = await locateArea(rect);
    // Hidden margin: the selection is usually tight around the face, but the
    // detector needs surrounding context to find it (see renderPadded).
    // Ladder: configured margin, then wider twice, then narrower (a loosely
    // drawn box around the whole head may do better with less).
    const base = clamp(cfg.marginPct, 0, MAX_MARGIN_PCT);
    const ladder = [...new Set([
      base,
      Math.min(MAX_MARGIN_PCT, base * 2 || 100),
      Math.min(MAX_MARGIN_PCT, base * 4 || 200),
      Math.round(base * 0.4),
    ])];
    return {
      what: 'selected area',
      variants: ladder.map((m) => async () => ({ ...(await renderPadded(src, crop, m)), margin: m })),
    };
  }

  // -------------------------------------------------------------- sidecar API

  async function callIdentify(base64, cfg, imageUrl) {
    const res = await gm({
      method: 'POST',
      url: `${cfg.sidecarUrl}/identify`,
      headers: { 'Content-Type': 'application/json' },
      data: JSON.stringify(imageUrl ? { image_url: imageUrl, top_k: cfg.topK } : { image_base64: base64, top_k: cfg.topK }),
      responseType: 'text',
      timeout: 90000,
    });
    if (res.status < 200 || res.status >= 300) {
      let detail = res.responseText || `HTTP ${res.status}`;
      try {
        const d = JSON.parse(res.responseText).detail;
        if (d) detail = typeof d === 'string' ? d : JSON.stringify(d);
      } catch (e) { /* keep raw text */ }
      throw new Error(`Identification failed: ${detail}`);
    }
    return JSON.parse(res.responseText);
  }

  // Mirrors the plugin: the sidecar lazily (re)loads its models after idle, so
  // poll /health to tell the user why the first request is slow.
  function pollHealth(cfg, onProgress) {
    let stopped = false;
    let shown = false;
    const tick = async () => {
      if (stopped) return;
      try {
        const r = await gm({ method: 'GET', url: `${cfg.sidecarUrl}/health`, timeout: 5000 });
        if (stopped) return;
        const hl = JSON.parse(r.responseText);
        if (hl.face_recognition_loading) {
          shown = true;
          onProgress('Loading face recognition models (first use after idle)...');
        } else if (shown) {
          shown = false;
          onProgress('Identifying performers...');
        }
      } catch (e) { /* best-effort feedback only */ }
    };
    tick();
    const id = setInterval(tick, 700);
    return () => { stopped = true; clearInterval(id); };
  }

  // ----------------------------------------------------------- result display

  function thumbUrl(url) {
    try {
      const p = new URL(url);
      const hostName = p.hostname.replace(/^www\./, '');
      if (hostName === 'stashdb.org' && p.pathname.startsWith('/images/')) {
        p.searchParams.set('size', '600');
        return p.toString();
      }
      if (hostName === 'media.seekfans.com') {
        return `https://seekfans.com/_next/image?url=${encodeURIComponent(url)}&w=256&q=75`;
      }
    } catch (e) { /* fall through */ }
    return url;
  }

  // Match photos are plain <img> hotlinks; a page CSP can block them, in
  // which case they are fetched through the userscript manager instead.
  function thumbnail(url, alt) {
    if (!url) return h('div', { class: 'no-image' }, 'No image');
    const img = h('img', { alt: alt || '', loading: 'lazy', referrerpolicy: 'no-referrer' });
    const wrap = h('div', { class: 'match-image' }, img);
    img.addEventListener('load', () => {
      if (img.naturalWidth > img.naturalHeight) wrap.classList.add('landscape');
    });
    let retried = false;
    img.addEventListener('error', async () => {
      if (retried) { wrap.replaceChildren(h('div', { class: 'no-image' }, 'No image')); return; }
      retried = true;
      try {
        const blob = await fetchBlob(thumbUrl(url));
        img.src = await new Promise((res, rej) => {
          const fr = new FileReader();
          fr.onload = () => res(fr.result);
          fr.onerror = rej;
          fr.readAsDataURL(blob);
        });
      } catch (e) {
        wrap.replaceChildren(h('div', { class: 'no-image' }, 'No image'));
      }
    });
    img.src = thumbUrl(url);
    return wrap;
  }

  function matchLinks(m, cfg) {
    const endpoint = m.endpoint || 'stashdb.org';
    if (m.source) {
      const href = m.profile_url || m.catalogue_url;
      if (!href) return [h('span', { class: 'link-disabled' }, `Source: ${m.source}`)];
      let label = `View on ${m.source}`;
      if (m.profile_url) {
        try { label = `View on ${new URL(m.profile_url).hostname.replace(/^www\./, '')}`; } catch (e) { label = 'View profile'; }
      }
      return [link(href, label)];
    }
    if (!m.local_performer_id) {
      if (!String(endpoint).includes('.')) return [h('span', { class: 'link-disabled' }, `Source: ${endpoint}`)];
      return [link(`https://${endpoint}/performers/${m.stashdb_id}`, `View on ${endpoint}`)];
    }
    const links = [];
    if (cfg.stashUrl) {
      links.push(link(`${cfg.stashUrl}/performers/${m.local_performer_id}`, 'View local performer'));
    } else {
      links.push(h('span', { class: 'link-disabled' }, 'In your library (set the Stash URL to link it)'));
    }
    if (m.stashdb_id && m.stashdb_id !== m.local_performer_id) {
      links.unshift(link(`https://stashdb.org/performers/${m.stashdb_id}`, 'View on stashdb.org'));
    }
    if (m.profile_url) {
      let label = 'View on source';
      try { label = `View on ${new URL(m.profile_url).hostname.replace(/^www\./, '')}`; } catch (e) { /* generic */ }
      links.unshift(link(m.profile_url, label));
    }
    return links;
  }

  function matchView(m, cfg) {
    const conf = Math.round((1 - clamp(m.distance, 0, 1)) * 100);
    const cls = conf >= 70 ? 'high' : conf >= 50 ? 'medium' : 'low';
    return h('div', { class: 'match' },
      thumbnail(m.image_url, m.name),
      h('div', { class: 'match-info' },
        h('h4', {}, m.name, m.disambiguation ? h('span', { class: 'disamb' }, ` (${m.disambiguation})`) : null, m.local_performer_id ? h('span', { class: 'badge' }, 'in your library') : null),
        m.original_name ? h('div', { class: 'orig-name' }, `aka ${m.original_name}`) : null,
        h('div', { class: `confidence ${cls}` }, `${conf}% match`),
        m.country ? h('div', { class: 'country' }, m.country) : null,
        h('div', { class: 'links' }, matchLinks(m, cfg))));
  }

  const ICON_INFO = 'M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z';
  const ICON_ERROR = 'M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z';

  // Built with DOM calls, not innerHTML: Trusted Types pages reject the latter.
  function iconBox(pathData) {
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('width', '48');
    svg.setAttribute('height', '48');
    svg.setAttribute('fill', 'currentColor');
    const path = document.createElementNS(ns, 'path');
    path.setAttribute('d', pathData);
    svg.appendChild(path);
    return h('div', { class: 'error-icon' }, svg);
  }

  function openResultModal() {
    const { modal, body, cleanups } = modalShell(`${NAME} Results`, 'result');
    const loadingText = h('p', { class: 'loading-text' }, `Connecting to ${NAME}...`);
    const loadingDetail = h('p', { class: 'loading-detail' });
    const loading = h('div', { class: 'loading' }, h('div', { class: 'spinner' }), loadingText, loadingDetail);
    const results = h('div', { style: 'display:none' });
    const error = h('div', { class: 'error', style: 'display:none' });
    body.append(loading, results, error);

    return {
      setLoading(text, detail = '') {
        loadingText.textContent = text;
        loadingDetail.textContent = detail;
      },

      showError(message, override) {
        loading.style.display = 'none';
        results.style.display = 'none';
        let title = 'Analysis Failed';
        let hint = `Check the settings and make sure ${NAME} is running.`;
        let detail = null;
        if (/Could not load the image|Could not encode/.test(message)) {
          title = 'This site blocks access to its images';
          hint = 'The site\'s bot protection (e.g. Cloudflare) refuses downloads by scripts, the sidecar could not fetch it either, '
            + 'and browsers do not allow reading the pixels of a cross-origin image. It works on sites that serve images normally.';
          detail = message;
        } else if (/No image or video|does not cover|too small/.test(message)) {
          title = 'Nothing to analyze';
          hint = 'Draw the area over an image or video (not over text or empty page background).';
        } else if (/connect|Connection/i.test(message)) {
          title = 'Connection Failed';
          hint = `Could not connect to the ${NAME} sidecar. Check the API URL in the settings and make sure the container is running.`;
        } else if (/timed out|timeout/i.test(message)) {
          title = 'Request Timed Out';
          hint = 'The sidecar took too long to answer.';
        }
        if (override) { title = override.title; hint = override.hint; detail = null; }
        error.replaceChildren(
          iconBox(ICON_ERROR),
          h('p', { class: 'error-title' }, title),
          detail
            ? h('details', { class: 'error-details' }, h('summary', {}, 'Technical details'), h('p', { class: 'error-message' }, detail))
            : h('p', { class: 'error-message' }, message),
          h('p', { class: 'error-hint' }, hint),
          h('button', { class: 'btn', onclick: () => { modal._close(); openSettings(); } }, 'Open settings'));
        error.style.display = 'block';
      },

      showResults(data, prepared, cfg) {
        loading.style.display = 'none';
        const faces = data.faces || [];
        if (!faces.length) {
          error.replaceChildren(
            iconBox(ICON_INFO),
            h('p', { class: 'error-title' }, 'No faces detected'),
            h('p', { class: 'error-hint' }, prepared.what === 'selected area'
              ? 'No face was found in the selected area. Try an area that covers the whole head.'
              : 'The image may not contain clear face shots.'));
          error.style.display = 'block';
          return;
        }

        const matched = faces.filter((f) => f.matches && f.matches.length);
        // No preview when the sidecar downloaded the image itself.
        let preview = null;
        if (prepared.blob) {
          const objUrl = URL.createObjectURL(prepared.blob);
          cleanups.push(() => URL.revokeObjectURL(objUrl));
          preview = h('div', { class: 'preview', title: `What was sent to ${NAME} (${prepared.w}×${prepared.h})` },
            h('img', { src: objUrl, alt: '', onerror: (e) => { e.target.parentElement.style.display = 'none'; } }));
        }
        if (preview) faces.forEach((f) => {
          const idx = matched.indexOf(f);
          const b = f.box;
          const box = h('div', {
            class: 'fbox' + (idx >= 0 ? ' matched' : ''),
            style: `left:${(b.x / prepared.w) * 100}%;top:${(b.y / prepared.h) * 100}%;width:${(b.width / prepared.w) * 100}%;height:${(b.height / prepared.h) * 100}%`,
          }, idx >= 0 ? h('span', {}, idx + 1) : null);
          preview.appendChild(box);
        });

        const persons = h('div', {});
        matched.forEach((face, i) => {
          const card = h('div', { class: 'person' },
            h('div', { class: 'person-header' }, h('span', { class: 'person-label' }, `Face ${i + 1}`)),
            matchView(face.matches[0], cfg));
          if (face.matches.length > 1) {
            const list = h('ul', {}, face.matches.slice(1).map((m) => h('li', { class: 'alt' }, matchView(m, cfg))));
            card.appendChild(h('details', { class: 'others' },
              h('summary', {}, `Other possible matches (${face.matches.length - 1})`), list));
          }
          persons.appendChild(card);
        });

        results.replaceChildren(...[
          h('p', { class: 'summary' }, 'Detected ', h('strong', {}, data.face_count ?? faces.length), ` face(s) in ${prepared.what}.`,
            prepared.remote ? h('span', { class: 'badge', title: prepared.remoteReason || '' }, 'downloaded by the sidecar') : null),
          preview,
          matched.length ? null : h('p', { class: 'no-match' }, 'No matches found in database'),
          persons,
        ].filter(Boolean));
        results.style.display = 'block';
      },
      close: () => modal._close(),
    };
  }

  // ----------------------------------------------------------------- settings

  function openSettings(opts = {}) {
    return new Promise((resolve) => {
      const cfg = loadConfig();
      const { modal, body } = modalShell(`${NAME} - userscript settings`, 'settings');
      modal._onClose = () => resolve();

      const sidecar = h('input', { type: 'text', value: cfg.sidecarUrl, placeholder: 'http://192.168.1.100:6961', spellcheck: 'false' });
      const stash = h('input', { type: 'text', value: cfg.stashUrl, placeholder: 'http://192.168.1.100:9999 (optional)', spellcheck: 'false' });
      const modifier = h('select', {},
        [['ctrl', 'Ctrl + right-click'], ['alt', 'Alt + right-click'], ['shift', 'Shift + right-click'], ['none', 'Plain right-click (replaces the browser menu)']]
          .map(([v, label]) => h('option', { value: v, selected: v === cfg.modifier }, label)));
      const topK = h('input', { type: 'number', min: 1, max: 20, value: cfg.topK });
      const translateTo = h('input', { type: 'text', value: cfg.translateTo, placeholder: 'en', spellcheck: 'false' });
      const margin = h('input', { type: 'number', min: 0, max: MAX_MARGIN_PCT, value: cfg.marginPct });
      const testResult = h('span', { class: 'test-result' });

      const field = (label, input, hint) => h('div', { class: 'field' }, h('label', {}, label), input, hint ? h('div', { class: 'hint' }, hint) : null);

      const read = () => ({
        sidecarUrl: normalizeUrl(sidecar.value),
        stashUrl: normalizeUrl(stash.value),
        modifier: modifier.value,
        topK: clamp(parseInt(topK.value, 10) || DEFAULTS.topK, 1, 20),
        translateTo: (translateTo.value || '').trim() || DEFAULTS.translateTo,
        marginPct: clamp(Number.isNaN(parseInt(margin.value, 10)) ? DEFAULTS.marginPct : parseInt(margin.value, 10), 0, MAX_MARGIN_PCT),
      });

      const testBtn = h('button', {
        class: 'btn',
        onclick: async () => {
          const url = normalizeUrl(sidecar.value);
          testResult.className = 'test-result';
          if (!url) { testResult.textContent = 'Enter the sidecar URL first.'; testResult.classList.add('bad'); return; }
          testResult.textContent = 'Testing...';
          try {
            const r = await gm({ method: 'GET', url: `${url}/health`, timeout: 8000 });
            const hl = JSON.parse(r.responseText);
            testResult.textContent = `Connected - sidecar v${hl.version}, ${hl.performer_count?.toLocaleString?.() ?? '?'} performers${hl.database_loaded ? '' : ' (database not loaded!)'}`;
            testResult.classList.add(hl.database_loaded ? 'ok' : 'bad');
          } catch (e) {
            testResult.textContent = e.message;
            testResult.classList.add('bad');
          }
        },
      }, 'Test connection');

      body.append(...[
        opts.firstRun ? h('p', { class: 'error-hint', style: 'text-align:left' }, `Enter the URL of your ${NAME} sidecar to get started.`) : null,
        field('Sidecar URL', sidecar, 'The Stash Sense 2 API, same address as the plugin\'s "API URL" setting.'),
        field('Stash URL (optional)', stash, 'Only used to link matches that are in your library ("View local performer").'),
        field('Menu trigger', modifier, 'Hold this key while right-clicking to open the Stash Sense 2 menu. A browser\'s own menu can\'t be extended by a userscript.'),
        field('Matches per face', topK),
        field('Translate text to (language code)', translateTo, 'Target language for "Translate text in area", e.g. en, de, fr, es, ja.'),
        field('Area-select margin (%)', margin, 'Context added around a drawn area on each side, as a share of the box size (gray where the image ends). The face detector finds nothing if the face fills the picture; 100 suits a box drawn tight on the face. If no face is found it retries with wider and then narrower margins.'),
        h('div', { class: 'actions' }, testResult, testBtn,
          h('button', { class: 'btn', onclick: () => modal._close() }, 'Cancel'),
          h('button', {
            class: 'btn primary',
            onclick: () => { saveConfig(read()); modal._close(); },
          }, 'Save')),
      ].filter(Boolean));
      sidecar.focus();
    });
  }

  async function ensureConfigured() {
    let cfg = loadConfig();
    if (!cfg.sidecarUrl) {
      await openSettings({ firstRun: true });
      cfg = loadConfig();
    }
    return cfg.sidecarUrl ? cfg : null;
  }

  // ------------------------------------------------------------------- flows

  async function identifyFlow(prepare, done) {
    const cfg = await ensureConfigured();
    if (!cfg) { done?.(); return; }
    const modal = openResultModal();
    const stopPoll = pollHealth(cfg, (msg) => modal.setLoading(msg));
    try {
      modal.setLoading('Preparing image...');
      const prepared = await prepare(cfg);
      done?.();
      done = null;
      let sent = null;
      let data = null;
      for (let i = 0; i < prepared.variants.length; i++) {
        modal.setLoading(
          i === 0 ? `Analyzing ${prepared.what}...` : 'No face found - retrying with a different margin...',
          'Detecting faces');
        sent = { ...(await prepared.variants[i]()), what: prepared.what, remoteReason: prepared.remoteReason };
        if (sent.remote) {
          try {
            data = await callIdentify(null, cfg, sent.remote);
          } catch (e) {
            throw new Error(`${e.message} -- the browser could not read this image either: ${prepared.remoteReason}`);
          }
        } else {
          data = await callIdentify(await blobToBase64(sent.blob), cfg);
        }
        if ((data.faces || []).length) break;
      }
      modal.setLoading('Processing results...');
      modal.showResults(data, sent, cfg);
    } catch (e) {
      console.error(`[${NAME}]`, e);
      modal.showError(e.message || String(e));
    } finally {
      done?.();
      stopPoll();
    }
  }

  // Draw / move / resize a box over the page. Resolves { rect, release } in
  // viewport CSS pixels (release() resumes videos paused for the selection),
  // or null when cancelled.
  function selectRegion(hintText, useLabel) {
    return new Promise((resolve) => {
      const root = ensureUi();
      const paused = [...document.querySelectorAll('video')].filter((v) => !v.paused && !v.ended);
      paused.forEach((v) => v.pause());
      const release = () => paused.forEach((v) => v.play().catch(() => {}));

      const useBtn = h('button', { class: 'btn primary', disabled: true }, useLabel || 'Use for identify');
      const cancelBtn = h('button', { class: 'btn' }, 'Cancel');
      const toolbar = h('div', { class: 'sel-toolbar', style: 'display:none' }, useBtn, cancelBtn);
      const box = h('div', { class: 'sel-box', style: 'display:none' },
        ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'].map((n) => h('div', { class: `sel-handle h-${n}`, 'data-h': n })));
      const hint = h('div', { class: 'sel-hint' }, hintText || 'Drag over the face · move the box or drag its handles to adjust · Enter = use, Esc = cancel');
      const layer = h('div', { class: 'sel-layer' }, hint, box, toolbar);
      root.appendChild(layer);

      let edges = null;
      let drag = null;
      const W = () => window.innerWidth;
      const H = () => window.innerHeight;
      const norm = (e) => ({ l: Math.min(e.l, e.r), r: Math.max(e.l, e.r), t: Math.min(e.t, e.b), b: Math.max(e.t, e.b) });
      const valid = () => edges && Math.abs(edges.r - edges.l) >= 12 && Math.abs(edges.b - edges.t) >= 12;

      function render() {
        layer.classList.toggle('has-box', !!edges);
        if (!edges) { box.style.display = 'none'; toolbar.style.display = 'none'; return; }
        const n = norm(edges);
        Object.assign(box.style, { display: 'block', left: `${n.l}px`, top: `${n.t}px`, width: `${n.r - n.l}px`, height: `${n.b - n.t}px` });
        useBtn.disabled = !valid();
        if (drag) { toolbar.style.display = 'none'; return; }
        toolbar.style.display = 'flex';
        const below = n.b + 52 < H();
        toolbar.style.top = `${below ? n.b + 10 : Math.max(8, n.t - 46)}px`;
        toolbar.style.left = `${clamp(n.l, 8, Math.max(8, W() - 230))}px`;
      }

      function finish(result) {
        window.removeEventListener('keydown', onKey, true);
        layer.remove();
        if (!result) release();
        resolve(result ? { rect: result, release } : null);
      }
      const confirm = () => {
        if (!valid()) return;
        const n = norm(edges);
        finish({ x: n.l, y: n.t, w: n.r - n.l, h: n.b - n.t });
      };
      const cancel = () => finish(null);

      function onKey(e) {
        if (e.key === 'Escape') { e.preventDefault(); e.stopImmediatePropagation(); cancel(); }
        else if (e.key === 'Enter' && valid()) { e.preventDefault(); e.stopImmediatePropagation(); confirm(); }
      }
      window.addEventListener('keydown', onKey, true);

      toolbar.addEventListener('pointerdown', (e) => e.stopPropagation());
      useBtn.addEventListener('click', confirm);
      cancelBtn.addEventListener('click', cancel);
      box.addEventListener('dblclick', confirm);
      layer.addEventListener('wheel', (e) => e.preventDefault(), { passive: false });
      layer.addEventListener('contextmenu', (e) => e.preventDefault());

      layer.addEventListener('pointerdown', (e) => {
        if (e.button !== 0) return;
        e.preventDefault();
        layer.setPointerCapture(e.pointerId);
        const start = { x: clamp(e.clientX, 0, W()), y: clamp(e.clientY, 0, H()) };
        const handle = e.target.dataset?.h;
        if (edges && handle) drag = { mode: 'resize', handle, start, orig: norm(edges) };
        else if (edges && e.target === box) drag = { mode: 'move', start, orig: norm(edges) };
        else { drag = { mode: 'draw', start }; edges = { l: start.x, t: start.y, r: start.x, b: start.y }; }
        render();
      });

      layer.addEventListener('pointermove', (e) => {
        if (!drag) return;
        const x = clamp(e.clientX, 0, W());
        const y = clamp(e.clientY, 0, H());
        if (drag.mode === 'draw') {
          edges = { l: drag.start.x, t: drag.start.y, r: x, b: y };
        } else if (drag.mode === 'move') {
          const o = drag.orig;
          const w = o.r - o.l;
          const hh = o.b - o.t;
          const l = clamp(o.l + (x - drag.start.x), 0, W() - w);
          const t = clamp(o.t + (y - drag.start.y), 0, H() - hh);
          edges = { l, t, r: l + w, b: t + hh };
        } else {
          const e2 = { ...drag.orig };
          if (drag.handle.includes('w')) e2.l = x;
          if (drag.handle.includes('e')) e2.r = x;
          if (drag.handle.includes('n')) e2.t = y;
          if (drag.handle.includes('s')) e2.b = y;
          edges = e2;
        }
        render();
      });

      const end = () => {
        if (!drag) return;
        drag = null;
        if (edges) edges = norm(edges);
        if (!valid()) edges = null;
        render();
      };
      layer.addEventListener('pointerup', end);
      layer.addEventListener('pointercancel', end);
    });
  }

  async function startAreaFlow() {
    const cfg = await ensureConfigured();
    if (!cfg) return;
    const sel = await selectRegion();
    if (!sel) return;
    await identifyFlow((c) => prepareArea(sel.rect, c), sel.release);
  }

  function blobToDataUrl(blob) {
    return new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onload = () => resolve(String(fr.result));
      fr.onerror = () => reject(new Error('Could not read image data'));
      fr.readAsDataURL(blob);
    });
  }

  // Pick the image an area shows: when one image/video/canvas fills (nearly)
  // the whole box, its native pixels are cropped exactly (best for text inside
  // pictures); anything else -- page text, a mix of text and pictures, overlays
  // -- is rendered from the page itself (rasterizeRect).
  async function captureArea(rect) {
    const total = rect.w * rect.h;
    const media = mediaCandidatesIn(rect)
      .filter((m) => m.kind !== 'bg')
      .some((m) => visibleArea(m, rect) >= total * 0.85);
    if (media) {
      try {
        const { src, crop } = await locateArea(rect);
        return { ...(await renderPadded(src, crop, 0, 'image/png')), how: 'image pixels' };
      } catch (e) {
        console.warn(`[${NAME}] native crop failed, capturing the page instead:`, e);
      }
    }
    return { ...(await rasterizeRect(rect)), how: 'page capture' };
  }

  // Render the part of the page under `rect` to a PNG with html2canvas.
  // Cross-origin images/backgrounds would be blank (the library cannot read
  // them), so those are downloaded here first and swapped into the library's
  // copy of the page as data URLs; <video> frames are snapshotted the same way.
  async function rasterizeRect(rect) {
    if (typeof html2canvas !== 'function') {
      throw new Error('Page capture is unavailable: the html2canvas library did not load. Update this script in your userscript manager so it downloads its @require.');
    }
    const MAX_ASSETS = 16;
    const touches = (el) => {
      const r = el.getBoundingClientRect();
      return r.width > 1 && r.height > 1 && intersect(rect, { x: r.left, y: r.top, w: r.width, h: r.height });
    };
    const crossOrigin = (u) => { try { return new URL(u, location.href).origin !== location.origin; } catch (e) { return false; } };
    const tagged = [];
    const replacements = [];
    const tag = (el, info) => {
      el.setAttribute('data-ssvm-i', String(replacements.length));
      tagged.push(el);
      replacements.push(info);
    };
    const toDataUrl = async (url) => blobToDataUrl(await fetchBlob(url));

    try {
      let assets = 0;
      for (const el of document.querySelectorAll('*')) {
        if (assets >= MAX_ASSETS) break;
        if (el === host || (host && host.contains(el))) continue;
        const tagName = el.tagName;
        if (tagName === 'IMG') {
          const url = el.currentSrc || el.src;
          if (!url || /^data:/i.test(url) || !crossOrigin(url) || !touches(el)) continue;
          try { tag(el, { kind: 'img', dataUrl: await toDataUrl(url) }); assets++; } catch (e) { /* left to the library, may stay blank */ }
        } else if (tagName === 'VIDEO') {
          if (!el.videoWidth || !touches(el)) continue;
          try {
            const snap = document.createElement('canvas');
            snap.width = el.videoWidth;
            snap.height = el.videoHeight;
            snap.getContext('2d').drawImage(el, 0, 0);
            const r = el.getBoundingClientRect();
            const cs = getComputedStyle(el);
            tag(el, { kind: 'video', dataUrl: snap.toDataURL('image/png'), w: r.width, h: r.height, fit: cs.objectFit, pos: cs.objectPosition });
            assets++;
          } catch (e) { /* cross-origin video: cannot be read */ }
        } else if (tagName !== 'CANVAS' && touches(el)) {
          const bg = getComputedStyle(el).backgroundImage;
          const m = bg && bg !== 'none' ? bg.match(/url\((['"]?)(.*?)\1\)/) : null;
          if (m && m[2] && !/^data:/i.test(m[2]) && crossOrigin(m[2])) {
            try { tag(el, { kind: 'bg', dataUrl: await toDataUrl(m[2]) }); assets++; } catch (e) { /* ignore */ }
          }
        }
      }

      const scrollX = window.scrollX;
      const scrollY = window.scrollY;
      const bodyBg = getComputedStyle(document.body || document.documentElement).backgroundColor;
      const htmlBg = getComputedStyle(document.documentElement).backgroundColor;
      const solid = (c) => c && c !== 'transparent' && !/rgba\(.*,\s*0\)$/.test(c);
      const scale = Math.min(Math.max(2, window.devicePixelRatio || 1), 6000 / Math.max(rect.w, rect.h, 1));
      const canvas = await html2canvas(document.documentElement, {
        // Document coordinates, with the library's copy of the page left
        // unscrolled: Firefox applies the copy's scroll on top of the crop
        // offset (the area came out shifted by the scroll distance).
        x: rect.x + scrollX,
        y: rect.y + scrollY,
        scrollX: 0,
        scrollY: 0,
        width: rect.w,
        height: rect.h,
        windowWidth: document.documentElement.clientWidth,
        windowHeight: document.documentElement.clientHeight,
        scale,
        useCORS: true,
        logging: false,
        imageTimeout: 8000,
        backgroundColor: solid(bodyBg) ? bodyBg : solid(htmlBg) ? htmlBg : '#ffffff',
        ignoreElements: (el) => el === host,
        onclone: (doc) => {
          replacements.forEach((info, i) => {
            const el = doc.querySelector(`[data-ssvm-i="${i}"]`);
            if (!el) return;
            if (info.kind === 'img') {
              el.removeAttribute('srcset');
              el.removeAttribute('loading');
              if (el.parentElement) el.parentElement.querySelectorAll('source').forEach((x) => x.remove());
              el.src = info.dataUrl;
            } else if (info.kind === 'bg') {
              el.style.backgroundImage = `url("${info.dataUrl}")`;
            } else if (info.kind === 'video') {
              const img = doc.createElement('img');
              img.src = info.dataUrl;
              img.style.cssText = `display:block;width:${info.w}px;height:${info.h}px;object-fit:${info.fit};object-position:${info.pos};`;
              el.replaceWith(img);
            }
          });
        },
      });
      const blob = await new Promise((resolve, reject) => {
        try {
          canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('Could not encode the captured page area.'))), 'image/png');
        } catch (e) { reject(new Error(e.name === 'SecurityError' ? SECURITY_MSG : e.message)); }
      });
      return { blob, w: canvas.width, h: canvas.height };
    } finally {
      tagged.forEach((el) => el.removeAttribute('data-ssvm-i'));
    }
  }

  // Draw an area and hand whatever is shown there (page text, images, video,
  // overlays) to Google Translate's image mode in a new tab, where
  // deliverTranslateJob uploads it.
  async function startTranslateFlow() {
    const sel = await selectRegion('Drag over the text to translate · move the box or drag its handles to adjust · Enter = use, Esc = cancel', 'Translate with Google');
    if (!sel) return;
    const cfg = loadConfig();
    try {
      const out = await captureArea(sel.rect);
      GM_setValue(TRANSLATE_JOB_KEY, { dataUrl: await blobToDataUrl(out.blob), ts: Date.now() });
      const lang = encodeURIComponent((cfg.translateTo || 'en').trim() || 'en');
      GM_openInTab(`https://translate.google.com/?sl=auto&tl=${lang}&op=images`, { active: true, insert: true });
    } catch (e) {
      console.error(`[${NAME}]`, e);
      openResultModal().showError(e.message || String(e), {
        title: 'Could not translate this area',
        hint: 'The area could not be captured. Try a smaller area, or one that is only text or only a single image.',
      });
    } finally {
      sel.release();
    }
  }

  async function startImageFlow(media) {
    await identifyFlow(() => prepareImage(media));
  }

  // ------------------------------------------------------------- context menu

  let menuEl = null;
  let menuCleanup = null;

  function closeMenu() {
    if (menuEl) { menuEl.remove(); menuEl = null; }
    if (menuCleanup) { menuCleanup(); menuCleanup = null; }
  }

  function showMenu(x, y, media) {
    closeMenu();
    const root = ensureUi();
    const run = (fn) => () => { closeMenu(); fn(); };
    const describe = (m) => (m.kind === 'video' ? 'Current video frame' : m.kind === 'canvas' ? 'Canvas' : (m.url || '').slice(0, 48));
    menuEl = h('div', { class: 'ctx' },
      h('div', { class: 'title' }, NAME),
      h('button', { disabled: !media, onclick: media ? run(() => startImageFlow(media)) : null },
        'Identify this image', h('small', {}, media ? describe(media) : 'No image under the cursor')),
      h('button', { onclick: run(startAreaFlow) },
        'Select area to identify…', h('small', {}, 'Draw a box over a face, then adjust it')),
      h('button', { onclick: run(startTranslateFlow) },
        'Translate text in area…', h('small', {}, 'Capture the box (page, image or video) for Google Translate')),
      h('hr'),
      h('button', { onclick: run(() => openSettings()) }, 'Settings…'));
    // Keep the page from seeing (and reacting to) presses on our menu, and keep
    // its focused element focused.
    for (const type of ['pointerdown', 'mousedown', 'mouseup', 'click', 'contextmenu']) {
      menuEl.addEventListener(type, (e) => {
        e.stopPropagation();
        if (type === 'mousedown' || type === 'contextmenu') e.preventDefault();
      });
    }
    root.appendChild(menuEl);
    const r = menuEl.getBoundingClientRect();
    menuEl.style.left = `${clamp(x, 4, window.innerWidth - r.width - 4)}px`;
    menuEl.style.top = `${clamp(y, 4, window.innerHeight - r.height - 4)}px`;

    const outside = (e) => { if (!e.composedPath().includes(menuEl)) closeMenu(); };
    const key = (e) => { if (e.key === 'Escape') { e.preventDefault(); closeMenu(); } };
    window.addEventListener('pointerdown', outside, true);
    window.addEventListener('keydown', key, true);
    // Capture-phase scroll/blur also report scrolling inside any element and
    // any element losing focus -- pressing the menu blurs a focused page input
    // (e.g. a search box), which used to close the menu before the click
    // landed. Only react to the document scrolling / the window losing focus.
    const onScroll = (e) => { if (e.target === document || e.target === window) closeMenu(); };
    const onBlur = (e) => { if (e.target === window) closeMenu(); };
    window.addEventListener('scroll', onScroll, true);
    window.addEventListener('resize', closeMenu, true);
    window.addEventListener('blur', onBlur, true);
    menuCleanup = () => {
      window.removeEventListener('pointerdown', outside, true);
      window.removeEventListener('keydown', key, true);
      window.removeEventListener('scroll', onScroll, true);
      window.removeEventListener('resize', closeMenu, true);
      window.removeEventListener('blur', onBlur, true);
    };
  }

  function triggerHeld(e, modifier) {
    switch (modifier) {
      case 'alt': return e.altKey;
      case 'shift': return e.shiftKey;
      case 'none': return true;
      default: return e.ctrlKey;
    }
  }

  window.addEventListener('contextmenu', (e) => {
    if (host && e.composedPath().includes(host)) return; // our own UI: leave alone
    const cfg = loadConfig();
    if (!triggerHeld(e, cfg.modifier)) { closeMenu(); return; }
    e.preventDefault();
    e.stopImmediatePropagation();
    const cands = mediaCandidatesAt(e.clientX, e.clientY);
    showMenu(e.clientX, e.clientY, cands[0] || null);
  }, true);

  if (typeof GM_registerMenuCommand === 'function') {
    GM_registerMenuCommand('Select area to identify', startAreaFlow);
    GM_registerMenuCommand('Translate text in area (Google)', startTranslateFlow);
    GM_registerMenuCommand('Settings', () => openSettings());
  }
})();
