// The book scanner page. Plain JavaScript, no build step, nothing loaded from elsewhere.
//
// Every piece of text that comes from the server (file names above all) is put
// on the page as text, never as HTML: el() only ever creates text nodes.
(() => {
  'use strict';

  const main = document.getElementById('main');
  const chip = document.getElementById('job-chip');

  let viewToken = 0;         // goes up on every navigation; work started for an older view is dropped
  let job = null;            // the latest scan, as the server last described it
  let jobTimer = null;
  let onJobChange = null;    // set by whichever view is showing the scan
  let finishedUnseen = null; // id of a scan that finished while another part of the page was open
  let upload = null;         // {request, name, loaded, total} while a dropped video is being copied
  let onUploadChange = null;

  // ---- small helpers -------------------------------------------------------

  function el(tag, props, ...kids) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key === 'class') node.className = value;
      else if (key === 'on') for (const [name, handler] of Object.entries(value)) node.addEventListener(name, handler);
      else if (key === 'data') for (const [name, data] of Object.entries(value)) node.dataset[name] = data;
      else if (value === true) node.setAttribute(key, '');
      else node.setAttribute(key, String(value));
    }
    for (const kid of kids.flat()) {
      if (kid === null || kid === undefined || kid === false) continue;
      node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return node;
  }

  class Problem extends Error {}

  async function api(path, options) {
    let response;
    try {
      response = await fetch(path, Object.assign({cache: 'no-store'}, options));
    } catch (error) {
      throw new Problem('This page cannot reach the book scanner on this PC. It may have been stopped.');
    }
    let data = null;
    try { data = await response.json(); } catch (error) { /* an answer without JSON */ }
    if (!response.ok) {
      const problem = new Problem((data && data.error) || 'Something went wrong. Try again.');
      problem.status = response.status;
      problem.data = data;
      throw problem;
    }
    return data;
  }

  function post(path, body) {
    return api(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})});
  }

  const enc = encodeURIComponent;
  const count = (n) => Number(n).toLocaleString('en-GB');
  const plural = (n, one, many) => `${count(n)} ${n === 1 ? one : (many || one + 's')}`;
  const lowerFirst = (text) => text.charAt(0).toLowerCase() + text.slice(1);
  const withoutExtension = (name) => name.replace(/\.[A-Za-z0-9]{2,4}$/, '');

  function duration(seconds) {
    if (seconds === null || seconds === undefined) return '';
    const total = Math.round(seconds);
    if (total < 60) return `${total} s`;
    const minutes = Math.floor(total / 60);
    if (minutes < 60) return total % 60 ? `${minutes} min ${total % 60} s` : `${minutes} min`;
    return minutes % 60 ? `${Math.floor(minutes / 60)} h ${minutes % 60} min` : `${Math.floor(minutes / 60)} h`;
  }

  function clock(seconds) {
    const total = Math.floor(seconds);
    return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
  }

  function size(bytes) {
    if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
    if (bytes >= 1e6) return `${Math.round(bytes / 1e6)} MB`;
    return `${Math.max(1, Math.round(bytes / 1e3))} KB`;
  }

  function resultHash(run, slug) { return `#/results/${enc(run)}/${enc(slug)}`; }

  function crumb(href, text) { return el('p', {class: 'crumb'}, el('a', {href}, `‹ ${text}`)); }

  function showProblem(problem, back) {
    main.replaceChildren(
      back || crumb('#/', 'Scan a video'),
      el('h1', {}, problem.status === 404 ? 'That is not here' : 'That did not work'),
      el('p', {class: 'problem', role: 'alert'}, problem.message || 'Something went wrong. Try again.'),
    );
  }

  function stateIcon(state) {
    const glyph = {complete: '✓', missing: '!', unfinished: '!', running: '…'}[state] || '?';
    return el('span', {class: 'state-icon', 'aria-hidden': 'true'}, glyph);
  }

  // ---- the scan in progress -------------------------------------------------

  async function refreshJob() {
    try {
      setJob((await api('/api/job')).job);
    } catch (error) {
      clearTimeout(jobTimer);
      if (job && job.active) jobTimer = setTimeout(refreshJob, 3000);
    }
  }

  function setJob(next) {
    const before = job;
    job = next;
    if (before && next && before.id === next.id && before.active && !next.active) finishedUnseen = next.id;
    renderChip();
    if (onJobChange) onJobChange(job, before);
    clearTimeout(jobTimer);
    if (job && job.active) jobTimer = setTimeout(refreshJob, 1000);
  }

  function renderChip() {
    if (job && job.active) {
      chip.hidden = false;
      chip.dataset.state = 'active';
      chip.href = '#/scan';
      chip.textContent = `Scanning: ${withoutExtension(job.name)}`;
    } else if (job && finishedUnseen === job.id) {
      chip.hidden = false;
      chip.dataset.state = job.status === 'done' ? 'done' : 'stopped';
      chip.href = job.status === 'done' ? resultHash(job.run, job.slug) : '#/scan';
      chip.textContent = job.status === 'done' ? 'Scan finished: see the result' : 'The scan stopped: see why';
    } else {
      chip.hidden = true;
    }
  }

  // ---- adding a video --------------------------------------------------------

  function setUpload(next) {
    upload = next;
    if (onUploadChange) onUploadChange();
  }

  async function startUpload(file, say, added) {
    if (upload) { say('A video is already being copied. Wait for it to finish.'); return; }
    say('');
    try {
      await post('/api/upload/check', {name: file.name, size: file.size});
    } catch (problem) {
      say(problem.message);
      return;
    }
    const request = new XMLHttpRequest();
    request.open('POST', `/api/upload?name=${enc(file.name)}`);
    request.responseType = 'json';
    request.upload.addEventListener('progress', (event) => {
      if (upload && event.lengthComputable) setUpload(Object.assign({}, upload, {loaded: event.loaded, total: event.total}));
    });
    request.addEventListener('load', () => {
      setUpload(null);
      if (request.status === 201 && request.response && request.response.video) added(request.response.video);
      else say((request.response && request.response.error) || 'The video could not be copied. Try again.');
    });
    request.addEventListener('error', () => { setUpload(null); say('The video could not be copied. Try again.'); });
    request.addEventListener('abort', () => { setUpload(null); });
    setUpload({request, name: file.name, loaded: 0, total: file.size});
    request.send(file);
  }

  // A file dropped anywhere else on the page must not make the browser open it.
  window.addEventListener('dragover', (event) => event.preventDefault());
  window.addEventListener('drop', (event) => event.preventDefault());

  // ---- view: pick a video ----------------------------------------------------

  async function showHome() {
    const token = viewToken;
    document.title = 'Scan a video · Book scanner';
    let videos = null;
    let highlight = null;

    const banner = el('div', {class: 'banner', hidden: true});
    const note = el('p', {class: 'problem', role: 'alert', hidden: true});
    const say = (text) => { note.textContent = text; note.hidden = !text; };

    const picker = el('input', {type: 'file', accept: '.mov,.m4v,.mp4,video/mp4,video/quicktime,video/x-m4v', class: 'out-of-sight', tabindex: '-1', 'aria-hidden': 'true'});
    const copyStatus = el('p', {class: 'small', 'aria-live': 'polite'});
    const copyBar = el('progress', {max: '100', value: '0', 'aria-label': 'Copying the video'});
    const copyStop = el('button', {type: 'button', class: 'plain', on: {click: () => upload && upload.request.abort()}}, 'Stop copying');
    const copying = el('div', {class: 'copying', hidden: true}, copyStatus, copyBar, copyStop);
    const invite = el('div', {class: 'drop-invite'},
      el('p', {class: 'drop-line'}, el('strong', {}, 'Drop a video here'), ' or ',
        el('button', {type: 'button', class: 'secondary', on: {click: () => picker.click()}}, 'Choose a file')),
      el('p', {class: 'small'}, 'MOV, M4V or MP4. It is copied onto this PC first; then you can scan it.'));
    const drop = el('section', {class: 'drop', 'aria-label': 'Add a video'}, invite, copying, picker);

    const added = async (video) => {
      highlight = video.name;
      await load();
    };
    picker.addEventListener('change', () => {
      if (picker.files.length) startUpload(picker.files[0], say, added);
      picker.value = '';
    });
    drop.addEventListener('dragover', (event) => { event.preventDefault(); drop.classList.add('over'); });
    drop.addEventListener('dragleave', () => drop.classList.remove('over'));
    drop.addEventListener('drop', (event) => {
      event.preventDefault();
      drop.classList.remove('over');
      const file = event.dataTransfer && event.dataTransfer.files[0];
      if (file) startUpload(file, say, added);
    });

    const search = el('input', {type: 'search', placeholder: 'Find a video by name', 'aria-label': 'Find a video by name', autocomplete: 'off'});
    const list = el('ul', {class: 'rows'});
    const empty = el('p', {class: 'quiet', hidden: true});
    search.addEventListener('input', renderRows);

    function renderUpload() {
      copying.hidden = !upload;
      invite.hidden = Boolean(upload);
      if (!upload) return;
      const share = upload.total ? Math.floor(100 * upload.loaded / upload.total) : 0;
      copyBar.value = share;
      copyStatus.textContent = `Copying ${upload.name}: ${share}% (${size(upload.loaded)} of ${size(upload.total)})`;
    }

    function renderBanner() {
      if (job && job.active) {
        banner.hidden = false;
        banner.replaceChildren(
          el('span', {}, `Scanning ${withoutExtension(job.name)} now. One video is scanned at a time.`),
          el('a', {href: '#/scan'}, 'See progress'));
      } else {
        banner.hidden = true;
      }
    }

    async function start(video, button) {
      say('');
      button.disabled = true;
      try {
        const data = await post('/api/jobs', {source: video.source, name: video.name});
        finishedUnseen = null;
        setJob(data.job);
        location.hash = '#/scan';
      } catch (problem) {
        if (problem.data && problem.data.job) setJob(problem.data.job);
        say(problem.message);
        button.disabled = Boolean(job && job.active);
      }
    }

    function row(video) {
      const busy = Boolean(job && job.active);
      const facts = [duration(video.seconds), size(video.bytes), video.source === 'added' ? 'added by you' : ''].filter(Boolean).join(' · ');
      const button = el('button', {type: 'button', class: 'primary', disabled: busy, 'aria-label': `Scan ${withoutExtension(video.name)}`,
        title: busy ? 'Wait for the scan in progress to finish' : null}, 'Scan');
      button.addEventListener('click', () => start(video, button));
      const earlier = video.earlier
        ? el('a', {class: 'row-earlier', href: resultHash(video.earlier.run, video.earlier.slug), data: {state: video.earlier.state}},
          `Scanned before: ${lowerFirst(video.earlier.short)}`)
        : null;
      return el('li', {class: video.name === highlight ? 'row fresh' : 'row'},
        el('div', {class: 'row-text'}, el('p', {class: 'row-title'}, withoutExtension(video.name)), el('p', {class: 'row-facts'}, facts), earlier),
        button);
    }

    function renderRows() {
      if (!videos) return;
      const wanted = search.value.trim().toLowerCase();
      const shown = videos.filter((video) => video.name.toLowerCase().includes(wanted));
      list.replaceChildren(...shown.map(row));
      empty.hidden = shown.length > 0;
      empty.textContent = videos.length ? 'No video has that in its name.' : 'There are no videos yet. Drop one above.';
      const fresh = list.querySelector('.fresh');
      if (fresh && highlight) { fresh.scrollIntoView({block: 'center'}); }
    }

    async function load() {
      try {
        const data = await api('/api/videos');
        if (token !== viewToken) return;
        videos = data.videos;
        setJob(data.job);
        renderRows();
        highlight = null;
      } catch (problem) {
        if (token === viewToken) say(problem.message);
      }
    }

    main.replaceChildren(
      el('h1', {}, 'Scan a video'),
      el('p', {class: 'lead'}, 'Pick a video of a book being paged through. You get one picture of every two-page spread, in order, and a check that no page is missing.'),
      banner, drop, note,
      el('div', {class: 'list-head'}, el('h2', {}, 'Videos on this PC'), search),
      list, empty);
    list.append(el('li', {class: 'quiet'}, 'Loading…'));

    onJobChange = (now, before) => {
      renderBanner();
      if (Boolean(now && now.active) !== Boolean(before && before.active)) {
        if (now && !now.active) load(); else renderRows();
      }
    };
    onUploadChange = renderUpload;
    renderUpload();
    renderBanner();
    await load();
  }

  // ---- view: the scan in progress ---------------------------------------------

  const STATUS_WORDS = {
    starting: 'Getting ready…',
    running: 'Scanning…',
    done: 'Finished.',
    cancelled: 'Stopped.',
    failed: 'This did not work.',
    busy: 'Not started.',
  };

  function showScan() {
    const token = viewToken;
    document.title = 'Scanning · Book scanner';
    let builtFor = null;
    let title, status, stages, message, actions, output;

    function build(current) {
      builtFor = current.id;
      title = el('h1', {}, withoutExtension(current.name));
      status = el('p', {class: 'scan-status', role: 'status'});
      stages = el('ol', {class: 'stages'});
      message = el('p', {class: 'problem', hidden: true});
      actions = el('div', {class: 'actions'});
      output = el('div', {});
      main.replaceChildren(crumb('#/', 'All videos'), title, status, stages, message, actions, output);
    }

    function stageRow(stage) {
      const words = {waiting: '', running: duration(stage.seconds), done: duration(stage.seconds), skipped: 'not needed', stopped: 'stopped here'}[stage.state] || '';
      return el('li', {class: 'stage', data: {state: stage.state}, 'aria-current': stage.state === 'running' ? 'step' : null},
        el('span', {class: 'stage-mark', 'aria-hidden': 'true'}, stage.state === 'done' ? '✓' : stage.state === 'stopped' ? '×' : ''),
        el('span', {class: 'stage-text'}, el('span', {class: 'stage-label'}, stage.label),
          stage.state === 'running' && stage.detail && stage.detail.toLowerCase() !== stage.label.toLowerCase() ? el('span', {class: 'stage-detail'}, stage.detail) : null),
        el('span', {class: 'stage-time'}, words));
    }

    async function stop(button) {
      button.disabled = true;
      button.textContent = 'Stopping…';
      try { setJob((await post(`/api/jobs/${enc(job.id)}/cancel`)).job); } catch (problem) { message.textContent = problem.message; message.hidden = false; }
    }

    async function again(current) {
      try {
        const data = await post('/api/jobs', {source: current.source, name: current.name});
        finishedUnseen = null;
        setJob(data.job);
      } catch (problem) {
        message.textContent = problem.message;
        message.hidden = false;
      }
    }

    function render(current, before) {
      if (token !== viewToken) return;
      if (!current) {
        builtFor = null;
        main.replaceChildren(crumb('#/', 'All videos'), el('h1', {}, 'Nothing is being scanned'), el('p', {class: 'lead'}, 'Pick a video to scan.'));
        return;
      }
      if (before && before.id === current.id && before.active && current.status === 'done') {
        finishedUnseen = null;
        renderChip();
        location.replace(resultHash(current.run, current.slug));   // straight to the result, without a stop on the way back
        return;
      }
      if (!current.active && finishedUnseen === current.id) { finishedUnseen = null; renderChip(); }
      const statusChanged = !before || before.id !== current.id || before.status !== current.status || before.cancelling !== current.cancelling;
      if (builtFor !== current.id) build(current);
      status.textContent = current.active
        ? `${current.cancelling ? 'Stopping…' : STATUS_WORDS[current.status]} ${clock(current.elapsed)} so far. A four-minute video takes a minute or two.`
        : `${STATUS_WORDS[current.status] || ''} ${current.status === 'done' ? `It took ${duration(current.elapsed)}.` : ''}`.trim();
      status.dataset.state = current.status;
      stages.replaceChildren(...current.stages.map(stageRow));
      message.textContent = current.message || '';
      message.hidden = !current.message;
      if (!statusChanged && actions.childElementCount) return;
      if (current.active) {
        const button = el('button', {type: 'button', class: 'secondary', disabled: current.cancelling}, current.cancelling ? 'Stopping…' : 'Stop');
        button.addEventListener('click', () => stop(button));
        actions.replaceChildren(button);
      } else if (current.status === 'done') {
        actions.replaceChildren(el('a', {class: 'button primary', href: resultHash(current.run, current.slug)}, 'See the result'));
      } else {
        const button = el('button', {type: 'button', class: 'primary'}, 'Try again');
        button.addEventListener('click', () => again(current));
        actions.replaceChildren(button, el('a', {class: 'button secondary', href: '#/'}, 'Back to the videos'));
      }
      output.replaceChildren();
      if (current.status === 'failed' && current.output && current.output.length) {
        output.append(el('details', {class: 'technical'}, el('summary', {}, 'Details for whoever looks after this'), el('pre', {}, current.output.join('\n'))));
      }
    }

    onJobChange = render;
    render(job, null);
    refreshJob();
  }

  // ---- view: earlier results ---------------------------------------------------

  function totalsLine(totals) {
    const states = totals.states || {};
    const checked = (states.complete || 0) + (states.missing || 0);
    const unchecked = (states.unreadable || 0) + (states.not_checked || 0);
    const parts = [`${plural(totals.videos, 'video')}, ${plural(totals.pictures, 'picture')}`];
    if (checked) {
      parts.push(states.missing
        ? `${count(states.complete || 0)} with every page there, ${count(states.missing)} with pages missing`
        : `${checked === 1 ? 'it has' : `all ${count(checked)} checked have`} every page there`);
    }
    if (unchecked) parts.push(`${count(unchecked)} with the page numbers not checked`);
    if (states.unfinished) parts.push(`${count(states.unfinished)} not finished`);
    return parts.join(' · ');
  }

  async function showRuns() {
    const token = viewToken;
    document.title = 'Earlier results · Book scanner';
    main.replaceChildren(el('h1', {}, 'Earlier results'), el('p', {class: 'quiet'}, 'Loading…'));
    let data;
    try { data = await api('/api/runs'); } catch (problem) { if (token === viewToken) showProblem(problem); return; }
    if (token !== viewToken) return;
    const rows = data.runs.map((run) => el('li', {class: 'row linked'},
      el('a', {class: 'row-link', href: `#/results/${enc(run.run)}`},
        el('span', {class: 'row-title'}, run.run_label),
        el('span', {class: 'row-facts'}, totalsLine(run.totals)),
        el('span', {class: 'row-facts folder'}, run.run))));
    main.replaceChildren(
      el('h1', {}, 'Earlier results'),
      el('p', {class: 'lead'}, 'Every batch of scans kept on this PC, newest first.'),
      rows.length ? el('ul', {class: 'rows'}, rows) : el('p', {class: 'quiet'}, 'Nothing has been scanned yet.'));
  }

  async function showRun(run) {
    const token = viewToken;
    const back = crumb('#/results', 'Earlier results');
    main.replaceChildren(back, el('p', {class: 'quiet'}, 'Loading…'));
    let data;
    try { data = await api(`/api/runs/${enc(run)}`); } catch (problem) { if (token === viewToken) showProblem(problem, back); return; }
    if (token !== viewToken) return;
    document.title = `${data.run_label} · Book scanner`;

    const groups = [
      {key: 'all', label: 'All', test: () => true},
      {key: 'complete', label: 'Every page there', test: (v) => v.verdict.state === 'complete'},
      {key: 'missing', label: 'Pages missing', test: (v) => v.verdict.state === 'missing'},
      {key: 'unchecked', label: 'Not checked', test: (v) => !['complete', 'missing'].includes(v.verdict.state)},
    ];
    let chosen = 'all';
    const search = el('input', {type: 'search', placeholder: 'Find a video by name', 'aria-label': 'Find a video by name', autocomplete: 'off'});
    const chips = el('div', {class: 'chips', role: 'group', 'aria-label': 'Show'});
    const list = el('ul', {class: 'rows'});
    const empty = el('p', {class: 'quiet', hidden: true}, 'No video here matches.');

    function render() {
      chips.replaceChildren(...groups.map((group) => {
        const n = data.videos.filter(group.test).length;
        if (!n && group.key !== 'all') return null;
        return el('button', {type: 'button', class: 'chip', 'aria-pressed': String(chosen === group.key),
          on: {click: () => { chosen = group.key; render(); }}}, `${group.label} `, el('span', {class: 'chip-count'}, count(n)));
      }).filter(Boolean));
      const wanted = search.value.trim().toLowerCase();
      const test = groups.find((group) => group.key === chosen).test;
      const shown = data.videos.filter((video) => test(video) && video.title.toLowerCase().includes(wanted));
      list.replaceChildren(...shown.map((video) => {
        const look = video.counts.needs_look;
        return el('li', {class: 'row linked'},
          el('a', {class: 'row-link', href: resultHash(run, video.slug)},
            el('span', {class: 'row-title'}, video.title),
            el('span', {class: 'badge', data: {state: video.verdict.state}}, stateIcon(video.verdict.state), video.verdict.short),
            el('span', {class: 'row-facts'}, `${plural(video.counts.pictures, 'picture')}${look ? ` · ${count(look)} need${look === 1 ? 's' : ''} a look` : ''}`)));
      }));
      empty.hidden = shown.length > 0;
    }
    search.addEventListener('input', render);

    main.replaceChildren(back, el('h1', {}, data.run_label), el('p', {class: 'lead'}, totalsLine(data.totals)),
      el('div', {class: 'list-head'}, chips, search), list, empty);
    render();
  }

  // ---- view: one video's result ---------------------------------------------------

  async function showResult(run, slug) {
    const token = viewToken;
    const back = crumb(`#/results/${enc(run)}`, 'All videos of this batch');
    main.replaceChildren(back, el('p', {class: 'quiet'}, 'Loading…'));
    let view;
    try { view = await api(`/api/runs/${enc(run)}/${enc(slug)}`); } catch (problem) { if (token === viewToken) showProblem(problem, back); return; }
    if (token !== viewToken) return;
    document.title = `${view.title} · Book scanner`;
    back.firstChild.textContent = `‹ ${view.run_label}`;
    if (job && finishedUnseen === job.id && job.run === run && job.slug === slug) { finishedUnseen = null; renderChip(); }

    const counts = view.counts;
    const pictures = view.items.filter((item) => item.kind === 'picture');
    const marks = view.items.filter((item) => item.kind !== 'picture');
    const base = `${enc(run)}/${enc(slug)}`;

    const verdict = el('section', {class: 'verdict', data: {state: view.verdict.state}, 'aria-label': 'Page-number check'},
      stateIcon(view.verdict.state),
      el('div', {}, el('h2', {class: 'verdict-headline'}, view.verdict.headline), view.verdict.details.map((line) => el('p', {}, line))));

    const stat = (value, label) => el('div', {class: 'stat'}, el('dt', {}, label), el('dd', {}, value));
    const stats = el('dl', {class: 'stats'},
      stat(count(counts.pictures), 'Pictures'),
      stat(count(counts.needs_look), counts.needs_look === 1 ? 'Needs a look' : 'Need a look'),
      counts.missing !== null ? stat(count(counts.missing), counts.missing === 1 ? 'Spread missing' : 'Spreads missing') : null,
      counts.repeats !== null ? stat(count(counts.repeats), counts.repeats === 1 ? 'Possible repeat' : 'Possible repeats') : null,
      counts.restarts ? stat(count(counts.restarts), counts.restarts === 1 ? 'Place the numbering starts again' : 'Places the numbering starts again') : null,
      view.video_seconds ? stat(duration(view.video_seconds), 'Length of the video') : null);

    // A mark is a gap, a possible repeat, or the numbering starting again; the last is not a fault.
    const isRestart = (mark) => mark.kind === 'restart';
    const beside = (item) => item.beside_kinds || [];
    const filters = [
      {key: 'all', label: 'All', n: counts.pictures, picture: () => true, mark: () => true},
      {key: 'look', label: 'Needs a look', n: counts.needs_look + marks.length, picture: (item) => item.flags.length > 0 || item.beside_mark, mark: () => true},
      {key: 'hand', label: 'Hand over the text', n: counts.hand, picture: (item) => item.flags.includes('hand_on_print'), mark: () => false},
      {key: 'moving', label: 'Page was moving', n: counts.moving, picture: (item) => item.flags.includes('never_at_rest'), mark: () => false},
      {key: 'marks', label: 'Gaps and repeats', n: marks.filter((mark) => !isRestart(mark)).length,
        picture: (item) => beside(item).some((kind) => kind !== 'restart'), mark: (mark) => !isRestart(mark)},
      {key: 'restarts', label: 'Numbering starts again', n: marks.filter(isRestart).length,
        picture: (item) => beside(item).includes('restart'), mark: isRestart},
    ];
    let chosen = 'all';
    const chips = el('div', {class: 'chips', role: 'group', 'aria-label': 'Show'});
    const grid = el('ol', {class: 'grid'});
    const empty = el('p', {class: 'quiet', hidden: true}, 'Nothing to show here.');
    let visible = pictures;

    const nodes = view.items.map((item) => {
      if (item.kind !== 'picture') {
        return el('li', {class: 'mark', data: {kind: item.kind}}, el('p', {class: 'mark-title'}, item.title), el('p', {class: 'mark-text'}, item.text));
      }
      const caption = el('span', {class: 'tile-caption'},
        el('span', {class: 'tile-number'}, String(item.number)),
        item.pages ? el('span', {class: 'tile-pages'}, item.pages) : null,
        item.seconds !== null ? el('span', {class: 'tile-time'}, clock(item.seconds)) : null);
      const flags = item.labels.length ? el('span', {class: 'tile-flags'}, item.labels.map((label) => el('span', {class: 'pill'}, label))) : null;
      if (!item.file || !item.on_disk) {
        return el('li', {class: 'tile absent'}, el('span', {class: 'tile-absent'}, 'The picture file is not there'), caption, flags);
      }
      const image = el('img', {src: `/thumb/${base}/${enc(item.file)}?v=${view.version}`, alt: '', loading: 'lazy', decoding: 'async', width: '480', height: '270'});
      image.addEventListener('error', () => image.classList.add('broken'), {once: true});
      const words = `Picture ${item.number}${item.pages ? `, ${item.pages}` : ''}${item.labels.length ? `, ${item.labels.join(', ')}` : ''}. Show full size.`;
      const button = el('button', {type: 'button', class: 'tile-button', 'aria-label': words}, image, caption, flags);
      button.addEventListener('click', () => openViewer(visible, visible.indexOf(item), view, base));
      return el('li', {class: item.flags.length ? 'tile flagged' : 'tile'}, button);
    });

    function render() {
      chips.replaceChildren(...filters.map((filter) => {
        if (!filter.n && !['all', 'look'].includes(filter.key)) return null;
        return el('button', {type: 'button', class: 'chip', 'aria-pressed': String(chosen === filter.key),
          on: {click: () => { chosen = filter.key; render(); }}}, `${filter.label} `, el('span', {class: 'chip-count'}, count(filter.n)));
      }).filter(Boolean));
      const filter = filters.find((candidate) => candidate.key === chosen);
      visible = [];
      let shown = 0;
      view.items.forEach((item, index) => {
        const show = item.kind === 'picture' ? filter.picture(item) : filter.mark(item);
        nodes[index].hidden = !show;
        if (show) shown += 1;
        if (show && item.kind === 'picture' && item.file && item.on_disk) visible.push(item);
      });
      empty.hidden = shown > 0;
    }

    grid.append(...nodes);
    const download = counts.pictures - counts.files_missing > 0
      ? el('a', {class: 'button primary', href: `/zip/${base}`, download: `${slug}.zip`}, 'Download all pictures (ZIP)')
      : null;
    const leftOut = view.left_out ? el('p', {class: 'small left-out'}, view.left_out.text) : null;
    main.replaceChildren(...[back, el('h1', {}, view.title), verdict, stats, leftOut,
      el('div', {class: 'actions'}, download, el('p', {class: 'small where-saved'}, 'The pictures are saved in ', el('span', {class: 'folder'}, view.folder))),
      chips, grid, empty].filter(Boolean));
    render();
  }

  // ---- the full-size picture ---------------------------------------------------------

  const viewer = {
    dialog: document.getElementById('viewer'),
    stage: document.getElementById('viewer-stage'),
    image: document.getElementById('viewer-image'),
    where: document.getElementById('viewer-where'),
    open: document.getElementById('viewer-open'),
    previous: document.getElementById('viewer-previous'),
    next: document.getElementById('viewer-next'),
    list: [], at: 0, view: null, base: '',
  };

  function openViewer(list, at, view, base) {
    if (at < 0) return;
    Object.assign(viewer, {list, at, view, base});
    showInViewer();
    if (!viewer.dialog.open) viewer.dialog.showModal();
  }

  function showInViewer() {
    const item = viewer.list[viewer.at];
    const address = `/image/${viewer.base}/${enc(item.file)}?v=${viewer.view.version}`;
    viewer.stage.classList.remove('zoomed');
    viewer.image.src = address;
    viewer.image.alt = `Picture ${item.number}`;
    viewer.open.href = address;
    const pieces = [`Picture ${item.number} of ${count(viewer.view.counts.pictures)}`];
    if (item.pages) pieces.push(item.pages);
    if (item.seconds !== null) pieces.push(`${clock(item.seconds)} into the video`);
    viewer.where.replaceChildren(pieces.join(' · '), ...item.labels.map((label) => el('span', {class: 'pill'}, label)));
    viewer.previous.disabled = viewer.at === 0;
    viewer.next.disabled = viewer.at === viewer.list.length - 1;
  }

  function stepViewer(by) {
    const to = viewer.at + by;
    if (to < 0 || to >= viewer.list.length) return;
    viewer.at = to;
    showInViewer();
  }

  function closeViewer() {
    if (viewer.dialog.open) viewer.dialog.close();
  }

  viewer.previous.addEventListener('click', () => stepViewer(-1));
  viewer.next.addEventListener('click', () => stepViewer(1));
  document.getElementById('viewer-close').addEventListener('click', closeViewer);
  viewer.image.addEventListener('click', () => viewer.stage.classList.toggle('zoomed'));
  viewer.dialog.addEventListener('close', () => { viewer.image.removeAttribute('src'); });
  viewer.dialog.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowLeft') { stepViewer(-1); event.preventDefault(); }
    if (event.key === 'ArrowRight') { stepViewer(1); event.preventDefault(); }
  });

  // ---- finding the view for the address ------------------------------------------------

  function route() {
    viewToken += 1;
    onJobChange = null;
    onUploadChange = null;
    closeViewer();
    let parts;
    try {
      parts = location.hash.replace(/^#\/?/, '').split('/').filter(Boolean).map(decodeURIComponent);
    } catch (error) {
      parts = ['?'];
    }
    const section = parts[0] === 'results' ? 'results' : 'scan';
    for (const link of document.querySelectorAll('[data-section]')) {
      if (link.dataset.section === section) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current');
    }
    window.scrollTo(0, 0);
    if (parts.length === 0) showHome();
    else if (parts[0] === 'scan' && parts.length === 1) showScan();
    else if (parts[0] === 'results' && parts.length === 1) showRuns();
    else if (parts[0] === 'results' && parts.length === 2) showRun(parts[1]);
    else if (parts[0] === 'results' && parts.length === 3) showResult(parts[1], parts[2]);
    else showProblem(Object.assign(new Problem('There is nothing at this address.'), {status: 404}));
  }

  window.addEventListener('hashchange', route);
  route();
  refreshJob();
})();
