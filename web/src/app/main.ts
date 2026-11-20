// The page: choose a video, watch the stages, look at and save the pictures.

import { NumberReader, STAGES, Scan, UnsupportedVideo, checkSupport, clearStore, type Picture, type Progress, type ScanResult, type StageId } from './run.ts';
import { captureName } from '../pipeline/repeats.ts';
import type { NumberCheck } from '../ocr/audit.ts';
import { buildZip } from './zip.ts';
import { BookMaker, type Book, type BookStep } from './book.ts';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const sections = { start: $('start'), work: $('work'), problem: $('problem'), done: $('done') };
function show(which: keyof typeof sections): void {
  for (const [k, el] of Object.entries(sections)) el.hidden = k !== which;
  window.scrollTo({ top: 0 });
}

// A small status object for automated end-to-end tests; it holds no picture data.
const status: { state: string; error?: string; result?: Omit<ScanResult, 'pictures' | 'region'> & { pictures: Omit<Picture, 'preview' | 'full'>[] }; numbers?: Record<string, unknown>; book?: Record<string, unknown> } = { state: 'idle' };
(window as unknown as { __bookScanner: typeof status }).__bookScanner = status;

let scan: Scan | null = null;
let pictures: Picture[] = [];
let videoName = 'video';
const urls: string[] = [];

function fmtTime(s: number): string {
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)} s`;
  const m = Math.floor(s / 60);
  return `${m} min ${Math.round(s - 60 * m)} s`;
}

// ---- support note on load ------------------------------------------------

(function supportNote() {
  const missing: string[] = [];
  if (!('VideoDecoder' in window)) missing.push('video decoding (WebCodecs)');
  if (!('OffscreenCanvas' in window)) missing.push('off-screen drawing');
  if (!('Worker' in window)) missing.push('background workers');
  if (missing.length) {
    const el = $('support');
    el.textContent = `This browser lacks ${missing.join(', ')}, which this page needs. Try a recent Safari, Chrome or Edge.`;
    el.hidden = false;
  }
})();

// ---- choosing a file -----------------------------------------------------

const input = $<HTMLInputElement>('file');
input.addEventListener('change', () => { if (input.files?.[0]) void begin(input.files[0]); });
const drop = $('drop');
for (const ev of ['dragenter', 'dragover']) drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); });
for (const ev of ['dragleave', 'drop']) drop.addEventListener(ev, () => drop.classList.remove('over'));
drop.addEventListener('drop', e => {
  e.preventDefault();
  const f = (e as DragEvent).dataTransfer?.files?.[0];
  if (f) void begin(f);
});
window.addEventListener('dragover', e => e.preventDefault());
window.addEventListener('drop', e => e.preventDefault());

// ---- running -------------------------------------------------------------

const stageEls = new Map<StageId, { li: HTMLLIElement; bar: HTMLSpanElement; time: HTMLSpanElement; started: number }>();

function buildStages(): void {
  const ol = $('stages');
  ol.textContent = '';
  stageEls.clear();
  for (const s of STAGES) {
    const li = document.createElement('li');
    li.className = 'stage';
    li.innerHTML = '<span class="dot" aria-hidden="true"></span><span class="label"></span><span class="time"></span><span class="bar"><span></span></span>';
    li.querySelector('.label')!.textContent = s.label;
    ol.append(li);
    stageEls.set(s.id, { li, bar: li.querySelector('.bar span') as HTMLSpanElement, time: li.querySelector('.time') as HTMLSpanElement, started: 0 });
  }
}

let current: StageId | null = null;
let clock = 0;
let startedAt = 0;

function onProgress(p: Progress): void {
  const order = STAGES.map(s => s.id);
  if (current !== p.stage) {
    const now = performance.now();
    for (const id of order) {
      const el = stageEls.get(id)!;
      const before = order.indexOf(id) < order.indexOf(p.stage);
      if (before && !el.li.classList.contains('done')) {
        el.li.classList.remove('active');
        el.li.classList.add('done');
        if (el.started) el.time.textContent = fmtTime((now - el.started) / 1000);
      }
    }
    const el = stageEls.get(p.stage)!;
    el.li.classList.add('active');
    el.started = now;
    current = p.stage;
  }
  const el = stageEls.get(p.stage)!;
  el.bar.style.width = `${Math.round(Math.min(1, p.fraction) * 100)}%`;
  if (p.fraction >= 1) {
    el.li.classList.remove('active');
    el.li.classList.add('done');
    el.time.textContent = fmtTime((performance.now() - el.started) / 1000);
  }
}

// Keep a phone's screen on while a scan runs: a locked phone pauses the page.
let wakeLock: { release(): Promise<void> } | null = null;
async function holdScreen(on: boolean): Promise<void> {
  try {
    if (on && !wakeLock && document.visibilityState === 'visible') {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      wakeLock = await (navigator as any).wakeLock?.request('screen') ?? null;
      (wakeLock as unknown as EventTarget | null)?.addEventListener?.('release', () => { wakeLock = null; });
    } else if (!on && wakeLock) {
      const lock = wakeLock;
      wakeLock = null;
      await lock.release();
    }
  } catch { wakeLock = null; }
}
document.addEventListener('visibilitychange', () => { if (scan && document.visibilityState === 'visible') void holdScreen(true); });

async function begin(file: File): Promise<void> {
  void holdScreen(true);
  videoName = file.name.replace(/\.[^.]+$/, '') || 'video';
  $('filename').textContent = file.name;
  buildStages();
  current = null;
  status.state = 'checking';
  show('work');
  startedAt = performance.now();
  clock = window.setInterval(() => { $('elapsed').textContent = `Time so far: ${fmtTime((performance.now() - startedAt) / 1000)}`; }, 500);
  try {
    const { track, hardware, decoder } = await checkSupport(file);
    status.state = 'running';
    const note = $('work-note');
    note.hidden = decoder !== 'built-in';
    note.textContent = decoder === 'built-in'
      ? 'This browser cannot decode this kind of video (HEVC) by itself, so the page is using its own decoder. That is several times slower: a 4-minute video takes a few minutes on a fast computer. In Safari on an iPhone, iPad or Mac it is much quicker.'
      : '';
    scan = new Scan(file, onProgress, decoder);
    scan.details.hardwareDecoding = hardware === null ? 'unknown' : hardware ? 'available' : 'not available';
    scan.details.browser = navigator.userAgent;
    const result = await scan.run(track);
    if (scan.debug) (window as unknown as { __bookScannerDebug: unknown }).__bookScannerDebug = scan.debug;
    finish(result);
  } catch (e) {
    if ((e as Error).message === 'stopped') return;
    fail(e);
  } finally {
    window.clearInterval(clock);
    input.value = '';
    void holdScreen(false);
  }
}

function fail(e: unknown): void {
  scan?.stop();
  scan = null;
  const message = e instanceof UnsupportedVideo ? e.message
    : `Something went wrong while working on the video: ${e instanceof Error ? e.message : String(e)}`;
  $('problem-text').textContent = message;
  status.state = 'error';
  status.error = message;
  show('problem');
}

$('stop').addEventListener('click', () => {
  scan?.stop();
  scan = null;
  window.clearInterval(clock);
  status.state = 'stopped';
  void clearStore();
  show('start');
});
$('problem-again').addEventListener('click', () => show('start'));

// ---- results -------------------------------------------------------------

let lastResult: ScanResult | null = null;
let reader: NumberReader | null = null;
let numbersChecked = false;

function summaryText(seconds: number): void {
  const marked = pictures.filter(p => p.flags.length).length;
  $('done-title').textContent = `${pictures.length} ${pictures.length === 1 ? 'picture' : 'pictures'}`;
  $('done-text').textContent = `Done in ${fmtTime(seconds)}. ` +
    (marked ? `${marked} ${marked === 1 ? 'is' : 'are'} marked for a closer look (a hand over the text, or the page still moving).` : 'None is marked for a closer look.') +
    (numbersChecked ? '' : ' Page numbers have not been checked yet (see below).');
  (document.querySelector('.filter') as HTMLElement).hidden = marked === 0;
}

function renderGallery(): void {
  for (const u of urls.splice(0)) URL.revokeObjectURL(u);
  const gallery = $('gallery');
  gallery.textContent = '';
  for (const p of pictures) {
    const li = document.createElement('li');
    li.className = 'card' + (p.flags.length ? ' flagged' : '');
    const url = URL.createObjectURL(p.preview);
    urls.push(url);
    li.innerHTML = `<button type="button"><img loading="lazy" alt=""><div class="meta"><strong></strong><span class="page"></span><span class="at"></span></div></button><div class="badges"></div>`;
    const img = li.querySelector('img')!;
    img.src = url;
    img.alt = `Picture ${p.number}`;
    li.querySelector('strong')!.textContent = `#${p.number}`;
    li.querySelector('.page')!.textContent = p.page != null ? `p. ${p.page}` : '';
    li.querySelector('.at')!.textContent = fmtClock(p.seconds);
    const badges = li.querySelector('.badges')!;
    for (const f of p.flags) {
      const b = document.createElement('span');
      b.className = 'badge';
      b.textContent = f;
      badges.append(b);
    }
    li.querySelector('button')!.addEventListener('click', () => openViewer(pictures.indexOf(p)));
    gallery.append(li);
  }
}

function publishStatus(): void {
  if (!lastResult) return;
  status.result = {
    setup: lastResult.setup, seconds: lastResult.seconds, details: lastResult.details,
    pictures: pictures.map(({ preview: _p, full: _f, ...rest }) => rest),
  };
}

function finish(result: ScanResult): void {
  scan = null;
  lastResult = result;
  pictures = result.pictures;
  numbersChecked = false;
  summaryText(result.seconds.total);
  document.querySelector<HTMLInputElement>('input[name="show"][value="all"]')!.checked = true;
  $('gallery').classList.remove('marked-only');
  renderGallery();
  resetNumbers();
  resetBook();
  const table = $('timings');
  table.textContent = '';
  const rows: [string, string][] = [
    ...Object.entries(result.seconds).map(([k, v]) => [`time: ${k}`, `${v} s`] as [string, string]),
    ...Object.entries(result.details).map(([k, v]) => [k, String(v)] as [string, string]),
  ];
  for (const [k, v] of rows) addDetail(k, v);
  status.state = 'done';
  publishStatus();
  show('done');
}

function addDetail(k: string, v: string): void {
  const tr = document.createElement('tr');
  tr.innerHTML = '<td></td><td></td>';
  tr.children[0].textContent = k;
  tr.children[1].textContent = v;
  $('timings').append(tr);
}

// ---- the page-number check ------------------------------------------------

function resetNumbers(): void {
  reader?.stop();
  reader = null;
  $('numbers').classList.remove('good');
  $('numbers-text').textContent = 'Not checked yet. This page can read the printed page numbers to find pages that are missing and pictures that show the same pages twice. It downloads about 31 MB the first time and takes about a second per picture on a computer, longer on a phone.';
  $('numbers-bar').hidden = true;
  $('numbers-list').hidden = true;
  $('numbers-list').textContent = '';
  $<HTMLButtonElement>('numbers-check').hidden = false;
  $<HTMLButtonElement>('numbers-check').disabled = false;
  $('numbers-stop').hidden = true;
}

/** "#12 (page 22)", a button that opens that picture. */
function pictureLink(i: number): HTMLElement {
  const b = document.createElement('button');
  b.type = 'button';
  const p = pictures[i];
  b.textContent = `#${p.number}${p.page != null ? ` (page ${p.page})` : ''}`;
  b.addEventListener('click', () => openViewer(i));
  return b;
}

function listItem(...parts: (string | HTMLElement)[]): HTMLLIElement {
  const li = document.createElement('li');
  li.append(...parts);
  return li;
}

let numbersRunning: Promise<void> | null = null;

/** The page-number check; while one is running, that one. */
function checkNumbers(): Promise<void> {
  if (!lastResult) return Promise.resolve();
  numbersRunning ??= runNumberCheck().finally(() => { numbersRunning = null; });
  return numbersRunning;
}

async function runNumberCheck(): Promise<void> {
  if (!lastResult || reader) return;
  const setup = lastResult.setup;
  const r = new NumberReader();
  reader = r;
  $<HTMLButtonElement>('numbers-check').disabled = true;
  $('numbers-stop').hidden = false;
  $('numbers-bar').hidden = false;
  const bar = $('numbers-bar').firstElementChild as HTMLElement;
  bar.style.width = '0%';
  status.numbers = { state: 'running' };
  try {
    const { check } = await r.run(pictures, setup, (done, total, loading) => {
      $('numbers-text').textContent = loading ? 'Getting the page-number reader ready…' : `Reading page numbers: ${done} of ${total} pictures`;
      bar.style.width = `${Math.round(100 * done / Math.max(1, total))}%`;
    });
    if (reader !== r) return;
    reader = null;
    // apply: drop the repeats, number the rest again, note the printed pages
    const removed = new Set(check.removed);
    const removedPictures = pictures.filter((_, k) => removed.has(k));
    pictures = pictures.filter((_, k) => !removed.has(k));
    pictures.forEach((p, k) => {
      p.number = k + 1;
      p.name = captureName(k + 1, p.frame, setup.fps);
      p.page = check.pages[k] ?? null;
    });
    numbersChecked = true;
    summaryText(lastResult.seconds.total);
    renderGallery();
    showNumbers(check, removedPictures.length);
    addDetail('page numbers: time (s)', String(Math.round(r.seconds.total * 10) / 10));
    addDetail('page numbers: read on', `${check.report.read} of ${pictures.length} pictures`);
    status.numbers = {
      state: 'done', seconds: r.seconds, removed: check.removed, report: check.report, before: check.before, vouched: check.vouched,
    };
    publishStatus();
  } catch (e) {
    if (reader === r) reader = null;
    if ((e as Error).message === 'stopped') return;
    $('numbers-text').textContent = `The page numbers could not be read: ${e instanceof Error ? e.message : String(e)}`;
    $<HTMLButtonElement>('numbers-check').disabled = false;
    status.numbers = { state: 'error', error: String(e) };
  } finally {
    $('numbers-stop').hidden = true;
    $('numbers-bar').hidden = true;
  }
}

function showNumbers(check: NumberCheck, removed: number): void {
  const rep = check.report;
  const list = $('numbers-list');
  list.textContent = '';
  const parts: string[] = [];
  const good = check.vouched && rep.missing === 0;
  if (!check.vouched) {
    parts.push(`A page number could be read on only ${rep.read} of the ${pictures.length} pictures, too few to check this book. Look through the pictures yourself.`);
  } else if (rep.missing === 0) {
    parts.push(`The printed page numbers run from ${rep.first_page} to ${rep.last_page} without a gap.`);
  } else {
    const places = rep.anomalies.filter(a => a.missing).length;
    parts.push(`Pages seem to be missing in ${places} ${places === 1 ? 'place' : 'places'}. The reader may have turned two leaves at once there; the numbers can also be misread, so look at those pictures.`);
  }
  if (removed) parts.push(`${removed} ${removed === 1 ? 'picture showed' : 'pictures showed'} the same pages as another and ${removed === 1 ? 'was' : 'were'} removed.`);
  if (check.vouched) {
    for (const a of rep.anomalies) {
      if (a.missing) list.append(listItem('Between ', pictureLink(a.after), ' and ', pictureLink(a.before), `: about ${a.missing} ${a.missing === 1 ? 'spread' : 'spreads'} missing.`));
      else if (a.extra) list.append(listItem('Between ', pictureLink(a.after), ' and ', pictureLink(a.before), `: ${a.extra} ${a.extra === 1 ? 'picture' : 'pictures'} more than the page numbers allow (a repeat, or a page shown out of order).`));
      else if (a.unclear) list.append(listItem('Between ', pictureLink(a.after), ' and ', pictureLink(a.before), ': a number was probably misread.'));
    }
    for (const a of rep.restarts) list.append(listItem('After ', pictureLink(a.after), `, the numbering starts again (page ${a.page_after}, then ${a.page_before} at `, pictureLink(a.before), ').'));
  }
  $('numbers-text').textContent = parts.join(' ');
  $('numbers').classList.toggle('good', good);
  list.hidden = list.childElementCount === 0;
  $<HTMLButtonElement>('numbers-check').hidden = true;
}

// ---- the book as a PDF -------------------------------------------------------

let bookMaker: BookMaker | null = null;
let book: Book | null = null;

function resetBook(): void {
  bookMaker?.stop();
  bookMaker = null;
  book = null;
  $('book').classList.remove('good');
  $('book-text').textContent = 'Not made yet. This page can cut every picture into its two pages, flatten them, read their text and make a PDF of the book you can search and copy from. It checks the page numbers first, if that has not been done, so that no picture of the same pages goes in twice. It downloads about 33 MB the first time and takes a few seconds per page on a computer, longer on a phone.';
  $('book-bar').hidden = true;
  $<HTMLButtonElement>('book-make').hidden = false;
  $<HTMLButtonElement>('book-make').disabled = false;
  for (const id of ['book-stop', 'book-pdf', 'book-txt']) $(id).hidden = true;
}

function save(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 120_000);
}

const bookName = (ext: string) => `${videoName.replace(/\.[^.]+$/, '').replace(/[\\/:*?"<>|]+/g, '_')}.${ext}`;

async function makeBook(): Promise<void> {
  if (!lastResult || bookMaker) return;
  const maker = new BookMaker();
  bookMaker = maker;
  $<HTMLButtonElement>('book-make').disabled = true;
  $('book-stop').hidden = false;
  $('book-bar').hidden = false;
  const bar = $('book-bar').firstElementChild as HTMLElement;
  status.book = { state: 'running' };
  const words: Record<BookStep, string> = { loading: 'Getting ready', pages: 'Flattening the pages', reading: 'Reading the text', joining: 'Putting the PDF together' };
  try {
    // the pictures the page numbers show to be repeats (a view of a spread
    // during a fumbled turn, with the clean view beside it) are left out first,
    // as the PC version does before it makes its pages
    if (!numbersChecked) {
      $('book-text').textContent = 'Checking the page numbers first, to leave out pictures of the same pages…';
      bar.style.width = '0%';
      await checkNumbers();
      if (bookMaker !== maker) return;
    }
    const made = await maker.run(pictures, lastResult.region, videoName.replace(/\.[^.]+$/, ''), (step, done, total) => {
      $('book-text').textContent = step === 'pages' ? `${words[step]}: ${done} of ${total} pictures` : step === 'reading' ? `${words[step]}: ${done} of ${total} pages` : `${words[step]}…`;
      bar.style.width = `${Math.round(100 * done / Math.max(1, total))}%`;
    });
    if (bookMaker !== maker) return;
    book = made;
    const unrolled = made.pages.filter(p => p.unrolled).length;
    const repeats = made.leftOut.length;
    $('book-text').textContent = `${made.pages.length} pages, ${unrolled} of them flattened from their lines of text (the others cut out straight), ${Math.round(made.pdf.size / 1e6)} MB. The text can be searched and copied in any PDF reader.` +
      (repeats ? ` ${repeats} ${repeats === 1 ? 'page was' : 'pages were'} left out because ${repeats === 1 ? 'it showed' : 'they showed'} the same page as a better picture near ${repeats === 1 ? 'it' : 'them'}.` : '');
    $('book').classList.add('good');
    $<HTMLButtonElement>('book-make').hidden = true;
    $('book-pdf').hidden = false;
    $('book-txt').hidden = false;
    for (const [k, v] of Object.entries(made.seconds)) addDetail(`book: ${k} (s)`, String(Math.round(v * 10) / 10));
    status.book = {
      state: 'done', pages: made.pages.length, unrolled, pdfBytes: made.pdf.size, seconds: made.seconds,
      leftOut: made.leftOut.map(r => ({ picture: r.page.picture, side: r.page.side, sameAs: { picture: r.sameAs.picture, side: r.sameAs.side }, share: r.share })),
    };
  } catch (e) {
    if (bookMaker !== maker) return;
    $('book-text').textContent = `The PDF could not be made: ${e instanceof Error ? e.message : String(e)}`;
    $<HTMLButtonElement>('book-make').disabled = false;
    status.book = { state: 'error', error: String(e) };
  } finally {
    if (bookMaker === maker) { bookMaker = null; $('book-stop').hidden = true; $('book-bar').hidden = true; }
  }
}

$('book-make').addEventListener('click', () => { void makeBook(); });
$('book-stop').addEventListener('click', () => { resetBook(); status.book = { state: 'stopped' }; });
$('book-pdf').addEventListener('click', () => { if (book) save(book.pdf, bookName('pdf')); });
$('book-txt').addEventListener('click', () => { if (book) save(book.text, bookName('txt')); });

$('numbers-check').addEventListener('click', () => { void checkNumbers(); });
$('numbers-stop').addEventListener('click', () => {
  reader?.stop();
  reader = null;
  resetNumbers();
  status.numbers = { state: 'stopped' };
});

function fmtClock(s: number): string {
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s - 60 * m)).padStart(2, '0')}`;
}

document.querySelectorAll<HTMLInputElement>('input[name="show"]').forEach(r => r.addEventListener('change', () => {
  $('gallery').classList.toggle('marked-only', r.checked && r.value === 'marked');
}));

$('download').addEventListener('click', () => {
  const zip = buildZip(pictures.map(p => ({ name: p.name, data: p.full, size: p.size, crc: p.crc })));
  const url = URL.createObjectURL(zip);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${videoName.replace(/[\\/:*?"<>|]+/g, '_')} pages.zip`;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 120_000);
  const note = $('download-note');
  note.textContent = `Saving ${Math.round(zip.size / 1e6)} MB. Your browser shows where the file went.`;
  note.hidden = false;
});

$('again').addEventListener('click', () => {
  closeViewer();
  resetNumbers();
  resetBook();
  lastResult = null;
  for (const u of urls.splice(0)) URL.revokeObjectURL(u);
  pictures = [];
  $('gallery').textContent = '';
  void clearStore();
  status.state = 'idle';
  show('start');
});

// ---- the viewer ------------------------------------------------------------

const viewer = $<HTMLDialogElement>('viewer');
const viewerImg = $<HTMLImageElement>('viewer-img');
let viewing = -1;
let viewerUrl = '';

function openViewer(i: number): void {
  if (i < 0 || i >= pictures.length) return;
  viewing = i;
  const p = pictures[i];
  if (viewerUrl) URL.revokeObjectURL(viewerUrl);
  viewerUrl = URL.createObjectURL(p.full);
  viewerImg.src = viewerUrl;
  viewerImg.alt = `Picture ${p.number}, full size`;
  $('viewer-title').textContent = `#${p.number} of ${pictures.length}, at ${fmtClock(p.seconds)}${p.page != null ? `, page ${p.page}` : ''}`;
  $('viewer-flags').textContent = p.flags.join(' · ');
  const save = $<HTMLAnchorElement>('viewer-save');
  save.href = viewerUrl;
  save.download = p.name;
  viewerImg.parentElement!.classList.remove('zoomed');
  if (!viewer.open) viewer.showModal();
}

function closeViewer(): void {
  if (viewer.open) viewer.close();
}

viewer.addEventListener('close', () => {
  if (viewerUrl) URL.revokeObjectURL(viewerUrl);
  viewerUrl = '';
  viewerImg.removeAttribute('src');
});
$('viewer-close').addEventListener('click', closeViewer);
$('viewer-prev').addEventListener('click', () => openViewer(viewing - 1));
$('viewer-next').addEventListener('click', () => openViewer(viewing + 1));
viewerImg.addEventListener('click', () => viewerImg.parentElement!.classList.toggle('zoomed'));
viewer.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft') openViewer(viewing - 1);
  else if (e.key === 'ArrowRight') openViewer(viewing + 1);
});
let touchX = 0;
viewer.addEventListener('touchstart', e => { touchX = e.touches[0].clientX; }, { passive: true });
viewer.addEventListener('touchend', e => {
  if (viewerImg.parentElement!.classList.contains('zoomed')) return;
  const dx = e.changedTouches[0].clientX - touchX;
  if (Math.abs(dx) > 60) openViewer(viewing + (dx < 0 ? 1 : -1));
});

window.addEventListener('beforeunload', e => {
  if (scan) { e.preventDefault(); }
});
