#!/usr/bin/env node
// Check that the files to be served contain nothing of the owner's: no local
// paths, no names or e-mail, no word from any video's file name, no media.
//
//   node tools/privacy-check.mjs [dist]
import { readdir, readFile, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';

const root = fileURLToPath(new URL('..', import.meta.url));
const dist = join(root, process.argv[2] ?? 'dist');

// What is the owner's alone (names, the home folder, where the list of videos is) is not in git: it is
// read from tools/privacy-check.local.json on the owner's machine,
//   {"markers": ["..."], "words": ["..."], "inventory": "path/to/video_inventory.csv"}
const localFile = join(root, 'tools', 'privacy-check.local.json');
const local = existsSync(localFile) ? JSON.parse(await readFile(localFile, 'utf8')) : {};
const markers = ['/Users/', '/tmp/claude', 'auto-camera', 'Videos/', 'Pictures/', 'scratchpad', ...(local.markers ?? [])];
// names as whole words, in any case
const words = (local.words ?? []).map(w => new RegExp(`\\b${w}\\b`, 'i'));
const exact = ['.M4V', '.MOV', 'IMG-00']; // as the owner's files are named; case matters
try {
  const email = execFileSync('git', ['config', 'user.email'], { cwd: root }).toString().trim();
  if (email) markers.push(email, email.split('@')[0]);
  const user = execFileSync('git', ['config', 'user.name'], { cwd: root }).toString().trim();
  if (user) markers.push(user);
} catch { /* no git */ }
// every word of six letters or more from the videos' file names (book titles, authors)
const inventory = local.inventory ?? join(root, '..', 'data', 'corpus', 'video_inventory.csv');
const allowed = new Set(['commentary', 'primary', 'sources', 'writings', 'through', 'church']); // checked below as well, just not fatal
const titleWords = new Set();
if (existsSync(inventory)) {
  for (const line of (await readFile(inventory, 'utf8')).split('\n').slice(1)) {
    const name = line.startsWith('"') ? line.slice(1, line.indexOf('"', 1)) : line.split(',')[0];
    for (const w of name.match(/[A-Za-z]{6,}/g) ?? []) titleWords.add(w.toLowerCase());
  }
}

const files = [];
async function walk(dir) {
  for (const e of await readdir(dir, { withFileTypes: true })) {
    if (e.name === '.vercel' || e.name === '.vercelignore') continue; // the CLI's link to the project and its ignore list; never uploaded
    const p = join(dir, e.name);
    if (e.isDirectory()) await walk(p); else files.push(p);
  }
}
await walk(dist);
let problems = 0;
const notes = [];
for (const f of files) {
  const rel = relative(dist, f);
  if (rel === 'source/ffmpeg-7.1.2.tar.xz') {
    // FFmpeg's release tarball, served for its licence: it must be exactly the published one
    const sum = createHash('sha256').update(await readFile(f)).digest('hex');
    if (sum !== '089bc60fb59d6aecc5d994ff530fd0dcb3ee39aa55867849a2bbc4e555f9c304') { console.log(`NOT THE RELEASE: ${rel}`); problems++; }
    continue;
  }
  if (!/\.(js|mjs|html|css|svg|json|txt|wasm|onnx|task|tflite|traineddata)$/.test(f) && !['source/hevcdec.c', 'source/build-hevc.sh'].includes(rel)) { console.log(`NOT ALLOWED (type): ${rel}`); problems++; continue; }
  // third-party binaries (FFmpeg, ONNX Runtime, the OCR models) come from their releases, untouched: an
  // ordinary English word in their string tables is reported, not failed; paths, names and e-mail still fail
  const binary = /\.(wasm|onnx|task|tflite|traineddata)$/.test(f) || /tesseract-core.*\.wasm\.js$/.test(f);
  // WebAssembly is binary; its strings (file names in error messages and the like) are read as Latin-1
  const text = await readFile(f, binary ? 'latin1' : 'utf8');
  const lower = text.toLowerCase();
  for (const m of markers) if (lower.includes(m.toLowerCase())) { console.log(`FOUND "${m}" in ${rel}`); problems++; }
  for (const m of exact) if (text.includes(m)) { console.log(`FOUND "${m}" in ${rel}`); problems++; }
  for (const re of words) if (re.test(text)) { console.log(`FOUND ${re} in ${rel}`); problems++; }
  for (const w of titleWords) {
    const re = new RegExp(`\\b${w}\\b`, 'i');
    if (re.test(text)) {
      if (allowed.has(w) || binary) notes.push(`${binary ? 'word' : 'common word'} "${w}" in ${rel}${binary ? ' (third-party binary)' : ''}`);
      else { console.log(`FOUND title word "${w}" in ${rel}`); problems++; }
    }
  }
  // connections to other hosts written into code: fetch/import/src of an absolute URL
  for (const m of text.matchAll(/(?:fetch\(|importScripts\(|src=|href=|url\()\s*["'`]?(https?:)?\/\/[^"'`)\s]+/g)) {
    console.log(`EXTERNAL REFERENCE in ${rel}: ${m[0].slice(0, 80)}`);
    problems++;
  }
}
let size = 0;
for (const f of files) size += (await stat(f)).size;
console.log(`${files.length} files, ${(size / 1e6).toFixed(1)} MB, ${titleWords.size} title words and ${markers.length} markers checked`);
for (const n of notes) console.log(`note: ${n}`);
if (problems) { console.log(`${problems} problem(s)`); process.exit(1); }
console.log('no personal data and no external references found');
