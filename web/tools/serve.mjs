#!/usr/bin/env node
// A small static file server for local testing, with the same headers as the
// production host (see ../vercel.json). Only listens on 127.0.0.1.
//
//   node tools/serve.mjs [--root dist] [--port 8765]
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = fileURLToPath(new URL('..', import.meta.url));
const args = process.argv.slice(2);
const option = (name, fallback) => {
  const at = args.indexOf(name);
  return at >= 0 ? args[at + 1] : fallback;
};
const root = resolve(here, option('--root', 'dist'));
const port = Number(option('--port', '8765'));

const types = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.txt': 'text/plain; charset=utf-8',
  '.webmanifest': 'application/manifest+json',
};

// The same header rules as the production host: every rule whose source
// matches applies, later rules overriding earlier keys.
const config = JSON.parse(await readFile(join(here, 'vercel.json'), 'utf8'));
const rules = config.headers.map(rule => ({
  test: new RegExp(`^${rule.source.includes('(') ? rule.source : rule.source.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}$`),
  headers: Object.fromEntries(rule.headers.map(({ key, value }) => [key, value])),
}));
export const headersFor = path => Object.assign({}, ...rules.filter(r => r.test.test(path)).map(r => r.headers));

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url ?? '/', 'http://localhost');
    let path = normalize(decodeURIComponent(url.pathname)).replace(/^(\.\.[/\\])+/, '');
    if (path.endsWith('/')) path += 'index.html';
    const file = join(root, path);
    if (!file.startsWith(root)) throw new Error('outside root');
    const info = await stat(file);
    if (!info.isFile()) throw new Error('not a file');
    const body = await readFile(file);
    res.writeHead(200, { ...headersFor(url.pathname), 'Content-Type': types[extname(file)] ?? 'application/octet-stream', 'Cache-Control': 'no-store' });
    res.end(body);
  } catch {
    res.writeHead(404, { ...headersFor('/404'), 'Content-Type': 'text/plain' });
    res.end('not found');
  }
});
// --port 0: any free port; the line below says which
server.listen(port, '127.0.0.1', () => console.log(`serving ${root} on http://127.0.0.1:${server.address().port}/`));
