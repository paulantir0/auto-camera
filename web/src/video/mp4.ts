// A small MP4 / QuickTime demuxer for the first video track: the sample
// table and the decoder configuration, read from a File by byte ranges so a
// 2 GB video is never held in memory.

export interface Sample {
  offset: number;
  size: number;
  dts: number; // in track timescale
  cts: number; // presentation time, in track timescale
  sync: boolean;
}

export interface VideoTrack {
  codec: string; // WebCodecs codec string
  codedWidth: number;
  codedHeight: number;
  description: Uint8Array | undefined; // hvcC / avcC payload
  timescale: number;
  duration: number; // seconds
  fps: number;
  samples: Sample[]; // decode order
  /** presentation index -> decode index */
  presentation: Int32Array;
  /** decode index -> presentation index */
  decodeToPresentation: Int32Array;
  format: string; // sample entry type, e.g. hvc1
}

type Reader = (offset: number, length: number) => Promise<DataView>;

export function fileReader(file: Blob): Reader {
  return async (offset, length) => {
    const buf = await file.slice(offset, offset + length).arrayBuffer();
    return new DataView(buf);
  };
}

interface Box { type: string; start: number; size: number; header: number }

function boxAt(view: DataView, pos: number, end: number): Box | null {
  if (pos + 8 > end) return null;
  let size = view.getUint32(pos);
  const type = String.fromCharCode(view.getUint8(pos + 4), view.getUint8(pos + 5), view.getUint8(pos + 6), view.getUint8(pos + 7));
  let header = 8;
  if (size === 1) {
    size = Number(view.getBigUint64(pos + 8));
    header = 16;
  } else if (size === 0) size = end - pos;
  if (size < header) return null;
  return { type, start: pos, size, header };
}

function children(view: DataView, start: number, end: number): Box[] {
  const out: Box[] = [];
  let pos = start;
  while (pos < end) {
    const box = boxAt(view, pos, end);
    if (!box) break;
    out.push(box);
    pos += box.size;
  }
  return out;
}

const find = (boxes: Box[], type: string) => boxes.find(b => b.type === type);

/** Locate the moov box at the top level of the file without reading the media data. */
async function readMoov(read: Reader, fileSize: number): Promise<DataView> {
  let pos = 0;
  while (pos + 8 <= fileSize) {
    const head = await read(pos, Math.min(16, fileSize - pos));
    let size = head.getUint32(0);
    const type = String.fromCharCode(head.getUint8(4), head.getUint8(5), head.getUint8(6), head.getUint8(7));
    if (size === 1) size = Number(head.getBigUint64(8));
    else if (size === 0) size = fileSize - pos;
    if (size < 8) throw new Error('This file does not look like an MP4 or MOV video.');
    if (type === 'moov') return read(pos, size);
    pos += size;
  }
  throw new Error('No video index (moov) found: the file may be incomplete.');
}

function fullBoxBody(box: Box): number { return box.start + box.header + 4; }

function hevcCodecString(hvcC: DataView): string {
  // ISO/IEC 14496-15 Annex E
  const b1 = hvcC.getUint8(1);
  const profileSpace = b1 >> 6;
  const tier = (b1 >> 5) & 1;
  const profileIdc = b1 & 0x1f;
  const compat = hvcC.getUint32(2);
  let reversed = 0;
  for (let i = 0; i < 32; i++) if (compat & (1 << i)) reversed |= 1 << (31 - i);
  const constraint: number[] = [];
  for (let i = 0; i < 6; i++) constraint.push(hvcC.getUint8(6 + i));
  while (constraint.length && constraint[constraint.length - 1] === 0) constraint.pop();
  const level = hvcC.getUint8(12);
  const space = ['', 'A', 'B', 'C'][profileSpace];
  let s = `.${space}${profileIdc}.${(reversed >>> 0).toString(16)}.${tier ? 'H' : 'L'}${level}`;
  for (const c of constraint) s += `.${c.toString(16).toUpperCase()}`;
  return s;
}

function avcCodecString(avcC: DataView): string {
  const hex = (v: number) => v.toString(16).padStart(2, '0');
  return `.${hex(avcC.getUint8(1))}${hex(avcC.getUint8(2))}${hex(avcC.getUint8(3))}`;
}

export async function readVideoTrack(file: Blob): Promise<VideoTrack> {
  const read = fileReader(file);
  const moov = await readMoov(read, file.size);
  const top = children(moov, 8, moov.byteLength);
  for (const trak of top.filter(b => b.type === 'trak')) {
    const tk = children(moov, trak.start + trak.header, trak.start + trak.size);
    const mdia = find(tk, 'mdia');
    if (!mdia) continue;
    const md = children(moov, mdia.start + mdia.header, mdia.start + mdia.size);
    const hdlr = find(md, 'hdlr');
    if (!hdlr) continue;
    const handler = String.fromCharCode(...[0, 1, 2, 3].map(i => moov.getUint8(fullBoxBody(hdlr) + 4 + i)));
    if (handler !== 'vide') continue;
    const mdhd = find(md, 'mdhd')!;
    const mv = moov.getUint8(mdhd.start + mdhd.header);
    const timescale = mv === 1 ? moov.getUint32(fullBoxBody(mdhd) + 16) : moov.getUint32(fullBoxBody(mdhd) + 8);
    const minf = find(md, 'minf')!;
    const mi = children(moov, minf.start + minf.header, minf.start + minf.size);
    const stbl = find(mi, 'stbl')!;
    const st = children(moov, stbl.start + stbl.header, stbl.start + stbl.size);
    const stsd = find(st, 'stsd')!;
    // first sample entry
    const entry = boxAt(moov, fullBoxBody(stsd) + 4, stsd.start + stsd.size)!;
    const format = entry.type;
    const e0 = entry.start + entry.header;
    const codedWidth = moov.getUint16(e0 + 24), codedHeight = moov.getUint16(e0 + 26);
    const inner = children(moov, e0 + 78, entry.start + entry.size);
    let codec = format, description: Uint8Array | undefined;
    const hvcC = find(inner, 'hvcC'), avcC = find(inner, 'avcC');
    if (hvcC) {
      const view = new DataView(moov.buffer, moov.byteOffset + hvcC.start + hvcC.header, hvcC.size - hvcC.header);
      codec = format + hevcCodecString(view);
      description = new Uint8Array(view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength));
    } else if (avcC) {
      const view = new DataView(moov.buffer, moov.byteOffset + avcC.start + avcC.header, avcC.size - avcC.header);
      codec = format + avcCodecString(view);
      description = new Uint8Array(view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength));
    }
    // sample table
    const u32 = (p: number) => moov.getUint32(p);
    const stszBox = find(st, 'stsz'), stz2 = find(st, 'stz2');
    const sizes: number[] = [];
    if (stszBox) {
      const b = fullBoxBody(stszBox);
      const fixed = u32(b), count = u32(b + 4);
      for (let i = 0; i < count; i++) sizes.push(fixed || u32(b + 8 + 4 * i));
    } else if (stz2) {
      const b = fullBoxBody(stz2);
      const field = moov.getUint8(b + 3), count = u32(b + 4);
      for (let i = 0; i < count; i++) {
        if (field === 16) sizes.push(moov.getUint16(b + 8 + 2 * i));
        else if (field === 8) sizes.push(moov.getUint8(b + 8 + i));
        else sizes.push((moov.getUint8(b + 8 + (i >> 1)) >> (i & 1 ? 0 : 4)) & 15);
      }
    }
    const n = sizes.length;
    const chunkOffsets: number[] = [];
    const stco = find(st, 'stco'), co64 = find(st, 'co64');
    if (stco) { const b = fullBoxBody(stco); const c = u32(b); for (let i = 0; i < c; i++) chunkOffsets.push(u32(b + 4 + 4 * i)); }
    if (co64) { const b = fullBoxBody(co64); const c = u32(b); for (let i = 0; i < c; i++) chunkOffsets.push(Number(moov.getBigUint64(b + 4 + 8 * i))); }
    const stsc = find(st, 'stsc')!;
    const sb = fullBoxBody(stsc), entries = u32(sb);
    const offsets = new Array<number>(n);
    let sample = 0;
    for (let e = 0; e < entries; e++) {
      const firstChunk = u32(sb + 4 + 12 * e) - 1;
      const perChunk = u32(sb + 8 + 12 * e);
      const nextFirst = e + 1 < entries ? u32(sb + 4 + 12 * (e + 1)) - 1 : chunkOffsets.length;
      for (let c = firstChunk; c < nextFirst && sample < n; c++) {
        let pos = chunkOffsets[c];
        for (let k = 0; k < perChunk && sample < n; k++) { offsets[sample] = pos; pos += sizes[sample]; sample++; }
      }
    }
    const stts = find(st, 'stts')!;
    const tb = fullBoxBody(stts);
    const dts = new Array<number>(n);
    let t = 0;
    sample = 0;
    for (let e = 0, c = u32(tb); e < c; e++) {
      const count = u32(tb + 4 + 8 * e), delta = u32(tb + 8 + 8 * e);
      for (let k = 0; k < count && sample < n; k++) { dts[sample++] = t; t += delta; }
    }
    while (sample < n) dts[sample++] = t;
    const cts = dts.slice();
    const ctts = find(st, 'ctts');
    if (ctts) {
      const version = moov.getUint8(ctts.start + ctts.header);
      const cb = fullBoxBody(ctts);
      sample = 0;
      for (let e = 0, c = u32(cb); e < c; e++) {
        const count = u32(cb + 4 + 8 * e);
        const raw = u32(cb + 8 + 8 * e);
        const off = version === 1 ? (raw | 0) : raw > 0x7fffffff ? raw - 0x100000000 : raw;
        for (let k = 0; k < count && sample < n; k++) { cts[sample] = dts[sample] + off; sample++; }
      }
    }
    const syncSet = new Set<number>();
    const stss = find(st, 'stss');
    if (stss) { const b = fullBoxBody(stss); for (let i = 0, c = u32(b); i < c; i++) syncSet.add(u32(b + 4 + 4 * i) - 1); }
    const samples: Sample[] = [];
    for (let i = 0; i < n; i++) samples.push({ offset: offsets[i], size: sizes[i], dts: dts[i], cts: cts[i], sync: stss ? syncSet.has(i) : true });
    const order = [...samples.keys()].sort((a, b) => samples[a].cts - samples[b].cts || a - b);
    const presentation = Int32Array.from(order);
    const decodeToPresentation = new Int32Array(n);
    order.forEach((d, p) => { decodeToPresentation[d] = p; });
    const span = n > 1 ? (samples[order[n - 1]].cts - samples[order[0]].cts) / timescale : 0;
    let fps = n > 1 ? (n - 1) / span : 0;
    if (Math.abs(fps - Math.round(fps)) < 0.01) fps = Math.round(fps);
    return { codec, codedWidth, codedHeight, description, timescale, duration: n / (fps || 1), fps, samples, presentation, decodeToPresentation, format };
  }
  throw new Error('No video track found in this file.');
}

/** Microsecond timestamp WebCodecs is given for a sample. */
export const sampleTimestamp = (track: VideoTrack, s: Sample) => Math.round(s.cts * 1e6 / track.timescale);

/**
 * Where decoding must start to show the frame at presentation index p: the
 * last sync sample at or before it in decode order, or the one before that
 * when the frame is a leading picture of that sync sample (earlier
 * presentation time, so it may need the previous group of pictures).
 */
export function decodeStart(track: VideoTrack, p: number): number {
  const d = track.presentation[p];
  let k = d;
  while (k > 0 && !track.samples[k].sync) k--;
  if (track.samples[d].cts < track.samples[k].cts) {
    let j = k - 1;
    while (j > 0 && !track.samples[j].sync) j--;
    return Math.max(0, j);
  }
  return k;
}

/** Read the bytes of samples [first, last] in decode order, in as few reads as the layout allows. */
export async function readSamples(file: Blob, track: VideoTrack, first: number, last: number): Promise<Uint8Array[]> {
  const out: Uint8Array[] = [];
  let i = first;
  while (i <= last) {
    // a run of samples that are contiguous in the file, up to 32 MB
    let j = i;
    let end = track.samples[i].offset + track.samples[i].size;
    while (j + 1 <= last && track.samples[j + 1].offset === end && end - track.samples[i].offset < 32 << 20) {
      j++;
      end += track.samples[j].size;
    }
    const start = track.samples[i].offset;
    const buf = new Uint8Array(await file.slice(start, end).arrayBuffer());
    for (let k = i; k <= j; k++) {
      const s = track.samples[k];
      out.push(buf.subarray(s.offset - start, s.offset - start + s.size));
    }
    i = j + 1;
  }
  return out;
}
