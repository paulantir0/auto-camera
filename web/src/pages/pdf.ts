// The book's PDF from Tesseract's one-page PDFs, and the JPEG detail that sets
// a page's printed size.
//
// Python runs `tesseract PAGE.jpg OUT pdf` per page and joins the results with
// pypdf; here Tesseract.js writes the same one-page PDFs (the page's JPEG as
// it is, with the recognised text invisibly over it in Tesseract's glyphless
// font) and joinPdfs puts them into one file. Each page's objects are copied
// with new numbers; objects that are the same in every page (the font) are
// written once. The result is a Blob made of the pieces, so a long book is
// never held twice in memory.

const latin1 = (bytes: Uint8Array) => {
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return s;
};
const toBytes = (s: string) => Uint8Array.from(s, c => c.charCodeAt(0) & 255);

/**
 * Set a JPEG's resolution (JFIF density, dots per inch), as Python's Pillow
 * writes it with save(dpi=...): Tesseract reads it to size the PDF page.
 * A JFIF header is added when the encoder wrote none.
 */
export function setJpegDpi(jpeg: Uint8Array, dpi: number): Uint8Array {
  const d = Math.max(1, Math.min(65535, roundHalfEven(dpi)));
  if (jpeg[0] !== 0xff || jpeg[1] !== 0xd8) throw new Error('not a JPEG');
  if (jpeg[2] === 0xff && jpeg[3] === 0xe0 && latin1(jpeg.subarray(6, 11)) === 'JFIF\0') {
    const out = jpeg.slice();
    out[13] = 1; // units: dots per inch
    out[14] = d >> 8; out[15] = d & 255;
    out[16] = d >> 8; out[17] = d & 255;
    return out;
  }
  const app0 = Uint8Array.from([0xff, 0xe0, 0, 16, 0x4a, 0x46, 0x49, 0x46, 0, 1, 1, 1, d >> 8, d & 255, d >> 8, d & 255, 0, 0]);
  const out = new Uint8Array(jpeg.length + app0.length);
  out.set(jpeg.subarray(0, 2), 0);
  out.set(app0, 2);
  out.set(jpeg.subarray(2), 2 + app0.length);
  return out;
}

function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

interface PdfObject { num: number; head: string; tail: [number, number] }  // head: up to the stream (or the whole object), as Latin-1; tail: where the rest lies in the file

/** The objects of a PDF with a classic cross-reference table (as Tesseract writes it). */
function readObjects(pdf: Uint8Array): Map<number, PdfObject> {
  const end = latin1(pdf.subarray(Math.max(0, pdf.length - 64)));
  const at = /startxref\s+(\d+)/.exec(end);
  if (!at) throw new Error('no startxref');
  const xref = latin1(pdf.subarray(Number(at[1]), Math.min(pdf.length, Number(at[1]) + 64 * 1024)));
  const m = /^xref\s+(\d+)\s+(\d+)\s+/.exec(xref);
  if (!m) throw new Error('no cross-reference table');
  const first = Number(m[1]), count = Number(m[2]);
  const entries = xref.slice(m[0].length).split(/\r?\n|\r/).filter(l => /^\d{10} \d{5} [nf]/.test(l.trim())).slice(0, count);
  const offsets: [number, number][] = [];
  entries.forEach((line, k) => { const [off, , kind] = line.trim().split(' '); if (kind === 'n') offsets.push([first + k, Number(off)]); });
  offsets.sort((a, b) => a[1] - b[1]);
  const objects = new Map<number, PdfObject>();
  offsets.forEach(([num, off], k) => {
    const next = k + 1 < offsets.length ? offsets[k + 1][1] : Number(at[1]);
    const bytes = pdf.subarray(off, next);
    // the object ends at its last 'endobj'
    let e = bytes.length - 6;
    while (e >= 0 && !(bytes[e] === 0x65 && bytes[e + 1] === 0x6e && bytes[e + 2] === 0x64 && bytes[e + 3] === 0x6f && bytes[e + 4] === 0x62 && bytes[e + 5] === 0x6a)) e--;
    if (e < 0) throw new Error(`object ${num} has no end`);
    const text = latin1(bytes.subarray(0, Math.min(e, 8192)));
    const stream = /\bstream(\r\n|\n)/.exec(text);
    const head = text.slice(0, stream ? stream.index : e).replace(/^\s*\d+\s+0\s+obj\s*/, '');
    const tail: [number, number] = stream ? [off + stream.index, off + e + 6] : [0, 0];
    objects.set(num, { num, head, tail });
  });
  return objects;
}

/** The numbers of the objects `head` refers to. */
const refs = (head: string) => [...head.matchAll(/(\d+)\s+0\s+R\b/g)].map(r => Number(r[1]));

const fnv = (s: string, bytes: Uint8Array) => {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193); }
  for (let i = 0; i < bytes.length; i++) { h ^= bytes[i]; h = Math.imul(h, 0x01000193); }
  return h >>> 0;
};

/** One PDF of the pages, in order, from Tesseract's one-page PDFs (read one at a time); `title` goes into its information. */
export async function joinPdfs(pages: Blob[], title: string): Promise<Blob> {
  const parts: (Blob | string)[] = [];
  let position = 0;
  const offsets: number[] = [0, 0, 0]; // 1: catalog, 2: page tree, then the copied objects
  const write = (piece: Blob | string) => { parts.push(piece); position += typeof piece === 'string' ? piece.length : piece.size; };
  write('%PDF-1.5\n%\xe2\xe3\xcf\xd3\n');
  let next = 3;
  const shared = new Map<string, number>();
  const kids: number[] = [];
  for (const file of pages) {
    const objects = readObjects(new Uint8Array(await file.arrayBuffer()));
    const page = [...objects.values()].find(o => /\/Type\s*\/Page\b(?!s)/.test(o.head));
    if (!page) throw new Error('a page PDF without a page');
    // what the page needs, without its parent
    const needed = new Set<number>();
    const walk = (num: number) => {
      if (needed.has(num) || !objects.has(num)) return;
      needed.add(num);
      const o = objects.get(num)!;
      for (const r of refs(o.head.replace(/\/Parent\s+\d+\s+0\s+R/, ''))) walk(r);
    };
    walk(page.num);
    const map = new Map<number, number>();
    for (const num of [...needed].sort((a, b) => a - b)) {
      const o = objects.get(num)!;
      // small objects the same in every page (the font) are written once; the page, its text and its image never are
      const size = o.tail[1] - o.tail[0];
      const small = num !== page.num && size <= 65536;
      const key = small ? `${num}:${size}:${fnv(o.head, new Uint8Array(await file.slice(o.tail[0], o.tail[1]).arrayBuffer()))}` : '';
      if (small && shared.has(key)) { map.set(num, shared.get(key)!); continue; }
      map.set(num, next);
      if (small) shared.set(key, next);
      next++;
    }
    for (const num of [...needed].sort((a, b) => a - b)) {
      const o = objects.get(num)!;
      const target = map.get(num)!;
      if (offsets[target] !== undefined) continue; // written with an earlier page
      let head = o.head.replace(/(\d+)\s+0\s+R\b/g, (all, n) => (map.has(Number(n)) ? `${map.get(Number(n))} 0 R` : all));
      if (num === page.num) head = head.replace(/\/Parent\s+\d+\s+0\s+R/, '/Parent 2 0 R');
      offsets[target] = position;
      write(`${target} 0 obj\n${head}`);
      if (o.tail[1] > o.tail[0]) write(file.slice(o.tail[0], o.tail[1])); else write('endobj');
      write('\n');
    }
    kids.push(map.get(page.num)!);
  }
  offsets[2] = position;
  write(`2 0 obj\n<< /Type /Pages /Kids [${kids.map(k => `${k} 0 R`).join(' ')}] /Count ${kids.length} >>\nendobj\n`);
  offsets[1] = position;
  write('1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n');
  const info = next++;
  offsets[info] = position;
  const utf16 = Array.from(title, c => c.charCodeAt(0).toString(16).padStart(4, '0')).join('').toUpperCase();
  write(`${info} 0 obj\n<< /Producer (Book Scanner with Tesseract) /Title <FEFF${utf16}> >>\nendobj\n`);
  const xref = position;
  let table = `xref\n0 ${next}\n0000000000 65535 f \n`;
  for (let n = 1; n < next; n++) table += `${String(offsets[n] ?? 0).padStart(10, '0')} 00000 n \n`;
  write(table);
  write(`trailer\n<< /Size ${next} /Root 1 0 R /Info ${info} 0 R >>\nstartxref\n${xref}\n%%EOF\n`);
  return new Blob(parts.map(p => (typeof p === 'string' ? toBytes(p) : p)) as BlobPart[], { type: 'application/pdf' });
}
