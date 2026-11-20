// Unpack a ZIP of stored (uncompressed) entries, checking each CRC-32: a
// test of the archive the page writes.
import { open, mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; }
  return t;
})();
const crc32 = b => { let c = ~0 >>> 0; for (let i = 0; i < b.length; i++) c = TABLE[(c ^ b[i]) & 0xff] ^ (c >>> 8); return ~c >>> 0; };

export async function unzipStored(path, dir) {
  await mkdir(dir, { recursive: true });
  const fh = await open(path, 'r');
  try {
    const { size } = await fh.stat();
    const tail = Buffer.alloc(Math.min(size, 65536 + 22));
    await fh.read(tail, 0, tail.length, size - tail.length);
    const eocd = tail.lastIndexOf(Buffer.from([0x50, 0x4b, 0x05, 0x06]));
    if (eocd < 0) throw new Error('no end of central directory');
    const count = tail.readUInt16LE(eocd + 10), cdSize = tail.readUInt32LE(eocd + 12), cdOffset = tail.readUInt32LE(eocd + 16);
    const cd = Buffer.alloc(cdSize);
    await fh.read(cd, 0, cdSize, cdOffset);
    const names = [];
    let p = 0;
    for (let i = 0; i < count; i++) {
      if (cd.readUInt32LE(p) !== 0x02014b50) throw new Error('bad central directory');
      const method = cd.readUInt16LE(p + 10), crc = cd.readUInt32LE(p + 16), csize = cd.readUInt32LE(p + 20);
      const nlen = cd.readUInt16LE(p + 28), elen = cd.readUInt16LE(p + 30), clen = cd.readUInt16LE(p + 32), off = cd.readUInt32LE(p + 42);
      const name = cd.toString('utf8', p + 46, p + 46 + nlen);
      if (method !== 0) throw new Error(`entry ${name} is compressed`);
      const lh = Buffer.alloc(30);
      await fh.read(lh, 0, 30, off);
      if (lh.readUInt32LE(0) !== 0x04034b50) throw new Error(`bad local header for ${name}`);
      const start = off + 30 + lh.readUInt16LE(26) + lh.readUInt16LE(28);
      const data = Buffer.alloc(csize);
      await fh.read(data, 0, csize, start);
      if (crc32(data) !== crc) throw new Error(`CRC mismatch in ${name}`);
      await writeFile(join(dir, name), data);
      names.push(name);
      p += 46 + nlen + elen + clen;
    }
    return names;
  } finally {
    await fh.close();
  }
}
