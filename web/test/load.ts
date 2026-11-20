// Read the fixtures: arrays and values written by the Python code.
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

type Typed = Uint8Array | Int32Array | Float32Array | Float64Array | Int16Array | Uint16Array;

export class Fixture {
  readonly manifest: Record<string, { dtype?: string; shape?: number[]; json?: unknown }>;
  readonly dir: string;
  constructor(dir: string) {
    this.dir = dir;
    this.manifest = JSON.parse(readFileSync(join(dir, 'manifest.json'), 'utf8'));
  }
  static exists(dir: string): boolean { return existsSync(join(dir, 'manifest.json')); }
  shape(name: string): number[] { return this.manifest[name].shape!; }
  value<T = unknown>(name: string): T { return this.manifest[name].json as T; }
  array(name: string): Typed {
    const entry = this.manifest[name];
    const buf = readFileSync(join(this.dir, `${name}.bin`));
    const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    switch (entry.dtype) {
      case 'uint8': return new Uint8Array(ab);
      case 'int32': return new Int32Array(ab);
      case 'int64': return new Int32Array(Array.from(new BigInt64Array(ab), Number));
      case 'float32': return new Float32Array(ab);
      case 'float64': return new Float64Array(ab);
      case 'int16': return new Int16Array(ab);
      case 'uint16': return new Uint16Array(ab);
      default: throw new Error(`dtype ${entry.dtype}`);
    }
  }
  u8(name: string): Uint8Array { return this.array(name) as Uint8Array; }
  f32(name: string): Float32Array { return this.array(name) as Float32Array; }
}

/** Count of positions where two arrays differ. */
export function differences(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) throw new Error(`length ${a.length} vs ${b.length}`);
  let n = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) n++;
  return n;
}

export function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}
