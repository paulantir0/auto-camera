// Python's round(x, n) for floats: the exact binary value rounded to n
// decimals, ties to even, then read back as the nearest float.

const TWO52 = 2n ** 52n;

function exactRational(x: number): [bigint, bigint] {
  // x = num / den exactly, den a power of two
  const buf = new DataView(new ArrayBuffer(8));
  buf.setFloat64(0, x);
  const bits = buf.getBigUint64(0);
  const sign = bits >> 63n ? -1n : 1n;
  const exp = Number((bits >> 52n) & 0x7ffn);
  const frac = bits & (TWO52 - 1n);
  if (exp === 0) return [sign * frac, 2n ** 1074n];
  const mant = frac | TWO52;
  const e = exp - 1075;
  return e >= 0 ? [sign * mant * 2n ** BigInt(e), 1n] : [sign * mant, 2n ** BigInt(-e)];
}

export function pyRound(x: number, digits: number): number {
  if (!Number.isFinite(x) || x === 0) return x;
  const [num, den] = exactRational(x);
  const scale = 10n ** BigInt(digits);
  const scaled = num * scale; // value * 10^digits = scaled / den
  const negative = scaled < 0n;
  const a = negative ? -scaled : scaled;
  let q = a / den;
  const r = a - q * den;
  const twice = 2n * r;
  if (twice > den || (twice === den && q % 2n === 1n)) q += 1n;
  const s = q.toString();
  const value = Number(`${negative ? '-' : ''}${s}e-${digits}`);
  return value === 0 && negative ? -0 : value;
}

/** Python's f"{x:0W.Pf}" for non-negative x: the exact value rounded half to even. */
export function pyFixed(x: number, width: number, digits: number): string {
  const [num, den] = exactRational(x);
  const scale = 10n ** BigInt(digits);
  const a = num * scale;
  let q = a / den;
  const r = a - q * den;
  if (2n * r > den || (2n * r === den && q % 2n === 1n)) q += 1n;
  let s = q.toString().padStart(digits + 1, '0');
  s = `${s.slice(0, s.length - digits)}.${s.slice(s.length - digits)}`;
  return s.padStart(width, '0');
}
