import type { FormatId, FormatSpec, Matrix, PresetId, QuantConfig, QuantResult } from "@/types/quantization";

export const FORMATS: FormatSpec[] = [
  { id: "BF16", label: "BF16", kind: "float", bits: 16, exponentBits: 8, mantissaBits: 7, maxValue: 3.39e38 },
  { id: "FP16", label: "FP16", kind: "float", bits: 16, exponentBits: 5, mantissaBits: 10, maxValue: 65504 },
  { id: "FP8", label: "FP8 (E4M3)", kind: "float", bits: 8, exponentBits: 4, mantissaBits: 3, maxValue: 448 },
  { id: "INT8", label: "INT8", kind: "int", bits: 8 },
  { id: "INT4", label: "INT4", kind: "int", bits: 4 },
  { id: "INT2", label: "INT2", kind: "int", bits: 2 },
];
export const getFormat = (id: FormatId) => FORMATS.find((f) => f.id === id)!;

function seeded(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}
function gauss(rnd: () => number) {
  const u = Math.max(rnd(), 1e-9), v = rnd();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
const r3 = (x: number) => Math.round(x * 1000) / 1000;

export function generatePreset(id: PresetId, seed = 7): Matrix {
  const rnd = seeded(id === "random" ? Date.now() : seed);
  const m: Matrix = Array.from({ length: 4 }, () => Array(4).fill(0));
  for (let r = 0; r < 4; r++)
    for (let c = 0; c < 4; c++) {
      let v: number;
      if (id === "normal") v = gauss(rnd) * 0.5;
      else if (id === "outliers") v = gauss(rnd) * 0.3;
      else if (id === "shifted") v = 0.8 + gauss(rnd) * 0.3;
      else if (id === "uniform") v = rnd() * 2 - 1;
      else v = (rnd() * 2 - 1) * (0.5 + rnd() * 1.5);
      m[r][c] = r3(v);
    }
  if (id === "outliers") { m[1][2] = 4.2; m[3][0] = -3.1; }
  return m;
}

export function intRange(bits: number, symmetric: boolean) {
  return symmetric
    ? { qmin: -(2 ** (bits - 1) - 1), qmax: 2 ** (bits - 1) - 1 }
    : { qmin: 0, qmax: 2 ** bits - 1 };
}

export function blockIndexer(g: QuantConfig["granularity"], cols: number) {
  if (g === "tensor") return () => 0;
  if (g === "channel") return (r: number) => r;
  return (r: number, c: number) => r * Math.ceil(cols / 2) + Math.floor(c / 2);
}

/** Round x to a float format with given mantissa bits (ignores subnormal detail, clamps to max). */
export function roundToFloat(x: number, mantissaBits: number, maxValue: number) {
  if (x === 0 || !isFinite(x)) return 0;
  const e = Math.floor(Math.log2(Math.abs(x)));
  const step = 2 ** (e - mantissaBits);
  const v = Math.round(x / step) * step;
  return Math.max(-maxValue, Math.min(maxValue, v));
}

export function quantize(W: Matrix, cfg: QuantConfig): QuantResult {
  const f = getFormat(cfg.format);
  const rows = W.length, cols = W[0].length;
  if (f.kind === "float") {
    const q = W.map((row) => row.map((x) => roundToFloat(x, f.mantissaBits!, f.maxValue!)));
    return { q, deq: q, params: null, blockOf: () => 0 };
  }
  const blockOf = blockIndexer(cfg.granularity, cols);
  const nBlocks = cfg.granularity === "tensor" ? 1 : cfg.granularity === "channel" ? rows : rows * Math.ceil(cols / 2);
  const mins = Array(nBlocks).fill(Infinity), maxs = Array(nBlocks).fill(-Infinity);
  W.forEach((row, r) => row.forEach((x, c) => {
    const b = blockOf(r, c);
    mins[b] = Math.min(mins[b], x); maxs[b] = Math.max(maxs[b], x);
  }));
  const { qmin, qmax } = intRange(f.bits, cfg.symmetric);
  const scales: number[] = [], zeros: number[] = [];
  for (let b = 0; b < nBlocks; b++) {
    if (cfg.symmetric) {
      const a = Math.max(Math.abs(mins[b]), Math.abs(maxs[b])) * cfg.alpha;
      scales.push(a / qmax || 1e-8); zeros.push(0);
    } else {
      const lo = Math.min(mins[b], 0) * cfg.alpha, hi = Math.max(maxs[b], 0) * cfg.alpha;
      const s = (hi - lo) / (qmax - qmin) || 1e-8;
      scales.push(s); zeros.push(Math.round(qmin - lo / s));
    }
  }
  const q = W.map((row, r) => row.map((x, c) => {
    const b = blockOf(r, c);
    return Math.max(qmin, Math.min(qmax, Math.round(x / scales[b]) + zeros[b]));
  }));
  const deq = q.map((row, r) => row.map((v, c) => { const b = blockOf(r, c); return (v - zeros[b]!) * scales[b]!; }));
  return { q, deq, params: { scales, zeros, qmin, qmax }, blockOf };
}

export function matrixRange(W: Matrix) {
  const flat = W.flat();
  return { min: Math.min(...flat), max: Math.max(...flat) };
}
export const matVec = (W: Matrix, x: number[]) => W.map((row) => row.reduce((s, w, i) => s + w * x[i]!, 0));
export const mse = (a: Matrix, b: Matrix) => {
  const fa = a.flat(), fb = b.flat();
  return fa.reduce((s, v, i) => s + (v - fb[i]!) ** 2, 0) / fa.length;
};
export const bytesFor = (n: number, bits: number) => (n * bits) / 8;
