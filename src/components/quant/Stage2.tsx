import { useMemo, useState } from "react";
import { ArrowLeft, ArrowRight, Pencil } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import {
  floatBitsFor, getFormat, mae, packNibbles, roundToFloat, quantRowSym, storageGB, twosComplement,
} from "@/utils/quantization";
import type { Matrix, QuantConfig, QuantResult } from "@/types/quantization";
import { heatStyle } from "./MatrixGrid";

type Cell = [number, number];
const GRAN = { tensor: "Per-Tensor", channel: "Per-Channel", group: "Per-Group(2)" } as const;
const fx = (x: number, d = 4) => (Math.abs(x) < 1e-3 && x !== 0 ? x.toExponential(2) : x.toFixed(d));

function cellInfo(W: Matrix, cfg: QuantConfig, res: QuantResult, [r, c]: Cell) {
  const f = getFormat(cfg.format);
  const w = W[r]![c]!;
  const wq = res.q[r]![c]!, deq = res.deq[r]![c]!;
  if (f.kind === "float") return { f, w, wq, deq, clipped: Math.abs(w) > f.maxValue!, S: 0, Z: 0, raw: 0, b: 0 };
  const b = res.blockOf(r, c), p = res.params!;
  const S = p.scales[b]!, Z = p.zeros[b]!, raw = Math.round(w / S) + Z;
  return { f, w, wq, deq, S, Z, raw, b, clipped: raw < p.qmin || raw > p.qmax };
}

export function Stage2({ W, setW, cfg, result, sel, setSel, onEdit, onBack, onNext }: {
  W: Matrix; setW: (m: Matrix) => void; cfg: QuantConfig; result: QuantResult;
  sel: Cell; setSel: (c: Cell) => void; onEdit: () => void; onBack: () => void; onNext: () => void;
}) {
  const f = getFormat(cfg.format);
  const info = cellInfo(W, cfg, result, sel);
  const maxAbs = Math.max(...W.flat().map(Math.abs));
  const storedLabel = f.kind === "int" ? `W_q (${f.label})` : f.id === "FP8" ? "FP8 Hex" : f.label;

  const cellBtn = (r: number, c: number, child: React.ReactNode, v: number) => {
    const ci = cellInfo(W, cfg, result, [r, c]);
    const active = sel[0] === r && sel[1] === c;
    return (
      <div key={`${r}${c}`} style={heatStyle(v, maxAbs)} onClick={() => setSel([r, c])}
        className={cn("relative cursor-pointer rounded-md ring-offset-1 ring-offset-card transition-all", active && "ring-2 ring-primary", ci.clipped && "outline-2 outline-dashed outline-clip")}>
        {child}
        {ci.clipped && <span className="absolute right-0.5 top-0.5 h-1.5 w-1.5 rounded-full bg-clip" />}
      </div>
    );
  };

  return (
    <div className="space-y-10">
      {/* Summary bar */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 rounded-lg border bg-muted/40 px-4 py-3 text-sm">
        <span className="text-muted-foreground">Stage 1</span>
        <span>Format <b className="font-mono">{f.label}</b></span>
        {f.kind === "int" && <>
          <span>{cfg.symmetric ? "Symmetric" : "Asymmetric"}</span>
          <span>{GRAN[cfg.granularity]}</span>
          <span>α <span className="font-mono">{cfg.alpha.toFixed(2)}</span></span>
          <span>S <span className="font-mono">{fx(result.params!.scales[0]!)}</span>{result.params!.scales.length > 1 && " …"}</span>
        </>}
        <Button variant="outline" size="sm" className="ml-auto" onClick={onEdit}><Pencil className="mr-1 h-3.5 w-3.5" />Edit Stage 1</Button>
      </div>

      {/* Side by side */}
      <div className="grid items-center gap-4 md:grid-cols-[1fr_auto_1fr]">
        <div>
          <div className="mb-2 font-mono text-xs text-muted-foreground">1. FP32 Weights (W)</div>
          <div className="grid grid-cols-4 gap-1.5 rounded-lg border bg-card p-2">
            {W.map((row, r) => row.map((v, c) => cellBtn(r, c,
              <input type="number" step="0.1" value={v} onFocus={() => setSel([r, c])}
                onChange={(e) => setW(W.map((rr, i) => rr.map((x, j) => (i === r && j === c ? parseFloat(e.target.value) || 0 : x))))}
                className="w-full rounded-md bg-transparent py-2 text-center font-mono text-sm outline-none [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none" />, v)))}
          </div>
        </div>
        <ArrowRight className="mx-auto h-5 w-5 rotate-90 text-muted-foreground md:rotate-0" />
        <div>
          <div className="mb-2 font-mono text-xs text-muted-foreground">2. Stored Format ({storedLabel})</div>
          <div className="grid grid-cols-4 gap-1.5 rounded-lg border bg-card p-2">
            {result.q.map((row, r) => row.map((v, c) => cellBtn(r, c,
              <div className="py-2 text-center font-mono text-sm">
                {f.kind === "int" ? v : f.id === "FP8" ? floatBitsFor("FP8", v).hex : v.toPrecision(4)}
              </div>, result.deq[r]![c]!)))}
          </div>
        </div>
      </div>
      <p className="-mt-6 flex items-center gap-2 text-xs text-muted-foreground">
        <span className="h-2 w-2 rounded-full bg-clip" /> Amber = clipped (value fell outside the representable range). Click any cell to inspect it below.
      </p>

      {/* Breakdown */}
      <div className="grid gap-6 lg:grid-cols-[1.3fr_1fr]">
        <div className="rounded-xl border bg-card p-5 shadow-card">
          <h3 className="font-serif text-xl">Number-line bin mapping</h3>
          <p className="mb-4 text-sm text-muted-foreground">Cell W[{sel[0]}][{sel[1]}] snaps to its nearest representable value.</p>
          <NumberLine info={info} result={result} W={W} cfg={cfg} />
        </div>
        <MathCard info={info} cfg={cfg} sel={sel} result={result} />
      </div>

      <Journey W={W} />

      <div className="flex justify-between">
        <Button variant="outline" size="lg" onClick={onBack}><ArrowLeft className="mr-1 h-4 w-4" />Back</Button>
        <Button size="lg" onClick={onNext}>Upload to GPU &amp; Dequantize <ArrowRight className="ml-1 h-4 w-4" /></Button>
      </div>
    </div>
  );
}

function NumberLine({ info, result, W, cfg }: { info: ReturnType<typeof cellInfo>; result: QuantResult; W: Matrix; cfg: QuantConfig }) {
  const Wd = 560, H = 185, pad = 30, y0 = 100;
  const { f, w, deq } = info;
  let ticks: { v: number; label: string | undefined }[] = [];
  let lo: number, hi: number, clipLo = 0, clipHi = 0;
  const blockVals = f.kind === "int" ? W.flatMap((row, r) => row.filter((_, c) => result.blockOf(r, c) === info.b)) : W.flat();
  const obsLo = Math.min(...blockVals, 0), obsHi = Math.max(...blockVals, 0);
  if (f.kind === "int") {
    const p = result.params!;
    clipLo = (p.qmin - info.Z) * info.S; clipHi = (p.qmax - info.Z) * info.S;
    lo = Math.min(obsLo, clipLo); hi = Math.max(obsHi, clipHi);
    const n = p.qmax - p.qmin;
    for (let q = p.qmin; q <= p.qmax; q++) {
      const lbl = q === p.qmin || q === p.qmax || q === info.Z || q === info.wq ? String(q) : n <= 16 ? String(q) : undefined;
      ticks.push({ v: (q - info.Z) * info.S, label: lbl });
    }
  } else {
    // neighbouring representable values around w
    const e = Math.floor(Math.log2(Math.abs(w) || 1)), step = 2 ** (e - f.mantissaBits!);
    for (let k = -4; k <= 4; k++) ticks.push({ v: roundToFloat(deq + k * step, f.mantissaBits!, f.maxValue!), label: k === 0 ? "Ŵ" : undefined });
    lo = ticks[0]!.v - step; hi = ticks.at(-1)!.v + step;
  }
  const span = hi - lo || 1;
  const X = (v: number) => pad + ((v - lo) / span) * (Wd - 2 * pad);
  const xw = X(w), xq = X(deq), mid = (xw + xq) / 2, lift = Math.max(28, Math.abs(xq - xw) * 0.6);
  return (
    <svg viewBox={`0 0 ${Wd} ${H}`} className="w-full">
      {f.kind === "int" && (
        <>
          <rect x={X(clipLo)} y={y0 - 14} width={X(clipHi) - X(clipLo)} height={28} rx={4} className="fill-primary/10 stroke-primary/40" strokeDasharray="3 3" />
          <text x={X(clipLo)} y={y0 - 20} className="fill-muted-foreground font-mono text-[10px]">clip lo {fx(clipLo, 2)}</text>
          <text x={X(clipHi)} y={y0 - 20} textAnchor="end" className="fill-muted-foreground font-mono text-[10px]">clip hi {fx(clipHi, 2)}</text>
          <rect x={X(obsLo)} y={y0 + 18} width={X(obsHi) - X(obsLo)} height={3} className="fill-muted-foreground/40" />
          <text x={X(obsLo)} y={y0 + 34} className="fill-muted-foreground font-mono text-[10px]">observed FP32 range</text>
        </>
      )}
      <line x1={pad} x2={Wd - pad} y1={y0} y2={y0} className="stroke-foreground/50" />
      {ticks.map((t, i) => (
        <g key={i}>
          <line x1={X(t.v)} x2={X(t.v)} y1={y0 - 6} y2={y0 + 6} className="stroke-foreground/60" />
          {t.label && <text x={X(t.v)} y={y0 + 52} textAnchor="middle" className="fill-muted-foreground font-mono text-[10px]">{t.label}</text>}
        </g>
      ))}
      {f.kind === "int" && (
        <>
          <line x1={X(0)} x2={X(0)} y1={y0 - 10} y2={y0 + 60} className="stroke-bit-exp" strokeDasharray="2 2" />
          <text x={X(0)} y={y0 + 72} textAnchor="middle" className="fill-bit-exp font-mono text-[10px]">Z = {info.Z}</text>
          {ticks.length > 1 && <text x={Wd - pad} y={H - 4} textAnchor="end" className="fill-muted-foreground font-mono text-[10px]">Δ = S = {fx(info.S)}</text>}
        </>
      )}
      <path d={`M ${xw} ${y0} Q ${mid} ${y0 - lift - 20} ${xq} ${y0}`} className="fill-none stroke-heat-neg" strokeWidth={2} markerEnd="url(#arr)" />
      <defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 z" className="fill-heat-neg" /></marker></defs>
      <circle cx={xw} cy={y0} r={5} className="fill-heat-pos" />
      <text x={xw} y={y0 - lift - 26} textAnchor="middle" className="fill-foreground font-mono text-[11px]">W {fx(w, 3)} → Ŵ {fx(deq, 3)}</text>
      <circle cx={xq} cy={y0} r={4} className="fill-heat-neg" />
    </svg>
  );
}

function Row({ k, formula, v }: { k: string; formula: string; v: string }) {
  return (
    <div className="border-b border-dashed py-3 last:border-0">
      <div className="flex justify-between text-sm"><span className="font-medium">{k}</span><span className="font-mono">{v}</span></div>
      <div className="mt-1 font-mono text-xs text-muted-foreground">{formula}</div>
    </div>
  );
}

function MathCard({ info, cfg, sel, result }: { info: ReturnType<typeof cellInfo>; cfg: QuantConfig; sel: Cell; result: QuantResult }) {
  const { f, w, deq, S, Z, raw, wq } = info;
  return (
    <div className="rounded-xl border bg-card p-5 shadow-card">
      <h3 className="font-serif text-xl">Step-by-step math</h3>
      <p className="mb-2 text-sm text-muted-foreground">W[{sel[0]}][{sel[1]}] = <span className="font-mono text-foreground">{w}</span></p>
      {f.kind === "int" ? (() => {
        const p = result.params!;
        return <>
          <Row k="Scale S" formula={cfg.symmetric ? `α·max|W| / ${p.qmax}` : `α·(max − min) / ${p.qmax - p.qmin}`} v={fx(S, 5)} />
          <Row k="Zero-point Z" formula={cfg.symmetric ? "symmetric → 0" : "round(qmin − α·min / S)"} v={String(Z)} />
          <Row k="Quantize W_q" formula={`clamp(round(${fx(w, 3)} / ${fx(S, 4)}) + ${Z}, ${p.qmin}, ${p.qmax}) = clamp(${raw})`} v={String(wq)} />
          <Row k="Dequantize Ŵ" formula={`${fx(S, 4)} · (${wq} − ${Z})`} v={fx(deq, 5)} />
          <Row k="Error e" formula={`W − Ŵ${info.clipped ? "  (clipped!)" : ""}`} v={fx(w - deq, 5)} />
        </>;
      })() : (() => {
        const b = floatBitsFor(f.id, w);
        const bias = 2 ** (f.exponentBits! - 1) - 1;
        return <>
          <div className="my-3 flex flex-wrap gap-0.5 font-mono text-xs">
            {b.binary.split("").map((ch, i) => (
              <span key={i} className={cn("flex h-7 w-5 items-center justify-center rounded-sm text-primary-foreground", i === 0 ? "bg-bit-sign" : i <= f.exponentBits! ? "bg-bit-exp" : "bg-bit-man")}>{ch}</span>
            ))}
          </div>
          <Row k="Layout" formula={`1S + ${f.exponentBits}E + ${f.mantissaBits}M`} v={b.hex} />
          <Row k="Sign" formula="0 = +, 1 = −" v={String(b.sign)} />
          <Row k="Exponent" formula={`stored − bias(${bias})`} v={`${b.exp} → 2^${b.exp ? b.exp - bias : 1 - bias}`} />
          <Row k="Mantissa" formula={`1 + m / 2^${f.mantissaBits}`} v={`${b.man} / ${2 ** f.mantissaBits!}`} />
          <Row k="Value Ŵ" formula="(−1)^S · 2^E · (1.M)" v={fx(deq, 6)} />
          <Row k="Error e" formula="W − Ŵ" v={fx(w - deq, 6)} />
        </>;
      })()}
    </div>
  );
}

const TRANSFORMER = [-1.0, -0.8, -0.3, -0.1, 0.1, 0.3, 0.8, 1.0];
const OUTLIER = [0.01, -0.02, 0.03, -0.04, 0.05, 0.01, 0.02, 1.0];

function Journey({ W }: { W: Matrix }) {
  const [src, setSrc] = useState<"transformer" | "outlier" | "matrix">("transformer");
  const [grouped, setGrouped] = useState(false);
  const [rowIdx, setRowIdx] = useState(0);
  const row = src === "transformer" ? TRANSFORMER : src === "outlier" ? OUTLIER : W[rowIdx]!;
  const groupSize = src === "outlier" && grouped ? 4 : row.length;
  const data = useMemo(() => {
    const fp16 = row.map((x) => roundToFloat(x, 10, 65504));
    const bf16 = row.map((x) => roundToFloat(x, 7, 3.39e38));
    const i8 = quantRowSym(row, 8), i4 = quantRowSym(row, 4, groupSize);
    return { fp16, bf16, i8, i4 };
  }, [row, groupSize]);
  const packed = packNibbles(data.i4.q);
  const P = 9e9;
  const bars = [
    { k: "FP32", v: storageGB(P, 32) }, { k: "FP16 / BF16", v: storageGB(P, 16) },
    { k: "INT8", v: storageGB(P, 8) }, { k: "INT4 Group-128", v: storageGB(P, 4, 128) },
  ];
  const Tab = ({ id, l }: { id: typeof src; l: string }) => (
    <button onClick={() => setSrc(id)} className={cn("rounded-md px-3 py-1.5 text-sm", src === id ? "bg-card shadow-card" : "text-muted-foreground hover:text-foreground")}>{l}</button>
  );
  return (
    <div className="space-y-6 rounded-xl border bg-muted/30 p-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h3 className="font-serif text-2xl">Progressive format journey</h3>
          <p className="text-sm text-muted-foreground">Track every weight of one row down the precision ladder.</p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex rounded-lg bg-muted p-1">
            <Tab id="transformer" l="Transformer Row" /><Tab id="outlier" l="Outlier Row" /><Tab id="matrix" l="Active 4×4 Row" />
          </div>
          {src === "outlier" && <label className="flex items-center gap-2 text-sm"><Switch checked={grouped} onCheckedChange={setGrouped} />Group-wise INT4 Scaling (g=4)</label>}
          {src === "matrix" && <div className="flex gap-1">{[0, 1, 2, 3].map((i) => <button key={i} onClick={() => setRowIdx(i)} className={cn("h-7 w-7 rounded font-mono text-xs", rowIdx === i ? "bg-primary text-primary-foreground" : "bg-card")}>{i}</button>)}</div>}
        </div>
      </div>

      <div className="overflow-x-auto rounded-lg border bg-card">
        <table className="w-full font-mono text-xs">
          <thead><tr className="border-b text-left text-muted-foreground">
            {["#", "FP32", "FP16", "BF16", "INT8 q", "INT8 bin", "INT4 q", "INT4 nibble", "INT4 Ŵ"].map((h) => <th key={h} className="px-3 py-2 font-medium">{h}</th>)}
          </tr></thead>
          <tbody>{row.map((x, i) => (
            <tr key={i} className="border-b border-dashed last:border-0">
              <td className="px-3 py-1.5 text-muted-foreground">{i}</td>
              <td className="px-3">{x.toFixed(4)}</td>
              <td className="px-3">{data.fp16[i]!.toPrecision(5)}</td>
              <td className="px-3">{data.bf16[i]!.toPrecision(4)}</td>
              <td className="px-3">{data.i8.q[i]}</td>
              <td className="px-3 text-muted-foreground">{twosComplement(data.i8.q[i]!, 8)}</td>
              <td className={cn("px-3", data.i4.q[i] === 0 && x !== 0 && "text-clip")}>{data.i4.q[i]}</td>
              <td className="px-3 text-primary">{twosComplement(data.i4.q[i]!, 4)}</td>
              <td className="px-3">{data.i4.deq[i]!.toFixed(4)}</td>
            </tr>
          ))}</tbody>
          <tfoot><tr className="border-t bg-muted/40">
            <td className="px-3 py-2 text-muted-foreground" colSpan={2}>MAE</td>
            <td className="px-3">{mae(row, data.fp16).toExponential(1)}</td>
            <td className="px-3">{mae(row, data.bf16).toExponential(1)}</td>
            <td className="px-3" colSpan={2}>{mae(row, data.i8.deq).toExponential(2)}</td>
            <td className="px-3" colSpan={3}>{mae(row, data.i4.deq).toExponential(2)}{src === "outlier" && !grouped && <span className="ml-2 text-clip">small weights collapse to 0 — try group-wise</span>}</td>
          </tr></tfoot>
        </table>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-lg border bg-card p-5">
          <h4 className="font-serif text-lg">Physical INT4 nibble bit-packing</h4>
          <p className="mb-4 text-xs text-muted-foreground">Two 4-bit two's-complement weights share one byte (first → high nibble).</p>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {packed.map((p, i) => (
              <div key={i} className="rounded-md border p-2 text-center font-mono">
                <div className="text-[10px] text-muted-foreground">w{2 * i}, w{2 * i + 1} = {data.i4.q[2 * i]}, {data.i4.q[2 * i + 1]}</div>
                <div className="my-1 flex justify-center gap-1 text-xs">
                  <span className="rounded bg-bit-man/20 px-1">{p.hi.toString(2).padStart(4, "0")}</span>
                  <span className="rounded bg-bit-exp/20 px-1">{p.lo.toString(2).padStart(4, "0")}</span>
                </div>
                <div className="text-base font-medium text-primary">{p.hex}</div>
              </div>
            ))}
          </div>
          <p className="mt-3 font-mono text-xs text-muted-foreground">{row.length} weights → {packed.length} bytes{data.i4.scales.length > 1 && ` + ${data.i4.scales.length} FP16 scales`}</p>
        </div>
        <div className="rounded-lg border bg-card p-5">
          <h4 className="font-serif text-lg">9B-parameter LLM storage</h4>
          <p className="mb-4 text-xs text-muted-foreground">Weights only. Group-128 adds one FP16 scale per 128 weights.</p>
          <div className="space-y-3">
            {bars.map((b) => (
              <div key={b.k}>
                <div className="mb-1 flex justify-between font-mono text-xs"><span>{b.k}</span><span>{Number(b.v.toFixed(2))} GB</span></div>
                <div className="h-3 rounded bg-muted"><div className="h-3 rounded bg-primary transition-all" style={{ width: `${(b.v / 36) * 100}%` }} /></div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
