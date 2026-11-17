import { useState } from "react";
import { ArrowRight, Cpu, HardDrive, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { BASIC_FORMATS, PROJECTIONS, basicProjection, initialProjections, sourceBF16 } from "@/utils/basic-quantization";
import type { BasicFormat, ProjectionId } from "@/utils/basic-quantization";
import { floatBitsFor, getFormat, packNibbles } from "@/utils/quantization";
import { MatrixGrid } from "./MatrixGrid";
import { QuantizationScene } from "./QuantizationScene";

const NAMES = { Q: "Query", K: "Key", V: "Value" };
const number = (value: number) => Number(value.toPrecision(7)).toString();

export function BasicMode({ active = true }: { active?: boolean }) {
  const [matrices, setMatrices] = useState(initialProjections);
  const [x, setX] = useState([1, -0.5, 0.25, 2]);
  const [format, setFormat] = useState<BasicFormat>("INT4");
  const [selected, setSelected] = useState<ProjectionId>("Q");
  const projections = Object.fromEntries(PROJECTIONS.map(id => [id, basicProjection(matrices[id], x, format)])) as Record<ProjectionId, ReturnType<typeof basicProjection>>;
  const current = projections[selected];
  const spec = getFormat(format);
  const description = BASIC_FORMATS.find(item => item.id === format)?.description;
  const scale = current.result.params?.scales[0];

  return (
    <main className="mx-auto max-w-6xl px-4 pb-20 sm:px-6">
      <QuantizationScene projection={current} format={format} selected={selected} active={active} />

      <aside className="flex items-start gap-3 rounded-lg border border-primary/20 bg-accent/40 p-4 text-sm" aria-label="Basic assumptions">
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
        <div className="space-y-1">
          <p className="font-medium">Our starting point: BF16 weights</p>
          <p className="leading-relaxed text-muted-foreground">One 4-element input vector x and three small 2×4 projection matrices: W_Q, W_K, W_V. Edited weights are rounded to BF16. Integer formats use one scale per matrix, symmetric codes, and no clipping.</p>
        </div>
      </aside>

      <section className="grid gap-6 border-b py-7 md:grid-cols-[1.2fr_1fr]">
        <div>
          <label htmlFor="basic-format" className="mb-3 block text-sm font-medium">Target weight format</label>
          <div className="flex flex-wrap items-center gap-3">
            <span className="font-mono text-sm text-muted-foreground">BF16</span>
            <ArrowRight className="h-4 w-4 text-muted-foreground" />
            <select id="basic-format" value={format} onChange={e => setFormat(e.target.value as BasicFormat)} className="h-10 rounded-md border bg-background px-3 text-sm outline-none focus:ring-2 focus:ring-ring">
              {BASIC_FORMATS.map(item => <option key={item.id} value={item.id}>{getFormat(item.id).label} · {(16 / getFormat(item.id).bits).toFixed(1)}×</option>)}
            </select>
            <span className="font-mono text-lg text-primary">{current.payloadRatio.toFixed(1)}× <span className="font-sans text-xs text-muted-foreground">less weight data</span></span>
          </div>
          <p className="mt-3 text-sm text-muted-foreground">{description}</p>
          <p className="mt-2 text-xs leading-relaxed text-muted-foreground">HBM weight payload vs BF16: {current.weightBytes} B instead of {current.baselineBytes} B per matrix.{scale !== undefined ? ` Plus ${current.scaleBytes} B for its FP32 scale (${current.weightBytes + current.scaleBytes} B total).` : " No scale metadata needed."}</p>
        </div>
        <div>
          <h3 className="mb-3 text-sm font-medium">Input activation x <span className="font-mono text-xs text-muted-foreground">FP32 · 4 elements</span></h3>
          <div className="grid grid-cols-4 gap-2">
            {x.map((value, i) => <label key={i} className="min-w-0">
              <span className="font-mono text-xs text-muted-foreground">x[{i}]</span>
              <input type="number" step="0.25" aria-label={`Basic x[${i}]`} value={value} onChange={e => {
                const next = Number(e.target.value);
                if (Number.isFinite(next)) setX(x.map((v, j) => j === i ? next : v));
              }} className="mt-1 h-10 w-full min-w-0 rounded-md border bg-card px-2 text-center font-mono text-sm outline-none focus:ring-2 focus:ring-ring" />
            </label>)}
          </div>
        </div>
      </section>

      <section className="border-b py-7" aria-label="Weight storage journey">
        <div className="mb-5 flex flex-wrap items-center justify-between gap-4">
          <h3 className="font-serif text-2xl">The weight journey <span className="font-mono text-sm text-muted-foreground">W_{selected}</span></h3>
          <div role="group" aria-label="Projection matrix" className="flex rounded-lg border p-1">
            {PROJECTIONS.map(id => <Button key={id} size="sm" variant={selected === id ? "secondary" : "ghost"} aria-pressed={selected === id} onClick={() => setSelected(id)}>{NAMES[id]} <span className="hidden font-mono text-xs sm:inline">W_{id}</span></Button>)}
          </div>
        </div>
        <div className="grid items-start gap-5 lg:grid-cols-3">
          <div className="min-w-0">
            <div className="mb-3 flex items-center gap-2 text-sm font-medium"><span className="font-mono text-primary">01</span> Source BF16 <span className="text-xs text-muted-foreground">Editable</span></div>
            <MatrixGrid m={current.source} editable onChange={(r, c, value) => setMatrices(previous => ({ ...previous, [selected]: sourceBF16(previous[selected].map((row, i) => row.map((v, j) => i === r && j === c ? value : v))) }))} format={number} />
            <p className="mt-3 text-xs leading-relaxed text-muted-foreground">Original 16-bit weights, before converting to {spec.label}.</p>
          </div>
          <div className="min-w-0">
            <div className="mb-3 flex items-center gap-2 text-sm font-medium"><HardDrive className="h-4 w-4 text-primary" /> Stored in GPU HBM</div>
            <MatrixGrid m={current.result.q} format={value => spec.kind === "int" ? String(value) : format === "FP8" ? floatBitsFor("FP8", value).hex : number(value)} />
            {format === "INT4" && <div className="mt-3">
              <p className="mb-2 text-xs text-muted-foreground">Packed bytes · row-major · first code in high nibble</p>
              <div className="grid grid-cols-4 gap-2">{packNibbles(current.result.q.flat()).map((byte, i) => <div key={i} className="rounded-md border bg-secondary py-2 text-center font-mono text-sm" title={`Weight codes ${current.result.q.flat()[i * 2]} and ${current.result.q.flat()[i * 2 + 1]}`}>{byte.hex}</div>)}</div>
            </div>}
            <p className="mt-3 break-words font-mono text-xs leading-relaxed text-muted-foreground">{scale !== undefined ? `S = ${number(scale)} · Z = 0` : `${spec.bits}-bit floating-point values`}</p>
          </div>
          <div className="min-w-0">
            <div className="mb-3 flex items-center gap-2 text-sm font-medium"><Cpu className="h-4 w-4 text-primary" /> {spec.kind === "int" ? "Dequantized" : "Reconstructed"} in SM Registers</div>
            <MatrixGrid m={current.result.deq} format={number} />
            <p className="mt-3 text-xs leading-relaxed text-muted-foreground">{scale !== undefined ? "Ŵ = (q − Z) × S. Integer codes become floating-point weights for the multiply." : "Ŵ uses the stored float values for the multiply; no integer scale is needed."}</p>
            <p className="mt-2 text-xs leading-relaxed text-muted-foreground">Values model the format conversion; SM arithmetic and accumulation rounding are not simulated.</p>
          </div>
        </div>
      </section>

      <section className="pt-7" aria-label="Projection output comparisons">
        <div className="mb-5 flex flex-wrap items-baseline justify-between gap-2">
          <h3 className="font-serif text-2xl">What changes in the output?</h3>
          <p className="font-mono text-xs text-muted-foreground">Δ = BF16 baseline − quantized output</p>
        </div>
        <div className="grid gap-4 lg:grid-cols-3">
          {PROJECTIONS.map(id => {
            const projection = projections[id];
            return <article key={id} className={cn("min-w-0 rounded-lg border bg-card p-4 shadow-card", selected === id && "border-primary/40")} aria-label={`${NAMES[id]} output`}>
              <h4 className="font-medium">{NAMES[id]} <span className="ml-2 font-mono text-sm text-primary">{id} = W_{id}·x</span></h4>
              <table className="mt-4 w-full table-fixed text-right font-mono text-xs">
                <thead><tr className="text-muted-foreground"><th className="w-8 pb-3 text-left font-normal" /><th className="pb-3 font-normal">BF16</th><th className="pb-3 font-normal">{id}̂ · {format}</th><th className="pb-3 font-normal">Δ</th></tr></thead>
                <tbody>{projection.baseline.map((value, i) => <tr key={i} className="border-t"><th className="py-3 text-left font-normal">{id}[{i}]</th><td className="py-3 text-heat-pos">{value.toFixed(4)}</td><td className="py-3 text-heat-neg">{(projection.output[i] ?? 0).toFixed(4)}</td><td className="py-3" title={String(projection.delta[i])}>{(projection.delta[i] ?? 0).toFixed(5)}</td></tr>)}</tbody>
              </table>
            </article>;
          })}
        </div>
      </section>
    </main>
  );
}