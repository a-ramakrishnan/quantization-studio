import { ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";
import { FORMATS, getFormat, intRange, matrixRange } from "@/utils/quantization";
import type { Matrix, PresetId, QuantConfig, QuantResult } from "@/types/quantization";
import { MatrixGrid } from "./MatrixGrid";

const PRESETS: { id: PresetId; label: string }[] = [
  { id: "normal", label: "Normal Weights" }, { id: "outliers", label: "With Outliers" },
  { id: "shifted", label: "Shifted / Biased" }, { id: "uniform", label: "Uniform Spread" }, { id: "random", label: "Random" },
];

function Pill({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button onClick={onClick} className={cn(
      "rounded-full border px-3 py-1.5 text-sm transition-all",
      active ? "border-primary bg-primary text-primary-foreground" : "bg-card text-foreground hover:border-primary/50 hover:bg-accent",
    )}>{children}</button>
  );
}

const Label = ({ children }: { children: React.ReactNode }) => (
  <div className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">{children}</div>
);

export function Stage1({ W, setW, preset, onPreset, cfg, setCfg, result, onNext, nextUnlocked }: {
  W: Matrix; setW: (m: Matrix) => void; preset: PresetId; onPreset: (p: PresetId) => void;
  cfg: QuantConfig; setCfg: (c: QuantConfig) => void; result: QuantResult; onNext: () => void; nextUnlocked: boolean;
}) {
  const f = getFormat(cfg.format);
  const { min, max } = matrixRange(W);
  const set = (p: Partial<QuantConfig>) => setCfg({ ...cfg, ...p });
  const idx = FORMATS.findIndex((x) => x.id === cfg.format);

  return (
    <div className="grid gap-8 lg:grid-cols-[1.15fr_1fr]">
      <div className="space-y-6">
        <div>
          <Label>Weight preset</Label>
          <div className="flex flex-wrap gap-2">
            {PRESETS.map((p) => <Pill key={p.id} active={preset === p.id} onClick={() => onPreset(p.id)}>{p.label}</Pill>)}
          </div>
        </div>
        <div>
          <Label>Target format · precision ladder</Label>
          <div className="flex flex-wrap items-center gap-1">
            {FORMATS.map((x, i) => (
              <div key={x.id} className="flex items-center gap-1">
                <Pill active={cfg.format === x.id} onClick={() => set({ format: x.id })}>
                  <span className="font-mono">{x.label}</span>
                </Pill>
                {i < FORMATS.length - 1 && <ArrowRight className={cn("h-3.5 w-3.5", i < idx ? "text-primary" : "text-muted-foreground/50")} />}
              </div>
            ))}
          </div>
        </div>
        {f.kind === "int" && (
          <div className="space-y-4 rounded-lg border bg-muted/40 p-4">
            <div className="flex flex-wrap gap-2">
              <Pill active={cfg.symmetric} onClick={() => set({ symmetric: true })}>Symmetric (Z=0)</Pill>
              <Pill active={!cfg.symmetric} onClick={() => set({ symmetric: false })}>Asymmetric</Pill>
            </div>
            <div className="flex flex-wrap gap-2">
              <Pill active={cfg.granularity === "tensor"} onClick={() => set({ granularity: "tensor" })}>Per-Tensor</Pill>
              <Pill active={cfg.granularity === "channel"} onClick={() => set({ granularity: "channel" })}>Per-Channel</Pill>
              <Pill active={cfg.granularity === "group"} onClick={() => set({ granularity: "group" })}>Per-Group(2)</Pill>
            </div>
            <div>
              <div className="mb-2 flex justify-between text-sm"><span>Outlier Clipping (α)</span><span className="font-mono">{cfg.alpha.toFixed(2)}</span></div>
              <Slider min={0.5} max={1} step={0.01} value={[cfg.alpha]} onValueChange={([a]) => set({ alpha: a ?? 1 })} />
            </div>
          </div>
        )}
        <div>
          <Label>FP32 weight matrix (W) · click a cell to edit</Label>
          <MatrixGrid m={W} editable onChange={(r, c, v) => setW(W.map((row, i) => row.map((x, j) => (i === r && j === c ? v : x))))} />
        </div>
      </div>

      <OutputCard f={f} cfg={cfg} min={min} max={max} result={result} onNext={onNext} nextUnlocked={nextUnlocked} />
    </div>
  );
}

function Stat({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-dashed py-2.5 last:border-0">
      <span className="text-sm text-muted-foreground">{k}</span>
      <span className="text-right font-mono text-sm">{v}</span>
    </div>
  );
}

function OutputCard({ f, cfg, min, max, result, onNext, nextUnlocked }: {
  f: ReturnType<typeof getFormat>; cfg: QuantConfig; min: number; max: number; result: QuantResult; onNext: () => void; nextUnlocked: boolean;
}) {
  const p = result.params;
  const n = p?.scales.length ?? 0;
  const fmt = (x: number) => (Math.abs(x) < 1e-3 && x !== 0 ? x.toExponential(2) : x.toFixed(4));
  let explain: string;
  if (f.kind === "float") {
    explain = `${f.label} keeps the floating-point idea but with fewer bits: ${f.exponentBits} exponent bits set the range and ${f.mantissaBits} mantissa bits set the precision. No scale is needed — each weight is simply rounded to the nearest representable ${f.label} value.`;
  } else {
    const r = intRange(f.bits, cfg.symmetric);
    explain = `Each weight will be mapped to an integer in [${r.qmin}, ${r.qmax}] via q = round(w / S) + Z. ${cfg.granularity === "tensor" ? "One scale covers the whole matrix" : cfg.granularity === "channel" ? "Each output row (channel) gets its own scale" : "Every pair of adjacent weights gets its own scale"}${cfg.alpha < 1 ? `, and the range is clipped to ${Math.round(cfg.alpha * 100)}% so outliers saturate instead of stretching the grid` : ""}. Only ${f.bits} bits per weight are stored, plus ${n} scale${n > 1 ? "s" : ""}.`;
  }
  return (
    <div className="h-fit rounded-xl border bg-card p-6 shadow-card lg:sticky lg:top-24">
      <div className="text-xs font-medium uppercase tracking-wider text-primary">First-level output</div>
      <h3 className="mt-1 font-serif text-2xl">Calibration for <span className="font-mono text-xl">{f.label}</span></h3>
      <div className="mt-4">
        <Stat k="Observed range" v={`[${min.toFixed(3)}, ${max.toFixed(3)}]`} />
        {f.kind === "int" && p ? (
          <>
            <Stat k="Target range" v={`[${p.qmin}, ${p.qmax}]  (${f.bits}-bit)`} />
            <Stat k={n > 1 ? `Scale S (${n} blocks)` : "Scale (S)"} v={n > 1 ? `${fmt(Math.min(...p.scales))} … ${fmt(Math.max(...p.scales))}` : fmt(p.scales[0]!)} />
            <Stat k="Zero-point (Z)" v={cfg.symmetric ? "0" : n > 1 ? `${Math.min(...p.zeros)} … ${Math.max(...p.zeros)}` : p.zeros[0]} />
          </>
        ) : (
          <>
            <Stat k="Target range" v={`±${f.maxValue! > 1e6 ? f.maxValue!.toExponential(2) : f.maxValue}`} />
            <div className="py-3">
              <div className="mb-2 text-sm text-muted-foreground">Bit-field layout ({f.bits} bits)</div>
              <div className="flex gap-0.5">
                <Bits n={1} cls="bg-bit-sign" />
                <Bits n={f.exponentBits!} cls="bg-bit-exp" />
                <Bits n={f.mantissaBits!} cls="bg-bit-man" />
              </div>
              <div className="mt-2 flex gap-4 font-mono text-xs text-muted-foreground">
                <span><i className="mr-1 inline-block h-2 w-2 rounded-sm bg-bit-sign" />Sign 1</span>
                <span><i className="mr-1 inline-block h-2 w-2 rounded-sm bg-bit-exp" />Exponent {f.exponentBits}</span>
                <span><i className="mr-1 inline-block h-2 w-2 rounded-sm bg-bit-man" />Mantissa {f.mantissaBits}</span>
              </div>
            </div>
          </>
        )}
      </div>
      <p className="mt-4 rounded-lg bg-muted/60 p-4 text-sm leading-relaxed text-muted-foreground">
        <span className="font-medium text-foreground">What happens next. </span>{explain}
      </p>
      <Button size="lg" className="mt-6 w-full" onClick={onNext}>
        {nextUnlocked ? "Go to Stage 2" : "Convert Weights → Stage 2"} <ArrowRight className="ml-1 h-4 w-4" />
      </Button>
    </div>
  );
}

function Bits({ n, cls }: { n: number; cls: string }) {
  return <>{Array.from({ length: n }).map((_, i) => <div key={i} className={cn("h-6 flex-1 rounded-[2px] opacity-85", cls)} />)}</>;
}
