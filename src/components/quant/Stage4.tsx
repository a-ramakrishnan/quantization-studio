import { ArrowLeft, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { bytesFor, cosine, getFormat, matVec, mse, sqnrDb } from "@/utils/quantization";
import type { Matrix, QuantConfig, QuantResult } from "@/types/quantization";

function Metric({ k, v, sub, text }: { k: string; v: string; sub?: string; text: string }) {
  return (
    <div className="rounded-xl border bg-card p-5 shadow-card">
      <div className="text-xs uppercase tracking-wider text-muted-foreground">{k}</div>
      <div className="mt-2 font-mono text-2xl">{v}</div>
      {sub && <div className="font-mono text-xs text-primary">{sub}</div>}
      <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{text}</p>
    </div>
  );
}

export function Stage4({
  W,
  cfg,
  result,
  x,
  setX,
  onBack,
  onRestart,
}: {
  W: Matrix;
  cfg: QuantConfig;
  result: QuantResult;
  x: number[];
  setX: (x: number[]) => void;
  onBack: () => void;
  onRestart: () => void;
}) {
  const f = getFormat(cfg.format);
  const y = matVec(W, x),
    yq = matVec(result.deq, x);
  const maxAbs = Math.max(...y.map(Math.abs), ...yq.map(Math.abs), 1e-6);
  const wMse = mse(W, result.deq),
    sq = sqnrDb(W, result.deq);
  const oMse = mse([y], [yq]),
    cos = cosine(y, yq);
  const meta = (result.params?.scales.length ?? 0) * 4;
  const fp32B = bytesFor(16, 32),
    qB = bytesFor(16, f.bits) + meta;
  const Bar = ({ v, cls }: { v: number; cls: string }) => {
    const pct = (Math.abs(v) / maxAbs) * 50;
    return (
      <div className="relative h-3 rounded bg-muted">
        <div className="absolute inset-y-0 left-1/2 w-px bg-foreground/30" />
        <div
          className={`absolute inset-y-0 rounded transition-all ${cls}`}
          style={v >= 0 ? { left: "50%", width: `${pct}%` } : { right: "50%", width: `${pct}%` }}
        />
      </div>
    );
  };
  return (
    <div className="space-y-8">
      <div className="rounded-xl border bg-card p-5 shadow-card">
        <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
          <h3 className="font-serif text-xl">
            Input vector x <span className="font-mono text-sm text-muted-foreground">(FP32)</span>
          </h3>
          <span className="text-xs text-muted-foreground">
            Edit any value — outputs update live
          </span>
        </div>
        <div className="grid max-w-md grid-cols-4 gap-2">
          {x.map((v, i) => (
            <label key={i} className="block">
              <span className="font-mono text-xs text-muted-foreground">x[{i}]</span>
              <input
                type="number"
                step="0.25"
                value={v}
                aria-label={`x[${i}]`}
                onChange={(e) =>
                  setX(x.map((xx, j) => (j === i ? parseFloat(e.target.value) || 0 : xx)))
                }
                className="mt-1 w-full rounded-md border bg-background py-2 text-center font-mono text-sm outline-none focus:ring-2 focus:ring-ring [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none"
              />
            </label>
          ))}
        </div>
      </div>

      <div className="rounded-xl border bg-card p-5 shadow-card">
        <h3 className="font-serif text-xl">Output comparison</h3>
        <div className="mt-1 mb-4 flex gap-4 text-xs text-muted-foreground">
          <span>
            <i className="mr-1 inline-block h-2 w-3 rounded-sm bg-heat-pos" />
            True y = Wx
          </span>
          <span>
            <i className="mr-1 inline-block h-2 w-3 rounded-sm bg-heat-neg" />
            Quantized ŷ = Ŵx
          </span>
        </div>
        <div className="space-y-4">
          {y.map((v, i) => (
            <div key={i} className="grid items-center gap-3 sm:grid-cols-[3rem_1fr_13rem]">
              <span className="font-mono text-sm">y[{i}]</span>
              <div className="space-y-1">
                <Bar v={v} cls="bg-heat-pos" />
                <Bar v={yq[i]!} cls="bg-heat-neg" />
              </div>
              <div className="font-mono text-xs leading-5">
                <div>
                  {v.toFixed(4)} vs {yq[i]!.toFixed(4)}
                </div>
                <div className="text-clip">Δ = {(v - yq[i]!).toFixed(5)}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Metric
          k="Weight MSE"
          v={wMse.toExponential(2)}
          text="Average squared gap between each FP32 weight and its reconstruction Ŵ. Lower means the grid fits the weights better."
        />
        <Metric
          k="Weight SQNR"
          v={isFinite(sq) ? `${sq.toFixed(1)} dB` : "∞ dB"}
          text="Signal power vs. quantization-noise power. Each extra bit adds roughly 6 dB; above ~30 dB is usually safe."
        />
        <Metric
          k="Output MSE"
          v={oMse.toExponential(2)}
          sub={`cosine ${(cos * 100).toFixed(3)}%`}
          text="What the next layer actually sees. Cosine similarity near 100% means ŷ points the same direction as y."
        />
        <Metric
          k="HBM savings"
          v={`${(fp32B / qB).toFixed(1)}×`}
          sub={`${f.bits} bits vs 32 bits`}
          text={`${qB} B instead of ${fp32B} B for this 4×4${meta ? `, including ${meta} B of scales/zero-points` : ""}. Less to read from HBM per token.`}
        />
      </div>

      <div className="flex justify-between gap-2">
        <Button variant="outline" size="lg" onClick={onBack}>
          <ArrowLeft className="mr-1 h-4 w-4" />
          Back
        </Button>
        <Button size="lg" onClick={onRestart}>
          <RotateCcw className="mr-1 h-4 w-4" />
          Restart Timeline
        </Button>
      </div>
    </div>
  );
}
