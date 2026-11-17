import { ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { bytesFor, getFormat, matVec, mse } from "@/utils/quantization";
import type { Matrix, QuantConfig, QuantResult } from "@/types/quantization";
import { MatrixGrid } from "./MatrixGrid";

const Next = ({ label, onClick }: { label: string; onClick: () => void }) => (
  <Button size="lg" onClick={onClick} className="mt-6">{label} <ArrowRight className="ml-1 h-4 w-4" /></Button>
);

export function Stage3({ cfg, result, onNext }: { cfg: QuantConfig; result: QuantResult; onNext: () => void }) {
  const f = getFormat(cfg.format);
  const n = 16, fp32 = bytesFor(n, 32), packed = bytesFor(n, f.bits), meta = (result.params?.scales.length ?? 0) * 2;
  const bar = (b: number) => ({ width: `${(b / fp32) * 100}%` });
  return (
    <div>
      <p className="max-w-2xl text-muted-foreground">Weights live in HBM in packed {f.label}. Each kernel tile loads them into registers and dequantizes on the fly with <span className="font-mono text-foreground">ŵ = (q − Z) · S</span> into FP16 — the W{f.bits}A16 pattern — so memory traffic shrinks while math stays in 16-bit.</p>
      <div className="mt-6 space-y-3 font-mono text-sm">
        <div><div className="mb-1 flex justify-between"><span>FP32 in HBM</span><span>{fp32} B</span></div><div className="h-3 rounded bg-muted-foreground/30" style={bar(fp32)} /></div>
        <div><div className="mb-1 flex justify-between"><span>{f.label} packed + {meta ? "FP16 scales" : "no scales"}</span><span>{packed + meta} B</span></div><div className="h-3 rounded bg-primary" style={bar(packed + meta)} /></div>
      </div>
      <div className="mt-6 max-w-sm"><MatrixGrid m={result.deq} label="Dequantized Ŵ in registers (FP16)" /></div>
      <Next label="Run forward pass → Stage 4" onClick={onNext} />
    </div>
  );
}

export function Stage4({ W, result, x }: { W: Matrix; result: QuantResult; x: number[] }) {
  const y = matVec(W, x), yq = matVec(result.deq, x);
  return (
    <div>
      <p className="max-w-2xl text-muted-foreground">With input <span className="font-mono text-foreground">x = [{x.join(", ")}]</span>, compare the exact FP32 output with the quantized one.</p>
      <div className="mt-6 overflow-x-auto">
        <table className="w-full max-w-xl font-mono text-sm">
          <thead><tr className="border-b text-left text-muted-foreground"><th className="py-2">row</th><th>y = Wx</th><th>ŷ = Ŵx</th><th>|Δ|</th></tr></thead>
          <tbody>{y.map((v, i) => (
            <tr key={i} className="border-b border-dashed"><td className="py-2">y[{i}]</td><td>{v.toFixed(4)}</td><td>{yq[i]!.toFixed(4)}</td><td className="text-heat-neg">{Math.abs(v - yq[i]!).toFixed(4)}</td></tr>
          ))}</tbody>
        </table>
      </div>
    </div>
  );
}
