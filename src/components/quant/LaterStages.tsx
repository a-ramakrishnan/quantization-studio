import { ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { bytesFor, getFormat, matVec, mse } from "@/utils/quantization";
import type { Matrix, QuantConfig, QuantResult } from "@/types/quantization";
import { MatrixGrid } from "./MatrixGrid";

const Next = ({ label, onClick }: { label: string; onClick: () => void }) => (
  <Button size="lg" onClick={onClick} className="mt-6">
    {label} <ArrowRight className="ml-1 h-4 w-4" />
  </Button>
);

export function Stage4({ W, result, x }: { W: Matrix; result: QuantResult; x: number[] }) {
  const y = matVec(W, x),
    yq = matVec(result.deq, x);
  return (
    <div>
      <p className="max-w-2xl text-muted-foreground">
        With input <span className="font-mono text-foreground">x = [{x.join(", ")}]</span>, compare
        the exact FP32 output with the quantized one.
      </p>
      <div className="mt-6 overflow-x-auto">
        <table className="w-full max-w-xl font-mono text-sm">
          <thead>
            <tr className="border-b text-left text-muted-foreground">
              <th className="py-2">row</th>
              <th>y = Wx</th>
              <th>ŷ = Ŵx</th>
              <th>|Δ|</th>
            </tr>
          </thead>
          <tbody>
            {y.map((v, i) => (
              <tr key={i} className="border-b border-dashed">
                <td className="py-2">y[{i}]</td>
                <td>{v.toFixed(4)}</td>
                <td>{yq[i]!.toFixed(4)}</td>
                <td className="text-heat-neg">{Math.abs(v - yq[i]!).toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
