import type { FormatId, Matrix } from "@/types/quantization";
import { bytesFor, getFormat, matVec, quantize } from "./quantization";

export type ProjectionId = "Q" | "K" | "V";
export type BasicFormat = Exclude<FormatId, "INT2">;
export const PROJECTIONS: ProjectionId[] = ["Q", "K", "V"];
export const BASIC_FORMATS: { id: BasicFormat; description: string }[] = [
  { id: "BF16", description: "Keep the original 16-bit weights unchanged." },
  { id: "FP16", description: "16-bit floats with more precision, but a smaller numeric range." },
  { id: "FP8", description: "8-bit floats: 1 sign, 4 exponent, and 3 mantissa bits." },
  { id: "INT8", description: "8-bit integer codes; a scale restores approximate weights." },
  { id: "INT4", description: "4-bit integer codes; two weights fit in each byte." },
];

export function sourceBF16(matrix: Matrix): Matrix {
  return quantize(matrix, { format: "BF16", symmetric: true, granularity: "tensor", alpha: 1 }).deq;
}

export function initialProjections(): Record<ProjectionId, Matrix> {
  return {
    Q: [[0.5, -0.25, 0.75, 0.125], [-0.375, 0.625, 0.25, -0.5]],
    K: [[0.25, 0.5, -0.625, 0.375], [0.75, -0.125, 0.5, 0.25]],
    V: [[-0.5, 0.375, 0.125, 0.625], [0.25, 0.75, -0.375, -0.125]],
  };
}

export function basicProjection(matrix: Matrix, x: number[], format: BasicFormat) {
  const source = sourceBF16(matrix);
  const result = quantize(source, { format, symmetric: true, granularity: "tensor", alpha: 1 });
  const baseline = matVec(source, x);
  const output = matVec(result.deq, x);
  const count = source.flat().length;
  const baselineBytes = bytesFor(count, 16);
  const weightBytes = bytesFor(count, getFormat(format).bits);
  return {
    source, result, baseline, output,
    delta: baseline.map((value, i) => value - (output[i] ?? 0)),
    baselineBytes, weightBytes, payloadRatio: baselineBytes / weightBytes,
    // One FP32 scale for this symmetric per-tensor teaching example; Z=0 is implicit.
    scaleBytes: result.params ? 4 : 0,
  };
}