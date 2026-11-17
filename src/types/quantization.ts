export type Matrix = number[][];
export type FormatId = "BF16" | "FP16" | "FP8" | "INT8" | "INT4" | "INT2";
export type Granularity = "tensor" | "channel" | "group";
export type PresetId = "normal" | "outliers" | "shifted" | "uniform" | "random";

export interface FormatSpec {
  id: FormatId;
  label: string;
  kind: "float" | "int";
  bits: number;
  exponentBits?: number;
  mantissaBits?: number;
  maxValue?: number;
}

export interface QuantConfig {
  format: FormatId;
  symmetric: boolean;
  granularity: Granularity;
  alpha: number; // clipping ratio 0.5..1
}

export interface QuantParams {
  /** one entry per block (tensor = 1, channel = rows, group = rows*cols/2) */
  scales: number[];
  zeros: number[];
  qmin: number;
  qmax: number;
}

export interface QuantResult {
  q: Matrix; // integer codes (int) or rounded values (float)
  deq: Matrix;
  params: QuantParams | null;
  blockOf: (r: number, c: number) => number;
}
