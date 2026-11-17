import { describe, it, expect } from "vitest";
import { quantize, intRange } from "./quantization";
const W = [[1, -1, 0.5, 0], [0.2, 0.3, -0.4, 2], [0, 0, 0, 0], [1, 1, 1, 1]];
describe("quantization", () => {
  it("INT8 symmetric range is -127..127 with Z=0", () => {
    expect(intRange(8, true)).toEqual({ qmin: -127, qmax: 127 });
    const r = quantize(W, { format: "INT8", symmetric: true, granularity: "tensor", alpha: 1 });
    expect(r.params!.zeros[0]).toBe(0);
    expect(r.params!.scales[0]).toBeCloseTo(2 / 127);
  });
  it("INT4 asymmetric range is 0..15", () => {
    expect(intRange(4, false)).toEqual({ qmin: 0, qmax: 15 });
  });
  it("per-group(2) yields 8 scales for 4x4", () => {
    const r = quantize(W, { format: "INT4", symmetric: true, granularity: "group", alpha: 1 });
    expect(r.params!.scales.length).toBe(8);
  });
});
