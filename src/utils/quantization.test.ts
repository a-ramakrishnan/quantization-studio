import { describe, it, expect } from "vitest";
import { quantize, intRange } from "./quantization";
const W = [
  [1, -1, 0.5, 0],
  [0.2, 0.3, -0.4, 2],
  [0, 0, 0, 0],
  [1, 1, 1, 1],
];
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
import { packNibbles, storageGB, floatBitsFor, quantRowSym } from "./quantization";
describe("format journey", () => {
  it("packs -7 and -6 into 0x9A", () => {
    expect(packNibbles([-7, -6])[0]!.hex).toBe("0x9A");
  });
  it("9B params storage sizes", () => {
    expect(storageGB(9e9, 32)).toBe(36);
    expect(storageGB(9e9, 16)).toBe(18);
    expect(storageGB(9e9, 8)).toBe(9);
    expect(storageGB(9e9, 4, 128)).toBeCloseTo(4.64, 2);
  });
  it("encodes 1.0 in FP16/BF16/FP8", () => {
    expect(floatBitsFor("FP16", 1).hex).toBe("0x3C00");
    expect(floatBitsFor("BF16", 1).hex).toBe("0x3F80");
    expect(floatBitsFor("FP8", 1).hex).toBe("0x38");
  });
  it("INT4 symmetric row max maps to 7", () => {
    expect(quantRowSym([-1, 0.5, 1], 4).q).toEqual([-7, 4, 7]);
  });
});
