import { describe, expect, it } from "vitest";
import { basicProjection, initialProjections, PROJECTIONS, sourceBF16 } from "./basic-quantization";

describe("Basic BF16 projection rules", () => {
  it("provides three 2×4 Q/K/V matrices for a four-element input", () => {
    const matrices = initialProjections();
    expect(PROJECTIONS).toEqual(["Q", "K", "V"]);
    for (const matrix of Object.values(matrices)) {
      expect(matrix).toHaveLength(2);
      expect(matrix.map(row => row.length)).toEqual([4, 4]);
      expect(basicProjection(matrix, [1, -0.5, 0.25, 2], "BF16").baseline).toHaveLength(2);
    }
  });
  it("rounds edited source weights to BF16 before using a target format", () => {
    expect(sourceBF16([[1.003, 0, 0, 0], [0, 0, 0, 0]])[0]?.[0]).toBe(1);
    expect(basicProjection([[1.003, 0, 0, 0], [0, 0, 0, 0]], [1, 0, 0, 0], "FP16").baseline).toEqual([1, 0]);
  });
  it("shows BF16 weight-payload ratios 1×, 2×, and 4× separately from scale overhead", () => {
    const matrix = initialProjections().Q;
    expect(["BF16", "FP16", "FP8", "INT8", "INT4"].map(format =>
      basicProjection(matrix, [1, 0, 0, 0], format as "BF16" | "FP16" | "FP8" | "INT8" | "INT4").payloadRatio,
    )).toEqual([1, 1, 2, 2, 4]);
    const int4 = basicProjection(matrix, [1, 0, 0, 0], "INT4");
    expect(int4.weightBytes).toBe(4);
    expect(int4.scaleBytes).toBe(4);
  });
  it("computes Q = W_Q·x and zero delta when BF16 is retained", () => {
    const projection = basicProjection(initialProjections().Q, [1, -0.5, 0.25, 2], "BF16");
    expect(projection.baseline).toEqual([1.0625, -1.625]);
    expect(projection.output).toEqual(projection.baseline);
    expect(projection.delta).toEqual([0, 0]);
  });
  it("computes the per-element delta as baseline minus reconstructed output", () => {
    const projection = basicProjection(initialProjections().Q, [1, -0.5, 0.25, 2], "INT4");
    expect(projection.output[0]).toBeCloseTo(1.044642857142857);
    expect(projection.delta[0]).toBeCloseTo(0.017857142857143);
  });
});