import { useState } from "react";
import { ArrowLeft, ArrowRight, Layers, Boxes } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { bytesFor, getFormat, matVec } from "@/utils/quantization";
import type { Matrix, QuantConfig, QuantResult } from "@/types/quantization";
import { MatrixGrid } from "./MatrixGrid";

const v4 = (a: number[], d = 3) =>
  `[${a.map((x) => (Number.isInteger(x) ? x : x.toFixed(d))).join(", ")}]`;

function Tensor({ name, value, note }: { name: string; value: string; note?: string }) {
  return (
    <div className="rounded-md border bg-background/60 px-3 py-2">
      <div className="flex justify-between gap-3 text-xs">
        <span className="font-mono text-primary">{name}</span>
        {note && <span className="text-muted-foreground">{note}</span>}
      </div>
      <div className="mt-1 break-all font-mono text-sm">{value}</div>
    </div>
  );
}

type BlockId = "cpu" | "pcie" | "hbm" | "l2" | "sram" | "reg" | "tc";
const INFO: Record<BlockId, { title: string; role: string; why: string }> = {
  cpu: {
    title: "Host CPU / Calibrator",
    role: "Runs offline calibration: finds ranges, computes S and Z, and quantizes FP32 weights once.",
    why: "Quantization happens once ahead of time, so the GPU never pays for it at inference.",
  },
  pcie: {
    title: "PCIe link",
    role: "Copies the packed checkpoint from host RAM into GPU VRAM.",
    why: "Fewer bits → shorter upload and faster model loading.",
  },
  hbm: {
    title: "HBM / VRAM",
    role: "Large but comparatively slow off-chip memory holding every weight of the model.",
    why: "LLM decoding is memory-bandwidth bound; 4-bit weights move ~4× fewer bytes than FP16 per token.",
  },
  l2: {
    title: "L2 Cache",
    role: "Chip-wide cache shared by all SMs, buffering tiles fetched from HBM.",
    why: "Compressed tiles mean more of the working set fits on-chip.",
  },
  sram: {
    title: "Shared Memory (SRAM)",
    role: "Per-SM scratchpad where tiles of W_q, S, Z and X are staged cooperatively.",
    why: "Packed tiles use less SRAM, allowing larger tiles or more concurrent warps.",
  },
  reg: {
    title: "Register File · Dequantizer",
    role: "Each thread unpacks nibbles and computes Ŵ = S·(W_q − Z) in FP16 registers.",
    why: "Ŵ lives only in registers — it's never written back to HBM, so the bandwidth saving is kept.",
  },
  tc: {
    title: "Tensor Cores + FP32 Accumulators",
    role: "Perform the FP16 × FP16 multiply with FP32 accumulation: ŷ_i = Σ Ŵ_ij · x_j.",
    why: "Math stays in 16-bit (A16), so accuracy is preserved while memory traffic shrinks.",
  },
};

export function Stage3({
  W,
  cfg,
  result,
  row,
  setRow,
  x,
  onBack,
  onNext,
}: {
  W: Matrix;
  cfg: QuantConfig;
  result: QuantResult;
  row: number;
  setRow: (r: number) => void;
  x: number[];
  onBack: () => void;
  onNext: () => void;
}) {
  const [mode, setMode] = useState<"boxes" | "together">("boxes");
  const [box, setBox] = useState(0);
  const f = getFormat(cfg.format);
  const isInt = f.kind === "int";
  const wRow = W[row]!,
    qRow = result.q[row]!,
    dRow = result.deq[row]!;
  const blocks = qRow.map((_, c) => result.blockOf(row, c));
  const S = isInt ? blocks.map((b) => result.params!.scales[b]!) : [];
  const Z = isInt ? blocks.map((b) => result.params!.zeros[b]!) : [];
  const nMeta = result.params?.scales.length ?? 0;
  const fp32B = bytesFor(16, 32),
    qB = bytesFor(16, f.bits) + nMeta * 4;
  const yi = matVec([dRow], x)[0]!;
  const sz = isInt ? `S=${v4(S, 4)}  Z=${v4(Z)}` : "no scale (native float)";

  const BOXES = [
    {
      title: "Host CPU / Calibrator",
      inp: [["W[row] FP32", v4(wRow)]],
      out: [
        [`W_q ${f.label}`, v4(qRow, 4)],
        ["S, Z", sz],
      ],
      text: "Offline, the CPU inspects the weight range, computes the scale and zero-point, and rounds every weight to the target format. This happens once, before deployment.",
    },
    {
      title: "GPU HBM / VRAM",
      inp: [["W_q + S, Z", `${16} weights + ${nMeta} meta`]],
      out: [["Stored bytes", `${qB} B vs ${fp32B} B FP32 — ${(fp32B / qB).toFixed(1)}× smaller`]],
      text: `The packed ${f.label} weights${nMeta ? ` and ${nMeta} scale/zero pair${nMeta > 1 ? "s" : ""}` : ""} sit in HBM. Every token generation reads all of them, so their size directly sets the memory bandwidth cost.`,
    },
    {
      title: "L2 Cache → SM Shared Memory (SRAM)",
      inp: [["HBM tile", v4(qRow, 4)]],
      out: [
        ["SRAM: W_q tile", v4(qRow, 4)],
        ["SRAM: S, Z", sz],
        ["SRAM: X", v4(x)],
      ],
      text: "The kernel streams a tile of packed weights through L2 into each SM's shared memory, alongside the matching scales and the activation vector X (FP16).",
    },
    {
      title: "SM Register File (Dequantizer)",
      inp: [["W_q, S, Z", isInt ? `${v4(qRow)} · S, Z` : v4(qRow, 4)]],
      out: [["Ŵ FP16 (registers)", v4(dRow, 4)]],
      text: isInt
        ? "Each thread unpacks its nibbles and computes Ŵ = S·(W_q − Z) right in registers. Ŵ is never written back to HBM — it exists only for this multiply."
        : `${f.label} values are up-converted to FP16 in registers on the fly.`,
    },
    {
      title: "Tensor Cores & FP32 Accumulators",
      inp: [
        ["Ŵ[row]", v4(dRow, 4)],
        ["x", v4(x)],
      ],
      out: [[`ŷ[${row}]`, yi.toFixed(5)]],
      text: `Tensor cores multiply FP16 Ŵ by FP16 x and accumulate in FP32: ŷ_${row} = Σ_j Ŵ_${row},j · x_j = ${dRow.map((d, j) => `${d.toFixed(2)}·${x[j]}`).join(" + ")}.`,
    },
  ];
  const B = BOXES[box]!;

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex rounded-lg bg-muted p-1">
          {(
            [
              ["boxes", "Break Down Boxes", Boxes],
              ["together", "Reconstruct Together", Layers],
            ] as const
          ).map(([id, l, I]) => (
            <button
              key={id}
              onClick={() => setMode(id)}
              className={cn(
                "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm",
                mode === id ? "bg-card shadow-card" : "text-muted-foreground hover:text-foreground",
              )}
            >
              <I className="h-4 w-4" />
              {l}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          Row
          {[0, 1, 2, 3].map((i) => (
            <button
              key={i}
              onClick={() => setRow(i)}
              className={cn(
                "h-7 w-7 rounded font-mono text-xs",
                row === i ? "bg-primary text-primary-foreground" : "bg-card border",
              )}
            >
              {i}
            </button>
          ))}
        </div>
      </div>

      {mode === "boxes" ? (
        <div className="space-y-5">
          <div className="grid grid-cols-5 gap-2">
            {BOXES.map((b, i) => (
              <button
                key={i}
                onClick={() => setBox(i)}
                className={cn(
                  "rounded-lg border p-2 text-left text-xs transition-all",
                  i === box
                    ? "border-primary bg-accent"
                    : i < box
                      ? "bg-card"
                      : "bg-card opacity-60",
                )}
              >
                <div className="font-mono text-primary">Box {i + 1}</div>
                <div className="mt-0.5 hidden font-medium sm:block">{b.title}</div>
              </button>
            ))}
          </div>
          <div key={box} className="animate-stage-in rounded-xl border bg-card p-6 shadow-card">
            <div className="font-mono text-xs text-primary">Box {box + 1} / 5</div>
            <h3 className="mt-1 font-serif text-2xl">{B.title}</h3>
            <div className="mt-5 grid items-center gap-4 md:grid-cols-[1fr_auto_1fr]">
              <div className="space-y-2">
                <div className="text-xs uppercase tracking-wider text-muted-foreground">
                  Input to this box
                </div>
                {B.inp.map(([n, v]) => (
                  <Tensor key={n} name={n!} value={v!} />
                ))}
              </div>
              <ArrowRight className="mx-auto h-5 w-5 rotate-90 text-muted-foreground md:rotate-0" />
              <div className="space-y-2">
                <div className="text-xs uppercase tracking-wider text-muted-foreground">
                  Output from this box
                </div>
                {B.out.map(([n, v]) => (
                  <Tensor key={n} name={n!} value={v!} />
                ))}
              </div>
            </div>
            <p className="mt-5 rounded-lg bg-muted/60 p-4 text-sm leading-relaxed text-muted-foreground">
              {B.text}
            </p>
            <div className="mt-5 flex flex-wrap justify-between gap-2">
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={box === 0}
                  onClick={() => setBox(box - 1)}
                >
                  <ArrowLeft className="mr-1 h-4 w-4" />
                  Prev box
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={box === 4}
                  onClick={() => setBox(box + 1)}
                >
                  Next box
                  <ArrowRight className="ml-1 h-4 w-4" />
                </Button>
              </div>
              <Button size="sm" onClick={() => setMode("together")}>
                <Layers className="mr-1 h-4 w-4" />
                Reconstruct All Boxes Together
              </Button>
            </div>
          </div>
        </div>
      ) : (
        <Together result={result} W={W} f={f} />
      )}

      <div className="flex justify-between gap-2">
        <Button variant="outline" size="lg" onClick={onBack}>
          <ArrowLeft className="mr-1 h-4 w-4" />
          Back
        </Button>
        <Button size="lg" onClick={onNext}>
          Provide Input Vector x &amp; Run Linear Layer <ArrowRight className="ml-1 h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}

function Together({
  result,
  W,
  f,
}: {
  result: QuantResult;
  W: Matrix;
  f: ReturnType<typeof getFormat>;
}) {
  const [sel, setSel] = useState<BlockId>("reg");
  const err = W.map((r, i) => r.map((w, j) => w - result.deq[i]![j]!));
  const Block = ({
    id,
    x,
    y,
    w,
    h,
    label,
    sub,
  }: {
    id: BlockId;
    x: number;
    y: number;
    w: number;
    h: number;
    label: string;
    sub?: string;
  }) => (
    <g onClick={() => setSel(id)} className="cursor-pointer">
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={8}
        className={cn(
          "transition-all",
          sel === id
            ? "fill-accent stroke-primary"
            : "fill-card stroke-border hover:stroke-primary/60",
        )}
        strokeWidth={sel === id ? 2 : 1.2}
      />
      <text
        x={x + w / 2}
        y={y + h / 2 - (sub ? 4 : -4)}
        textAnchor="middle"
        className="fill-foreground text-[12px] font-medium"
      >
        {label}
      </text>
      {sub && (
        <text
          x={x + w / 2}
          y={y + h / 2 + 12}
          textAnchor="middle"
          className="fill-muted-foreground font-mono text-[9px]"
        >
          {sub}
        </text>
      )}
    </g>
  );
  const Flow = ({ d }: { d: string }) => (
    <path
      d={d}
      className="flow-dash fill-none stroke-primary"
      strokeWidth={1.8}
      markerEnd="url(#fa)"
    />
  );
  const info = INFO[sel];
  return (
    <div className="space-y-6">
      <div className="grid gap-5 lg:grid-cols-[1.7fr_1fr]">
        <div className="rounded-xl border bg-card p-4 shadow-card">
          <svg viewBox="0 0 720 300" className="w-full">
            <defs>
              <marker
                id="fa"
                viewBox="0 0 10 10"
                refX="8"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto"
              >
                <path d="M0 0L10 5L0 10z" className="fill-primary" />
              </marker>
            </defs>
            <Block id="cpu" x={10} y={40} w={110} h={60} label="Host CPU" sub="FP32 → W_q, S, Z" />
            <Block id="pcie" x={150} y={55} w={70} h={30} label="PCIe" />
            <Block
              id="hbm"
              x={250}
              y={40}
              w={120}
              h={60}
              label="HBM / VRAM"
              sub={`packed ${f.label}`}
            />
            <Block id="l2" x={400} y={40} w={100} h={60} label="L2 Cache" sub="shared by SMs" />
            <rect
              x={150}
              y={150}
              width={560}
              height={135}
              rx={12}
              className="fill-muted/40 stroke-border"
              strokeDasharray="4 4"
            />
            <text
              x={165}
              y={172}
              className="fill-muted-foreground text-[11px] uppercase tracking-wider"
            >
              Streaming Multiprocessor (SM)
            </text>
            <Block
              id="sram"
              x={170}
              y={190}
              w={150}
              h={70}
              label="Shared Memory"
              sub="SRAM · W_q, S, Z, X"
            />
            <Block
              id="reg"
              x={350}
              y={190}
              w={160}
              h={70}
              label="Register File"
              sub="Ŵ = S·(W_q − Z)"
            />
            <Block
              id="tc"
              x={540}
              y={190}
              w={150}
              h={70}
              label="Tensor Cores"
              sub="FP16×FP16 → FP32"
            />
            <Flow d="M120 70 L148 70" />
            <Flow d="M220 70 L248 70" />
            <Flow d="M370 70 L398 70" />
            <Flow d="M450 100 C450 140 245 140 245 188" />
            <Flow d="M320 225 L348 225" />
            <Flow d="M510 225 L538 225" />
            {[
              { x: 310, l: "HBM" },
              { x: 245, y: 278, l: "SRAM" },
              { x: 430, y: 278, l: "Registers" },
            ].map((h, i) => (
              <text
                key={i}
                x={h.x}
                y={h.y ?? 28}
                textAnchor="middle"
                className="fill-primary font-mono text-[10px]"
              >
                {i + 1}. {h.l}
              </text>
            ))}
          </svg>
        </div>
        <div key={sel} className="animate-stage-in h-fit rounded-xl border bg-card p-5 shadow-card">
          <div className="text-xs uppercase tracking-wider text-primary">Inspector</div>
          <h4 className="mt-1 font-serif text-xl">{info.title}</h4>
          <p className="mt-3 text-sm leading-relaxed">{info.role}</p>
          <p className="mt-3 rounded-lg bg-muted/60 p-3 text-sm text-muted-foreground">
            <span className="font-medium text-foreground">Why quantization helps: </span>
            {info.why}
          </p>
          <p className="mt-3 text-xs text-muted-foreground">Click any block in the diagram.</p>
        </div>
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        <MatrixGrid
          m={result.q}
          label={`W_q (${f.label})`}
          format={(v) => (f.kind === "int" ? String(v) : v.toPrecision(4))}
        />
        <MatrixGrid m={result.deq} label="Dequantized Ŵ" />
        <MatrixGrid m={err} label="Error (W − Ŵ)" />
      </div>
    </div>
  );
}
