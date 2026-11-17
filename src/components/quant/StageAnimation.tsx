import { useEffect, useId, useRef, useState } from "react";
import { Pause, Play, RotateCcw, SkipBack, SkipForward } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { Matrix, QuantConfig, QuantResult } from "@/types/quantization";
import { getFormat, matVec, matrixRange, packNibbles, twosComplement } from "@/utils/quantization";

const LESSONS = [
  { title: "Find the scale of the weights", chapters: ["The matrix", "The range", "The scale"] },
  { title: "Turn a weight into a stored code", chapters: ["Divide & round", "Store the bits", "See the error"] },
  { title: "Follow the weights through the GPU", chapters: ["Upload to HBM", "Move to the SM", "Reconstruct"] },
  { title: "Watch the linear layer compute", chapters: ["Multiply", "Accumulate", "Compare"] },
];
const n = (v: number) => Number(v.toFixed(4)).toString();
const ease = (t: number) => t * t * (3 - 2 * t);

export function StageAnimation({ stage, W, cfg, result, x, selected, active }: {
  stage: number; W: Matrix; cfg: QuantConfig; result: QuantResult; x: number[];
  selected: [number, number]; active: boolean;
}) {
  const lesson = LESSONS[stage];
  const ref = useRef<HTMLElement>(null);
  const grid = useId().replace(/:/g, "");
  const [onscreen, setOnscreen] = useState(false);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new IntersectionObserver(([entry]) => setOnscreen(entry?.isIntersecting ?? false), { threshold: 0.1 });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => { setReduced(media.matches); if (media.matches) setPlaying(false); };
    update(); media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  useEffect(() => {
    if (!playing || !active || !onscreen || reduced) return;
    let frame = 0;
    let previous = performance.now();
    const tick = (now: number) => {
      const dt = Math.min(now - previous, 100) / 1000;
      previous = now;
      setTime(t => Math.min(18, t + dt * speed));
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, active, onscreen, reduced, speed]);
  useEffect(() => { if (time >= 18) setPlaying(false); }, [time]);
  if (!lesson) return null;
  const chapter = Math.min(2, Math.floor(time / 6));
  const phase = ease(Math.max(0, Math.min(1, (time - chapter * 6) / 4)));
  const spec = getFormat(cfg.format);
  const [row, col] = selected;
  const w = W[row]?.[col] ?? 0;
  const q = result.q[row]?.[col] ?? 0;
  const deq = result.deq[row]?.[col] ?? 0;
  const block = result.blockOf(row, col);
  const scale = result.params?.scales[block] ?? 1;
  const zero = result.params?.zeros[block] ?? 0;
  const range = matrixRange(W);
  const max = Math.max(...W.flat().map(Math.abs), 0.001);
  const px = (v: number) => 480 + v / max * 330;
  const y = matVec(W, x);
  const yq = matVec(result.deq, x);
  const narrative = [
    ["Sixteen original FP32 weights. Blue is positive; rose is negative.", `The observed range is ${n(range.min)} to ${n(range.max)}. Outliers stretch the grid.`, spec.kind === "int" ? `The selected weight uses block ${block}, with S = ${n(scale)} and Z = ${zero}.` : `${spec.label} stores sign, exponent and mantissa instead of integer scales.`],
    [spec.kind === "int" ? `Divide by the selected block's scale, add Z, round, and clamp to the code range.` : `Round the selected FP32 weight to ${spec.label} precision.`, cfg.format === "INT4" ? "Two 4-bit codes share one byte. The first occupies the high nibble." : `Each stored weight occupies ${spec.bits} bits.`, `The reconstructed value is ${n(deq)}. The original minus reconstruction is ${n(w - deq)}.`],
    ["The CPU uploads compact weights and their metadata to GPU memory.", "Weights travel through L2 to shared memory. Compact codes reduce the bytes being moved.", spec.kind === "int" ? "Registers reconstruct (q − Z) × S just before the floating-point multiply." : "Floating-point weights are loaded into registers for the multiply; no integer scale is needed."],
    ["Each weight in the selected row multiplies the matching input element.", "Add the four products. This is one output of the linear layer.", "Repeat for every row and compare the original output with the reconstructed-weight output."],
  ][stage]?.[chapter];
  const jump = (next: number) => {setTime(Math.max(0, Math.min(2, next)) * 6);setPlaying(false);};

  return <section ref={ref} className="math-theater mb-7" aria-label={`Stage ${stage + 1} animated lesson`}>
    <div className="px-4 pt-6 sm:px-6">
      <div className="flex justify-between gap-2 font-mono text-xs text-theater-muted"><span>STAGE 0{stage + 1} · {spec.label}</span><span className="text-theater-gold">0{chapter + 1} / 03</span></div>
      <h3 className="mt-3 font-serif text-2xl sm:text-3xl">{lesson.title}</h3>
      <div className="stage-math-viewport mt-4">
        <svg viewBox="0 0 960 400" className="h-full w-full" role="img" aria-label={`${lesson.chapters[chapter]} for Stage ${stage + 1}`}>
          <defs><pattern id={grid} width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40 0H0V40" fill="none" className="stroke-theater-grid" strokeWidth="0.5" /></pattern></defs>
          <rect width="960" height="400" fill={`url(#${grid})`} />
          {stage === 0 && <g>
            {chapter === 0 ? <g>
              <text x="480" y="48" textAnchor="middle" className="fill-theater-ink font-serif" fontSize="30">W ∈ ℝ⁴ˣ⁴</text>
              <path d="M285 82H270V330H285 M675 82H690V330H675" fill="none" className="stroke-theater-blue" strokeWidth="2" />
              {W.flat().map((v, i) => <text key={i} x={330 + i % 4 * 100} y={126 + Math.floor(i / 4) * 58} textAnchor="middle" fontSize="22" opacity={0.65 + 0.35 * Math.min(1, phase * 2 + (15 - i) / 16)} className={v >= 0 ? "fill-theater-blue font-mono" : "fill-theater-rose font-mono"}>{n(v)}</text>)}
            </g> : chapter === 1 ? <g>
              <text x="480" y="54" textAnchor="middle" className="fill-theater-ink font-serif" fontSize="28">The range sets the spacing</text>
              <line x1="100" y1="240" x2="860" y2="240" className="stroke-theater-muted" />
              <line x1={px(range.min)} y1="250" x2={px(range.max)} y2="250" className="stroke-theater-gold" strokeWidth="3" />
              {W.flat().map((v, i) => <circle key={i} cx={480 + (px(v) - 480) * phase} cy={216 - i % 3 * 15} r="6" className={v >= 0 ? "fill-theater-blue" : "fill-theater-rose"} />)}
              <text x={px(range.min)} y="294" textAnchor="middle" className="fill-theater-rose font-mono" fontSize="20">{n(range.min)}</text><text x={px(range.max)} y="294" textAnchor="middle" className="fill-theater-blue font-mono" fontSize="20">{n(range.max)}</text>
              <text x="480" y="354" textAnchor="middle" className="fill-theater-muted font-mono" fontSize="17">α = {cfg.alpha.toFixed(2)} · {cfg.granularity} granularity</text>
            </g> : <g>
              <text x="480" y="100" textAnchor="middle" className="fill-theater-blue font-serif" fontSize="30">w[{row},{col}] = {n(w)}</text>
              <text x="480" y="190" textAnchor="middle" className="fill-theater-gold font-mono" fontSize="34">{spec.kind === "int" ? `S = ${n(scale)}     Z = ${zero}` : `1 + ${spec.exponentBits} + ${spec.mantissaBits} bits`}</text>
              <text x="480" y="272" textAnchor="middle" className="fill-theater-green font-serif" fontSize="26">{spec.kind === "int" ? `q ∈ [${result.params?.qmin}, ${result.params?.qmax}]` : "Sign + exponent + mantissa"}</text>
              <text x="480" y="342" textAnchor="middle" className="fill-theater-muted font-mono" fontSize="17">{spec.kind === "int" ? `${cfg.symmetric ? "Symmetric" : "Asymmetric"} · block ${block}` : `${spec.label} · ${spec.bits} bits per weight`}</text>
            </g>}
          </g>}
          {stage === 1 && <g>
            {chapter === 0 ? <g>
              <text x="480" y="60" textAnchor="middle" className="fill-theater-ink font-serif" fontSize="28">{spec.kind === "int" ? "q = clamp(round(w / S) + Z)" : `FP32 → ${spec.label}`}</text>
              <text x="215" y="183" textAnchor="middle" className="fill-theater-blue font-mono" fontSize="38">{n(w)}</text>
              <path d="M320 174H640 M630 166L640 174L630 182" fill="none" className="stroke-theater-muted" />
              <circle cx={340 + phase * 280} cy="174" r="6" className="fill-theater-gold" />
              <text x="752" y="183" textAnchor="middle" className="fill-theater-gold font-mono" fontSize="38">{n(q)}</text>
              <text x="480" y="284" textAnchor="middle" className="fill-theater-muted font-mono" fontSize="19">{spec.kind === "int" ? `w / S = ${n(w / scale)} · Z = ${zero}` : `${spec.mantissaBits} mantissa bits retained`}</text>
            </g> : chapter === 1 ? <g>
              <text x="480" y="70" textAnchor="middle" className="fill-theater-ink font-serif" fontSize="29">{cfg.format === "INT4" ? "Two weights. One byte." : `${spec.bits} bits per stored weight`}</text>
              {cfg.format === "INT4" ? <g>
                <text x="480" y="143" textAnchor="middle" className="fill-theater-gold font-mono" fontSize="32">{twosComplement(result.q[row]?.[0] ?? 0, 4)} {twosComplement(result.q[row]?.[1] ?? 0, 4)}</text>
                <text x="480" y="206" textAnchor="middle" className="fill-theater-green font-mono" fontSize="34">{packNibbles(result.q[row] ?? []).map(b => b.hex).join("    ")}</text>
              </g> : <g>{Array.from({length:spec.bits}, (_, i) => <rect key={i} x={150 + i * (660 / spec.bits)} y="145" width={600 / spec.bits} height="55" rx="2" className={i < Math.round(phase * spec.bits) ? "fill-theater-gold" : "fill-theater-grid"} />)}</g>}
              <text x="480" y="302" textAnchor="middle" className="fill-theater-muted font-mono" fontSize="20">FP32: 64 B weights → {16 * spec.bits / 8} B weight payload</text>
              <text x="480" y="347" textAnchor="middle" className="fill-theater-muted font-mono" fontSize="16">Integer scales and zero-points are additional metadata</text>
            </g> : <g>
              <text x="480" y="67" textAnchor="middle" className="fill-theater-ink font-serif" fontSize="30">Rounding leaves a small gap</text>
              <line x1="100" y1="234" x2="860" y2="234" className="stroke-theater-muted" />
              <circle cx={px(w)} cy="215" r="9" className="fill-theater-blue" /><circle cx={px(w + (deq - w) * phase)} cy="250" r="9" className="fill-theater-green" />
              <text x="250" y="145" textAnchor="middle" className="fill-theater-blue font-mono" fontSize="25">w = {n(w)}</text><text x="710" y="145" textAnchor="middle" className="fill-theater-green font-mono" fontSize="25">ŵ = {n(deq)}</text>
              <text x="480" y="332" textAnchor="middle" className="fill-theater-rose font-mono" fontSize="28">w − ŵ = {n(w - deq)}</text>
            </g>}
          </g>}
          {stage === 2 && <g>
            <text x="480" y="53" textAnchor="middle" className="fill-theater-ink font-serif" fontSize="28">{chapter === 2 ? (spec.kind === "int" ? "ŵ = (q − Z) × S" : "Stored floats → register values") : "Compact weights travel together"}</text>
            {chapter < 2 ? <g>
              {["CPU", "HBM", "L2", "SRAM", "Registers"].map((label, i) => <g key={label}>
                <rect x={55 + i * 182} y="125" width="126" height="115" rx="4" fill="none" className={i <= (chapter === 0 ? 1 : 4) ? "stroke-theater-gold" : "stroke-theater-grid"} strokeWidth="2" />
                <text x={118 + i * 182} y="191" textAnchor="middle" className="fill-theater-ink font-mono" fontSize="18">{label}</text>
                {i < 4 && <path d={`M ${181+i*182} 182 H ${237+i*182}`} className="stroke-theater-muted" strokeDasharray="3 5" />}
              </g>)}
              {[0,1,2,3].map(i => <circle key={i} cx={chapter === 0 ? 184 + phase * 48 : 367 + ((phase + i / 4) % 1) * 544} cy={265 + i % 2 * 18} r="5" className="fill-theater-gold" />)}
              <text x="480" y="346" textAnchor="middle" className="fill-theater-muted font-mono" fontSize="18">{spec.bits}-bit weights + metadata · row {row}</text>
            </g> : <g>
              {(result.q[row] ?? []).map((v, i) => <g key={i}>
                <text x={170 + i * 205} y="135" textAnchor="middle" className="fill-theater-gold font-mono" fontSize="28">{n(v)}</text>
                <path d={`M ${170+i*205} 160 V 222`} className="stroke-theater-muted" />
                <circle cx={170+i*205} cy={163+phase*55} r="4" className="fill-theater-green" />
                <text x={170 + i * 205} y="268" textAnchor="middle" className="fill-theater-green font-mono" fontSize="25">{n(result.deq[row]?.[i] ?? 0)}</text>
              </g>)}
              <text x="480" y="345" textAnchor="middle" className="fill-theater-muted font-mono" fontSize="18">Register values → Tensor Cores × activations</text>
            </g>}
          </g>}
          {stage === 3 && <g>
            <text x="480" y="52" textAnchor="middle" className="fill-theater-ink font-serif" fontSize="28">{chapter === 0 ? `Row ${row}: multiply each weight by x` : chapter === 1 ? `y[${row}] = Σⱼ w[${row},j] x[j]` : "y = Wx    vs    ŷ = Ŵx"}</text>
            {chapter < 2 ? <g>
              {(W[row] ?? []).map((v, i) => <g key={i}>
                <text x={170+i*205} y="119" textAnchor="middle" className="fill-theater-blue font-mono" fontSize="23">{n(v)} × {n(x[i] ?? 0)}</text>
                <path d={`M ${170+i*205} 140 V ${174+phase*20}`} className="stroke-theater-gold" />
                <text x={170+i*205 + (480-(170+i*205))*(chapter === 1 ? phase : 0)} y={215+(chapter === 1 ? phase*58 : 0)} textAnchor="middle" opacity={chapter === 1 ? 1-phase : 1} className="fill-theater-gold font-mono" fontSize="25">{n(v*(x[i] ?? 0))}</text>
              </g>)}
              <text x="480" y="320" textAnchor="middle" className="fill-theater-green font-mono" fontSize="36" opacity={chapter === 1 ? phase : 0.5}>y[{row}] = {n(y[row] ?? 0)}</text>
            </g> : <g>
              <text x="240" y="105" textAnchor="middle" className="fill-theater-blue font-mono" fontSize="20">FP32 baseline</text><text x="570" y="105" textAnchor="middle" className="fill-theater-green font-mono" fontSize="20">{spec.label} weights</text><text x="810" y="105" textAnchor="middle" className="fill-theater-rose font-mono" fontSize="20">Δ</text>
              {y.map((v, i) => <g key={i}>
                <text x="240" y={155+i*54} textAnchor="middle" className="fill-theater-blue font-mono" fontSize="25">{v.toFixed(4)}</text>
                <text x="570" y={155+i*54} textAnchor="middle" className="fill-theater-green font-mono" fontSize="25">{(v+((yq[i] ?? 0)-v)*phase).toFixed(4)}</text>
                <text x="810" y={155+i*54} textAnchor="middle" className="fill-theater-rose font-mono" fontSize="21">{(v-(yq[i] ?? 0)).toFixed(5)}</text>
              </g>)}
            </g>}
          </g>}
        </svg>
      </div>
      <p className="mx-auto min-h-16 max-w-3xl text-center font-serif text-lg leading-relaxed text-theater-ink" aria-live={playing ? "off" : "polite"}>{narrative}</p>
      <div className="mt-3 grid grid-cols-3 gap-2 border-b border-theater-grid pb-3">
        {lesson.chapters.map((label, i) => <Button key={label} variant="ghost" aria-pressed={chapter === i} onClick={() => jump(i)} className={cn("h-auto min-h-10 whitespace-normal px-1 text-xs sm:text-sm", chapter === i ? "text-theater-gold" : "text-theater-muted")}>{label}</Button>)}
      </div>
      <div className="flex flex-wrap items-center gap-2 py-4 sm:gap-3">
        <Button variant="ghost" size="icon" title="Restart animation" aria-label="Restart animation" onClick={() => jump(0)}><RotateCcw /></Button>
        <Button variant="ghost" size="icon" title="Previous chapter" aria-label="Previous chapter" disabled={chapter === 0} onClick={() => jump(chapter-1)}><SkipBack /></Button>
        <Button variant="secondary" size="icon" title={playing ? "Pause animation" : "Play animation"} aria-label={playing ? "Pause animation" : "Play animation"} disabled={reduced} onClick={() => {if(time>=18)setTime(0);setPlaying(!playing);}}>{playing ? <Pause /> : <Play />}</Button>
        <Button variant="ghost" size="icon" title="Next chapter" aria-label="Next chapter" disabled={chapter === 2} onClick={() => jump(chapter+1)}><SkipForward /></Button>
        <input type="range" min="0" max="18" step="0.05" value={time} aria-label="Animation progress" className="min-w-20 flex-1 accent-theater-gold" onChange={e => {setTime(Number(e.target.value));setPlaying(false);}} />
        <span className="w-20 text-right font-mono text-xs text-theater-muted">{time.toFixed(1)} / 18s</span>
        <select aria-label="Playback speed" value={speed} onChange={e => setSpeed(Number(e.target.value))} className="rounded-md border border-theater-grid bg-theater px-2 py-1 text-xs text-theater-ink"><option value="0.5">0.5×</option><option value="1">1×</option><option value="2">2×</option></select>
        {reduced && <span className="w-full text-xs text-theater-muted">Reduced motion: select a chapter to view each step.</span>}
      </div>
    </div>
  </section>;
}