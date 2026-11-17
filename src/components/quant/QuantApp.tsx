import { useEffect, useMemo, useRef, useState } from "react";
import { Check, Info, Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import { generatePreset, quantize } from "@/utils/quantization";
import type { PresetId, QuantConfig } from "@/types/quantization";
import { Stage1 } from "./Stage1";
import { Stage3, Stage4 } from "./LaterStages";
import { Stage2 } from "./Stage2";

const STAGES = ["Weights & Calibrate", "Quantize & Format Journey", "GPU HBM & On-the-Fly Dequantize", "Forward Pass (y = Wx)"];
const X = [1, -0.5, 0.25, 2];

export function QuantApp() {
  const [dark, setDark] = useState(false);
  const [showAll, setShowAll] = useState(false);
  const [unlocked, setUnlocked] = useState(1);
  const [preset, setPreset] = useState<PresetId>("normal");
  const [W, setW] = useState(() => generatePreset("normal"));
  const [cfg, setCfg] = useState<QuantConfig>({ format: "INT4", symmetric: true, granularity: "tensor", alpha: 1 });
  const result = useMemo(() => quantize(W, cfg), [W, cfg]);
  const [sel, setSel] = useState<[number, number]>([0, 0]);
  const refs = useRef<(HTMLElement | null)[]>([]);

  useEffect(() => { document.documentElement.classList.toggle("dark", dark); }, [dark]);

  const visible = showAll ? 4 : unlocked;
  const go = (s: number) => {
    setUnlocked((u) => Math.max(u, s));
    setTimeout(() => refs.current[s - 1]?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-30 border-b bg-background/85 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-6 py-3">
          <h1 className="font-serif text-lg font-medium sm:text-xl">Linear Layer Quantization &amp; GPU Execution</h1>
          <div className="flex items-center gap-2">
            <div className="hidden rounded-lg border p-0.5 text-sm sm:flex">
              {["Timeline View", "Show All"].map((l, i) => (
                <button key={l} onClick={() => setShowAll(i === 1)} className={cn("rounded-md px-3 py-1 transition-colors", showAll === (i === 1) ? "bg-secondary text-foreground" : "text-muted-foreground hover:text-foreground")}>{l}</button>
              ))}
            </div>
            <AboutDialog />
            <Button variant="ghost" size="icon" onClick={() => setDark(!dark)} aria-label="Toggle dark mode">{dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}</Button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 pb-32">
        <section className="py-12">
          <p className="font-mono text-sm text-primary">y = Wx</p>
          <h2 className="mt-2 max-w-3xl font-serif text-4xl leading-tight sm:text-5xl">How a 4×4 weight matrix shrinks, travels to the GPU, and still computes the right answer.</h2>
        </section>

        <nav className="mb-10 grid grid-cols-2 gap-2 md:grid-cols-4">
          {STAGES.map((s, i) => {
            const n = i + 1, open = n <= visible, done = n < unlocked;
            return (
              <button key={s} disabled={!open} onClick={() => go(n)} className={cn("rounded-lg border p-3 text-left transition-all", open ? "bg-card hover:border-primary/50" : "opacity-45")}>
                <div className="flex items-center gap-2">
                  <span className={cn("flex h-6 w-6 items-center justify-center rounded-full font-mono text-xs", open ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground")}>{done ? <Check className="h-3.5 w-3.5" /> : n}</span>
                  <span className="text-xs uppercase tracking-wider text-muted-foreground">Stage {n}</span>
                </div>
                <div className="mt-1.5 text-sm font-medium">{s}</div>
              </button>
            );
          })}
        </nav>

        <div className="space-y-16">
          {STAGES.slice(0, visible).map((s, i) => (
            <section key={s} ref={(el) => { refs.current[i] = el; }} className="scroll-mt-24 animate-stage-in">
              <div className="mb-6 flex items-baseline gap-3 border-b pb-3">
                <span className="font-mono text-sm text-primary">0{i + 1}</span>
                <h2 className="font-serif text-3xl">{s}</h2>
              </div>
              {i === 0 && <Stage1 W={W} setW={setW} preset={preset} onPreset={(p) => { setPreset(p); setW(generatePreset(p)); }} cfg={cfg} setCfg={setCfg} result={result} onNext={() => go(2)} nextUnlocked={visible >= 2} />}
              {i === 1 && <Stage2 W={W} setW={setW} cfg={cfg} result={result} sel={sel} setSel={setSel} onEdit={() => go(1)} onBack={() => go(1)} onNext={() => go(3)} />}
              {i === 2 && <Stage3 cfg={cfg} result={result} onNext={() => go(4)} />}
              {i === 3 && <Stage4 W={W} result={result} x={X} />}
            </section>
          ))}
        </div>
      </main>
    </div>
  );
}

function AboutDialog() {
  return (
    <Dialog>
      <DialogTrigger asChild><Button variant="ghost" size="sm"><Info className="mr-1 h-4 w-4" />About</Button></DialogTrigger>
      <DialogContent className="max-w-lg">
        <DialogHeader><DialogTitle className="font-serif text-2xl font-normal">About this explainer</DialogTitle></DialogHeader>
        <div className="space-y-3 text-sm leading-relaxed text-muted-foreground">
          <p><span className="text-foreground font-medium">Quantization</span> stores neural-network weights with fewer bits — e.g. 4-bit integers instead of 32-bit floats — using a scale S and zero-point Z so that w ≈ (q − Z)·S.</p>
          <p><span className="text-foreground font-medium">W4A16</span> means 4-bit weights and 16-bit activations: weights are kept compact in GPU memory and dequantized to FP16 on the fly inside the matmul kernel, cutting memory bandwidth roughly 4× versus FP16.</p>
          <p>Source &amp; notes: <a className="text-primary underline underline-offset-4" href="https://github.com/a-ramakrishnan/quantization" target="_blank" rel="noreferrer">github.com/a-ramakrishnan/quantization</a></p>
        </div>
      </DialogContent>
    </Dialog>
  );
}
