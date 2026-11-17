# Quantization Studio

Build a sleek, interactive educational web app in React + TypeScript + Tailwind CSS (Distill.pub / Stripe-docs aesthetic) for visualizing **Linear Layer Weight Quantization & GPU Execution** (`y = Wx`).

If `src/utils/quantization.ts` and `src/types/quantization.ts` already exist in the repo, keep them as the math source of truth and import from them.

Start with:

1. **Header & Navigation**:

   - Title: "Linear Layer Quantization & GPU Execution"

   - Light / Dark mode toggle

   - "Timeline View / Show All" toggle

   - "About" modal explaining quantization, W4A16, and linking to `https://github.com/a-ramakrishnan/quantization`

2. **4-Stage Progressive Timeline Stepper** (on initial load, ONLY Stage 1 is visible so it feels light and uncluttered):

   - Stage 1: Weights & Calibrate

   - Stage 2: Quantize & Format Journey (`FP32 → Target Format`)

   - Stage 3: GPU HBM & On-the-Fly Dequantize

   - Stage 4: Forward Pass (`y = Wx`)

3. **Stage 1 UI (Inputs → First-Level Output)**:

   - Left column: Weight Preset pills (`Normal Weights`, `With Outliers`, `Shifted / Biased`, `Uniform Spread`, `Random`) + Target Format selector along the precision ladder: `BF16` → `FP16` → `FP8 (E4M3)` → `INT8` → `INT4` → `INT2`. For integer formats, show `Symmetric (Z=0)` vs `Asymmetric`, `Per-Tensor / Per-Channel / Per-Group(2)`, and `Outlier Clipping (α)` slider. Below that, an editable 4×4 **FP32 Weight Matrix (W)** with a subtle blue/rose heatmap.

   - Right column: **First-Level Output Card** showing Observed Range `[min, max]`, Target Range, computed **Scale (S)** and **Zero-Point (Z)** (or bit-field layout `Sign + Exponent + Mantissa` for BF16/FP16/FP8), a plain-English explanation of what happens next, and a primary CTA button: **"Convert Weights → Stage 2"**.

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/4803b597-537b-4ca5-a049-7a26538ab36b).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
