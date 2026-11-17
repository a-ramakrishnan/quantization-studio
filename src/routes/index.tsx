import { createFileRoute } from "@tanstack/react-router";
import { QuantApp } from "@/components/quant/QuantApp";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Linear Layer Quantization & GPU Execution" },
      { name: "description", content: "Interactive visual guide to quantizing y = Wx weights (BF16 to INT2) and running them on a GPU." },
      { property: "og:title", content: "Linear Layer Quantization & GPU Execution" },
      { property: "og:description", content: "Interactive visual guide to quantizing y = Wx weights and running them on a GPU." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: QuantApp,
});
