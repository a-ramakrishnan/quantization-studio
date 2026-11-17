import { createFileRoute } from "@tanstack/react-router";
import { QuantApp } from "@/components/quant/QuantApp";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Linear Layer Quantization & GPU Execution" },
      {
        name: "description",
        content:
          "Explore BF16 Query, Key, and Value weight quantization in Basic mode, or follow the full four-stage GPU execution journey in Advanced mode.",
      },
      { property: "og:title", content: "Linear Layer Quantization & GPU Execution" },
      {
        property: "og:description",
        content: "Compare BF16 and quantized Q/K/V projections, then explore weight formats and GPU execution in depth.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: QuantApp,
});
