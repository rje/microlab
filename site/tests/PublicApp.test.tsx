import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { PublicApp } from "../src/PublicApp";

const library = {
  phases: [
    {
      id: "phase-0",
      title: "Phase 0: Evaluation",
      papers: [
        {
          id: "mmlu",
          title: "Measuring Massive Multitask Language Understanding",
          authors: "Hendrycks et al.",
          year: 2020,
          topic: "evaluation",
          sourceUrl: "https://arxiv.org/abs/2009.03300",
          pdfUrl: "/public/pdf/mmlu",
          overview: { paperId: "mmlu", tldr: "Broad 57-subject benchmark.", sections: [], readingFocus: [] }
        }
      ]
    }
  ],
  additional: []
};

describe("PublicApp", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("renders the phase-grouped reading list with summaries", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, json: async () => library })));
    render(<PublicApp />);
    expect(await screen.findByText("Phase 0: Evaluation")).toBeInTheDocument();
    expect(screen.getByText("Measuring Massive Multitask Language Understanding")).toBeInTheDocument();
    expect(screen.getByText("Broad 57-subject benchmark.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /read/i })).toBeInTheDocument();
  });
});
