import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { App } from "../src/App";
import type { MicrolabState } from "../src/state";

const state: MicrolabState = {
  phases: [
    {
      id: "phase-0",
      title: "Phase 0: Evaluation Harness",
      status: "current",
      goal: "Build reproducible evaluation before training.",
      summary: "Create the measurement layer for the lab.",
      tasks: [
        {
          id: "environment",
          title: "Environment and repo setup",
          status: "done",
          why: "Every experiment needs a stable base.",
          links: ["plans/environment-setup.md"]
        },
        {
          id: "eval-schema",
          title: "Evaluation schema and suite loader",
          status: "queued",
          why: "Stable task records make comparison possible.",
          links: ["plans/phase-0.md"]
        }
      ],
      readingPaperIds: ["mmlu"]
    }
  ],
  papers: [
    {
      id: "mmlu",
      topic: "evaluation",
      title: "Measuring Massive Multitask Language Understanding",
      authors: "Hendrycks et al.",
      year: 2020,
      sourceUrl: "https://arxiv.org/abs/2009.03300",
      pdfUrl: "/papers/evaluation/mmlu.pdf",
      filename: "mmlu.pdf"
    }
  ],
  synopses: {
    mmlu: {
      paperId: "mmlu",
      oneSentence: "MMLU measures broad multitask knowledge.",
      coreIdeas: ["Breadth matters.", "Aggregate scores need context."],
      whyItMatters: "It is the canonical broad knowledge eval.",
      phaseConnection: "It informs category-level Phase 0 reports.",
      suggestedReadingFocus: ["Subject construction", "Prompt settings"]
    }
  },
  evalRuns: []
};

describe("App", () => {
  it("renders the phase dashboard with work, reading, and result areas", () => {
    render(<App initialState={state} />);

    expect(screen.getByRole("heading", { name: "Phase 0: Evaluation Harness" })).toBeInTheDocument();
    expect(screen.getByText("Environment and repo setup")).toBeInTheDocument();
    expect(screen.getByText("Measuring Massive Multitask Language Understanding")).toBeInTheDocument();
    expect(screen.getByText("Awaiting first eval run")).toBeInTheDocument();
  });
});
