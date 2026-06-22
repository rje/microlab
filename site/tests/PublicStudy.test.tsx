import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { PublicApp } from "../src/PublicApp";

const library = {
  phases: [
    {
      id: "phase-0",
      title: "Phase 0",
      papers: [
        {
          id: "mmlu",
          title: "MMLU",
          authors: "H",
          year: 2020,
          topic: "evaluation",
          sourceUrl: "https://arxiv.org/abs/2009.03300",
          pdfUrl: "/public/pdf/mmlu",
          overview: { paperId: "mmlu", tldr: "t", sections: [], readingFocus: [] }
        }
      ]
    }
  ],
  additional: []
};

describe("Public study mode", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("flips through flashcards with no grading", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      if (String(url).includes("/cards")) {
        return { ok: true, json: async () => ({ cards: [
          { id: "mmlu#1", question: "Why MMLU?", answer: "Breadth." },
          { id: "mmlu#2", question: "How scored?", answer: "Letter probs." }
        ] }) };
      }
      return { ok: true, json: async () => library };
    }));

    render(<PublicApp />);
    fireEvent.click(await screen.findByRole("button", { name: /read/i }));
    fireEvent.click(await screen.findByRole("button", { name: /flashcards/i }));
    expect(await screen.findByText("Why MMLU?")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /show answer/i }));
    expect(screen.getByText("Breadth.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /next/i }));
    expect(await screen.findByText("How scored?")).toBeInTheDocument();
    // no grading controls in public study mode
    expect(screen.queryByRole("button", { name: /^good$/i })).toBeNull();
  });
});
