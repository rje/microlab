import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "../src/App";
import type { MicrolabState } from "../src/state";

const state: MicrolabState = {
  phases: [
    { id: "phase-0", title: "Phase 0", status: "current", goal: "g", summary: "s", tasks: [], readingPaperIds: [] }
  ],
  papers: [],
  synopses: {},
  evalRuns: [],
  csrfToken: "tok"
};

describe("Flashcard recall", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("shows due count and runs a review", async () => {
    const fetchMock = vi.fn(async (url: string) => {
      if (String(url).includes("/api/recall/due")) {
        return {
          ok: true,
          json: async () => ({
            total: 1,
            cards: [{ id: "mmlu#1", paperId: "mmlu", paperTitle: "MMLU", question: "Why MMLU?", answer: "Breadth.", status: "new" }]
          })
        };
      }
      return { ok: true, json: async () => ({}), text: async () => "" };
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App initialState={state} />);
    expect(await screen.findByText("cards due")).toBeInTheDocument();
    fireEvent.click(await screen.findByRole("button", { name: /start review/i }));
    fireEvent.click(await screen.findByRole("button", { name: /show answer/i }));
    expect(screen.getByText("Breadth.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /^good$/i }));
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/recall/review",
        expect.objectContaining({ method: "POST", headers: expect.objectContaining({ "X-CSRF-Token": "tok" }) })
      );
    });
  });
});
