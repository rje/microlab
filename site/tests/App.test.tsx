import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

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
      summary:
        "MMLU is a broad multiple-choice benchmark that makes uneven model knowledge visible across subjects.",
      coreIdeas: ["Breadth matters.", "Aggregate scores need context."],
      whyItMatters: "It is the canonical broad knowledge eval.",
      phaseConnection: "It informs category-level Phase 0 reports.",
      suggestedReadingFocus: ["Subject construction", "Prompt settings"]
    }
  },
  evalRuns: [],
  csrfToken: "tok"
};

describe("App", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders the phase dashboard with work, reading, and result areas", () => {
    render(<App initialState={state} />);

    expect(screen.getByRole("heading", { name: "Phase 0: Evaluation Harness" })).toBeInTheDocument();
    expect(screen.getByText("Environment and repo setup")).toBeInTheDocument();
    expect(screen.getByText("Measuring Massive Multitask Language Understanding")).toBeInTheDocument();
    expect(
      screen.getByText(/MMLU is a broad multiple-choice benchmark/i)
    ).toBeInTheDocument();
    expect(screen.getByText("Breadth matters.")).toBeInTheDocument();
    expect(screen.getByText("It is the canonical broad knowledge eval.")).toBeInTheDocument();
    expect(screen.getByText("Awaiting first eval run")).toBeInTheDocument();
  });

  it("opens markdown links in a rendered document viewer", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        path: "plans/environment-setup.md",
        title: "Environment Setup",
        content: "# Environment Setup\n\n- Use the `microlab` conda environment.\n\n```bash\nconda run -n microlab pytest\n```"
      })
    }));
    vi.stubGlobal("fetch", fetchMock);

    render(<App initialState={state} />);
    fireEvent.click(screen.getByRole("link", { name: /environment-setup\.md/i }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/markdown?path=plans%2Fenvironment-setup.md"
      );
    });
    expect(
      await screen.findByRole("heading", { name: "Environment Setup" })
    ).toBeInTheDocument();
    expect(screen.getByText(/Use the.*conda environment/i)).toBeInTheDocument();
    expect(screen.getByText("microlab")).toBeInTheDocument();
    expect(screen.getByText("conda run -n microlab pytest")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /raw markdown/i })).toHaveAttribute(
      "href",
      "/plans/environment-setup.md"
    );
  });

  it("shows a read-state selector and saves on change", async () => {
    const fetchMock = vi.fn(async () => ({ ok: true, json: async () => ({}), text: async () => "" }));
    vi.stubGlobal("fetch", fetchMock);

    render(<App initialState={state} />);
    const select = screen.getByLabelText(/reading state for/i) as HTMLSelectElement;
    fireEvent.change(select, { target: { value: "mapped" } });

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/papers/mmlu/progress",
        expect.objectContaining({
          method: "POST",
          headers: expect.objectContaining({ "X-CSRF-Token": "tok" })
        })
      );
    });
  });

  it("loads existing notes into the editor when opened", async () => {
    const fetchMock = vi.fn(async (url: string) =>
      String(url).includes("/notes")
        ? {
            ok: true,
            json: async () => ({ paperId: "mmlu", content: "loaded note text" }),
            text: async () => ""
          }
        : { ok: true, json: async () => ({}), text: async () => "" }
    );
    vi.stubGlobal("fetch", fetchMock);

    render(<App initialState={state} />);
    fireEvent.click(screen.getByRole("button", { name: /^notes$/i }));
    expect(await screen.findByDisplayValue("loaded note text")).toBeInTheDocument();
  });
});
