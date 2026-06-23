import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";

export type PdfRect = { x: number; y: number; w: number; h: number };
export type PdfHighlight = { id: number | string; page: number; rects: PdfRect[]; text: string };
export type PdfSelection = {
  page: number;
  rects: PdfRect[];
  text: string;
  anchor: { x: number; y: number };
};
export type PdfViewHandle = { scrollToHighlight: (page: number, normY: number) => void };

type PdfViewProps = {
  url: string;
  highlights?: PdfHighlight[];
  onSelect?: (selection: PdfSelection | null) => void;
};

/**
 * Renders a PDF inline with pdf.js: a canvas per page plus an invisible text layer
 * (selectable text → copy/paste) and a highlight overlay. `highlights` are drawn as
 * normalized rectangles; `onSelect` (passed only in the authed workspace) fires with
 * the selection geometry so the caller can offer to persist a highlight. The public
 * reader passes neither, so it gets copy/paste but no highlight wiring.
 */
export const PdfView = forwardRef<PdfViewHandle, PdfViewProps>(function PdfView(
  { url, highlights, onSelect },
  ref
) {
  const viewRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const wrapsRef = useRef<Map<number, HTMLDivElement>>(new Map());
  const highlightsRef = useRef<PdfHighlight[]>(highlights ?? []);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState("");

  const drawHighlights = () => {
    wrapsRef.current.forEach((wrap, page) => {
      const layer = wrap.querySelector<HTMLDivElement>(".pdf-hl-layer");
      if (!layer) {
        return;
      }
      layer.innerHTML = "";
      const w = wrap.clientWidth;
      const h = wrap.clientHeight;
      for (const hl of highlightsRef.current) {
        if (hl.page !== page) {
          continue;
        }
        for (const r of hl.rects) {
          const d = document.createElement("div");
          d.className = "pdf-hl";
          d.style.left = `${r.x * w}px`;
          d.style.top = `${r.y * h}px`;
          d.style.width = `${r.w * w}px`;
          d.style.height = `${r.h * h}px`;
          d.dataset.hid = String(hl.id);
          layer.appendChild(d);
        }
      }
    });
  };

  useImperativeHandle(ref, () => ({
    scrollToHighlight(page, normY) {
      const wrap = wrapsRef.current.get(page);
      const scroller = viewRef.current;
      if (!wrap || !scroller) {
        return;
      }
      const wrapRect = wrap.getBoundingClientRect();
      const scRect = scroller.getBoundingClientRect();
      scroller.scrollTop += wrapRect.top - scRect.top + normY * wrapRect.height - 60;
    },
  }));

  useEffect(() => {
    let cancelled = false;
    const root = containerRef.current;
    if (root) {
      root.innerHTML = "";
    }
    wrapsRef.current = new Map();
    setStatus("loading");
    setError("");

    (async () => {
      try {
        const pdfjs = await import("pdfjs-dist");
        const workerUrl = (await import("pdfjs-dist/build/pdf.worker.min.mjs?url")).default;
        pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;

        const pdf = await pdfjs.getDocument({ url, withCredentials: true }).promise;
        if (cancelled) {
          return;
        }
        const target = containerRef.current;
        if (!target) {
          return;
        }
        const dpr = window.devicePixelRatio || 1;
        const cssWidth = Math.max(280, (target.clientWidth || 800) - 4);

        for (let n = 1; n <= pdf.numPages; n += 1) {
          if (cancelled) {
            return;
          }
          const page = await pdf.getPage(n);
          const baseVp = page.getViewport({ scale: 1 });
          const scale = cssWidth / baseVp.width;
          const cssViewport = page.getViewport({ scale });
          const canvasViewport = page.getViewport({ scale: scale * dpr });
          const pageW = Math.floor(cssViewport.width);
          const pageH = Math.floor(cssViewport.height);

          const wrap = document.createElement("div");
          wrap.className = "pdf-page-wrap";
          wrap.style.width = `${pageW}px`;
          wrap.style.height = `${pageH}px`;
          wrap.dataset.page = String(n);
          target.appendChild(wrap);
          wrapsRef.current.set(n, wrap);

          const canvas = document.createElement("canvas");
          canvas.className = "pdf-page";
          canvas.width = Math.floor(canvasViewport.width);
          canvas.height = Math.floor(canvasViewport.height);
          canvas.style.width = `${pageW}px`;
          canvas.style.height = `${pageH}px`;
          wrap.appendChild(canvas);
          const ctx = canvas.getContext("2d");
          if (ctx) {
            await page.render({ canvasContext: ctx, viewport: canvasViewport, canvas }).promise;
          }

          const hlLayer = document.createElement("div");
          hlLayer.className = "pdf-hl-layer";
          wrap.appendChild(hlLayer);

          const textLayerDiv = document.createElement("div");
          textLayerDiv.className = "textLayer";
          textLayerDiv.style.setProperty("--scale-factor", String(scale));
          textLayerDiv.style.setProperty("--total-scale-factor", String(scale));
          textLayerDiv.style.width = `${pageW}px`;
          textLayerDiv.style.height = `${pageH}px`;
          wrap.appendChild(textLayerDiv);
          const textLayer = new pdfjs.TextLayer({
            textContentSource: await page.getTextContent(),
            container: textLayerDiv,
            viewport: cssViewport,
          });
          await textLayer.render();

          drawHighlights();
          if (n === 1 && !cancelled) {
            setStatus("ready");
          }
        }
        if (!cancelled) {
          setStatus("ready");
          drawHighlights();
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "PDF failed to render");
          setStatus("error");
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [url]);

  useEffect(() => {
    highlightsRef.current = highlights ?? [];
    drawHighlights();
  }, [highlights]);

  const handleMouseUp = () => {
    if (!onSelect) {
      return;
    }
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || sel.rangeCount === 0) {
      onSelect(null);
      return;
    }
    const range = sel.getRangeAt(0);
    const startNode = range.startContainer;
    const startEl =
      startNode.nodeType === Node.TEXT_NODE
        ? startNode.parentElement
        : (startNode as Element);
    const wrap = startEl?.closest<HTMLElement>(".pdf-page-wrap");
    if (!wrap) {
      onSelect(null);
      return;
    }
    const page = Number(wrap.dataset.page);
    const wr = wrap.getBoundingClientRect();
    const clientRects = Array.from(range.getClientRects()).filter(
      (r) => r.width > 1 && r.height > 1
    );
    const rects = clientRects
      .map((r) => ({
        x: (r.left - wr.left) / wr.width,
        y: (r.top - wr.top) / wr.height,
        w: r.width / wr.width,
        h: r.height / wr.height,
      }))
      .filter((r) => r.x >= -0.02 && r.x <= 1.02 && r.y >= -0.02 && r.y <= 1.02);
    const text = sel.toString().trim();
    if (rects.length === 0 || !text) {
      onSelect(null);
      return;
    }
    const last = clientRects[clientRects.length - 1];
    onSelect({ page, rects, text, anchor: { x: last.right, y: last.bottom } });
  };

  return (
    <div className="pdf-view" ref={viewRef} onMouseUp={handleMouseUp}>
      {status === "loading" && <p className="pdf-status">Loading PDF…</p>}
      {status === "error" && (
        <p className="pdf-status">
          Couldn’t render the PDF inline ({error}). Use “Open PDF” above.
        </p>
      )}
      <div className="pdf-canvases" ref={containerRef} />
    </div>
  );
});
