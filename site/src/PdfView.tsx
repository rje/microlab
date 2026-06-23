import { useEffect, useRef, useState } from "react";

/**
 * Renders a PDF inline by drawing each page to a canvas with pdf.js, plus an
 * invisible pdf.js text layer over each page so text is selectable (copy/paste)
 * and can later be highlighted. Unlike an <iframe>, this works on mobile browsers
 * (which usually refuse to render PDFs inline). pdf.js is loaded lazily.
 *
 * The `.textLayer` CSS this relies on lives in BOTH styles.css (authed app) and
 * public.css (the isolated public bundle) — the public bundle doesn't import
 * styles.css, so it needs its own copy.
 */
export function PdfView({ url }: { url: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    const root = containerRef.current;
    if (root) {
      root.innerHTML = "";
    }
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
          const base = page.getViewport({ scale: 1 });
          const scale = cssWidth / base.width;
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

          if (n === 1 && !cancelled) {
            setStatus("ready");
          }
        }
        if (!cancelled) {
          setStatus("ready");
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

  return (
    <div className="pdf-view">
      {status === "loading" && <p className="pdf-status">Loading PDF…</p>}
      {status === "error" && (
        <p className="pdf-status">
          Couldn’t render the PDF inline ({error}). Use “Open PDF” above.
        </p>
      )}
      <div className="pdf-canvases" ref={containerRef} />
    </div>
  );
}
