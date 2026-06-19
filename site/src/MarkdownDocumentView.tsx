import { ExternalLink, FileText, X } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import type { MarkdownDocument } from "./state";

type MarkdownDocumentViewProps = {
  document: MarkdownDocument | null;
  error: string | null;
  loadingPath: string | null;
  onClose: () => void;
};

function rawMarkdownHref(path: string) {
  if (path.startsWith("runs/evals/")) {
    return `/artifacts/${path}`;
  }
  return `/${path}`;
}

export function MarkdownDocumentView({
  document,
  error,
  loadingPath,
  onClose
}: MarkdownDocumentViewProps) {
  if (!document && !error && !loadingPath) {
    return null;
  }

  const path = document?.path ?? loadingPath ?? "";

  return (
    <div className="document-overlay" role="dialog" aria-modal="true" aria-label="Markdown document">
      <div className="document-shell">
        <header className="document-toolbar">
          <div>
            <p className="eyebrow">Project document</p>
            <p className="document-title">{document?.title ?? "Loading document"}</p>
            {path && <span>{path}</span>}
          </div>
          <div className="document-actions">
            {document && (
              <a href={rawMarkdownHref(document.path)}>
                <ExternalLink aria-hidden="true" />
                Raw markdown
              </a>
            )}
            <button type="button" onClick={onClose} aria-label="Close document">
              <X aria-hidden="true" />
            </button>
          </div>
        </header>

        <article className="markdown-document">
          {loadingPath && <p>Loading markdown document...</p>}
          {error && (
            <div className="document-error">
              <strong>Could not load document</strong>
              <p>{error}</p>
            </div>
          )}
          {document && (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{document.content}</ReactMarkdown>
          )}
        </article>
      </div>
    </div>
  );
}
