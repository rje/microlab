import "@testing-library/jest-dom/vitest";

// jsdom doesn't implement window.scrollTo — the public reader calls it on navigation,
// and jsdom's async "Not implemented" throw can intermittently fail the vitest process
// even though every test passes. Stub it to keep the guardrail deterministic.
window.scrollTo = () => {};
