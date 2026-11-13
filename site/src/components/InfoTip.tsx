"use client";
import { useEffect, useId, useRef, useState } from "react";

// A muted info icon beside a heading whose note shows on hover, focus or tap. The note is the
// button's description, so a screen reader reads it with the button. It is placed under the icon
// and kept inside the window, wherever the icon sits, and closes when the page scrolls.
export function InfoTip({ label, children }: { label: string; children: React.ReactNode }) {
  const id = useId();
  const button = useRef<HTMLButtonElement>(null);
  const [at, setAt] = useState<{ left: number; top: number; width: number } | null>(null);
  const open = () => {
    const r = button.current?.getBoundingClientRect();
    if (!r) return;
    const width = Math.min(280, window.innerWidth - 32);
    const left = Math.max(16, Math.min(r.left + r.width / 2 - width / 2, window.innerWidth - 16 - width));
    setAt({ left, top: r.bottom + 6, width });
  };
  const close = () => setAt(null);
  useEffect(() => {
    if (!at) return;
    window.addEventListener("scroll", close, { passive: true, capture: true });
    return () => window.removeEventListener("scroll", close, { capture: true });
  }, [at]);
  return (
    <span className="infotip" onMouseEnter={open} onMouseLeave={close}>
      <button
        ref={button}
        type="button"
        className="infotip-btn"
        aria-label={label}
        aria-describedby={id}
        aria-expanded={at !== null}
        onClick={() => (at ? close() : open())}
        onFocus={open}
        onBlur={close}
        onKeyDown={(e) => e.key === "Escape" && close()}
      >
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" aria-hidden="true">
          <circle cx="8" cy="8" r="6.5" /><path d="M8 7.2v4M8 4.8v.01" />
        </svg>
      </button>
      <span role="tooltip" id={id} className="infotip-body" hidden={!at} style={at ? { left: at.left, top: at.top, width: at.width } : undefined}>
        {children}
      </span>
    </span>
  );
}
