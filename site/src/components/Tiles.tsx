"use client";
import { TILES, type Status } from "@/lib/format";

export type Fill = Status | "home" | "none";

// The state tile grid. A click, tap or keyboard focus selects a tile and it stays selected;
// hovering another tile previews it, and leaving the map goes back to the selection. What a
// tile shows is the caller's: it shows `hovered ?? selected`.
export function TileMap({
  fills,
  selected,
  onSelect,
  onHover,
  label,
  aria,
}: {
  fills: Record<string, Fill>;
  selected: string | null;
  onSelect: (code: string) => void;
  onHover: (code: string | null) => void;
  label: string;
  aria: (code: string, name: string, fill: Fill) => string;
}) {
  return (
    <div className="tiles" role="group" aria-label={label} onMouseLeave={() => onHover(null)}>
      {TILES.map(([code, name, col, row]) => {
        const fill = fills[code] ?? "none";
        return (
          <button
            key={code}
            type="button"
            className={`tile ${fill}`}
            style={{ gridColumn: col, gridRow: row }}
            aria-pressed={selected === code}
            aria-label={aria(code, name, fill)}
            onClick={() => { onSelect(code); onHover(null); }}
            onFocus={() => onSelect(code)}
            onMouseEnter={() => onHover(code)}
          >
            {code}
          </button>
        );
      })}
    </div>
  );
}

export function MiniMap({ fills }: { fills: Record<string, Fill> }) {
  return (
    <div className="tiles mini" aria-hidden="true">
      {TILES.map(([code, , col, row]) => (
        <span key={code} className={`tile ${fills[code] ?? "none"}`} style={{ gridColumn: col, gridRow: row }} />
      ))}
    </div>
  );
}
