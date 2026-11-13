"use client";
import { TILES, type Status } from "@/lib/format";

export type Fill = Status | "home" | "none";

// The state tile grid. Selecting a tile is the map's only interaction; what it shows is the
// caller's.
export function TileMap({
  fills,
  selected,
  onSelect,
  label,
  aria,
}: {
  fills: Record<string, Fill>;
  selected: string | null;
  onSelect: (code: string) => void;
  label: string;
  aria: (code: string, name: string, fill: Fill) => string;
}) {
  return (
    <div className="tiles" role="group" aria-label={label}>
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
            onClick={() => onSelect(code)}
            onFocus={() => onSelect(code)}
            onMouseEnter={() => onSelect(code)}
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
