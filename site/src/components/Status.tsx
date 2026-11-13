import { STATUS_LABEL, type Status } from "@/lib/format";

const ICON: Partial<Record<Status, string>> = { passed: "M3 8.5l3.2 3L13 4.5", died: "M4 4l8 8M12 4l-8 8" };

export function StatusBadge({ status }: { status: Status | null }) {
  if (status === null) return <span className="small">Status not recorded</span>;
  const d = ICON[status];
  return (
    <span className={`status ${status}`}>
      {d && (
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d={d} />
        </svg>
      )}
      {STATUS_LABEL[status]}
    </span>
  );
}

// The line every map and list carries (DESIGN.md#known-failure-modes).
export const STANDING =
  "Status is the bill's legislative status, not its current legal standing.";
