import type { Metadata } from "next";
import { Suspense } from "react";
import { Footer, Header } from "@/components/Chrome";
import { Browse } from "@/components/Browse";
import { billList, meta } from "@/lib/data";
import { session, stateName } from "@/lib/format";

export const metadata: Metadata = { title: "Browse bills" };

export default function BillsPage() {
  const m = meta();
  const state = stateName(m.state);
  const bills = billList().bills;
  const sessions = [...new Set(bills.map((b) => b.bill.session))].sort();
  return (
    <>
      <Header current="bills" />
      <main className="wrap" style={{ paddingTop: 56 }}>
        <h1 className="display">{state} bills, 2023–2026</h1>
        <p className="lede">
          All {bills.length.toLocaleString("en-US")} bills from the{" "}
          {sessions.map(session).join(", ").replace(/, ([^,]*)$/, " and $1")} sessions, with the number of
          bills in other states that share at least one of its provisions. Open a bill to see which of
          its sections have relatives, and where.
        </p>
        <Suspense fallback={<p className="note">Loading the bill list…</p>}>
          <Browse sessions={sessions} total={bills.length} state={state} />
        </Suspense>
        <p className="note" style={{ marginTop: 20 }}>
          <strong style={{ color: "var(--ink)" }}>Related bills</strong> are bills in other states with
          at least one section that matches a section of the {state} bill, by exact wording or similar
          meaning. <strong style={{ color: "var(--ink)" }}>Closest match</strong> is the related bill with
          the highest similarity. Status is the bill&apos;s legislative status, not its current legal
          standing.
        </p>
      </main>
      <Footer />
    </>
  );
}
