import type { Metadata } from "next";
import { Suspense } from "react";
import { Footer, Header, REPOSITORY } from "@/components/Chrome";
import { CompareView } from "@/components/CompareView";
import { meta } from "@/lib/data";

export const metadata: Metadata = { title: "Comparison" };

// One page serves every comparison, named in its query (`?c=`): the comparison and the two
// bills' texts are fetched in the browser, so a section's text is downloaded once however many
// comparisons show it (DESIGN.md#export). The page fills the window: the rows scroll beneath a
// fixed title and above the legend, as in the 3b mock.
export default function ComparePage() {
  return (
    <div className="cmp-shell">
      <Header current="bills" />
      <Suspense fallback={<main className="cmp-main"><p className="note cmp-wrap">Loading the comparison…</p></main>}>
        <CompareView hosted={meta().state} issues={`${REPOSITORY}/issues`} footer={<Footer />} />
      </Suspense>
    </div>
  );
}
