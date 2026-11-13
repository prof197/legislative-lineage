// Each featured story's copy. The anti-SLAPP and deepfake stories' lines are the mocks'; the
// Georgia story had none in the mocks and is written plainly from the export's facts. The
// anti-SLAPP story compares each enactment with Utah SB 18, the earliest in the index, not with
// the act, whose text is never shown; it links to the act on the commission's site instead.

// The act's page on the commission's site, from its catalog of current acts.
export const ACT_PAGE =
  "https://www.uniformlaws.org/committees/community-home?CommunityKey=4f486460-199c-49d7-9fac-05570be1e7b1";
export type StoryCopy = {
  eyebrow: string;
  card: string;
  title: string;
  headline: string;
  pageEyebrow: string;
  actPage: string | null; // the act on its drafter's site, where a story has one
  listNoun: string; // what each comparison sets the bill beside
  listHeading: string; // "{n} {listHeading}"
  compareLabel: string;
};

export const STORY_ORDER = ["anti-slapp", "deepfakes", "georgia-social-media"] as const;

export const STORY_COPY: Record<string, StoryCopy> = {
  "anti-slapp": {
    eyebrow: "Uniform act · Anti-SLAPP",
    card: "The Uniform Public Expression Protection Act",
    title: "Uniform Public Expression Protection Act",
    headline: "A model anti-SLAPP law, and the state bills that share its language",
    pageEyebrow: "Featured story · Uniform act",
    actPage: ACT_PAGE,
    listNoun: "Utah SB 18",
    listHeading: "bills the Uniform Law Commission lists as enacting the act",
    compareLabel: "Compare with Utah SB 18",
  },
  deepfakes: {
    eyebrow: "Elections · Synthetic media",
    card: "Election deepfake disclosure laws",
    title: "Election deepfake disclosure laws",
    headline: "Election deepfake disclosure laws, and the state bills that share their language",
    pageEyebrow: "Featured story · Elections",
    actPage: null,
    listNoun: "Washington SB 5152",
    listHeading: "bills that share language with Washington SB 5152",
    compareLabel: "Compare with Washington SB 5152",
  },
  "georgia-social-media": {
    eyebrow: "Georgia",
    card: "Protecting Georgia's Children on Social Media Act",
    title: "Georgia SB 351",
    headline: "Georgia SB 351's social-media provisions, and the bills in other states that share their language",
    pageEyebrow: "Featured story · Georgia",
    actPage: null,
    listNoun: "SB 351",
    listHeading: "bills that share language with SB 351",
    compareLabel: "Compare with Georgia SB 351",
  },
};
