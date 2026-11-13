// Each featured story's copy. The anti-SLAPP and deepfake stories' lines are the mocks'; the
// Georgia story had none in the mocks and is written plainly from the export's facts.
export type StoryCopy = {
  eyebrow: string;
  card: string;
  title: string;
  headline: string;
  pageEyebrow: string;
  origin: string | null;
  listNoun: string; // "{n} bills that share language with {listNoun}"
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
    origin: "Uniform Law Commission, Public Expression Protection Act (2020)",
    listNoun: "the act",
    compareLabel: "Compare with the model act",
  },
  deepfakes: {
    eyebrow: "Elections · Synthetic media",
    card: "Election deepfake disclosure laws",
    title: "Election deepfake disclosure laws",
    headline: "Election deepfake disclosure laws, and the state bills that share their language",
    pageEyebrow: "Featured story · Elections",
    origin: null,
    listNoun: "Washington SB 5152",
    compareLabel: "Compare with Washington SB 5152",
  },
  "georgia-social-media": {
    eyebrow: "Georgia",
    card: "Protecting Georgia's Children on Social Media Act",
    title: "Georgia SB 351",
    headline: "Georgia SB 351's social-media provisions, and the bills in other states that share their language",
    pageEyebrow: "Featured story · Georgia",
    origin: null,
    listNoun: "SB 351",
    compareLabel: "Compare with Georgia SB 351",
  },
};
