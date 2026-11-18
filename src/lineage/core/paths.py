"""Where local data lives: `LINEAGE_DATA`, or `data` beside the checkout when it is unset.

Every default path to local data starts here, so git worktrees, which are fresh checkouts
without the gitignored data, can share one directory by setting `LINEAGE_DATA` to its absolute
path. It is read once, at import, as every default is.
"""

import os
from pathlib import Path

DATA = Path(os.environ.get("LINEAGE_DATA") or "data")
