"""One-time repair: extend earlier duplicate-scene dismissals to the pairs they never covered.

Dismissing duplicate scenes used to record only the scene pairs that were on screen. A pair that scored
just under the threshold had no recommendation and so no dismissal, and re-appeared as "new" after a
database update re-matched performers and lifted its score. Group dismissals now cover every pair among
the scenes they connect (see RecommendationsDB.dismiss_duplicate_scene_group); this applies the same
rule to dismissals made before that change: two scenes that were each dismissed against the same third
scene were reviewed together, so the pair between them is dismissed too.

    python backfill_duplicate_scene_dismissals.py            # dry run: counts only
    python backfill_duplicate_scene_dismissals.py --apply    # write (back up stash_sense.db first)

DATA_DIR selects the data directory (default /data); the database is stash_sense.db inside it.
"""
import argparse
import json
import os
from pathlib import Path

from recommendations_db import RecommendationsDB


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--apply", action="store_true", help="write the dismissals (default: dry run)")
    args = parser.parse_args()
    db_path = Path(os.environ.get("DATA_DIR", "/data")) / "stash_sense.db"
    result = RecommendationsDB(db_path).backfill_duplicate_scene_dismissals(dry_run=not args.apply)
    print(json.dumps({"database": str(db_path), **result}, indent=2))


if __name__ == "__main__":
    main()
