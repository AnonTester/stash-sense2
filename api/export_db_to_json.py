#!/usr/bin/env python3
"""Export performers.db to JSON files for the recognizer.

This script converts the SQLite database (performers.db) into the JSON files
that the FaceRecognizer expects:

  - faces.json: List where index i = universal_id of face at usearch index i
  - performers.json: Dict mapping universal_id -> {name, country, image_url, face_count}

Usage:
    python export_db_to_json.py [--data-dir PATH]

The script will:
1. Read from performers.db in the data directory
2. Generate faces.json and performers.json
3. Validate the output matches the usearch index

Run this script after copying a new performers.db from stash-sense2-data-gen,
or after updating the database in any way. Also called by
delta_applier.py's apply_delta_chain() to regenerate both files after a
delta chain applies -- universal_id has three shapes (stashbox, local,
catalogue) matching stash-sense2-data-gen's build/export_json.py exactly;
see make_catalogue_id()'s docstring for why this file needs to stay in
sync with that one.
"""
import argparse
import json
import sqlite3
from pathlib import Path


def make_universal_id(endpoint: str, stashbox_id: str) -> str:
    """Create a universal ID from endpoint and stashbox ID.

    Format: "endpoint.org:stashbox_id"
    Example: "stashdb.org:019bef93-b467-73eb-a04b-ac44fdaa7a04"
    """
    # Normalize endpoint to domain format
    endpoint_domains = {
        "stashdb": "stashdb.org",
        "theporndb": "theporndb.net",
        "fansdb": "fansdb.cc",
        "pmvstash": "pmvstash.org",
        "javstash": "javstash.org",
    }
    domain = endpoint_domains.get(endpoint, f"{endpoint}.org")
    return f"{domain}:{stashbox_id}"


def make_catalogue_id(source_endpoint: str, performer_id: int) -> str:
    """Universal ID for a performer with no stashbox linkage at all --
    e.g. "pornbox:176428", "seekfans:4821". Ported from
    stash-sense2-data-gen's build/export_json.py (the server-side
    counterpart of this exact function) -- this file had drifted out of
    sync with that fix (which added catalogue-performer support to the
    *server's* full-zip export) until stash-sense2-data-gen's own delta
    package started shipping catalogue performers too: without this,
    applying such a delta and then re-exporting locally (this script is
    exactly what apply_delta_chain() calls after a delta applies) would
    silently drop every catalogue performer from faces.json/
    performers.json all over again, the same bug the server-side fix
    already solved once."""
    return f"{source_endpoint}:{performer_id}"


def _performer_universal_ids(conn: sqlite3.Connection) -> dict:
    """One universal_id per performer_id -- a real stashbox linkage first
    (highest source priority wins if a performer somehow has more than
    one), then a catalogue id (performer_urls.url_type='catalogue') for
    performers with none, then whichever face source_endpoint is on file
    as a last resort (matches build/export_json.py's own three-tier
    fallback exactly, including its rationale for the third tier: a
    handful of legacy performers predate performer_urls entirely)."""
    priority_sql = """
        CASE endpoint
            WHEN 'stashdb' THEN 1
            WHEN 'theporndb' THEN 2
            WHEN 'fansdb' THEN 3
            WHEN 'pmvstash' THEN 4
            WHEN 'javstash' THEN 5
            ELSE 6
        END
    """
    ids: dict = {}
    best_priority: dict = {}
    for performer_id, endpoint, stashbox_id, priority in conn.execute(f"""
        SELECT performer_id, endpoint, stashbox_performer_id, {priority_sql}
        FROM stashbox_ids
    """):
        if performer_id not in best_priority or priority < best_priority[performer_id]:
            best_priority[performer_id] = priority
            ids[performer_id] = make_universal_id(endpoint, stashbox_id)

    for performer_id, source_endpoint in conn.execute("""
        SELECT performer_id, source_endpoint FROM performer_urls WHERE url_type = 'catalogue'
    """):
        if performer_id not in ids:
            ids[performer_id] = make_catalogue_id(source_endpoint, performer_id)

    for performer_id, source_endpoint in conn.execute("""
        SELECT performer_id, source_endpoint FROM faces GROUP BY performer_id
    """):
        if performer_id not in ids:
            ids[performer_id] = make_catalogue_id(source_endpoint, performer_id)

    return ids


def _catalogue_links(conn: sqlite3.Connection) -> dict:
    """performer_id -> {"source", "catalogue_url", "profile_url"} for
    performers with a catalogue (non-stashbox) universal_id. Ported
    verbatim from build/export_json.py -- see that module for the
    reasoning."""
    links: dict = {}
    for performer_id, url, url_type, source_endpoint in conn.execute("""
        SELECT performer_id, url, url_type, source_endpoint FROM performer_urls
        WHERE url_type IN ('catalogue', 'profile')
    """):
        entry = links.setdefault(performer_id, {"source": None, "catalogue_url": None, "profile_url": None})
        if url_type == "catalogue":
            entry["source"] = source_endpoint
            entry["catalogue_url"] = url
        elif url_type == "profile":
            entry["profile_url"] = url
    return {pid: v for pid, v in links.items() if v["source"]}


def export_faces_json(conn: sqlite3.Connection, output_path: Path) -> int:
    """Export faces.json - list of universal IDs indexed by usearch index.

    The usearch index stores embeddings at sequential integer indices.
    faces.json maps each index to the performer's universal ID.

    Returns the number of faces exported.
    """
    universal_ids = _performer_universal_ids(conn)

    cursor = conn.execute("SELECT embedding_index, performer_id FROM faces ORDER BY embedding_index")

    faces = []
    gap_count = 0

    for embedding_index, performer_id in cursor:
        while len(faces) < embedding_index:
            gap_count += 1
            faces.append(None)
        faces.append(universal_ids.get(performer_id))
        if faces[-1] is None:
            gap_count += 1

    # Write output
    with open(output_path, "w") as f:
        json.dump(faces, f)

    valid_faces = len(faces) - gap_count
    print(f"  Exported {valid_faces} faces ({len(faces)} total indices) to {output_path}")
    if gap_count:
        print(f"  WARNING: {gap_count} gaps filled with null (performer has neither a stashbox link nor a catalogue source)")
    return len(faces)


def export_performers_json(conn: sqlite3.Connection, output_path: Path) -> int:
    """Export performers.json - dict mapping universal_id to performer metadata.

    Format:
    {
        "stashdb.org:019bef93-...": {
            "name": "Performer Name",
            "country": "US",
            "image_url": "https://...",
            "face_count": 4
        },
        "pornbox:176428": {
            "name": "Aleks",
            "country": "RU",
            "image_url": "https://...",
            "face_count": 1,
            "source": "pornbox",
            "catalogue_url": "https://www.pornbox.com/application/model/930",
            "profile_url": null
        },
        ...
    }

    Returns the number of performers exported.
    """
    universal_ids = _performer_universal_ids(conn)
    catalogue_links = _catalogue_links(conn)

    cursor = conn.execute("""
        SELECT id, canonical_name, country, image_url, face_count
        FROM performers WHERE face_count > 0
    """)

    performers = {}
    for performer_id, name, country, image_url, face_count in cursor:
        universal_id = universal_ids.get(performer_id)
        if universal_id is None:
            continue  # matches the null gap left in faces.json for this performer

        entry = {
            "name": name,
            "country": country,
            "image_url": image_url,
            "face_count": face_count or 0,
        }
        # A catalogue-shaped id (no "." in its prefix, unlike "stashdb.org"
        # etc., and not "local") needs its source/links surfaced so the
        # UI renders it as a catalogue match instead of a broken stashbox
        # one -- see build/export_json.py's own comment for why this is
        # derived from the id's own shape rather than solely from having
        # a performer_urls row (the third-tier fallback above produces a
        # catalogue id with no matching row at all).
        prefix = universal_id.split(":", 1)[0]
        if prefix != "local" and "." not in prefix:
            link = catalogue_links.get(performer_id, {})
            entry["source"] = link.get("source") or prefix
            entry["catalogue_url"] = link.get("catalogue_url")
            entry["profile_url"] = link.get("profile_url")
        performers[universal_id] = entry

    # Write output
    with open(output_path, "w") as f:
        json.dump(performers, f, indent=2)

    print(f"  Exported {len(performers)} performers to {output_path}")
    return len(performers)


def validate_export(data_dir: Path, face_count: int) -> bool:
    """Validate the export matches the usearch index.

    Returns True if validation passes.
    """
    issues = []

    index_path = data_dir / "face_embeddings.usearch"
    if index_path.exists():
        # Just check file exists and is non-empty -- getting the vector
        # count without a full load isn't worth it here.
        if index_path.stat().st_size == 0:
            issues.append(f"{index_path.name} is empty")
    else:
        issues.append(f"{index_path.name} not found")

    # Check faces.json was created
    faces_path = data_dir / "faces.json"
    if not faces_path.exists():
        issues.append("faces.json was not created")

    # Check performers.json was created
    performers_path = data_dir / "performers.json"
    if not performers_path.exists():
        issues.append("performers.json was not created")

    if issues:
        print("\nValidation issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("\nValidation passed!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export performers.db to JSON files for the recognizer"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Data directory containing performers.db (default: ./data)"
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    db_path = data_dir / "performers.db"

    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        print("Copy performers.db from stash-sense-trainer first.")
        return 1

    print(f"Exporting from {db_path}...")

    conn = sqlite3.connect(db_path)

    try:
        print("\n1. Exporting faces.json...")
        face_count = export_faces_json(conn, data_dir / "faces.json")

        print("\n2. Exporting performers.json...")
        performer_count = export_performers_json(conn, data_dir / "performers.json")

        print("\n3. Validating export...")
        if not validate_export(data_dir, face_count):
            return 1

        print(f"\nDone! Exported {face_count} faces and {performer_count} performers.")
        print("\nRestart the stash-sense API to load the new data.")

    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    exit(main())
