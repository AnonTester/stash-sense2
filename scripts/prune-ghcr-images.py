#!/usr/bin/env python3
"""Keep only the newest N release images per GHCR package; delete the rest.

Why not "keep the last N package versions": every release pushes several package versions per image -- the tagged
image index plus untagged ones (the platform manifest and its provenance/SBOM attestations, which the index
references), and each build re-points the `buildcache` tag, orphaning the previous cache manifest. Counting versions
would keep ~1 release instead of N, and deleting an untagged version that a kept index still references breaks that
image.

What this does, per image:
  * a *release* is a package version carrying a full semver tag (0.47.1, 0.48.0-beta.1); the N newest are kept,
    older ones are deleted;
  * a version carrying `latest`, `beta` or `buildcache` is never deleted;
  * every digest referenced by a kept tagged version's index is protected (read from the registry);
  * untagged versions that nothing kept references -- the children of deleted releases, old build caches -- are
    deleted once they are older than --min-untagged-age-hours, so a build that is still running is left alone.

Dry run unless --delete is given. Needs `gh` authenticated (GH_TOKEN) with permission to list and delete package
versions. Standard library only.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

RELEASE_TAG = re.compile(r"^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$")
PINNED_TAGS = {"latest", "beta", "buildcache"}
INDEX_TYPES = {"application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json"}
ACCEPT = ", ".join(sorted(INDEX_TYPES) + [
    "application/vnd.oci.image.manifest.v1+json", "application/vnd.docker.distribution.manifest.v2+json"])


def gh_lines(args: list[str]) -> list[dict]:
    out = subprocess.run(["gh", "api", "--paginate", *args, "--jq", ".[]"], check=True, capture_output=True, text=True)
    return [json.loads(line) for line in out.stdout.splitlines() if line.strip()]


def registry_token(owner: str, name: str) -> str:
    url = f"https://ghcr.io/token?service=ghcr.io&scope=repository:{owner}/{name}:pull"
    req = urllib.request.Request(url)
    secret = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if secret:
        req.add_header("Authorization", "Basic " + base64.b64encode(f"{owner}:{secret}".encode()).decode())
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)["token"]


def referenced_digests(owner: str, name: str, token: str, digest: str) -> set[str]:
    """Digests an image index lists (platform manifests, attestations); empty for a plain manifest."""
    req = urllib.request.Request(f"https://ghcr.io/v2/{owner}/{name}/manifests/{digest}",
                                 headers={"Authorization": f"Bearer {token}", "Accept": ACCEPT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        manifest = json.load(resp)
    if manifest.get("mediaType") in INDEX_TYPES or "manifests" in manifest:
        return {m["digest"] for m in manifest.get("manifests", [])}
    return set()


def prune_image(owner: str, name: str, keep: int, apply: bool, min_untagged_age: timedelta) -> int:
    try:
        versions = gh_lines([f"/users/{owner}/packages/container/{name}/versions?per_page=100"])
    except subprocess.CalledProcessError as exc:
        if "404" in (exc.stderr or ""):
            print(f"{name}: no such package yet -- skipped")
            return 0
        print(f"{name}: cannot list versions: {(exc.stderr or '').strip()[:200]}")
        return 1
    for v in versions:
        v["tags"] = (v.get("metadata", {}).get("container", {}) or {}).get("tags") or []
        v["created"] = datetime.fromisoformat(v["created_at"].replace("Z", "+00:00"))

    def is_release(v: dict) -> bool:
        return any(RELEASE_TAG.match(t) for t in v["tags"]) and "buildcache" not in v["tags"]

    releases = sorted((v for v in versions if is_release(v)), key=lambda v: v["created"], reverse=True)
    dropped = [v for v in releases[keep:] if not (set(v["tags"]) & PINNED_TAGS)]
    dropped_ids = {v["id"] for v in dropped}
    kept_tagged = [v for v in versions if v["tags"] and v["id"] not in dropped_ids]

    # (API paths are case-insensitive; registry references are not -- hence the lower() below)
    token = registry_token(owner.lower(), name.lower())
    protected = {v["name"] for v in kept_tagged}
    for v in kept_tagged:
        try:
            protected |= referenced_digests(owner.lower(), name.lower(), token, v["name"])
        except (urllib.error.URLError, KeyError, ValueError) as exc:
            print(f"  ! cannot read the manifest of kept version {v['tags']} ({exc}) -- nothing is deleted for {name}")
            return 1

    now = datetime.now(timezone.utc)
    orphans = [v for v in versions if not v["tags"] and v["name"] not in protected
               and now - v["created"] >= min_untagged_age]
    to_delete = dropped + orphans
    print(f"{name}: {len(releases)} releases, keeping {min(keep, len(releases))}; "
          f"deleting {len(dropped)} old release(s) and {len(orphans)} unreferenced untagged version(s)")
    for v in dropped:
        print(f"  release  {','.join(v['tags'])}  {v['created']:%Y-%m-%d}  id={v['id']}")
    for v in orphans:
        print(f"  untagged {v['name'][:19]}  {v['created']:%Y-%m-%d %H:%M}  id={v['id']}")

    failures = 0
    if apply:
        for v in to_delete:
            res = subprocess.run(["gh", "api", "-X", "DELETE", f"/users/{owner}/packages/container/{name}/versions/{v['id']}"],
                                 capture_output=True, text=True)
            if res.returncode:
                failures += 1
                print(f"  ! delete {v['id']} failed: {res.stderr.strip()[:200]}")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--keep", type=int, default=5, help="releases to keep per image (default 5)")
    ap.add_argument("--min-untagged-age-hours", type=float, default=24)
    ap.add_argument("--delete", action="store_true", help="actually delete (default: dry run)")
    args = ap.parse_args()
    if args.keep < 1:
        ap.error("--keep must be at least 1")
    print(("DELETING" if args.delete else "DRY RUN (nothing is deleted)") + f" -- keep {args.keep} release(s) per image")
    rc = 0
    for image in args.images:
        rc |= prune_image(args.owner, image, args.keep, args.delete, timedelta(hours=args.min_untagged_age_hours))
    return rc


if __name__ == "__main__":
    sys.exit(main())
