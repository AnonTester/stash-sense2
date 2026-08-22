#!/usr/bin/env bash
# Validate that the sidecar's version files agree with each other (and,
# if given, with a release tag), and separately that the plugin's version
# files agree with each other.
#
# Sidecar and plugin versions are independent tracks -- the sidecar is
# what a git tag/GHCR image build is versioned against (plugin updates
# ship through AnonTester/stash-plugin-repo instead, with no git tag
# here), so a plugin-only release can legitimately sit ahead of the
# sidecar version without failing this check.
#
# Usage: ./scripts/check-version.sh [expected-sidecar-version]
# If expected-sidecar-version is given (e.g. from a git tag), also checks
# the sidecar pair against it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Extract versions from each file
main_ver=$(grep -oP 'version="\K[^"]+' "$ROOT/api/main.py")
settings_ver=$(grep -oP '_version: str = "\K[^"]+' "$ROOT/api/settings_router.py")
plugin_ver=$(grep -oP '^version: \K.+' "$ROOT/plugin/stash-sense.yml")
core_js_ver=$(grep -oP "const PLUGIN_VERSION = '\K[^']+" "$ROOT/plugin/stash-sense-core.js")

echo "Sidecar:"
echo "  api/main.py:                $main_ver"
echo "  api/settings_router.py:     $settings_ver"
echo "Plugin:"
echo "  plugin/stash-sense.yml:     $plugin_ver"
echo "  plugin/stash-sense-core.js: $core_js_ver"

errors=0

if [[ "$main_ver" != "$settings_ver" ]]; then
  echo "ERROR: api/main.py ($main_ver) != api/settings_router.py ($settings_ver)"
  errors=1
fi

if [[ "$plugin_ver" != "$core_js_ver" ]]; then
  echo "ERROR: plugin/stash-sense.yml ($plugin_ver) != plugin/stash-sense-core.js ($core_js_ver)"
  errors=1
fi

# If an expected version was passed (e.g. from a git tag), it's checked
# against the sidecar pair only -- tags/GHCR image builds are sidecar
# releases, not plugin releases.
if [[ "${1:-}" != "" ]]; then
  expected="$1"
  echo "Expected sidecar version: $expected"
  if [[ "$main_ver" != "$expected" ]]; then
    echo "ERROR: Sidecar files have $main_ver but expected $expected"
    errors=1
  fi
fi

if [[ $errors -eq 0 ]]; then
  echo "OK: sidecar ($main_ver) and plugin ($plugin_ver) version pairs are each internally consistent"
else
  exit 1
fi
