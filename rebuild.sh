#!/bin/sh
# Rebuilds the sidecar image defined by the local (git-ignored)
# docker-stashsense2.yml -- which Dockerfile it builds (Dockerfile.cuda for an
# NVIDIA GPU, Dockerfile.rocm for AMD, or the CPU variant) is set there -- and
# redeploys the plugin install.
docker compose -f docker-stashsense2.yml down
docker compose -f docker-stashsense2.yml up -d --force-recreate --build

# Where the plugin gets installed. Stash can run on the same machine as the
# sidecar (a plain directory) or on another one (an rsync-over-ssh target,
# "host:/path/to/stash/config/plugins/stash-sense2"). Set PLUGIN_DEST in the
# environment or in rebuild.local.env next to this script (git-ignored by the
# *.env rule); unset falls back to the original local path.
[ -f "$(dirname "$0")/rebuild.local.env" ] && . "$(dirname "$0")/rebuild.local.env"
PLUGIN_DEST="${PLUGIN_DEST:-/opt/stash-storage/config/plugins/stash-sense2}"

# The installed plugin's manifest is named stash-sense2.yml (matching its
# stash-plugin-repo release), not this repo's own plugin/stash-sense.yml --
# rename on the way out so it actually overwrites the live manifest instead
# of leaving a second, unused stash-sense.yml alongside it.
case "$PLUGIN_DEST" in
  *:*)
    rsync -t plugin/*.js plugin/*.css plugin/*.py "$PLUGIN_DEST/"
    rsync -t plugin/stash-sense.yml "$PLUGIN_DEST/stash-sense2.yml"
    ;;
  *)
    cp plugin/*.js plugin/*.css plugin/*.py "$PLUGIN_DEST/" --preserve 2>&1 | grep -v 'plugin/__pycache__'
    cp plugin/stash-sense.yml "$PLUGIN_DEST/stash-sense2.yml" --preserve
    ;;
esac

echo "Deployed. In Stash: Settings > Plugins > reload this plugin (or restart Stash) to pick up the JS changes."
