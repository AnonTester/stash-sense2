#!/bin/sh
# Rebuilds the ROCm (primary/tested) sidecar image and redeploys the local
# plugin install. For the CPU variant, swap docker-stashsense2.yml for
# docker-stashsense2-cpu.yml on the two docker compose lines below.
docker compose -f docker-stashsense2.yml down
docker compose -f docker-stashsense2.yml up -d --force-recreate --build

# The installed plugin's manifest is named stash-sense2.yml (matching its
# stash-plugin-repo release), not this repo's own plugin/stash-sense.yml --
# rename on the way out so it actually overwrites the live manifest instead
# of leaving a second, unused stash-sense.yml alongside it.
PLUGIN_DEST=/opt/stash-storage/config/plugins/stash-sense2
cp plugin/*.js plugin/*.css plugin/*.py "$PLUGIN_DEST/" --preserve 2>&1 | grep -v 'plugin/__pycache__'
cp plugin/stash-sense.yml "$PLUGIN_DEST/stash-sense2.yml" --preserve

echo "Deployed. In Stash: Settings > Plugins > reload this plugin (or restart Stash) to pick up the JS changes."
