#!/usr/local/bin/bash

# Stop if any command fails.
set -e

# Stop on unset variables.
set -u

# Make jq working.
PATH="$PATH:/usr/local/bin"

# Debug.
echo "$1"
say "$1"
