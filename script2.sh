#!/usr/local/bin/bash

# Stop if any command fails.
set -e

# Stop on unset variables.
set -u

# Make jq working.
PATH="$PATH:/usr/local/bin"

# Check version.
version=$(echo "$1" | jq -r '.[0]')
if [ "$version" != "v1" ]; then
  osascript -e 'tell app "System Events" to display dialog "Version is not equal to v1."'
  exit 1
fi

# Decode JSON.
readarray -t directories <<< "$(echo "$1" | jq -r '.[1][]')"
file=$(echo "$1" | jq -r '.[2]')
notes=$(echo "$1" | jq -r '.[3]')

# Find source file.
for directory in "${directories[@]}"; do
  readarray -t files < <(find "$directory" -name "$file")
  candidates=("${candidates[@]}" "${files[@]}")
done

if [ ${#candidates[@]} -eq 0 ]; then
  osascript -e 'tell app "System Events" to display dialog "File not found."'
  exit 1
fi

if [ ${#candidates[@]} -gt 1 ]; then
  osascript -e 'tell app "System Events" to display dialog "Multiple files found."'
  exit 1
fi

filepath="${candidates[0]}"

# Check drive.
input_dir="$HOME/My Drive/AI Cut Pro/input"
if [ ! -d "$input_dir" ]; then
  osascript -e 'tell app "System Events" to display dialog "Please create folder \"~/My Drive/AI Cut Pro/input\"."'
  exit 1
fi

# Copy
filename="$(shasum --algorithm 256 "$filepath" | awk '{print $1}').${filepath##*.}"
cp "$filepath" "$input_dir/$filename"
say "Done."
