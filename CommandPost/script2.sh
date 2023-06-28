#!/usr/local/bin/bash

# Stop if any command fails.
set -e

# Stop on unset variables.
set -u

# Make jq working.
PATH="$PATH:/usr/local/bin"

# Check version.
version=$(echo "$1" | jq -r '.[0]')
if [ "$version" != "_dx2rgq3ln9kfsl_wdv9vzlng" ]; then
  osascript -e 'tell app "System Events" to display dialog "Version mismatch."'
  exit 1
fi

# Decode JSON.
readarray -t directories <<< "$(echo "$1" | jq -r '.[1][]')"
file=$(echo "$1" | jq -r '.[2]')
name=$(echo "$1" | jq -r '.[3]')
clipStart=$(echo "$1" | jq -r '.[4]')
clipDuration=$(echo "$1" | jq -r '.[5]')
notes=$(echo "$1" | jq -r '.[6]')

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
input_dir="$HOME/My Drive/AI Cut Ultra/input"
if [ ! -d "$input_dir" ]; then
  osascript -e 'tell app "System Events" to display dialog "Please create folder \"~/My Drive/AI Cut Ultra/input\"."'
  exit 1
fi
output_dir="$HOME/My Drive/AI Cut Ultra/output"
if [ ! -d "$output_dir" ]; then
  osascript -e 'tell app "System Events" to display dialog "Please create folder \"~/My Drive/AI Cut Ultra/output\"."'
  exit 1
fi

# Copy file.
input_filename="$(shasum --algorithm 256 "$filepath" | awk '{print $1}').${filepath##*.}"
cp "$filepath" "$input_dir/$input_filename"

# Create JSON.
json=$(echo "[]" | jq '$ARGS.positional' --args "tdqt9rkbrsv7bf5bz16gy2p19" "$input_filename" "$name" "$clipStart" "$clipDuration" "$notes")
echo "$json" > "$output_dir/$(date +%s)-$RANDOM.json"

# Done.
say "Done."
