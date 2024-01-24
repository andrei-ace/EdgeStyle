#!/bin/bash

# Script to find top-level directories that are either empty or only contain empty subdirectories

# Change directory to the one provided as an argument, or use the current directory
cd "${1:-.}"

# Function to check if a directory contains only empty directories
contains_only_empty_dirs() {
    local dir=$1

    # If directory is empty, return true
    if [ -z "$(ls -A "$dir")" ]; then
        return 0
    fi

    # Check each item in the directory
    for item in "$dir"/*; do
        # If item is a file, return false
        if [ -f "$item" ]; then
            return 1
        fi
        # If item is a non-empty directory, return false
        if [ -d "$item" ] && [ -n "$(ls -A "$item")" ]; then
            return 1
        fi
    done

    # If all checks pass, return true
    return 0
}

# Find and process all directories
find . -type d | while read -r dir; do
    if contains_only_empty_dirs "$dir"; then
        # Print the directory, stripping the leading './'
        echo "${dir#./}"

        # Prevent further exploration of its subdirectories
        find "$dir" -mindepth 1 -type d -printf "%P\n" | sort -r | xargs -I{} echo "$dir/{}/"
    fi
done | sort -r | awk -F/ '!seen[$1]++'
