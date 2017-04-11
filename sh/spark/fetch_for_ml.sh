#!/bin/sh

## Copyright 2017 Eugenio Gianniti
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

root="${1?error: missing root directory}"
out="${2?error: missing output directory}"

echo warning: this script is not robust to directory structure changes 1>&2

find "$root" -name '*.csv' | while IFS= read -r filename; do
    query="$(echo "$filename" | awk -F / '{ print $2 }')"
    base="$(basename "$filename")"

    dir="$out/$query"
    mkdir -p "$dir"
    finalfile="$dir/$base"
    if ls "$dir" | grep -q "/$base"; then
        tmp="$dir/aux.csv"
        {
            cat "$finalfile"
            tail -n +3 "$filename"
        } > "$tmp"
        mv "$tmp" "$finalfile"
    else
        cp "$filename" "$finalfile"
    fi
done
