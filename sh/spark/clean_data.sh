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

find "$root" -name '*.csv' | while IFS= read -r filename; do
    auxfile="$filename".aux

    colnum="$(awk -F , '{ print NF }' "$filename" | sort -n | tail -n 1)"
    head -n 1 "$filename" > "$auxfile"
    awk -F , -v cols="$colnum" '( NF == cols ) { $1 = $1; print }' OFS=, "$filename" >> "$auxfile"

    head -n 2 "$auxfile" | tail -n 1 | tr , '\n' | grep -ivn max \
        | grep -iv job | cut -d : -f 1 | xargs | awk '{ $1 = $1; print }' OFS=, \
        | xargs -J % cut -d , -f % "$auxfile" > "$filename"

    rm "$auxfile"
done
