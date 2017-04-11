#!/bin/sh

## Copyright 2016 Eugenio Gianniti
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

root="${1?you should provide a root folder}"

error_aux ()
{
    echo "$0: line $1: ${2:-unknown error}" >&2
    exit 1
}
alias error='error_aux $LINENO '

test $# -gt 1 && error "too many input arguments"

test "x$root" = x. && error ". is not supported as root directory"

baseroot="${root%_reordered}"

find "$root" -name '*.csv' -type f | while read filename; do
    onlyfile="$(basename "$filename")"
    query="${onlyfile%.csv}"
    fieldno=$(head -n1 "$filename" | awk -F , '{
              for (i = 1; i <= NF; ++i) if ($i == "users") print i }')
    users=$(tail -n+2 "$filename" | \
                   awk -F , -v field=$fieldno '{ print $field }' | \
                   sort | uniq | grep -v ^$ | tail -n1)
    if [ "$users" -gt 1 ]; then
        dir="${baseroot}_organized/$query/multi"
    else
        dir="${baseroot}_organized/$query/single"
    fi
    mkdir -p "$dir"
    newfilename="$(mktemp -p "$dir" XXXXX.csv)"
    cp "$filename" "$newfilename"
done
