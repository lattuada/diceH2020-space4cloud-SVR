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

table_parameters ()
{
    local query="${1?missing query}"
    case "$query" in
        R1)
            awkvars='-v d250=36 -v d500=72 -v d750=109 -v d1000=148'
            ;;
        R[25])
            awkvars='-v d250=0.953 -v d500=0.355 -v d750=0.612 -v d1000=16'
            ;;
        R3)
            awkvars='-v d250=96 -v d500=190 -v d750=287 -v d1000=390'
            ;;
        R4)
            awkvars='-v d250=72 -v d500=144 -v d750=217 -v d1000=296'
            ;;
        *)
            awkvars='-v d250=250 -v d500=500 -v d750=750 -v d1000=1000'
            ;;
    esac
}

test "x$root" = x. && error ". is not supported as root directory"

find "$root" -name runs.csv -type f | while read filename; do
    relname="${filename#$root}"
    id=$(echo "$relname" | tr / "\n" | grep -v ^$ | grep -e data -e dependencies -B2 | \
             grep -v data | grep -v dependencies | grep -v session | head -n1)
    query="$(grep Application "$filename" | awk '{ print $NF }' | sort | uniq)"
    if [ $(echo $query | wc -w) -gt 1 ]; then
        echo $id is a multi-class experiment and will not be considered >&2
    else
        dir="${root}_reordered/${id}"
        mkdir -p "$dir"
        table_parameters "$query"
        cat "$filename" | grep , | \
            awk -F , $awkvars 'BEGIN { OFS = "," }
                           ( $(NF - 1) == 250 ) { $(NF - 1) = d250 }
                           ( $(NF - 1) == 500 ) { $(NF - 1) = d500 }
                           ( $(NF - 1) == 750 ) { $(NF - 1) = d750 }
                           ( $(NF - 1) == 1000 ) { $(NF - 1) = d1000 }
                           { $1 = $1; print }' \
                               > "$dir"/"$query".csv
    fi
done
