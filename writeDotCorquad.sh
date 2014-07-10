#!/bin/bash
out=$1
shift
echo "$1" > $out
shift
for f in "$@"
    do
        echo $f >> $out
    done
