#!/bin/bash

set -e

find src -type f -printf '%P\n' | while read path ; do

    install -v -D "src/$path" "/$path"

done

echo "done"
