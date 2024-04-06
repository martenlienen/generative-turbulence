#!/bin/bash

if [[ $1 = "--with-raw" ]]; then
  opts=""
else
  opts="--exclude=*-raw.tar.xz"
fi

RSYNC_PASSWORD=m1737748 rsync -r --progress $opts rsync://m1737748@dataserv.ub.tum.de/m1737748/ data/shapes/download
