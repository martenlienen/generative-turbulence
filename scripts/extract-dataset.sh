#!/bin/bash

for archive in data/shapes/download/*.tar.xz; do
  tar --extract -f "$archive" --directory data/shapes
done
