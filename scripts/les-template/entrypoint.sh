#!/bin/bash

# Entrypoint for the OpenFOAM docker container

# Set up OpenFOAM environment
source /opt/openfoam10/etc/bashrc

# Run the simulation
exec ./Allrun
