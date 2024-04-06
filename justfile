of-docker:
    udocker pull openfoam/openfoam10-paraview56
    udocker create --name=of openfoam/openfoam10-paraview56:latest

of-solve case:
    udocker run --volume=$(pwd)/{{case}}:/home/openfoam --entrypoint "/home/openfoam/entrypoint.sh" of

postprocess case:
    scripts/foam2h5.py {{case}}
    scripts/grid-embedding.py {{case}}
    scripts/homogeneous-regions.py -k 64 --max-cluster-size 512 {{case}}
    scripts/mean-flow.py {{case}}
