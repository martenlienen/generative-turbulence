# From Zero to Turbulence: Generative Modeling for 3D Flow Simulation

Marten Lienen, David Lüdke, Jan Hansen-Palmus, Stephan Günnemann

![](./figures/generated-sample-curl-norm.png)

This repository contains the code used to produce the results in our paper: [openreview](https://openreview.net/forum?id=ZhlwoC1XaN), [arxiv](https://arxiv.org/abs/2306.01776).

Besides the model, data loading and training code, this repository also contains code to configure and run OpenFOAM and postprocess its outputs. These tools could be an immensely useful starting point for other researchers in the field. In particular, there is
- a [lark](https://github.com/lark-parser/lark) [grammar for the OpenFOAM configuration format](./turbdiff/openfoam.lark),
- a [python module](./turbdiff/openfoam.py) using this grammar to load, edit and save these configuration files,
- a script to [generate new OpenFOAM cases](./scripts/les-case.py) from a template,
- a script to [generate regular meshes with arbitrary axis-aligned shapes cut out](./scripts/channel-3d.py),
- a script to [run simulations on a SLURM cluster](./scripts/solve-slurm.py) in a docker container via [udocker](https://github.com/indigo-dc/udocker),
- a script to [convert OpenFOAM outputs into much more performant HDF5 files](./scripts/foam2h5.py),
- and a script to [precompute an embedding of the sparse 3D data](./scripts/grid-embedding.py) into dense 3D tensors with padding layers to encode boundary information.
Our [data loader](./turbdiff/data/ofles.py) lets you load all this information into easily digestible data classes.

## Installation

```sh
# Clone the repository
git clone https://github.com/martenlienen/generative-turbulence.git

# Change into the repository
cd generative-turbulence

# Install package editably with dependencies
pip install -e .

# If you need a specific pytorch version, e.g. CPU-only or an older CUDA version, check
#
#     https://pytorch.org/get-started/locally/
#
# and run, for example,
#
# pip install torch --extra-index-url https://download.pytorch.org/whl/cu117
#
# before installing this package.
```

## Dataset

Because of the size of the dataset, we are currently in the process of organizing online storage. As soon as the dataset is online, we will add instructions here on how to download and use it.

In the meantime, you can generate your own data by following the steps in the next section.

### Data generation with OpenFOAM in docker

To generate data for new OpenFOAM simulations, first make sure that you have installed the extra dependencies and have [`just`](https://just.systems/) available:
```sh
pip install -e ".[data]"
```
If you don't want to use `just`, you can also read the [`justfile`](./justfile) and run the commands yourself.

Begin by creating a docker container with OpenFOAM installed:
```sh
just of-docker
```

Now generate a bunch of new cases. For example, the following sets up all the OpenFOAM cases (simulations) from our dataset:
```sh
./scripts/generate-shapes.py data/shapes
```
Of course, you can adapt the script to create other shapes or completely new datasets.

Now you can solve the case (run the simulation) with OpenFOAM locally
```sh
just of-solve path/to/case
```
or submit a whole bunch of them to your own SLURM cluster:
```sh
./scripts/solve-slurm.py data/shapes/data/*
```

Afterwards, apply the postprocessing, e.g. the HDF5 conversion, to each case
```sh
just postprocess path/to/case
```

Finally, compute the training set statistics for feature normalization:
```sh
./scripts/dataset-stats.py data/shapes
```

## Training

To start a training, call `train.py` with the your settings, for example
```sh
./train.py data.batch_size=128
```
The training script uses [hydra](https://hydra.cc) for configuration, so check out the files in the `config` directory, to learn about all available settings.

To re-run the experiments from the paper, execute
```sh
./train.py -cn shapes_experiment -m
```
which starts training with the settings in [`config/shapes_experiment.yaml`](./config/shapes_experiment.yaml). If you don't have a SLURM cluster available, remove the settings related to `launcher`.

## Citation

If you build upon this work, please cite our paper as follows.

```
@inproceedings{lienen2024zero,
  title = {From {{Zero}} to {{Turbulence}}: {{Generative Modeling}} for {{3D Flow Simulation}}},
  author = {Lienen, Marten and L{\"u}dke, David and {Hansen-Palmus}, Jan and G{\"u}nnemann, Stephan},
  booktitle = {International {{Conference}} on {{Learning Representations}}},
  year = {2024},
}
```
