# EvoTSC

An individual-based evolutionary simulation of the transcription-supercoiling coupling at the whole genome scale.

## Usage

### Installing

First, install the required packages (preferably inside a virtual environment):
```
pip install -r requirements.txt
```

### Running

To run an experiment with the default parameters, just run:
```
python3 path/to/evotsc_run.py -o `output_folder` -n `final_generation`
```

You can change parameter values at the top of the `evotsc_run.py` file.

Simulations can be seamlessly restarted at the last checkpoint (either at the end of a run or after a crash), by running the exact same command again, and completed runs can be extended by passing a larger `final_generation` argument.

## Reproducibility

The seed used for each simulation is output in the `output_folder/params.txt` file.

To reproduce a run, you can pass the seed to the program with the `-s SEED` parameter.

## License

EvoTSC is licensed under the [3-clause BSD licence](./LICENSE.txt), including all Python source files as well as notebooks.

## Publications

### PLOS Computational Biology paper

The exact code used for the upcoming PLOS Computational Biology paper, as well as the Jupyter notebooks analyzing the resulting data, can be found in the [ploscb](https://gitlab.inria.fr/tgrohens/evotsc/-/tree/ploscb) branch.

### PCI Math Comp Biol peer-reviewed preprint

The exact code used for the [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2022.09.23.509185v2) peer-reviewed at PCI Math Comp Biol, as well as the Jupyter notebooks analyzing the resulting data, can be found in the [pci](https://gitlab.inria.fr/tgrohens/evotsc/-/tree/pci) branch.

### PhD thesis

The exact code used for the simulations in Chapters 6 and 7 of my [PhD thesis](https://gitlab.inria.fr/tgrohens/phd), as well as the Jupyter notebooks analyzing the resulting data, can be found in the [phd](https://gitlab.inria.fr/tgrohens/evotsc/-/tree/phd) branch.

### Artificial Life journal paper

The exact code used for the simulations in the [Artificial Life journal paper](https://direct.mit.edu/artl/article-abstract/28/4/440/112557/A-Genome-Wide-Evolutionary-Simulation-of-the) (corresponding to Chapter 4 of my PhD thesis), as well as the Jupyter notebooks analyzing the resulting data, can be found in the [alife-journal](https://gitlab.inria.fr/tgrohens/evotsc/-/tree/alife-journal) branch.

### ALIFE 21 conference paper

The exact code used for the simulations in the [Alife21 conference paper](https://direct.mit.edu/isal/proceedings/isal/33/97/102928), as well as the Jupyter notebooks analyzing the resulting data, can be found in the [alife-model](https://gitlab.inria.fr/tgrohens/evotsc/-/tree/alife-model) branch.
