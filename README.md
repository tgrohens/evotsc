# EvoTSC

An individual-based evolutionary simulation of the transcription-supercoiling coupling at the whole genome scale.

## Usage

### Installing

In order to install the package, you should clone the repository, and run the Python installation process (preferably inside a virtual environment):
```
git clone https://gitlab.inria.fr/tgrohens/evotsc.git # Clone the repository
cd evotsc
pip install .                                         # Install the package itself
pip install -r requirements.txt                       # Install requirements
```

### Running

To run an experiment with the default parameters, just run:
```
python -m evotsc -o `output_folder` -n `final_generation`
```

You can change the parameter values at the top of the `evotsc/run.py` file.

Simulations can be seamlessly restarted at the last checkpoint (either at the end of a run or after a crash), by running the exact same command again, and completed runs can be extended by passing a larger `final_generation` argument.

## Reproducibility

The seed used for each simulation, as well as the commit of the code being run, are output to the `{output_folder}/params.txt` file.

To reproduce a run, you can pass the seed to the program with the `-s SEED` parameter.

## License

EvoTSC is licensed under the [3-clause BSD licence](./LICENSE.txt), including all source files.

## Publications

### ALIFE journal submission

The exact code used for the simulations in the ALIFE journal submission, as well as the Jupyter notebooks analyzing the resulting data, can be found in the [alife-journal](https://gitlab.inria.fr/tgrohens/evotsc/-/tree/alife-journal) branch.

### ALIFE 21 paper

The exact code used for the simulations in the Alife21 paper, as well as the Jupyter notebooks analyzing the resulting data, can be found in the [alife-model](https://gitlab.inria.fr/tgrohens/evotsc/-/tree/alife-model) branch.
