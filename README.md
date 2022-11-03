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

## PhD manuscript

This branch contains the code and notebooks used for chapters 6 and 7 of my PhD manuscript, which is available [here](https://gitlab.inria.fr/tgrohens/phd).

