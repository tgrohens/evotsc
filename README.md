Individual-based evolutionary simulation of the transcription-supercoiling coupling at the whole genome scale.

## Usage

To run an experiment with the default parameters, just run:
```
python3 path/to/evotsc_run.py -o `output_folder` -n `final_generation`
```

You can change parameter values at the top of the `evotsc_run.py` file.

Simulations can be seamlessly restarted at the last checkpoint (either at the end of a run or after a crash), by running the exact same command again, and completed runs can be extended by passing a larger `final_generation` argument.
## Reproducibility

The seed used for each simulation is output in the `output_folder/params.txt` file.

To reproduce a run, you can pass the seed to the program with the `-s SEED` parameter.

## ALIFE21 paper

The exact code used for the simulations in the [Alife21 paper](https://direct.mit.edu/isal/proceedings/isal2021/33/97/102928), as well as the Jupyter notebooks analyzing the resulting data, can be found in the [alife-model](https://gitlab.inria.fr/tgrohens/evotsc/-/tree/alife-model) branch.

## ALIFE journal paper

The exact code used for the simulations in the [ALIFE journal paper](https://direct.mit.edu/artl/article-abstract/28/4/440/112557/A-Genome-Wide-Evolutionary-Simulation-of-the), as well as the Jupyter notebooks analyzing the resulting data, can be found in the [alife-journal](https://gitlab.inria.fr/tgrohens/evotsc/-/tree/alife-journal) branch.
