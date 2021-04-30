Individual-based evolutionary simulation of the transcription-supercoiling coupling at the whole genome scale.

## Usage

To run an experiment with the default parameters, just run:
```
python3 path/to/evotsc_run.py -o `output_folder` -n `nb_generations`
```

You can change parameter values at the top of the `evotsc_run.py` file.

Simulations can be seamlessly restarted at the last checkpoint (either at the end of a run or after a crash), by running the command again with the new number of generations to run for.
## Reproducibility

The seed used for each simulation is output in the `output_folder/params.txt` file.

To reproduce a run, you can pass the seed to the program with the `-s SEED` parameter.

## ALIFE21 submission

The exact code used for the simulations in the Alife21 paper, as well as the Jupyter notebooks analyzing the resulting data, can be found at 30403808.