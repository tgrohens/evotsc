Individual-based evolutionary simulation of the transcription-supercoiling coupling at the whole-genome scale.

## Usage

To run an experiment with the default parameters, just run:
```
python3 path/to/run_evotsc.py `output_folder` `nb_generations`
```

You can change parameter values at the top of the `run_evotsc.py` file.

Simulations can be seamlessly restarted at the last checkpoint (either at the end of a run or after a crash), by running the command again with the new number of generations to run for.
## Reproducibility

The parameters used for each simulation are saved in the `output_folder/params.txt` file.
