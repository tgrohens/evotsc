import os
import argparse
import pickle
import pathlib
import re
import subprocess

import numpy as np

import evotsc

## Constants
# Population
nb_indivs = 100
nb_genes = 60

# Genome
intergene = 125
gene_length = 1000
interaction_dist = 5000
interaction_coef = 0.03
sigma_basal = -0.066
sigma_opt = -0.042
epsilon = 0.005
m = 2.5
default_basal_expression = (1 + np.exp(- m)) / 2 # Average of the maximum and minimum expression levels in the model

# Fitness
selection_coef = 50

# Selection
selection_method = "fit-prop" # Choices: "fit-prop", "rank", "exp-rank"

# Environment
sigma_A = 0.01
sigma_B = -0.01

# Mutations
inversion_poisson_lam = 2.0
intergene_poisson_lam = 0.0 #2.0
intergene_mutation_var = 0.0 #1e1
basal_sc_mutation_prob = 0.0 #1e-1
basal_sc_mutation_var = 0.0 #1e-4

# Logging
save_step = 50_000


def get_git_hash():
    git_path = pathlib.Path(__file__).parent.absolute()
    git_ref_raw = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=git_path)
    git_ref = str(git_ref_raw, "utf-8").strip()
    status_raw = subprocess.check_output(['git', 'status', '-s'], cwd=git_path)
    status = str(status_raw, "utf-8").strip()

    if status != '':
        git_ref += '-dirty'

    return git_ref


def print_params(output_dir, seed, neutral):
    with open(f'{output_dir}/params.txt', 'w') as params_file:
        # Meta
        params_file.write(f'commit: {get_git_hash()}\n')
        params_file.write(f'seed: {seed}\n')
        params_file.write(f'neutral: {neutral}\n')
        # Population
        params_file.write(f'nb_indivs: {nb_indivs}\n')
        params_file.write(f'nb_genes: {nb_genes}\n')
        # Genome
        params_file.write(f'intergene: {intergene}\n')
        params_file.write(f'gene_length: {gene_length}\n')
        params_file.write(f'interaction_dist: {interaction_dist}\n')
        params_file.write(f'interaction_coef: {interaction_coef}\n')
        params_file.write(f'sigma_basal: {sigma_basal}\n')
        params_file.write(f'sigma_opt: {sigma_opt}\n')
        params_file.write(f'epsilon: {epsilon}\n')
        params_file.write(f'default_basal_expression: {default_basal_expression}\n')
        # Fitness
        params_file.write(f'selection_coef: {selection_coef}\n')
        # Selection
        params_file.write(f'selection_method: {selection_method}\n')
        # Environment
        params_file.write(f'sigma_A: {sigma_A}\n')
        params_file.write(f'sigma_B: {sigma_B}\n')
        # Mutations
        params_file.write(f'inversion_poisson_lam: {inversion_poisson_lam}\n')
        params_file.write(f'intergene_poisson_lam: {intergene_poisson_lam}\n')
        params_file.write(f'intergene_mutation_var: {intergene_mutation_var}\n')
        params_file.write(f'basal_sc_mutation_prob: {basal_sc_mutation_prob}\n')
        params_file.write(f'basal_sc_mutation_var: {basal_sc_mutation_var}\n')
        # Logging
        params_file.write(f'save_step: {save_step}\n')


def save_pop(output_dir, pop, gen):
    # At this stage, we have a new non-evaluated population, so let's
    # evaluate everyone to have consistent save files
    pop.evaluate()
    with open(f'{output_dir}/pop_gen_{gen:06}.evotsc', 'wb') as save_file:
        pickle.dump(pop, save_file)


def load_pop(pop_path):
    with open(pop_path, 'rb') as save_file:
        return pickle.load(save_file)


def write_stats(stats_file, indiv, avg_fit, gen):
    on_genes_A, off_genes_A, on_genes_B, off_genes_B = indiv.summarize(sigma_A, sigma_B)
    stats_file.write(f'{gen},{indiv.fitness},{avg_fit},'
                    f'{on_genes_A[0]},{off_genes_A[0]},{on_genes_A[1]},{off_genes_A[1]},{on_genes_A[2]},{off_genes_A[2]},'
                    f'{on_genes_B[0]},{off_genes_B[0]},{on_genes_B[1]},{off_genes_B[1]},{on_genes_B[2]},{off_genes_B[2]}\n')
    stats_file.flush()


def main():

    # Parse CLI arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-n', '--generations', type=int, required=True,
                            help='number of generations')
    arg_parser.add_argument('-o', '--output_dir', required=True,
                            help='output directory')
    arg_parser.add_argument('--neutral', action='store_true',
                            help='run without selection')
    arg_parser.add_argument('-s', '--seed', type=int,
                            help='seed for the RNG')
    args = arg_parser.parse_args()

    nb_generations = int(args.generations)
    output_dir = pathlib.Path(args.output_dir)


    first_start = True
    start_gen = 0

    # Setup the experiment folder
    try:
        os.mkdir(output_dir)
    except FileExistsError: # We're restarting a population
        first_start = False

    if first_start:

        # Create the RNG seed
        seed = args.seed
        if not seed:
            seed = np.random.randint(1e9)
        rng = np.random.default_rng(seed=seed)

        # Save the parameters for reproducibility
        print_params(output_dir, seed, args.neutral)

        # Setup the initial individual and population
        init_genes = evotsc.Gene.generate(intergene=intergene,
                                          length=gene_length,
                                          nb_genes=nb_genes,
                                          default_basal_expression=default_basal_expression,
                                          rng=rng)


        init_indiv = evotsc.Individual(genes=init_genes,
                                       interaction_dist=interaction_dist,
                                       interaction_coef=interaction_coef,
                                       sigma_basal=sigma_basal,
                                       sigma_opt=sigma_opt,
                                       epsilon=epsilon,
                                       m=m,
                                       selection_coef=selection_coef,
                                       rng=rng)
        # Evaluate the initial individual before creating the clonal population
        init_indiv.evaluate(sigma_A, sigma_B)


        mutation = evotsc.Mutation(basal_sc_mutation_prob=basal_sc_mutation_prob,
                                   basal_sc_mutation_var=basal_sc_mutation_var,
                                   intergene_poisson_lam=intergene_poisson_lam,
                                   intergene_mutation_var=intergene_mutation_var,
                                   inversion_poisson_lam=inversion_poisson_lam)

        population = evotsc.Population(init_indiv=init_indiv,
                                       nb_indivs=nb_indivs,
                                       mutation=mutation,
                                       sigma_A=sigma_A,
                                       sigma_B=sigma_B,
                                       selection_method=selection_method,
                                       rng=rng)

        if not args.neutral:
            stats_file = open(f'{output_dir}/stats.csv', 'w')
            stats_file.write('Gen,Fitness,Avg Fit,ABon_A,ABoff_A,Aon_A,Aoff_A,Bon_A,Boff_A,'
                                        'ABon_B,ABoff_B,Aon_B,Aoff_B,Bon_B,Boff_B\n')

            save_pop(output_dir, population, 0)
            write_stats(stats_file, init_indiv, init_indiv.fitness, 0)

    else:
        save_files = [f for f in output_dir.iterdir() if 'pop_gen' in f.name]
        last_save_path = sorted(save_files)[-1]
        start_gen = int(re.search(r'\d+', last_save_path.name).group(0))
        population = load_pop(last_save_path)
        # Make sure the population has been evaluated before stepping
        population.evaluate()

        if not args.neutral:
            # Get rid of the stats that happened between the last save and the crash
            os.rename(f'{output_dir}/stats.csv', f'{output_dir}/old_stats.csv')

            old_stats_path = f'{output_dir}/old_stats.csv'
            old_stats_file = open(old_stats_path)
            stats_file = open(f'{output_dir}/stats.csv', 'w')
            # We need to save lines 0 to start_gen -> start_gen + 1 lines
            for gen in range(start_gen+2):
                stats_file.write(old_stats_file.readline())

            old_stats_file.close()
            os.remove(old_stats_path)


    if not args.neutral:
        for gen in range(start_gen+1, nb_generations+1):
            best_indiv, avg_fit = population.step()

            print(f'Gen {gen}: best fit {best_indiv.fitness:.5}, avg fit {avg_fit:.5}')
            write_stats(stats_file, best_indiv, avg_fit, gen)

            if gen % save_step == 0:
                save_pop(output_dir, population, gen)

        stats_file.close()

    else: # Neutral evolution -- no selection
        for gen in range(start_gen, nb_generations+1):
            population.neutral_step()
            print(f'Generation {gen}')

            if gen % save_step == 0:
                save_pop(output_dir, population, gen)


if __name__ == "__main__":
    main()
