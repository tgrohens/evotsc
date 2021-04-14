import os
import argparse
import pickle
import pathlib
import re
import subprocess

import numpy as np

import evotsc
import evotsc_plot

# Constants
intergene = 1000
interaction_dist = 2500
interaction_coef = 0.3
sigma_basal = -0.06
sigma_opt = -0.06
epsilon = 0.03
default_basal_expression = 0.5
nb_eval_steps = 51
sigma_A = 0.1
sigma_B = -0.1
nb_genes = 60
nb_indivs = 100
inversion_param = 2.0
basal_sc_mutation_prob = 1e-1
basal_sc_mutation_var = 1e-4
save_best_step = 500
save_full_step = 5000

def get_git_hash():
    git_path = pathlib.Path(__file__).parent.absolute()
    git_ref_raw = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=git_path)
    git_ref = str(git_ref_raw, "utf-8").strip()
    status_raw = subprocess.check_output(['git', 'status', '-s'], cwd=git_path)
    status = str(status_raw, "utf-8").strip()

    if status != '':
        git_ref += '-dirty'

    return git_ref


def print_params(output_dir, seed):
    with open(f'{output_dir}/params.txt', 'w') as params_file:
        # Meta
        params_file.write(f'commit: {get_git_hash()}\n')
        params_file.write(f'seed: {seed}\n')
        # Actual parameters
        params_file.write(f'intergene: {intergene}\n')
        params_file.write(f'interaction_dist: {interaction_dist}\n')
        params_file.write(f'interaction_coef: {interaction_coef}\n')
        params_file.write(f'sigma_basal: {sigma_basal}\n')
        params_file.write(f'sigma_opt: {sigma_opt}\n')
        params_file.write(f'epsilon: {epsilon}\n')
        params_file.write(f'default_basal_expression: {default_basal_expression}\n')
        params_file.write(f'nb_eval_steps: {nb_eval_steps}\n')
        params_file.write(f'inversion_param: {inversion_param}\n')
        params_file.write(f'basal_sc_mutation_prob: {basal_sc_mutation_prob}\n')
        params_file.write(f'basal_sc_mutation_var: {basal_sc_mutation_var}\n')
        params_file.write(f'sigma_A: {sigma_A}\n')
        params_file.write(f'sigma_B: {sigma_B}\n')
        params_file.write(f'nb_genes: {nb_genes}\n')
        params_file.write(f'nb_indivs: {nb_indivs}\n')
        params_file.write(f'save_best_step: {save_best_step}\n')
        params_file.write(f'save_full_step: {save_full_step}\n')


def save_indiv(output_dir, indiv, gen):
    evotsc_plot.plot_expr_AB(indiv=indiv,
                             sigma_A=sigma_A,
                             sigma_B=sigma_B,
                             plot_title=f'best generation {gen:06}',
                             plot_name=f'{output_dir}/plot_best_gen_{gen:06}.png')

    evotsc_plot.explain(indiv, sigma_A, sigma_B)

    with open(f'{output_dir}/best_gen_{gen:06}.evotsc', 'wb') as save_file:
        pickle.dump(indiv, save_file)


def save_pop(output_dir, pop, gen):
    with open(f'{output_dir}/pop_gen_{gen:06}.evotsc', 'wb') as save_file:
        pickle.dump(pop, save_file)


def load_pop(pop_path):
    with open(pop_path, 'rb') as save_file:
        return pickle.load(save_file)


def write_stats(stats_file, indiv, gen):
    on_genes_A, off_genes_A, on_genes_B, off_genes_B = indiv.summarize(sigma_A, sigma_B)
    stats_file.write(f'{gen},{indiv.fitness},'
                    f'{on_genes_A[0]},{off_genes_A[0]},{on_genes_A[1]},{off_genes_A[1]},{on_genes_A[2]},{off_genes_A[2]},'
                    f'{on_genes_B[0]},{off_genes_B[0]},{on_genes_B[1]},{off_genes_B[1]},{on_genes_B[2]},{off_genes_B[2]},'
                    f'{indiv.sigma_basal}\n')
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
    start_gen = 1

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
        print_params(output_dir, seed)

        # Setup the initial individual and population
        init_genes = evotsc.Gene.generate(intergene=intergene,
                                          nb_genes=nb_genes,
                                          default_basal_expression=default_basal_expression,
                                          rng=rng)


        init_indiv = evotsc.Individual(genes=init_genes,
                                       interaction_dist=interaction_dist,
                                       interaction_coef=interaction_coef,
                                       nb_eval_steps=nb_eval_steps,
                                       sigma_basal=sigma_basal,
                                       sigma_opt=sigma_opt,
                                       epsilon=epsilon,
                                       rng=rng)


        mutation = evotsc.Mutation(basal_sc_mutation_prob=basal_sc_mutation_prob,
                                   basal_sc_mutation_var=basal_sc_mutation_var,
                                   inversion_param=inversion_param)

        population = evotsc.Population(init_indiv=init_indiv,
                                       nb_indivs=nb_indivs,
                                       mutation=mutation,
                                       sigma_A=sigma_A,
                                       sigma_B=sigma_B,
                                       rng=rng)


        if not args.neutral:
            stats_file = open(f'{output_dir}/stats.csv', 'w')
            stats_file.write('Gen,Fitness,ABon_A,ABoff_A,Aon_A,Aoff_A,Bon_A,Boff_A,'
                                        'ABon_B,ABoff_B,Aon_B,Aoff_B,Bon_B,Boff_B,'
                                        'basal_sc\n')

            save_indiv(output_dir, init_indiv, 0)

    else:
        save_files = [f for f in output_dir.iterdir() if 'pop_gen' in f.name]
        last_save_path = sorted(save_files)[-1]
        start_gen = int(re.search(r'\d+', last_save_path.name).group(0)) + 1
        population = load_pop(last_save_path)

        if not args.neutral:
            # Get rid of the stats that happened between the last save and the crash
            os.rename(f'{output_dir}/stats.csv', f'{output_dir}/old_stats.csv')

            old_stats_path = f'{output_dir}/old_stats.csv'
            old_stats_file = open(old_stats_path)
            stats_file = open(f'{output_dir}/stats.csv', 'w')
            for gen in range(start_gen+1):
                stats_file.write(old_stats_file.readline())

            old_stats_file.close()
            os.remove(old_stats_path)


    if not args.neutral:
        for gen in range(start_gen, nb_generations+1):
            best_indiv, avg_fit = population.step()

            print(f'Gen {gen}: best fit {best_indiv.fitness:.5}, avg fit {avg_fit:.5}')
            write_stats(stats_file, best_indiv, gen)

            if gen % save_best_step == 0:
                save_indiv(output_dir, best_indiv, gen)

            if gen % save_full_step == 0:
                save_pop(output_dir, population, gen)

        stats_file.close()

    else: # Neutral evolution -- no selection
        for gen in range(start_gen, nb_generations+1):
            population.neutral_step()
            print(f'Generation {gen}')

            if gen % save_full_step == 0:
                save_pop(output_dir, population, gen)


if __name__ == "__main__":
    main()