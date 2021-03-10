import sys
import os
import pickle

import evotsc
import evotsc_plot

# Constants
intergene = 1000
interaction_dist = 2500
interaction_coef = 0.3
default_basal_expression = 1.0
nb_eval_steps = 51
beta_A = 0.0
beta_B = 0.25
nb_genes = 60
nb_indivs = 250
save_step = 50

def print_params(output_dir):
    with open(f'{output_dir}/params.txt', 'w') as params_file:
        params_file.write(f'intergene: {intergene}\n')
        params_file.write(f'interaction_dist: {interaction_dist}\n')
        params_file.write(f'interaction_coef: {interaction_coef}\n')
        params_file.write(f'default_basal_expression: {default_basal_expression}\n')
        params_file.write(f'nb_eval_steps: {nb_eval_steps}\n')
        params_file.write(f'beta_A: {beta_A}\n')
        params_file.write(f'beta_B: {beta_B}\n')
        params_file.write(f'nb_genes: {nb_genes}\n')
        params_file.write(f'nb_indivs: {nb_indivs}\n')
        params_file.write(f'save_step: {save_step}\n')


def save(output_dir, indiv, gen):
    evotsc_plot.plot_expr_AB(indiv=indiv,
                             plot_title=f'best generation {gen:05}',
                             plot_name=f'{output_dir}/plot_best_gen_{gen:05}.png')
    evotsc_plot.explain(indiv)

    with open(f'{output_dir}/best_gen_{gen:05}.evotsc', 'wb') as save_file:
        pickle.dump(indiv, save_file)


def main():
    # Parse CLI arguments: output directory and number of generations
    if len(sys.argv) != 3:
        print('Usage: `run_evotsc output_dir nb_generations`. Exiting.')
        return

    output_dir = sys.argv[1]
    nb_generations = int(sys.argv[2])

    # Setup the experiment folder
    os.mkdir(output_dir)

    # Save the parameters for reproducibility
    print_params(output_dir)

    # Setup the initial individual and population
    init_genes = evotsc.Gene.generate(intergene=intergene,
                                      nb_genes=nb_genes,
                                      default_basal_expression=default_basal_expression)

    init_indiv = evotsc.Individual(genes=init_genes,
                                   interaction_dist=interaction_dist,
                                   interaction_coef=interaction_coef,
                                   nb_eval_steps=nb_eval_steps,
                                   beta_A=beta_A,
                                   beta_B=beta_B)

    mutation = evotsc.Mutation(intergene_mutation_prob=0.0,
                                      intergene_mutation_var=0.0,
                                      inversion_prob=1e-1)


    population = evotsc.Population(init_indiv=init_indiv,
                                   nb_indivs=nb_indivs,
                                   mutation=mutation)


    save(output_dir, init_indiv, 0)

    gen = 1 # We start at 1 since the population at time 0 already exists

    while gen < nb_generations:
        best_indivs = population.evolve(save_step, start_time=gen)
        gen += save_step

        cur_best = best_indivs[-1]
        save(output_dir, cur_best, gen)


if __name__ == "__main__":
    main()