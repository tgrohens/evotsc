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
nb_eval_steps=51
beta_A = 0.0
beta_B = 0.25
nb_genes = 10
nb_indivs = 250
save_step = 50

def main():
    # Parse CLI arguments: output directory and number of generations
    if len(sys.argv) != 3:
        print('Usage: `run_evotsc output_dir nb_generations`. Exiting.')
        return

    output_dir = sys.argv[1]
    nb_generations = int(sys.argv[2])

    # Setup the experiment folder
    os.mkdir(output_dir)

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

    # Evaluate the first individual
    evotsc_plot.plot_expr_AB(indiv=init_indiv,
                             plot_title='best generation 0',
                             plot_name=f'{output_dir}/best generation 0.png')
    evotsc_plot.explain(init_indiv)

    best_indivs = []
    gen = 0

    while gen < nb_generations:
        best_indivs += population.evolve(save_step, start_time=gen)
        gen += save_step

        cur_best = best_indivs[-1]
        evotsc_plot.plot_expr_AB(indiv=cur_best,
                                plot_title=f'best generation {gen}',
                                plot_name=f'{output_dir}/best generation {gen}.png')
        evotsc_plot.explain(best_indivs[-1])

        with open(f'{output_dir}/best_gen_{gen}.evotsc', 'wb') as save_file:
            pickle.dump(cur_best, save_file)



if __name__ == "__main__":
    main()