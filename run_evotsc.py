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
inversion_prob=2.0 # It's the parameter of a Poisson law
beta_A = 0.0
beta_B = 0.25
nb_genes = 60
nb_indivs = 100
save_step = 100

def print_params(output_dir):
    with open(f'{output_dir}/params.txt', 'w') as params_file:
        params_file.write(f'intergene: {intergene}\n')
        params_file.write(f'interaction_dist: {interaction_dist}\n')
        params_file.write(f'interaction_coef: {interaction_coef}\n')
        params_file.write(f'default_basal_expression: {default_basal_expression}\n')
        params_file.write(f'nb_eval_steps: {nb_eval_steps}\n')
        params_file.write(f'inversion_prob: {inversion_prob}\n')
        params_file.write(f'beta_A: {beta_A}\n')
        params_file.write(f'beta_B: {beta_B}\n')
        params_file.write(f'nb_genes: {nb_genes}\n')
        params_file.write(f'nb_indivs: {nb_indivs}\n')
        params_file.write(f'save_step: {save_step}\n')


def save_indiv(output_dir, indiv, gen):
    evotsc_plot.plot_expr_AB(indiv=indiv,
                             plot_title=f'best generation {gen:05}',
                             plot_name=f'{output_dir}/plot_best_gen_{gen:05}.png')

    evotsc_plot.explain(indiv)

    with open(f'{output_dir}/best_gen_{gen:05}.evotsc', 'wb') as save_file:
        pickle.dump(indiv, save_file)


def write_stats(stats_file, indiv, gen):
    on_genes_A, off_genes_A, on_genes_B, off_genes_B = indiv.summarize()
    stats_file.write(f'{gen},{indiv.fitness},'
                    f'{on_genes_A[0]},{off_genes_A[0]},{on_genes_A[1]},{off_genes_A[1]},{on_genes_A[2]},{off_genes_A[2]},'
                    f'{on_genes_B[0]},{off_genes_B[0]},{on_genes_B[1]},{off_genes_B[1]},{on_genes_B[2]},{off_genes_B[2]}\n')
    stats_file.flush()



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
                                      inversion_prob=inversion_prob)


    population = evotsc.Population(init_indiv=init_indiv,
                                   nb_indivs=nb_indivs,
                                   mutation=mutation)

    stats_file = open(f'{output_dir}/stats.csv', 'w')
    stats_file.write('Gen,Fitness,ABon_A,ABoff_A,Aon_A,Aoff_A,Bon_A,Boff_A,'
                                 'ABon_B,ABoff_B,Aon_B,Aoff_B,Bon_B,Boff_B\n')

    save_indiv(output_dir, init_indiv, 0)

    for gen in range(1, nb_generations+1):
        best_indiv, avg_fit = population.step()

        print(f'Gen {gen}: best fit {best_indiv.fitness:.5}, avg fit {avg_fit:.5}')
        write_stats(stats_file, best_indiv, gen)

        if gen % save_step == 0:
            save_indiv(output_dir, best_indiv, gen)

    stats_file.close()


if __name__ == "__main__":
    main()
