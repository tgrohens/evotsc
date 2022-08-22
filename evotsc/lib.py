import pickle

from . import evotsc

def read_params(rep_dir):

    with open(rep_dir.joinpath('params.txt'), 'r') as params_file:
        param_lines = params_file.readlines()

    params = {}
    for line in param_lines:
        param_name = line.split(':')[0]
        if param_name == 'commit':
            param_val = line.split(':')[1].strip()
        elif param_name == 'neutral':
            param_val = (line.split(':')[1] == True)
        elif param_name == 'selection_method':
            param_val = line.split(':')[1].strip()
        else:
            param_val = float(line.split(':')[1])

        params[param_name] = param_val

    return params


def get_best_indiv(rep_path, gen):

    with open(rep_path.joinpath(f'pop_gen_{gen:06}.evotsc'), 'rb') as save_file:
        pop_rep = pickle.load(save_file)

    pop_rep.evaluate()

    best_fit = 0
    best_indiv = pop_rep.individuals[0]

    try:
        for indiv in pop_rep.individuals:
            if indiv.fitness > best_fit:
                best_fit = indiv.fitness
                best_indiv = indiv
     # In the neutral control, individuals are not evaluated, so there is no
     # fitness field; in that case, just return the first individual
    except AttributeError:
        pass

    return best_indiv

def make_random_indiv(intergene,
                      gene_length,
                      nb_genes,
                      default_basal_expression,
                      interaction_dist,
                      interaction_coef,
                      sigma_basal,
                      sigma_opt,
                      epsilon,
                      m,
                      selection_coef,
                      mutation,
                      rng,
                      nb_mutations=0):

    genes = evotsc.Gene.generate(intergene=intergene,
                                 length=gene_length,
                                 nb_genes=nb_genes,
                                 default_basal_expression=default_basal_expression,
                                 rng=rng)

    indiv = evotsc.Individual(genes=genes,
                              interaction_dist=interaction_dist,
                              interaction_coef=interaction_coef,
                              sigma_basal=sigma_basal,
                              sigma_opt=sigma_opt,
                              epsilon=epsilon,
                              m=m,
                              selection_coef=selection_coef,
                              rng=rng)

    for i_mut in range(nb_mutations):
        indiv.mutate(mutation)

    return indiv
