import copy
import numpy as np

# Class that holds all the mutation parameters
class Mutation:
    def __init__(self, intergene_mutation_prob, intergene_mutation_var):
        self.intergene_mutation_prob = intergene_mutation_prob
        self.intergene_mutation_var = intergene_mutation_var

class Gene:
    def __init__(self, intergene, orientation, basal_expression):
        self.intergene = intergene                # Distance to the next gene
        self.orientation = orientation            # Leading or lagging strand
        self.basal_expression = basal_expression  # Initial expression level

    def __repr__(self):
        return f'{self.intergene}, {["LEADING", "LAGGING"][self.orientation]}, {self.basal_expression}'

    # Generate a list of random genes
    @classmethod
    def generate(cls, intergene, nb_genes, default_basal_expression=None):
        genes = []

        for i_gene in range(nb_genes):
            if default_basal_expression is None:
                basal_expression = np.random.random()
            else:
                basal_expression = default_basal_expression
            new_gene = cls(intergene=intergene,
                           orientation=np.random.randint(2),
                           basal_expression=basal_expression)
            genes.append(new_gene)

        return genes


class Individual:
    def __init__(self, genes, interaction_dist, interaction_coef, nb_eval_steps, activation):
        self.genes = genes
        self.nb_genes = len(genes)
        self.interaction_dist = interaction_dist
        self.interaction_coef = interaction_coef
        self.nb_eval_steps = nb_eval_steps
        self.already_evaluated = False
        self.activation = activation


    def __repr__(self):
        gene_pos, total_len = self.compute_gene_positions()
        repr_str = f'length: {total_len}\n'
        for i_gene, gene in enumerate(self.genes):
            repr_str += f'Gene {i_gene}: pos {gene_pos[i_gene]}, intergene {gene}\n'
        return repr_str


    def clone(self):
        new_genes = [copy.copy(gene) for gene in self.genes]
        return Individual(new_genes,
                          self.interaction_dist,
                          self.interaction_coef,
                          self.nb_eval_steps,
                          self.activation)

    ############ Individual evaluation

    def compute_gene_positions(self):
        positions = np.zeros(self.nb_genes, dtype=int)
        cur_pos = 0

        for i_gene, gene in enumerate(self.genes):
            positions[i_gene] = cur_pos
            cur_pos += gene.intergene

        return positions, cur_pos


    def compute_inter_matrix(self):
        inter_matrix = np.zeros((self.nb_genes, self.nb_genes))

        for i in range(self.nb_genes):
            for j in range(self.nb_genes):

                if i == j: # It's the same gene
                    inter_matrix[i, j] = 1.0
                    continue

                pos_1 = self.gene_positions[i]
                pos_2 = self.gene_positions[j]

                ## On veut savoir si le gène 1 est avant le gène 2 ou après
                # Avant : -------1--2-------- ou -2---------------1-
                # Après : -------2--1-------- ou -1---------------2-

                if pos_1 < pos_2: # -------1--2-------- ou -1---------------2-
                    if pos_2 - pos_1 < self.genome_size + pos_1 - pos_2: # -------1--2--------
                        distance = pos_2 - pos_1
                        is_before = True
                    else: # -1---------------2-
                        distance = self.genome_size + pos_1 - pos_2
                        is_before = False

                else: # -------2--1-------- ou -2---------------1-
                    if pos_1 - pos_2 < self.genome_size + pos_2 - pos_1: # -------2--1--------
                        distance = pos_1 - pos_2
                        is_before = False
                    else:
                        distance = self.genome_size + pos_2 - pos_1
                        is_before = True

                ## Orientations relatives
                if ((is_before and self.genes[i].orientation == 0) or
                    (not is_before and self.genes[j].orientation == 1)):
                    sign = +1
                else:
                    sign = -1

                strength = max(1 - distance/self.interaction_dist, 0)

                inter_matrix[i, j] = sign * strength * self.interaction_coef

        return inter_matrix


    def run_system(self, nb_steps):
        temporal_expr = np.zeros((self.nb_genes, nb_steps))

        # Initial values at t = 0
        temporal_expr[:, 0] = np.array([gene.basal_expression for gene in self.genes])

        # Iterate the system
        for t in range(1, nb_steps):
            temporal_expr[:, t] = self.activation(self.inter_matrix @ temporal_expr[:, t-1])

        return temporal_expr


    def compute_fitness(self, expr_levels):
        # On renvoie la somme des valeurs moyennes d'expression sur les 5 derniers pas de temps
        target_steps = 5
        nb_genes, nb_steps = expr_levels.shape
        target = np.ones((nb_genes, target_steps)) * 2 # Target: tous les gènes activés au maximum

        delta = np.sum(np.abs(expr_levels[:, nb_steps-target_steps:] - target))
        fitness = np.exp(-delta)

        return fitness


    def evaluate(self):
        if self.already_evaluated:
            return self.expr_levels, self.fitness

        self.gene_positions, self.genome_size = self.compute_gene_positions()
        self.inter_matrix = self.compute_inter_matrix()

        self.expr_levels = self.run_system(self.nb_eval_steps)
        self.fitness = self.compute_fitness(self.expr_levels)

        self.already_evaluated = True

        return self.expr_levels, self.fitness

    ############ Mutational operators

    def mutate(self, mutation):
        self.mutate_intergene_distances(mutation)
        self.already_evaluated = False


    def mutate_intergene_distances(self, mutation):
        for gene in self.genes:
            # Mutate the intergenic distance
            if np.random.random() < mutation.intergene_mutation_prob:
                intergene_delta = np.random.normal(loc=0, scale=mutation.intergene_mutation_var)
                intergene_delta = np.fix(intergene_delta).astype(int) # Round toward 0

                if gene.intergene + intergene_delta >= 0:
                    gene.intergene += intergene_delta


class Population:
    def __init__(self, init_indiv, nb_indivs, mutation):
        # Individuals
        self.individuals = []
        self.nb_indivs = nb_indivs
        for i_indiv in range(nb_indivs):
            indiv = init_indiv.clone()
            self.individuals.append(indiv)

        # Mutation operators
        self.mutation = mutation


    def evaluate(self):
        for indiv in self.individuals:
            indiv.evaluate()


    def evolve(self, nb_steps):
        self.best_indivs = []
        for t in range(nb_steps):
            # On évalue tous les individus
            fitnesses = np.zeros(self.nb_indivs)
            for i_indiv, indiv in enumerate(self.individuals):
                _, fitness = indiv.evaluate()
                fitnesses[i_indiv] = fitness

            # Sauvegarde du meilleur individu
            best_indiv = self.individuals[np.argmax(fitnesses)]
            self.best_indivs.append(best_indiv.clone())

            # Sélection de l'ancêtre de chaque individu de la nouvelle génération
            total_fitness = np.sum(fitnesses)
            ancestors = np.random.choice(np.arange(self.nb_indivs),
                                         size=self.nb_indivs,
                                         p=fitnesses/total_fitness)

            # Création de la nouvelle génération avec mutation
            new_indivs = []
            for i_new_indiv in range(self.nb_indivs):
                ancestor = self.individuals[ancestors[i_new_indiv]]
                new_indiv = ancestor.clone()
                new_indiv.mutate(self.mutation)
                new_indivs.append(new_indiv)

            self.individuals = new_indivs

            if t % 10 == 0:
                print(f'Time {t}: avg fit {total_fitness/self.nb_indivs}')

        return self.best_indivs
