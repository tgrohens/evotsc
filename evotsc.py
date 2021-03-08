import copy
import numpy as np

# Class that holds all the mutation parameters
class Mutation:
    def __init__(self,
                 intergene_mutation_prob,
                 intergene_mutation_var,
                 inversion_prob):
        self.intergene_mutation_prob = intergene_mutation_prob
        self.intergene_mutation_var = intergene_mutation_var
        self.inversion_prob = inversion_prob

class Gene:
    def __init__(self, intergene, orientation, basal_expression, id):
        self.intergene = intergene                # Distance to the next gene
        self.orientation = orientation            # Leading or lagging strand
        self.basal_expression = basal_expression  # Initial expression level
        self.id = id                              # Track genes through inversions

    def __repr__(self):
        return (f'ID: {self.id}, '
                f'intergene: {self.intergene}, '
                f'{["LEADING", "LAGGING"][self.orientation]}, '
                f'expr: {self.basal_expression:.3}')

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
                           basal_expression=basal_expression,
                           id=i_gene)
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
            repr_str += f'Gene {i_gene}: pos {gene_pos[i_gene]}, {gene}\n'
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
        if self.already_evaluated:
            return self.gene_positions, self.genome_size

        positions = np.zeros(self.nb_genes, dtype=int)
        cur_pos = 0

        for i_gene, gene in enumerate(self.genes):
            positions[i_gene] = cur_pos
            cur_pos += gene.intergene

        self.gene_positions = positions
        self.genome_size = cur_pos

        return self.gene_positions, self.genome_size


    def compute_inter_matrix(self):
        self.compute_gene_positions()
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

        self.inter_matrix = self.compute_inter_matrix()

        self.expr_levels = self.run_system(self.nb_eval_steps)
        self.fitness = self.compute_fitness(self.expr_levels)

        self.already_evaluated = True

        return self.expr_levels, self.fitness

    ############ Mutational operators

    def mutate(self, mutation):
        self.generate_inversion(mutation)
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


    def generate_inversion(self, mutation):
        if np.random.random() < mutation.inversion_prob:
            self.compute_gene_positions()
            start_pos = np.random.randint(0, self.genome_size)
            end_pos = np.random.randint(0, self.genome_size)

            #print(f'Generated inversion: {start_pos} -> {end_pos}')

            # Inverting between start and end or between end and start is equivalent
            if end_pos < start_pos:
                start_pos, end_pos = end_pos, start_pos

            self.perform_inversion(start_pos, end_pos)


    def perform_inversion(self, start_pos, end_pos):
        gene_positions, total_length = self.compute_gene_positions()

        # Dernier gène avant l'inversion
        cur_pos = 0
        for i_gene, gene in enumerate(self.genes):
            if start_pos < cur_pos + gene.intergene:
                if start_pos > cur_pos:
                    start_i = i_gene     # The inversion starts just after the gene
                else:
                    start_i = i_gene - 1 # The inversion starts right at the gene, so start_i is the one before
                break
            cur_pos += gene.intergene

        # start_i can be -1 but Python arrays handle this correctly

        # Dernier gène de l'inversion
        cur_pos = 0
        for i_gene, gene in enumerate(self.genes):
            if end_pos < cur_pos + gene.intergene:
                end_i = i_gene
                break
            cur_pos += gene.intergene

        #print(f'start_i: {start_i}, end_i: {end_i}')

        if start_i == end_i: # Pas de gène à inverser
            return

        # Avant :     start_pos                    end_pos
        # ----[---]------||-----[---]-...-[---]------||-----[---]---
        #    start_i  a      b            end_i   c      d

        # Après :
        # ----[---]------||-----[---]-...-[---]------||-----[---]---
        #    start_i  a      c  end_i             b      d

        if start_i == -1: # We are inverting at position 0, so a == the intergene after the last gene
            a = self.genes[start_i].intergene
        else:
            a = start_pos - gene_positions[start_i]
        b = self.genes[start_i].intergene - a
        c = end_pos - gene_positions[end_i]
        d = self.genes[end_i].intergene - c


        #print(f'start_i: {start_i}, position: {gene_positions[start_i]}')
        #print(f'end_i: {end_i}, position: {gene_positions[end_i]}')
        #print(f'a: {a}, b: {b}, c: {c}, d: {d}')

        # Copy all the genes before the inversion
        new_genes = [copy.copy(gene) for gene in self.genes[:start_i + 1]]

        # Perform the actual inversion
        for invert_i in range(end_i - start_i):
            inverted_gene = copy.copy(self.genes[end_i - invert_i])

            # Get the new intergene
            if invert_i < end_i - start_i - 1:
                inverted_gene.intergene = self.genes[end_i - invert_i - 1].intergene # l'intergène du précédent
            else:
                inverted_gene.intergene = b + d # l'intergène du dernier gène post-inversion est b + d

            # Switch orientations
            inverted_gene.orientation = 1 - inverted_gene.orientation

            new_genes.append(inverted_gene)

        # Wrap up the remaining genes
        if end_i < self.nb_genes:
            new_genes += [copy.copy(gene) for gene in self.genes[end_i+1:]]

        # Change the intergene of the last gene before the inversion to a + c
        # We do this last because start_i could be -1 if inverting the first gene
        new_genes[start_i].intergene = a + c

        self.genes = new_genes


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
