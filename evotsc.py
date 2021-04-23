import numpy as np
from typing import List, Tuple

# Class that holds all the mutation parameters
class Mutation:
    def __init__(self,
                 intergene_poisson_lam: float = 0.0,
                 intergene_mutation_var: float = 0.0,
                 basal_sc_mutation_prob: float = 0.0,
                 basal_sc_mutation_var: float = 0.0,
                 inversion_poisson_lam: float = 0.0) -> None:
        self.intergene_poisson_lam = intergene_poisson_lam
        self.intergene_mutation_var = intergene_mutation_var
        self.basal_sc_mutation_prob = basal_sc_mutation_prob
        self.basal_sc_mutation_var = basal_sc_mutation_var
        self.inversion_poisson_lam = inversion_poisson_lam

class Gene:
    def __init__(self,
                 intergene: int,
                 orientation: int,
                 basal_expression: float,
                 gene_type: int,
                 id: int) -> None:
        self.intergene = intergene                # Distance to the next gene
        self.orientation = orientation            # Leading or lagging strand
        self.basal_expression = basal_expression  # Initial expression level
        self.gene_type = gene_type                # Should the gene be active in env. A, B, or both
        self.id = id                              # Track genes through inversions

    def __repr__(self) -> str:
        return (f'ID: {self.id}, '
                f'intergene: {self.intergene}, '
                f'{["LEADING", "LAGGING"][self.orientation]}, '
                f'type: {["A & B", "A", "B"][self.gene_type]}, '
                f'expr: {self.basal_expression:.3}')

    # Generate a list of random genes
    @classmethod
    def generate(cls,
                 intergene: int,
                 nb_genes: int,
                 default_basal_expression: float = None,
                 rng: np.random.Generator = None) -> 'Gene':

        if not rng:
            rng = np.random.default_rng()

        genes = []

        # Randomly assign 1/3 of genes to type A, B, and AB respectively
        nb_genes_A = nb_genes // 3
        nb_genes_B = nb_genes // 3
        nb_genes_AB = nb_genes - (nb_genes_A + nb_genes_B)

        gene_types = [0] * nb_genes_AB + [1] * nb_genes_A + [2] * nb_genes_B
        gene_types = rng.permutation(gene_types)

        for i_gene in range(nb_genes):
            if default_basal_expression is None:
                basal_expression = rng.random()
            else:
                basal_expression = default_basal_expression
            new_gene = cls(intergene=intergene,
                           orientation=rng.integers(2),
                           basal_expression=basal_expression,
                           gene_type=gene_types[i_gene],
                           id=i_gene)
            genes.append(new_gene)

        return genes


    def clone(self) -> 'Gene':
        return Gene(intergene=self.intergene,
                    orientation=self.orientation,
                    basal_expression=self.basal_expression,
                    gene_type=self.gene_type,
                    id=self.id)


class Individual:
    def __init__(self,
                 genes: List[Gene],
                 interaction_dist: float,
                 interaction_coef: float,
                 nb_eval_steps: int,
                 sigma_basal: float,
                 sigma_opt: float,
                 epsilon: float,
                 rng: np.random.Generator = None) -> None:
        self.genes = genes
        self.nb_genes = len(genes)
        self.interaction_dist = interaction_dist
        self.interaction_coef = interaction_coef
        self.sigma_basal = sigma_basal
        self.sigma_opt = sigma_opt
        self.epsilon = epsilon
        self.nb_eval_steps = nb_eval_steps

        if rng:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

        self.already_evaluated = False


    def __repr__(self) -> str:
        gene_pos, total_len = self.compute_gene_positions()
        repr_str = f'length: {total_len}\n'
        for i_gene, gene in enumerate(self.genes):
            repr_str += f'Gene {i_gene}: pos {gene_pos[i_gene]}, {gene}\n'
        return repr_str


    def clone(self) -> 'Individual':
        new_genes = [gene.clone() for gene in self.genes]
        new_indiv = Individual(new_genes,
                               self.interaction_dist,
                               self.interaction_coef,
                               self.nb_eval_steps,
                               self.sigma_basal,
                               self.sigma_opt,
                               self.epsilon,
                               self.rng)

        new_indiv.already_evaluated = self.already_evaluated

        if self.already_evaluated:
            new_indiv.inter_matrix = np.copy(self.inter_matrix)
            expr_A, expr_B = self.expr_levels
            new_indiv.expr_levels = np.copy(expr_A), np.copy(expr_B)
            new_indiv.fitness = self.fitness

        return new_indiv

    ############ Individual evaluation

    def compute_gene_positions(self) -> Tuple[int, np.ndarray]:

        positions = np.zeros(self.nb_genes, dtype=int)
        cur_pos = 0

        for i_gene, gene in enumerate(self.genes):
            positions[i_gene] = cur_pos
            cur_pos += gene.intergene

        return positions, cur_pos


    def compute_inter_matrix(self) -> np.ndarray:
        gene_positions, genome_size = self.compute_gene_positions()
        inter_matrix = np.zeros((self.nb_genes, self.nb_genes))

        for i in range(self.nb_genes):
            for j in range(i, self.nb_genes):

                # We compute the influence of gene 2/j on gene 1/i

                if i == j: # It's the same gene
                    inter_matrix[i, j] = 0.0
                    continue

                pos_1_minus_2 = gene_positions[i] - gene_positions[j]
                pos_2_minus_1 = - pos_1_minus_2

                ## On veut savoir si le gène 1 est avant le gène 2 ou après
                # Avant : -------1--2-------- ou -2---------------1-
                # Après : -------2--1-------- ou -1---------------2-

                if pos_1_minus_2 < 0: # -------1--2-------- ou -1---------------2-
                    if pos_2_minus_1 < genome_size + pos_1_minus_2: # -------1--2--------
                        distance = pos_2_minus_1
                        i_before_j = True
                    else: # -1---------------2-
                        distance = genome_size + pos_1_minus_2
                        i_before_j = False

                else: # -------2--1-------- ou -2---------------1-
                    if pos_1_minus_2 < genome_size + pos_2_minus_1: # -------2--1--------
                        distance = pos_1_minus_2
                        i_before_j = False
                    else:
                        distance = genome_size + pos_2_minus_1
                        i_before_j = True

                # Exit early if genes are too far
                if distance > self.interaction_dist:
                    inter_matrix[i, j] = 0.0
                    inter_matrix[j, i] = 0.0
                    continue

                if i_before_j:
                    if self.genes[j].orientation == 0: # j leading: +
                        sign_2_on_1 = +1
                    else:
                        sign_2_on_1 = -1
                    if self.genes[i].orientation == 1: # i lagging : +
                        sign_1_on_2 = +1
                    else:
                        sign_1_on_2 = -1
                else:
                    if self.genes[j].orientation == 1: # j lagging : +
                        sign_2_on_1 = +1
                    else:
                        sign_2_on_1 = -1
                    if self.genes[i].orientation == 0: # i leading : +
                        sign_1_on_2 = +1
                    else:
                        sign_1_on_2 = -1

                # Here, we know that distance <= self.interaction_dist
                strength = 1.0 - distance/self.interaction_dist

                inter_matrix[i, j] = sign_2_on_1 * strength * self.interaction_coef
                inter_matrix[j, i] = sign_1_on_2 * strength * self.interaction_coef

        inter_matrix = -inter_matrix # Negative sigma -> more transcription

        return inter_matrix


    def run_system(self, sigma_env: float) -> np.ndarray:
        temporal_expr = np.zeros((self.nb_genes, self.nb_eval_steps))

        # Initial values at t = 0
        temporal_expr[:, 0] = np.array([gene.basal_expression for gene in self.genes])

        # Iterate the system
        for t in range(1, self.nb_eval_steps):
            sigma_local = self.inter_matrix @ temporal_expr[:, t-1]
            sigma_total = self.sigma_basal + sigma_local + sigma_env
            temporal_expr[:, t] = 1.0 / (1.0 + np.exp((sigma_total - self.sigma_opt)/self.epsilon))

        return temporal_expr


    def compute_fitness(self) -> float:
        # On renvoie la moyenne de (valeur d'expression - target)
        # sur les 5 derniers pas de temps et sur les gènes
        target_steps = 5
        selection_coef = 50

        expr_levels_A, expr_levels_B = self.expr_levels

        target_A = np.zeros(self.nb_genes)
        for i_gene in range(self.nb_genes):
            if self.genes[i_gene].gene_type == 0 or self.genes[i_gene].gene_type == 1:
                target_A[i_gene] = 1.0
        full_target_A = np.repeat([target_A], repeats=target_steps, axis=0).transpose()
        delta_A = np.sum(np.square(expr_levels_A[:, self.nb_eval_steps-target_steps:] - full_target_A)) / (target_steps * self.nb_genes)

        target_B = np.zeros(self.nb_genes)
        for i_gene in range(self.nb_genes):
            if self.genes[i_gene].gene_type == 0 or self.genes[i_gene].gene_type == 2:
                target_B[i_gene] = 1.0
        full_target_B = np.repeat([target_B], repeats=target_steps, axis=0).transpose()
        delta_B = np.sum(np.square(expr_levels_B[:, self.nb_eval_steps-target_steps:] - full_target_B)) / (target_steps * self.nb_genes)

        fitness = np.exp(- selection_coef * (delta_A + delta_B))

        return fitness


    def evaluate(self, sigma_A: float, sigma_B: float) -> Tuple[np.ndarray, float]:
        if self.already_evaluated:
            return self.expr_levels, self.fitness

        self.inter_matrix = self.compute_inter_matrix()

        self.expr_levels = self.run_system(sigma_A), self.run_system(sigma_B)
        self.fitness = self.compute_fitness()

        self.already_evaluated = True

        return self.expr_levels, self.fitness

    ############ Mutational operators

    def mutate(self, mutation: Mutation) -> None:
        did_mutate = False

        if self.generate_inversions(mutation):
            did_mutate = True

        if self.mutate_basal_sc(mutation):
            did_mutate = True

        if self.mutate_intergene_distances(mutation):
            did_mutate = True

        if did_mutate:
            self.already_evaluated = False


    def mutate_intergene_distances(self, mutation: Mutation) -> bool:
        did_mutate = False

        nb_mutations = self.rng.poisson(mutation.intergene_poisson_lam)

        for i_mut in range(nb_mutations):

            i_gene = self.rng.integers(self.nb_genes)
            intergene_delta = self.rng.normal(loc=0, scale=mutation.intergene_mutation_var)
            intergene_delta = np.fix(intergene_delta).astype(int) # Round toward 0

            # We don't allow an intergene of 0 because otherwise the genes can't be separated anymore
            if self.genes[i_gene].intergene + intergene_delta > 0:
                self.genes[i_gene].intergene += intergene_delta
                did_mutate = True

        return did_mutate


    def mutate_basal_sc(self, mutation: Mutation) -> bool:
        did_mutate = False

        if self.rng.random() < mutation.basal_sc_mutation_prob:
            basal_sc_delta = self.rng.normal(loc=0, scale=mutation.basal_sc_mutation_var)

            self.sigma_basal += basal_sc_delta

            did_mutate = True

        return did_mutate


    def generate_inversions(self, mutation: Mutation) -> bool:
        did_mutate = False

        nb_inversions = self.rng.poisson(mutation.inversion_poisson_lam)

        for inv in range(nb_inversions):
            gene_positions, genome_size = self.compute_gene_positions()

            start_pos = self.rng.integers(0, genome_size)
            end_pos = self.rng.integers(0, genome_size)

            # Inverting between start and end or between end and start is equivalent
            if end_pos < start_pos:
                start_pos, end_pos = end_pos, start_pos

            if self.perform_inversion(start_pos, end_pos):
                did_mutate = True

        return did_mutate


    def perform_inversion(self, start_pos: int, end_pos: int) -> bool:
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

         # Pas de gène à inverser
        if (start_i == end_i) or (start_i == -1 and end_i == self.nb_genes - 1):
            return False

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
        new_genes = [gene.clone() for gene in self.genes[:start_i + 1]]

        # Perform the actual inversion
        for invert_i in range(end_i - start_i):
            inverted_gene = self.genes[end_i - invert_i].clone()

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
            new_genes += [gene.clone() for gene in self.genes[end_i+1:]]

        # Change the intergene of the last gene before the inversion to a + c
        # We do this last because start_i could be -1 if inverting the first gene
        new_genes[start_i].intergene = a + c

        self.genes = new_genes

        return True


    def summarize(self,
                  sigma_A: float,
                  sigma_B: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (temporal_expr_A, temporal_expr_B), fitness = self.evaluate(sigma_A, sigma_B)

        ### Environment A
        on_genes_A = np.zeros(3, dtype=int)
        off_genes_A = np.zeros(3, dtype=int)
        for i_gene, gene in enumerate(self.genes):
            if temporal_expr_A[i_gene, self.nb_eval_steps-1] > 0.5:
                on_genes_A[gene.gene_type] += 1
            else:
                off_genes_A[gene.gene_type] += 1

        ### Environment B
        on_genes_B = np.zeros(3, dtype=int)
        off_genes_B = np.zeros(3, dtype=int)
        for i_gene, gene in enumerate(self.genes):
            if temporal_expr_B[i_gene, self.nb_eval_steps-1] > 0.5:
                on_genes_B[gene.gene_type] += 1
            else:
                off_genes_B[gene.gene_type] += 1

        return on_genes_A, off_genes_A, on_genes_B, off_genes_B


class Population:
    def __init__(self,
                 init_indiv: Individual,
                 nb_indivs: int,
                 mutation: Mutation,
                 sigma_A: float,
                 sigma_B: float,
                 rng: np.random.Generator = None) -> None:
        # Individuals
        self.individuals = []
        self.nb_indivs = nb_indivs
        for i_indiv in range(nb_indivs):
            indiv = init_indiv.clone()
            self.individuals.append(indiv)

        # Mutation operators
        if rng:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

        self.mutation = mutation

        # Environment
        self.sigma_A = sigma_A
        self.sigma_B = sigma_B


    def evaluate(self) -> None:
        for indiv in self.individuals:
            indiv.evaluate(self.sigma_A, self.sigma_B)


    def step(self) -> Tuple[Individual, float]:

        # On évalue tous les individus
        fitnesses = np.zeros(self.nb_indivs)
        for i_indiv, indiv in enumerate(self.individuals):
            _, fitness = indiv.evaluate(self.sigma_A, self.sigma_B)
            fitnesses[i_indiv] = fitness

        # Sauvegarde du meilleur individu
        best_indiv = self.individuals[np.argmax(fitnesses)].clone()

        # Sélection de l'ancêtre de chaque individu de la nouvelle génération
        total_fitness = np.sum(fitnesses)
        ancestors = self.rng.choice(np.arange(self.nb_indivs),
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

        avg_fit = total_fitness/self.nb_indivs

        return best_indiv, avg_fit

    # Perform a step without selection
    def neutral_step(self) -> Tuple[Individual, float]:

        # Pick random ancestors
        ancestors = self.rng.choice(np.arange(self.nb_indivs), size=self.nb_indivs)

        # New generation
        new_indivs = []
        for i_new_indiv in range(self.nb_indivs):
            ancestor = self.individuals[ancestors[i_new_indiv]]
            new_indiv = ancestor.clone()
            new_indiv.mutate(self.mutation)
            new_indivs.append(new_indiv)

        self.individuals = new_indivs

        return self.individuals[0], 0.0
