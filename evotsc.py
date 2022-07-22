import copy
from typing import List, Tuple

import numpy as np

import evotsc_core

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
                 length: int,
                 basal_expression: float,
                 gene_type: int,
                 id: int) -> None:
        self.intergene = intergene                # Distance to the next gene
        self.orientation = orientation            # Leading or lagging strand
        self.length = length                      # Length of the gene
        self.basal_expression = basal_expression  # Initial expression level
        self.gene_type = gene_type                # Should the gene be active in env. A, B, or both
        self.id = id                              # Track genes through inversions

    def __repr__(self) -> str:
        return (f'ID: {self.id:02}, '
                f'length: {self.length}, '
                f'intergene: {self.intergene}, '
                f'{["LEADING", "LAGGING"][self.orientation]}, '
                f'type: {["AB", " A", " B"][self.gene_type]}, '
                f'expr: {self.basal_expression:.3}')

    # Generate a list of random genes
    @classmethod
    def generate(cls,
                 intergene: int,
                 length: int,
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
                # Note: this can be smaller than the minimum expression level
                # (exp(-m)), but this setting is not actually used in the
                # evolutionary runs, so it doesn't really matter.
                basal_expression = rng.random()
            else:
                basal_expression = default_basal_expression
            new_gene = cls(intergene=intergene,
                           orientation=rng.integers(2),
                           length=length,
                           basal_expression=basal_expression,
                           gene_type=gene_types[i_gene],
                           id=i_gene)
            genes.append(new_gene)

        return genes


    def clone(self) -> 'Gene':
        return Gene(intergene=self.intergene,
                    orientation=self.orientation,
                    length=self.length,
                    basal_expression=self.basal_expression,
                    gene_type=self.gene_type,
                    id=self.id)


class Individual:
    def __init__(self,
                 genes: List[Gene],
                 interaction_dist: float,
                 interaction_coef: float,
                 sigma_basal: float,
                 sigma_opt: float,
                 epsilon: float,
                 m: float,
                 selection_coef: int,
                 rng: np.random.Generator = None) -> None:
        self.genes = genes
        self.nb_genes = len(genes)
        self.interaction_dist = interaction_dist
        self.interaction_coef = interaction_coef
        self.sigma_basal = sigma_basal
        self.sigma_opt = sigma_opt
        self.epsilon = epsilon
        self.m = m
        self.selection_coef = selection_coef

        if rng:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

        self.inter_matrix = None
        self.expr_levels = None
        self.fitness = None
        self.already_evaluated = False


    def __repr__(self) -> str:
        gene_pos, total_len = self.compute_gene_positions(include_coding=True)
        repr_str = f'length: {total_len}\n'
        for i_gene, gene in enumerate(self.genes):
            repr_str += f'Gene {i_gene}: pos {gene_pos[i_gene]}, {gene}\n'
        return repr_str


    def clone(self) -> 'Individual':
        new_genes = [gene.clone() for gene in self.genes]
        new_indiv = Individual(new_genes,
                               self.interaction_dist,
                               self.interaction_coef,
                               self.sigma_basal,
                               self.sigma_opt,
                               self.epsilon,
                               self.m,
                               self.selection_coef,
                               self.rng)

        new_indiv.already_evaluated = self.already_evaluated

        if self.already_evaluated:
            new_indiv.inter_matrix = np.copy(self.inter_matrix)
            expr_A, expr_B = self.expr_levels
            new_indiv.expr_levels = np.copy(expr_A), np.copy(expr_B)
            new_indiv.fitness = self.fitness
        else:
            self.inter_matrix = None
            self.expr_levels = None
            self.fitness = None

        return new_indiv

    ############ Individual evaluation

    def compute_gene_positions(self,
                               include_coding: bool) -> Tuple[np.ndarray, int]:

        # include_coding = True is used when displaying genes, and when
        # computing gene interaction distances to compute the SC level.
        # include_coding = False is used in inversion-related code, as it makes
        # it simpler to handle mutations (inversions never fall inside a gene).

        positions = np.zeros(self.nb_genes, dtype=int)
        cur_pos = 0

        for i_gene, gene in enumerate(self.genes):
            positions[i_gene] = cur_pos

            if include_coding:
                # If the gene is lagging, add the gene length minus one to
                # the start position to get the promoter position
                if gene.orientation == 1: # Lagging
                    positions[i_gene] += gene.length - 1

                cur_pos += gene.length

            cur_pos += gene.intergene

        return positions, cur_pos


    def compute_inter_matrix(self) -> np.ndarray:
        gene_positions, genome_size = self.compute_gene_positions(include_coding=True)
        gene_orientations = np.array([gene.orientation for gene in self.genes])
        gene_lengths = np.array([gene.length for gene in self.genes])
        return evotsc_core.compute_inter_matrix_numba(nb_genes=self.nb_genes,
                                                    gene_positions=gene_positions,
                                                    gene_orientations=gene_orientations,
                                                    gene_lengths=gene_lengths,
                                                    genome_size=genome_size,
                                                    interaction_dist=self.interaction_dist,
                                                    interaction_coef=self.interaction_coef)

    # Fixpoint solver with a simple step by step iteration
    def run_system(self, sigma_env: float) -> np.ndarray:

        init_expr = np.array([gene.basal_expression for gene in self.genes])

        return evotsc_core.run_system_numba(nb_genes=self.nb_genes,
                                            init_expr=init_expr,
                                            inter_matrix=self.inter_matrix,
                                            sigma_basal=self.sigma_basal,
                                            sigma_opt=self.sigma_opt,
                                            epsilon=self.epsilon,
                                            m=self.m,
                                            sigma_env=sigma_env)


    def compute_fitness(self):
        expr_levels_A, expr_levels_B = self.expr_levels
        gene_types = np.array([gene.gene_type for gene in self.genes])
        return evotsc_core.compute_fitness_numba(nb_genes=self.nb_genes,
                                                 expr_levels_A=expr_levels_A,
                                                 expr_levels_B=expr_levels_B,
                                                 gene_types=gene_types,
                                                 m=self.m,
                                                 selection_coef=self.selection_coef)

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

        # Exit early if we don't have intergene mutations in this run
        if mutation.intergene_poisson_lam == 0.0:
            return False

        did_mutate = False

        nb_mutations = self.rng.poisson(mutation.intergene_poisson_lam)

        for i_mut in range(nb_mutations):

            intergene_delta = self.rng.normal(loc=0, scale=mutation.intergene_mutation_var)
            intergene_delta = np.fix(intergene_delta).astype(int) # Round toward 0

            # Try genes until we find one where we can perform the indel
            found_gene = False

            while not found_gene:
                i_gene = self.rng.integers(self.nb_genes)
                if self.genes[i_gene].intergene + intergene_delta > 0:
                    self.genes[i_gene].intergene += intergene_delta
                    found_gene = True

            did_mutate = True

        return did_mutate


    def mutate_basal_sc(self, mutation: Mutation) -> bool:

        # Exit early if we don't have SC mutations in this run
        if mutation.basal_sc_mutation_prob == 0.0:
            return False

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
            # Here, we are only looking at intergenic distances, and do not care
            # about gene lengths, so we only count non-coding bases.
            gene_positions, noncoding_size = self.compute_gene_positions(include_coding=False)

            start_pos = self.rng.integers(0, noncoding_size)
            end_pos = self.rng.integers(0, noncoding_size)

            # Inverting between start and end or between end and start is equivalent
            if end_pos < start_pos:
                start_pos, end_pos = end_pos, start_pos

            if self.perform_inversion(gene_positions, start_pos, end_pos):
                did_mutate = True

        return did_mutate


    def perform_inversion(self,
                          gene_positions: np.ndarray,
                          start_pos: int,
                          end_pos: int) -> bool:

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
        new_genes = [gene for gene in self.genes[:start_i + 1]]

        # Perform the actual inversion
        for invert_i in range(end_i - start_i):
            inverted_gene = self.genes[end_i - invert_i]

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
            new_genes += [gene for gene in self.genes[end_i+1:]]

        # Change the intergene of the last gene before the inversion to a + c
        # We do this last because start_i could be -1 if inverting the first gene
        new_genes[start_i].intergene = a + c

        self.genes = new_genes

        return True


    def summarize(self,
                  sigma_A: float,
                  sigma_B: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (temporal_expr_A, temporal_expr_B), fitness = self.evaluate(sigma_A, sigma_B)


        half_expr = (1 + np.exp(- self.m)) / 2

        ### Environment A
        on_genes_A = np.zeros(3, dtype=int)
        off_genes_A = np.zeros(3, dtype=int)
        final_expr_A = temporal_expr_A[-1, :]
        for i_gene, gene in enumerate(self.genes):
            if final_expr_A[i_gene] > half_expr:
                on_genes_A[gene.gene_type] += 1
            else:
                off_genes_A[gene.gene_type] += 1

        ### Environment B
        on_genes_B = np.zeros(3, dtype=int)
        off_genes_B = np.zeros(3, dtype=int)
        final_expr_B = temporal_expr_B[-1, :]
        for i_gene, gene in enumerate(self.genes):
            if final_expr_B[i_gene] > half_expr:
                on_genes_B[gene.gene_type] += 1
            else:
                off_genes_B[gene.gene_type] += 1

        return on_genes_A, off_genes_A, on_genes_B, off_genes_B

    # Compute the final supercoiling level at positions `positions`
    # under external supercoiling `sigma`
    def compute_final_sc_at(self,
                            sigma: float,
                            positions: np.ndarray) -> np.ndarray:

        gene_positions, genome_size = self.compute_gene_positions(include_coding=True)

        nb_pos = len(positions)
        sc_tsc = np.zeros(nb_pos)

        # Run the individual
        if self.inter_matrix is None:
            self.inter_matrix = self.compute_inter_matrix()
        temporal_expr = self.run_system(sigma)
        self.already_evaluated = False # Reset in case the individual is reused
        gene_expr = temporal_expr[-1, :]

        for i_pos, x in enumerate(positions):

            pos_tsc = 0.0

            for i_gene, gene in enumerate(self.genes):

                # We compute the influence of gene i at position x

                # If x is inside gene i, the computation is slightly different
                if ((gene.orientation == 0 and  # Leading
                     x >= gene_positions[i_gene] and
                     x < gene_positions[i_gene] + gene.length) or
                    (gene.orientation == 1 and
                     x >= gene_positions[i_gene] - gene.length + 1 and
                     x < gene_positions[i_gene] + 1)):

                    if gene.orientation == 0:  # Leading
                        pos_inside = (x - gene_positions[i_gene]) / gene.length
                    else:
                        pos_inside = (x - (gene_positions[i_gene] - gene.length + 1)) / gene.length

                    # The negative supercoiling is maximal at the promoter
                    # (pos_inside = 0), null at half the gene
                    # (pos_inside = 1/2), and minimal (ie the positive
                    # supercoiling is maximal) at the other end of the gene
                    # (pos_inside = 1).

                    delta_sc = ((2 * pos_inside - 1) *
                                (1 - gene.length / (2 * self.interaction_dist)) *
                                self.interaction_coef * gene_expr[i_gene])

                    if gene.orientation == 1:  # Lagging
                        delta_sc = -delta_sc

                    pos_tsc += delta_sc

                else:

                    # We use the distance to the middle of the gene to compute
                    # the interaction level.
                    if gene.orientation == 0:  # Leading
                        pos_i = gene_positions[i_gene] + gene.length // 2
                    else:
                        pos_i = gene_positions[i_gene] - gene.length // 2

                    pos_i_minus_x = pos_i - x
                    pos_x_minus_i = - pos_i_minus_x

                    # We want to know whether gene 1/x comes before or after gene 2
                    # Before: -------1--2-------- or -2---------------1-
                    # After:  -------2--1-------- or -1---------------2-

                    if pos_x_minus_i < 0: # -------1--2-------- ou -1---------------2-
                        if pos_i_minus_x < genome_size + pos_x_minus_i: # -------1--2--------
                            distance = pos_i_minus_x
                            x_before_i = True
                        else: # -1---------------2-
                            distance = genome_size + pos_x_minus_i
                            x_before_i = False

                    else: # -------2--1-------- ou -2---------------1-
                        if pos_x_minus_i < genome_size + pos_i_minus_x: # -------2--1--------
                            distance = pos_x_minus_i
                            x_before_i = False
                        else:
                            distance = genome_size + pos_i_minus_x
                            x_before_i = True

                    # Exit early if genes are too far
                    if distance > self.interaction_dist:
                        continue

                    if x_before_i:
                        if gene.orientation == 1: # i lagging: +
                            sign_1_on_x = +1
                        else:
                            sign_1_on_x = -1
                    else:
                        if gene.orientation == 0: # i leading: +
                            sign_1_on_x = +1
                        else:
                            sign_1_on_x = -1

                    # Here, we know that distance <= self.interaction_dist
                    strength = 1.0 - distance/self.interaction_dist

                    # Supercoiling variations are additive
                    delta_sc = sign_1_on_x * strength * self.interaction_coef * gene_expr[i_gene]
                    pos_tsc += delta_sc

            sc_tsc[i_pos] = pos_tsc

        return sc_tsc + sigma + self.sigma_basal


class Population:
    def __init__(self,
                 init_indiv: Individual,
                 nb_indivs: int,
                 mutation: Mutation,
                 sigma_A: float,
                 sigma_B: float,
                 selection_method: str,
                 rng: np.random.Generator = None) -> None:
        # Individuals
        self.individuals = []
        self.nb_indivs = nb_indivs
        for i_indiv in range(nb_indivs):
            indiv = init_indiv.clone()
            self.individuals.append(indiv)

        if rng:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

        # Mutation operators
        self.mutation = mutation

        # Selection
        self.selection_method = selection_method

        # Environment
        self.sigma_A = sigma_A
        self.sigma_B = sigma_B


    def evaluate(self) -> None:
        for indiv in self.individuals:
            indiv.evaluate(self.sigma_A, self.sigma_B)


    def step(self) -> Tuple[Individual, float]:

        # We start from an already-evaluated population, from which we:
        # - select reproducers
        # - create new individuals with mutations
        # - evaluate the new individuals

        # Compute the probability of being the ancestor to an individual of the
        # new population.

        # Probability is directly proportional to fitness
        if self.selection_method == 'fit-prop':
            old_fitnesses = np.array([indiv.fitness for indiv in self.individuals])
            total_fitness = old_fitnesses.sum()
            prob = old_fitnesses/total_fitness

        # Probability is proportional to the rank in the population, sorted by
        # fitness
        elif self.selection_method == 'rank':
            indivs_with_ids = list(zip(self.individuals, range(self.nb_indivs)))

            sorted_indivs = sorted(indivs_with_ids,
                                   key=lambda x : x[0].fitness, reverse=True)

            ranks = np.zeros(self.nb_indivs, dtype=int)
            prob = np.zeros(self.nb_indivs)
            for rank, (_, id) in enumerate(sorted_indivs):
                ranks[id] = rank

            prob = 2 * (self.nb_indivs - ranks) / (self.nb_indivs * (self.nb_indivs + 1))

        # Probability decreases exponentionally with the rank, according to
        # $c \in [0, 1]$. The ranking gets flatter as $c$ gets closer to 1.
        elif self.selection_method == 'exp-rank':
            c = 0.9
            indivs_with_ids = list(zip(self.individuals, range(self.nb_indivs)))

            sorted_indivs = sorted(indivs_with_ids,
                                   key=lambda x : x[0].fitness, reverse=True)

            ranks = np.zeros(self.nb_indivs, dtype=int)
            prob = np.zeros(self.nb_indivs)
            for rank, (_, id) in enumerate(sorted_indivs):
                ranks[id] = rank

            prob = (c - 1) / (np.power(c, self.nb_indivs) - 1) * np.power(c, ranks)

        else:
            raise ValueError('Unknown selection method')

        ancestors = self.rng.choice(np.arange(self.nb_indivs),
                                    size=self.nb_indivs,
                                    p=prob)

        # Création de la nouvelle génération avec mutation et évaluation
        new_indivs = []
        for i_new_indiv in range(self.nb_indivs):
            ancestor = self.individuals[ancestors[i_new_indiv]]
            new_indiv = ancestor.clone()
            new_indiv.mutate(self.mutation)
            new_indiv.evaluate(self.sigma_A, self.sigma_B)
            new_indivs.append(new_indiv)

        self.individuals = new_indivs

        # Meilleur individu et fitness moyenne
        new_fitnesses = np.array([indiv.fitness for indiv in new_indivs])
        best_indiv = new_indivs[np.argmax(new_fitnesses)].clone()
        avg_fit = new_fitnesses.sum() / self.nb_indivs

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
