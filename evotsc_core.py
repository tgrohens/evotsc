# File that contains numba-optimized code

from numba import jit

import numpy as np


@jit(nopython=True)
def run_system_numba(nb_genes: int,
                  init_expr: np.ndarray,
                  inter_matrix: np.ndarray,
                  sigma_basal: float,
                  sigma_opt: float,
                  epsilon: float,
                  m: float,
                  sigma_env: float) -> np.ndarray:

    step_size = 0.5
    stop_dist = 1e-7
    max_eval_steps = 200

    temporal_expr = np.zeros((max_eval_steps+1, nb_genes))

    # Initial values at t = 0
    temporal_expr[0, :] = init_expr

    # Iterate the system
    it = 1
    cont = True
    while cont:
        prev_expr = temporal_expr[it-1, :]
        sigma_local = inter_matrix @ prev_expr
        sigma_total = sigma_basal + sigma_local + sigma_env

        promoter_activity = 1.0 / (1.0 + np.exp((sigma_total - sigma_opt)/epsilon))

        # We subtract 1 to rescale between exp(-m) and 1
        iter_expr = np.exp(m * (promoter_activity - 1.0))

        nouv_expr = step_size * iter_expr + (1 - step_size) * prev_expr

        temporal_expr[it, :] = nouv_expr

        # Check if we're done
        dist = np.abs(nouv_expr - prev_expr).sum() / nb_genes

        prev_expr = nouv_expr

        if dist < stop_dist:
            cont = False

        if it == max_eval_steps:
            cont = False
        it += 1

    temporal_expr = temporal_expr[:it, :]

    return temporal_expr

@jit(nopython=True)
def compute_inter_matrix_numba(nb_genes: int,
                               gene_positions: np.ndarray,
                               gene_orientations: np.ndarray,
                               gene_lengths: np.ndarray,
                               genome_size: int,
                               interaction_dist: int,
                               interaction_coef: float) -> np.ndarray:
    inter_matrix = np.zeros((nb_genes, nb_genes))

    for i in range(nb_genes):
        for j in range(nb_genes):

            # We compute the influence of gene 2/j on gene 1/i.

            if i == j:  # It's the same gene
                continue

            # As genes have a non-zero length, the relevant distance is
            # between the middle of gene j and the promoter of gene i.

            if gene_orientations[j] == 0:  # Leading
                pos_j = gene_positions[j] + gene_lengths[j] // 2
            else:  # Lagging
                pos_j = gene_positions[j] - gene_lengths[j] // 2


            pos_1_minus_2 = gene_positions[i] - pos_j
            pos_2_minus_1 = - pos_1_minus_2

            # We want to know whether gene 1 comes before or after gene 2
            # Before: -------1--2-------- or -2---------------1-
            # After:  -------2--1-------- or -1---------------2-

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
            if distance > interaction_dist:
                # inter_matrix[i, j] and inter_matrix[j, i] are already 0.0
                continue

            if i_before_j:
                if gene_orientations[j] == 0: #  j leading: +
                    sign_2_on_1 = +1
                else:
                    sign_2_on_1 = -1
            else:
                if gene_orientations[j] == 1: #  j lagging : +
                    sign_2_on_1 = +1
                else:
                    sign_2_on_1 = -1

            # Here, we know that distance <= self.interaction_dist
            strength = 1.0 - distance / interaction_dist

            inter_matrix[i, j] = sign_2_on_1 * strength * interaction_coef

    # Negative sigma -> more transcription
    inter_matrix = - inter_matrix

    return inter_matrix

@jit(nopython=True)
def compute_fitness_numba(nb_genes: int,
                          expr_levels: np.ndarray,
                          gene_targets: np.ndarray,
                          selection_coef: float) -> float:
    # For each environment, we compute the gap g, which is the mean distance
    # to the optimal phenotype:
    # sum ((gene expression - expected expression) ^ 2) / nb_genes
    # The fitness f is then given by: f = exp(- k g) where k is the
    # selection coefficient.

    expr_levels_A, expr_levels_B = expr_levels
    gene_targets_A, gene_targets_B = gene_targets

    # Measure at the last generation
    gap_A = np.square(expr_levels_A[-1, :] - gene_targets_A).sum() / nb_genes
    gap_B = np.square(expr_levels_B[-1, :] - gene_targets_B).sum() / nb_genes

    fitness = np.exp(- selection_coef * (gap_A + gap_B))

    return fitness
