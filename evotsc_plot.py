import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_expr(indiv, sigma_env, plot_title, plot_name):

    # Plot only environment A
    (temporal_expr, _), fitness = indiv.evaluate(sigma_env, sigma_env)

    nb_genes, nb_steps = temporal_expr.shape

    colors = mpl.cm.get_cmap('viridis', nb_genes)(range(nb_genes))

    plt.figure(figsize=(9, 8), dpi=200)

    plt.ylim(-0.05, 1.05)

    for gene in range(nb_genes):
        linestyle = 'solid' if indiv.genes[gene].orientation == 0 else 'dashed'
        plt.plot(temporal_expr[indiv.genes[gene].id, :],
                 linestyle=linestyle,
                 color=colors[indiv.genes[gene].id],
                 label=f'Gene {indiv.genes[gene].id}')

    plt.grid(linestyle=':')
    plt.xlabel('Time', fontsize='large')
    plt.ylabel('Expression level', fontsize='large')

    plt.legend(loc='center right')
    plt.title(plot_title + f' fitness: {fitness:.5}')

    plt.savefig(plot_name, dpi=300, bbox_inches='tight')

    plt.close()


def plot_expr_AB(indiv, sigma_A, sigma_B, plot_title, plot_name):

    (temporal_expr_A, temporal_expr_B), fitness = indiv.evaluate(sigma_A, sigma_B)

    colors_A = ['tab:green', 'tab:green', 'tab:red'] # A & B or A are valid
    colors_B = ['tab:green', 'tab:red', 'tab:green'] # A & B or B are valid

    plt.figure(figsize=(9, 6), dpi=200)

    ## First subplot: environment A
    plt.subplot(2, 1, 1)
    plt.ylim(-0.05, 1.05)

    for gene in range(indiv.nb_genes):
        linestyle = 'solid' if indiv.genes[gene].orientation == 0 else 'dashed'
        plt.plot(temporal_expr_A[indiv.genes[gene].id, :],
                 linestyle=linestyle,
                 color=colors_A[indiv.genes[gene].gene_type],
                 alpha=0.25,
                 label=f'Gene {indiv.genes[gene].id}')

    plt.grid(linestyle=':')
    #plt.xlabel('Time', fontsize='large')
    plt.ylabel('Expression level', fontsize='large')

    #plt.legend(loc='center right')
    plt.title('Environment A')

    ## Second subplot: environment B
    plt.subplot(2, 1, 2)
    plt.ylim(-0.05, 1.05)

    for gene in range(indiv.nb_genes):
        linestyle = 'solid' if indiv.genes[gene].orientation == 0 else 'dashed'
        plt.plot(temporal_expr_B[indiv.genes[gene].id, :],
                 linestyle=linestyle,
                 color=colors_B[indiv.genes[gene].gene_type],
                 alpha=0.25,
                 label=f'Gene {indiv.genes[gene].id}')

    plt.grid(linestyle=':')
    plt.xlabel('Time', fontsize='large')
    plt.ylabel('Expression level', fontsize='large')

    #plt.legend(loc='center right')
    plt.title('Environment B')

    ## Final stuff
    plt.suptitle(f'{plot_title} fitness: {fitness:.5}')

    plt.tight_layout()

    plt.savefig(plot_name, dpi=300, bbox_inches='tight')

    plt.close()


def explain(indiv, sigma_A, sigma_B):
    (temporal_expr_A, temporal_expr_B), fitness = indiv.evaluate(sigma_A, sigma_B)
    nb_eval_steps = indiv.nb_eval_steps
    on_genes_A, off_genes_A, on_genes_B, off_genes_B = indiv.summarize(sigma_A, sigma_B)

    print(f'Fitness: {fitness:.5}')

    ### Environment A
    print('Environment A')
    print(f'  A & B genes: {on_genes_A[0]} on, {off_genes_A[0]} off')
    print(f'  A genes:     {on_genes_A[1]} on, {off_genes_A[1]} off')
    print(f'  B genes:     {on_genes_A[2]} on, {off_genes_A[2]} off')

    ### Environment B
    print('Environment B')
    print(f'  A & B genes: {on_genes_B[0]} on, {off_genes_B[0]} off')
    print(f'  A genes:     {on_genes_B[1]} on, {off_genes_B[1]} off')
    print(f'  B genes:     {on_genes_B[2]} on, {off_genes_B[2]} off')

