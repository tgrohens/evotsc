import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_expr(indiv, plot_title, plot_name):

    temporal_expr, fitness = indiv.evaluate()

    nb_genes, nb_steps = temporal_expr.shape

    colors = mpl.cm.get_cmap('viridis', nb_genes)(range(nb_genes))

    plt.figure(figsize=(9, 8), dpi=200)

    plt.ylim(-0.1, 2.1)

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


def plot_expr_AB(indiv, plot_title, plot_name):

    (temporal_expr_A, temporal_expr_B), fitness = indiv.evaluate()

    colors_A = ['tab:green', 'tab:green', 'tab:red'] # A & B or A are valid
    colors_B = ['tab:green', 'tab:red', 'tab:green'] # A & B or B are valid

    plt.figure(figsize=(9, 6), dpi=200)

    ## First subplot: environment A
    plt.subplot(2, 1, 1)
    plt.ylim(-0.1, 2.1)

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
    plt.ylim(-0.1, 2.1)

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


def explain(indiv):
    (temporal_expr_A, temporal_expr_B), fitness = indiv.evaluate()
    nb_eval_steps = indiv.nb_eval_steps

    print(f'Fitness: {fitness:.5}')

    ### Environment A
    print('Environment A')
    on_genes = np.zeros(3, dtype=int)
    off_genes = np.zeros(3, dtype=int)
    for i_gene, gene in enumerate(indiv.genes):
        if temporal_expr_A[i_gene, nb_eval_steps-1] > 1:
            on_genes[gene.gene_type] += 1
        else:
            off_genes[gene.gene_type] += 1
    print(f'  A & B genes: {on_genes[0]} on, {off_genes[0]} off')
    print(f'  A genes:     {on_genes[1]} on, {off_genes[1]} off')
    print(f'  B genes:     {on_genes[2]} on, {off_genes[2]} off')

    ### Environment B
    print('Environment B')
    on_genes = np.zeros(3, dtype=int)
    off_genes = np.zeros(3, dtype=int)
    for i_gene, gene in enumerate(indiv.genes):
        if temporal_expr_B[i_gene, nb_eval_steps-1] > 1:
            on_genes[gene.gene_type] += 1
        else:
            off_genes[gene.gene_type] += 1
    print(f'  A & B genes: {on_genes[0]} on, {off_genes[0]} off')
    print(f'  A genes:     {on_genes[1]} on, {off_genes[1]} off')
    print(f'  B genes:     {on_genes[2]} on, {off_genes[2]} off')
