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


def plot_genome(indiv, print_ids=False, name=None):

    # Compute gene positions
    gene_pos = np.zeros(len(indiv.genes), dtype=int)
    cur_pos = 0

    for i_gene, gene in enumerate(indiv.genes):
        gene_pos[i_gene] = cur_pos
        cur_pos += gene.intergene
        #print(f'Position gène {i_gene}: {cur_pos}')
    genome_length = cur_pos

    # Plot
    fig, ax = plt.subplots(figsize=(9,9), dpi=200)

    rect_width = 0.04
    rect_height = 0.1

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    circle = plt.Circle(xy=(0, 0), radius=1, linestyle='-', fill=False)
    ax.add_patch(circle)
    ax.set_axis_off()


    colors = ['tab:blue', 'tab:red', 'tab:green'] # AB: blue, A: red, B: green
    labels = ['AB', 'A', 'B']

    for i_gene, gene in enumerate(indiv.genes):
        pos_angle = 360 * gene_pos[i_gene] / genome_length
        orient_angle = 360 - pos_angle
        pos_rad = np.radians(pos_angle)
        orient_rad = np.radians(orient_angle)

        ## Plot the gene rectangle

        x0 = (1.0 - rect_height / 2.0) * np.sin(pos_rad)
        y0 = (1.0 - rect_height / 2.0) * np.cos(pos_rad)


        if gene.orientation == 0:
            final_width = rect_width
        else:
            final_width = -rect_width


        rect = plt.Rectangle(xy=(x0, y0),
                             width=final_width,
                             height=rect_height,
                             angle=orient_angle, #in degrees anti-clockwise about xy.
                             facecolor=colors[gene.gene_type],
                             edgecolor='black',
                             label=f'Gene {i_gene}')

        ax.add_patch(rect)

        ## Plot the orientation bar and arrow

        # Bar
        x_lin = (1.0 + (np.array([0.5, 1.0])) * rect_height) * np.sin(pos_rad)
        y_lin = (1.0 + (np.array([0.5, 1.0])) * rect_height) * np.cos(pos_rad)

        plt.plot(x_lin, y_lin, color='black', linewidth=1)

        # Arrow
        dx_arr = rect_width * np.cos(pos_rad) / 3.0
        dy_arr = - rect_width * np.sin(pos_rad) / 3.0

        if gene.orientation == 1: # Reverse
            dx_arr, dy_arr = -dx_arr, -dy_arr

        plt.arrow(x_lin[1], y_lin[1], dx_arr, dy_arr, head_width=0.02, color='black')

        ## Print gene ID
        if print_ids:
            plt.text(x=0.92*x0, y=0.92*y0, s=f'{gene.id}', rotation=orient_angle, ha='left', va='bottom',
                     rotation_mode='anchor', fontweight='bold')


    ## Legend
    patches = [mpl.patches.Patch(facecolor=color, edgecolor='black', label=label)
               for color, label in zip(colors, labels)]
    plt.legend(handles=patches, title='Gene type', loc='center')

    line_len = np.pi*indiv.interaction_dist/genome_length
    line_y = -0.3
    plt.plot([-line_len, line_len], [line_y, line_y],
             color='black',
             linewidth=1)
    plt.text(0, line_y - 0.07, 'Gene interaction distance', ha='center')

    plt.show()

    if name:
        plt.savefig(name, dpi=300, bbox_inches='tight')

    plt.close()
