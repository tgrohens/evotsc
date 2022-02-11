import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import evotsc

label_fontsize=20
tick_fontsize=15
legend_fontsize=15
dpi=300

def plot_expr(indiv, sigma_env, plot_title, plot_name):

    # Plot only environment A
    (temporal_expr, _), fitness = indiv.evaluate(sigma_env, sigma_env)

    nb_genes, nb_steps = temporal_expr.shape

    colors = mpl.cm.get_cmap('viridis', nb_genes)(range(nb_genes))

    plt.figure(figsize=(9, 8), dpi=dpi)

    plt.ylim(-0.05, 1.05)

    for gene in range(nb_genes):
        linestyle = ['solid', 'dashed'][indiv.genes[gene].orientation]
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

    colors = ['tab:blue', 'tab:red', 'tab:green'] # AB: blue, A: red, B: green

    plt.figure(figsize=(9, 8), dpi=dpi)

    ## First subplot: environment A
    plt.subplot(2, 1, 1)
    plt.ylim(-0.05, 1.05)

    for i_gene, gene in enumerate(indiv.genes):
        linestyle = ['solid', 'dashed'][gene.orientation]
        plt.plot(temporal_expr_A[i_gene, :],
                 linestyle=linestyle,
                 linewidth=2,
                 color=colors[gene.gene_type],
                 #alpha=0.25,
                 label=f'Gene {gene.id}')

    plt.grid(linestyle=':')
    #plt.xlabel('Time', fontsize='large')
    plt.ylabel('Expression level', fontsize=label_fontsize)

    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    #plt.legend(loc='center right')
    #plt.title('Environment A')

    ## Second subplot: environment B
    plt.subplot(2, 1, 2)
    plt.ylim(-0.05, 1.05)

    for i_gene, gene in enumerate(indiv.genes):
        linestyle = ['solid', 'dashed'][gene.orientation]
        plt.plot(temporal_expr_B[i_gene, :],
                 linestyle=linestyle,
                 linewidth=2,
                 color=colors[gene.gene_type],
                 #alpha=0.25,
                 label=f'Gene {gene.id}')

    plt.grid(linestyle=':')
    plt.xlabel('Time', fontsize=label_fontsize)
    plt.ylabel('Expression level', fontsize=label_fontsize)

    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    #plt.legend(loc='center right')
    #plt.title('Environment B')

    ## Final stuff
    plt.suptitle(f'{plot_title} fitness: {fitness:.5}')

    plt.tight_layout()

    plt.savefig(plot_name, dpi=dpi, bbox_inches='tight')

    plt.show()

    plt.close()


def explain(indiv, sigma_A, sigma_B):
    (temporal_expr_A, temporal_expr_B), fitness = indiv.evaluate(sigma_A, sigma_B)
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
    gene_pos, genome_length = indiv.compute_gene_positions(include_coding=True)

    # Plot
    pos_rect = [0.1, 0.1, 0.8, 0.8]
    fig = plt.figure(figsize=(9,9), dpi=dpi)
    ax = fig.add_axes(pos_rect)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    circle = plt.Circle(xy=(0, 0), radius=1, linestyle='-', fill=False)
    ax.add_patch(circle)
    ax.set_axis_off()

    gene_type_color = ['tab:blue', 'tab:red', 'tab:green'] # AB, A, B
    gene_types = ['AB', 'A', 'B']

    ## Plot the genes themselves
    for i_gene, gene in enumerate(indiv.genes):
        ## Compute the angles of the boundaries of the gene
        start_pos_deg = 360 * gene_pos[i_gene] / genome_length
        if gene.orientation == evotsc.Orient.LEADING:
            end_pos_deg = 360 * (gene_pos[i_gene] + gene.length - 1) / genome_length
        else:
            end_pos_deg = 360 * (gene_pos[i_gene] - (gene.length - 1)) / genome_length
        orient_angle = 360 - (start_pos_deg + end_pos_deg) / 2
        start_pos_rad = np.radians(start_pos_deg)
        end_pos_rad = np.radians(end_pos_deg)
        mid_pos_rad = (start_pos_rad + end_pos_rad) / 2

        ## Plot the gene rectangle
        rect_width = 2 * np.sin((end_pos_rad - start_pos_rad) / 2.0)
        rect_height = 0.1

        x0 = np.sin(start_pos_rad) - 0.5 * rect_height * np.sin(mid_pos_rad)
        y0 = np.cos(start_pos_rad) - 0.5 * rect_height * np.cos(mid_pos_rad)

        rect = plt.Rectangle(xy=(x0, y0),
                             width=rect_width,
                             height=rect_height,
                             angle=orient_angle, #in degrees anti-clockwise about xy.
                             facecolor=gene_type_color[gene.gene_type],
                             edgecolor='black',
                             label=f'Gene {i_gene}')

        ax.add_patch(rect)

        ## Plot the orientation bar and arrow

        # Bar
        x_lin = np.sin(start_pos_rad) + np.array([0.5, 1.0]) * rect_height * np.sin(mid_pos_rad)
        y_lin = np.cos(start_pos_rad) + np.array([0.5, 1.0]) * rect_height * np.cos(mid_pos_rad)

        ax.plot(x_lin, y_lin, color='black', linewidth=1)

        # Arrow
        dx_arr = rect_width * np.cos(mid_pos_rad) / 3.0
        dy_arr = - rect_width * np.sin(mid_pos_rad) / 3.0

        ax.arrow(x_lin[1], y_lin[1], dx_arr, dy_arr, head_width=0.02, color='black')

        ## Print gene ID
        if print_ids and (i_gene % 5 == 0):
            ha = 'left'
            if gene.orientation == evotsc.Orient.LAGGING:
                ha = 'right'
            ax.text(x=0.92*x0, y=0.92*y0, s=f'{i_gene}', rotation=orient_angle, ha=ha, va='bottom',
                     rotation_mode='anchor')

    ## Plot the legend
    patches = [mpl.patches.Patch(facecolor=color, edgecolor='black', label=label)
               for color, label in zip(gene_type_color, gene_types)]
    ax.legend(handles=patches, title='Gene type', loc='center',
              title_fontsize=15, fontsize=15)

    line_len = np.pi*indiv.interaction_dist/genome_length
    line_y = -0.3
    ax.plot([-line_len, line_len], [line_y, line_y],
             color='black',
             linewidth=1)
    ax.text(0, line_y - 0.07, 'Gene interaction distance', ha='center')

    if name:
        plt.savefig(name, dpi=300, bbox_inches='tight')

    plt.show()

    plt.close()


def plot_genome_and_tsc(indiv,
                        sigma,
                        show_bar=False,
                        print_ids=False,
                        name=None):

    # Compute gene positions
    gene_pos, genome_length = indiv.compute_gene_positions(include_coding=True)

    # Plot
    pos_rect = [0, 0, 1, 1]
    fig = plt.figure(figsize=(9,9), dpi=dpi)
    ax = fig.add_axes(pos_rect)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    circle = plt.Circle(xy=(0, 0), radius=1, linestyle='-', fill=False)
    ax.add_patch(circle)
    ax.set_axis_off()

    ## Plot the genes themselves

    gene_type_color = ['tab:blue', 'tab:red', 'tab:green'] # AB, A, B
    gene_types = ['AB', 'A', 'B']

    for i_gene, gene in enumerate(indiv.genes):
        ## Compute the angles of the boundaries of the gene
        start_pos_deg = 360 * gene_pos[i_gene] / genome_length
        if gene.orientation == evotsc.Orient.LEADING:
            end_pos_deg = 360 * (gene_pos[i_gene] + gene.length - 1) / genome_length
        else:
            end_pos_deg = 360 * (gene_pos[i_gene] - (gene.length - 1)) / genome_length
        orient_angle = 360 - (start_pos_deg + end_pos_deg) / 2
        start_pos_rad = np.radians(start_pos_deg)
        end_pos_rad = np.radians(end_pos_deg)
        mid_pos_rad = (start_pos_rad + end_pos_rad) / 2

        ## Plot the gene rectangle
        rect_width = 2 * np.sin((end_pos_rad - start_pos_rad) / 2.0)
        rect_height = 0.1

        x0 = np.sin(start_pos_rad) - 0.5 * rect_height * np.sin(mid_pos_rad)
        y0 = np.cos(start_pos_rad) - 0.5 * rect_height * np.cos(mid_pos_rad)

        rect = plt.Rectangle(xy=(x0, y0),
                             width=rect_width,
                             height=rect_height,
                             angle=orient_angle, #in degrees anti-clockwise about xy.
                             facecolor=gene_type_color[gene.gene_type],
                             edgecolor='black',
                             label=f'Gene {i_gene}')

        ax.add_patch(rect)

        ## Plot the orientation bar and arrow

        # Bar
        x_lin = np.sin(start_pos_rad) + np.array([0.5, 1.0]) * rect_height * np.sin(mid_pos_rad)
        y_lin = np.cos(start_pos_rad) + np.array([0.5, 1.0]) * rect_height * np.cos(mid_pos_rad)

        ax.plot(x_lin, y_lin, color='black', linewidth=1)

        # Arrow
        dx_arr = rect_width * np.cos(mid_pos_rad) / 3.0
        dy_arr = - rect_width * np.sin(mid_pos_rad) / 3.0

        ax.arrow(x_lin[1], y_lin[1], dx_arr, dy_arr, head_width=0.02, color='black')

        ## Print gene ID
        if print_ids and (i_gene % 5 == 0):
            ha = 'left'
            if gene.orientation == evotsc.Orient.LAGGING:
                ha = 'right'
            ax.text(x=0.92*x0, y=0.92*y0, s=f'{i_gene}',
                    rotation=orient_angle, ha=ha, va='bottom', rotation_mode='anchor',
                    fontsize=15)

    ## Plot local supercoiling along the genome, at the end of the individual's lifecycle
    sc_ax = fig.add_axes(pos_rect, projection='polar', frameon=False)
    sc_ax.set_ylim(0, 1)

    n = 1000  # the number of data points

    # theta values (see
    # https://matplotlib.org/devdocs/gallery/images_contours_and_fields/pcolormesh_grids.html)
    # To have the crisp version: put n+1 in theta and [data] as the 3rd argument of pcolormesh()
    # To have the blurry version: put n in theta and [data, data] ----------------------------
    theta = np.linspace(0, 2 * np.pi, n)
    radius = np.linspace(.6, .72, 2)

    #data = np.array([theta[:-1]]) #np.array([np.random.random(n) * 2 * np.pi])
    positions = np.linspace(0, genome_length, n, dtype=int)
    data = indiv.compute_final_sc_at(sigma, positions)

    min_sc = -1.0
    max_sc = 1.0
    norm = mpl.colors.Normalize(min_sc, max_sc) # Extremum values for the SC level

    if np.min(data) < min_sc or np.max(data) > max_sc:
        raise ValueError(f'SC values out of bounds! min {np.min(data)}, max {np.max(data)}')

    mesh = sc_ax.pcolormesh(theta, radius, [data, data], shading='gouraud',
                            norm=norm, cmap=plt.get_cmap('seismic'))
    sc_ax.set_yticklabels([])
    sc_ax.set_xticklabels([])
    #sc_ax.spines['polar'].set_visible(False)
    sc_ax.set_theta_zero_location('N')
    sc_ax.set_theta_direction('clockwise')

    # Color bar for the SC level
    if show_bar:
        cbar = fig.colorbar(mesh, ax=[ax, sc_ax], shrink=0.7, pad=0.0, location='left')
        cbar.set_label('σ', fontsize=20)
        cbar.ax.invert_yaxis()
        cbar.ax.tick_params(labelsize=15)

    ## Legend: gene types and interaction distance
    patches = [mpl.patches.Patch(facecolor=color, edgecolor='black', label=label)
               for color, label in zip(gene_type_color, gene_types)]
    ax.legend(handles=patches, title='Gene type', loc='center',
              title_fontsize=15, fontsize=15)

    line_len = np.pi*indiv.interaction_dist/genome_length
    line_y = -0.3
    ax.plot([-line_len, line_len], [line_y, line_y],
             color='black',
             linewidth=1)
    ax.text(0, line_y - 0.07, 'Gene interaction distance', ha='center', fontsize=15)

    ## Wrapping up
    if name:
        plt.savefig(name, dpi=300, bbox_inches='tight')

    plt.show()

    plt.close()
