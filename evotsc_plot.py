import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


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
    plt.xlabel('Iteration steps', fontsize='large')
    plt.ylabel('Expression level', fontsize='large')

    plt.legend(loc='center right')
    plt.title(plot_title + f' fitness: {fitness:.5}')

    plt.savefig(plot_name, dpi=300, bbox_inches='tight')

    plt.close()


def plot_expr_AB(indiv, sigma_A, sigma_B, color_by_type=True, plot_title='', plot_name=''):

    (temporal_expr_A, temporal_expr_B), fitness = indiv.evaluate(sigma_A, sigma_B)

    type_colors = ['tab:blue', 'tab:red', 'tab:green'] # AB: blue, A: red, B: green
    gene_colors = mpl.cm.get_cmap('viridis', indiv.nb_genes)(range(indiv.nb_genes))


    plt.figure(figsize=(9, 8), dpi=dpi)

    ## First subplot: environment A
    plt.subplot(2, 1, 1)
    plt.ylim(-0.05, 1.05)

    for i_gene, gene in enumerate(indiv.genes):
        linestyle = 'solid' if gene.orientation == 0 else 'dashed'
        if color_by_type:
            color = type_colors[gene.gene_type]
        else:
            color = gene_colors[gene.id]
        plt.plot(temporal_expr_A[:, i_gene],
                 linestyle=linestyle,
                 linewidth=2,
                 color=color,
                 #alpha=0.25,
                 label=f'Gene {gene.id}')

    x_min, x_max = plt.xlim()
    t_max = temporal_expr_A.shape[0] - 1
    half_expr = (1+np.exp(-indiv.m))/2
    plt.hlines(half_expr, 0, t_max, linestyle=':', linewidth=2,
           color='tab:pink', label='Half transcription level')
    plt.xlim(x_min, x_max)

    plt.grid(linestyle=':')
    #plt.xlabel('Iteration steps', fontsize='large')
    plt.ylabel('Expression level', fontsize=label_fontsize)

    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    #plt.legend(loc='center right')
    #plt.title('Environment A')

    ## Second subplot: environment B
    plt.subplot(2, 1, 2)
    plt.ylim(-0.05, 1.05)

    for i_gene, gene in enumerate(indiv.genes):
        linestyle = 'solid' if gene.orientation == 0 else 'dashed'
        if color_by_type:
            color = type_colors[gene.gene_type]
        else:
            color = gene_colors[gene.id]
        plt.plot(temporal_expr_B[:, i_gene],
                 linestyle=linestyle,
                 linewidth=2,
                 color=color,
                 #alpha=0.25,
                 label=f'Gene {gene.id}')

    x_min, x_max = plt.xlim()
    half_expr = (1+np.exp(-indiv.m))/2
    t_max = temporal_expr_B.shape[0] - 1
    plt.hlines(half_expr, 0, t_max, linestyle=':', linewidth=2,
           color='tab:pink', label='Half transcription level')
    plt.xlim(x_min, x_max)

    plt.grid(linestyle=':')
    plt.xlabel('Iteration steps', fontsize=label_fontsize)
    plt.ylabel('Expression level', fontsize=label_fontsize)

    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    #plt.legend(loc='center right')
    #plt.title('Environment B')

    ## Final stuff

    plt.tight_layout()
    plt.savefig(plot_name + '.pdf', dpi=dpi, bbox_inches='tight')
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


def plot_genome_and_tsc(indiv,
                        sigma,
                        show_bar=False,
                        coloring_type='type',
                        use_letters=False,
                        print_ids=False,
                        id_interval=5,
                        plot_name=None):

    # Compute gene positions and activation levels
    gene_pos, genome_length = indiv.compute_gene_positions(include_coding=True)

    # Boolean array, True if a gene's final expression level is > half the max
    if indiv.inter_matrix is None:
        indiv.inter_matrix = indiv.compute_inter_matrix()
    activated_genes = indiv.run_system(sigma)[-1, :] > (1 + np.exp(- indiv.m)) / 2

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

    text_size = 18

    ## Plot the genes themselves

    if coloring_type == 'type':
        gene_type_color = ['tab:blue', 'tab:red', 'tab:green']
    elif coloring_type == 'on-off':
        colors = plt.cm.get_cmap('tab20').colors
        #                   AB: blue   A:  red    B: green
        gene_type_color = [[colors[1], colors[7], colors[5]],  # light: off
                           [colors[0], colors[6], colors[4]]]  # normal: on
    else:
        gene_colors = mpl.cm.get_cmap('viridis', indiv.nb_genes)(range(indiv.nb_genes))
    gene_types = ['AB', 'A', 'B']

    if use_letters:
        if indiv.nb_genes > 26:
            raise ValueError('Trying to plot with letters on an individual with too many genes')
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    for i_gene, gene in enumerate(indiv.genes):
        ## Compute the angles of the boundaries of the gene
        start_pos_deg = 360 * gene_pos[i_gene] / genome_length
        if gene.orientation == 0:  # Leading
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

        if coloring_type == 'type':
            gene_color = gene_type_color[gene.gene_type]
        elif coloring_type == 'on-off':
            gene_color = gene_type_color[activated_genes[i_gene]][gene.gene_type]
        else:
            gene_color = gene_colors[i_gene]

        rect = plt.Rectangle(xy=(x0, y0),
                             width=rect_width,
                             height=rect_height,
                             angle=orient_angle, #in degrees anti-clockwise about xy.
                             facecolor=gene_color,
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
        if print_ids and (i_gene % id_interval == 0):
            if use_letters:
                gene_name = letters[i_gene]
            else:
                gene_name = i_gene
            if orient_angle < 120 or orient_angle > 240:  # Top part
                ha = 'left'
                if gene.orientation == 1:  # Lagging
                    ha = 'right'
                ax.text(x=0.915*x0, y=0.915*y0, s=gene_name, rotation=orient_angle,
                        ha=ha, va='bottom', rotation_mode='anchor', fontsize=text_size)
            else:  # Bottom part
                ha = 'right'
                if gene.orientation == 1:  # Lagging
                    ha = 'left'
                ax.text(x=0.93*x0, y=0.93*y0, s=gene_name, rotation=orient_angle+180,
                        ha=ha, va='top', rotation_mode='anchor', fontsize=text_size)


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
    data = indiv.compute_final_sc_at(sigma, positions) - sigma - indiv.sigma_basal

    min_sc = -0.15
    max_sc = 0.15
    norm = mpl.colors.Normalize(min_sc, max_sc) # Extremum values for the SC level

    if np.min(data) < min_sc or np.max(data) > max_sc:
        print(f'SC values out of bounds! min {np.min(data)}, max {np.max(data)}', file=sys.stderr)

    mesh = sc_ax.pcolormesh(theta, radius, [data, data], shading='gouraud',
                            norm=norm, cmap=plt.get_cmap('seismic'))
    sc_ax.set_yticklabels([])
    sc_ax.set_xticklabels([])
    #sc_ax.spines['polar'].set_visible(False)
    sc_ax.set_theta_zero_location('N')
    sc_ax.set_theta_direction('clockwise')

    # Color bar for the SC level
    if show_bar:
        height = 0.83
        #          [left,  bottom,    width,  height]
        pos_rect = [-0.15, (1 - height)/2, 1, height]
        cbar_ax = fig.add_axes(pos_rect, frameon=False)
        cbar_ax.set_axis_off()

        cbar = fig.colorbar(mesh, ax=cbar_ax, pad=0.0, location='left')
        cbar.set_label('$\sigma_{TSC}$', fontsize=30)
        cbar.ax.invert_yaxis()
        cbar.ax.tick_params(labelsize=text_size)

    ## Legend: gene types and interaction distance

    if coloring_type == 'type':
        draw_legend = True
        patches = ([mpl.patches.Patch(facecolor=color, edgecolor='black', label=label)
            for color, label in zip(gene_type_color, gene_types)])
        ncol = 1

    elif coloring_type == 'on-off':
        draw_legend = True
        patches = ([mpl.patches.Patch(facecolor=color, edgecolor='black', label=label + ' (on)')
                    for color, label in zip(gene_type_color[1], gene_types)] +
                   [mpl.patches.Patch(facecolor=color, edgecolor='black', label=label + ' (off)')
                    for color, label in zip(gene_type_color[0], gene_types)])
        ncol = 2
    else:
        draw_legend = False

    if draw_legend:
        ax.legend(handles=patches, title='Gene type', loc='center', ncol=ncol,
                  handletextpad=0.6, #columnspacing=1.0,
                  title_fontsize=text_size, fontsize=text_size)

    line_len = np.pi*indiv.interaction_dist/genome_length
    if draw_legend:
        line_y = -0.3
    else:
        line_y = -0.1
    ax.plot([-line_len, line_len], [line_y, line_y],
             color='black',
             linewidth=1)
    ax.text(0, line_y - 0.07, 'Gene interaction distance', ha='center', fontsize=text_size)

    ## Wrapping up
    if plot_name:
        plt.savefig(plot_name, dpi=300, bbox_inches='tight')

    plt.show()

    plt.close()
