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

    nb_genes = indiv.nb_genes

    colors = mpl.cm.get_cmap('viridis', nb_genes)(range(nb_genes))

    plt.figure(figsize=(9, 4), dpi=dpi)

    plt.ylim(-0.05, 1.05)

    for gene in range(nb_genes):
        linestyle = ['solid', 'dashed'][indiv.genes[gene].orientation]
        plt.plot(temporal_expr[:, indiv.genes[gene].id],
                 linestyle=linestyle,
                 color=colors[indiv.genes[gene].id],
                 label=f'Gene {indiv.genes[gene].id}')

    plt.grid(linestyle=':')
    plt.xlabel('Iteration steps', fontsize='large')
    plt.ylabel('Expression level', fontsize='large')

    plt.legend(loc='center right')
    plt.title(plot_title + f' fitness: {fitness:.5}')

    plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    plt.show()

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


def _plot_inter_graph(ax, indiv, radius, mid_pos_rad, inter_graph):

    for edge in inter_graph.edges:

        i_gene = edge[0]
        i_target = edge[1]

        x_start = radius * np.sin(mid_pos_rad[i_gene])
        y_start = radius * np.cos(mid_pos_rad[i_gene])
        x_end = radius * np.sin(mid_pos_rad[i_target])
        y_end = radius * np.cos(mid_pos_rad[i_target])

        pos_1_minus_2 = i_gene - i_target
        pos_2_minus_1 = - pos_1_minus_2

        # We want to know whether gene 1 comes before or after gene 2
        # Before: -------1--2-------- or -2---------------1-
        # After:  -------2--1-------- or -1---------------2-
        if pos_1_minus_2 < 0: # -------1--2-------- ou -1---------------2-
            if pos_2_minus_1 < indiv.nb_genes + pos_1_minus_2: # -------1--2--------
                i_before_j = True
            else: # -1---------------2-
                i_before_j = False

        else: # -------2--1-------- ou -2---------------1-
            if pos_1_minus_2 < indiv.nb_genes + pos_2_minus_1: # -------2--1--------
                i_before_j = False
            else:
                i_before_j = True

        if i_before_j:
            connectionstyle='arc3,rad=0.1'
        else:
            connectionstyle='arc3,rad=-0.1'

        if inter_graph[i_gene][i_target]['kind'] == 'activ':
            color = 'tab:green'
        else:
            color = 'tab:red'

        arrow = mpl.patches.FancyArrowPatch((x_start, y_start), (x_end, y_end),
                                            connectionstyle=connectionstyle,
                                            arrowstyle='-|>', color=color, mutation_scale=15)
        ax.add_patch(arrow)


def _plot_gene_ring(fig,
                    indiv,
                    sigma,
                    shift,
                    inter_graph,
                    coloring_type,
                    naming_type,
                    hatched_genes,
                    print_ids,
                    mid_gene_id,
                    id_interval,
                    id_ko,
                    id_central,
                    text_size):

    pos_rect = [0, 0, 1, 1]
    ax = fig.add_axes(pos_rect)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    circle = plt.Circle(xy=(0, 0), radius=1, linestyle='-', fill=False)
    ax.add_patch(circle)
    ax.set_axis_off()

    # Compute gene positions and activation levels
    gene_pos, genome_length = indiv.compute_gene_positions(include_coding=True)

    gene_pos = (gene_pos - shift) % genome_length

    # Boolean array, True if a gene's final expression level is > half the max
    if indiv.inter_matrix is None:
        indiv.inter_matrix = indiv.compute_inter_matrix()
    activated_genes = indiv.run_system(sigma)[-1, :] > (1 + np.exp(- indiv.m)) / 2


    ## Constants
    gene_types = ['AB', 'A', 'B']
    if coloring_type == 'type':
        gene_type_color = ['tab:blue', 'tab:red', 'tab:green']
    elif coloring_type == 'on-off':
        colors = plt.cm.get_cmap('tab20').colors
        #                   AB: blue   A:  red    B: green
        gene_type_color = [[colors[1], colors[7], colors[5]],  # light: off
                           [colors[0], colors[6], colors[4]]]  # normal: on
    elif coloring_type == 'by-id':
        gene_colors = mpl.cm.get_cmap('viridis', indiv.nb_genes)(range(indiv.nb_genes))

    if naming_type == 'alpha':
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    rect_height = 0.1

    ## Compute the angles of the boundaries of each gene
    orient_angle = np.zeros((indiv.nb_genes))
    start_pos_rad = np.zeros((indiv.nb_genes))
    mid_pos_rad = np.zeros((indiv.nb_genes))
    end_pos_rad = np.zeros((indiv.nb_genes))

    for i_gene, gene in enumerate(indiv.genes):
        start_pos_deg = 360 * gene_pos[i_gene] / genome_length
        if gene.orientation == 0:  # Leading
            end_pos_deg = 360 * (gene_pos[i_gene] + gene.length - 1) / genome_length
        else:
            end_pos_deg = 360 * (gene_pos[i_gene] - (gene.length - 1)) / genome_length
        orient_angle[i_gene] = 360 - (start_pos_deg + end_pos_deg) / 2
        start_pos_rad[i_gene] = np.radians(start_pos_deg)
        end_pos_rad[i_gene] = np.radians(end_pos_deg)
        mid_pos_rad[i_gene] = (start_pos_rad[i_gene] + end_pos_rad[i_gene]) / 2

    for i_gene, gene in enumerate(indiv.genes):

        ## Plot the gene rectangle
        rect_width = 2 * np.sin((end_pos_rad[i_gene] - start_pos_rad[i_gene]) / 2.0)

        x0 = np.sin(start_pos_rad[i_gene]) - 0.5 * rect_height * np.sin(mid_pos_rad[i_gene])
        y0 = np.cos(start_pos_rad[i_gene]) - 0.5 * rect_height * np.cos(mid_pos_rad[i_gene])

        if coloring_type == 'type':
            gene_color = gene_type_color[gene.gene_type]
        elif coloring_type == 'on-off':
            gene_color = gene_type_color[activated_genes[i_gene]][gene.gene_type]
        else:
            gene_color = gene_colors[i_gene]

        if i_gene == id_ko: # Override colors and set KO gene to white
            gene_color = 'white'

        hatch = None
        if (hatched_genes is not None) and hatched_genes[i_gene] and (i_gene != id_ko):
            hatch = '..'

        linewidth = 1
        zorder = 1
        if i_gene == id_central or i_gene == id_ko:
            linewidth = 2.5
            zorder = 10

        rect = plt.Rectangle(xy=(x0, y0),
                             width=rect_width,
                             height=rect_height,
                             angle=orient_angle[i_gene], #in degrees anti-clockwise about xy.
                             facecolor=gene_color,
                             edgecolor='black',
                             linewidth=linewidth,
                             hatch=hatch,
                             zorder=zorder,
                             label=f'Gene {i_gene}')

        ax.add_patch(rect)

        ## Plot the orientation bar and arrow

        # Bar
        x_lin = (np.sin(start_pos_rad[i_gene]) +
                 np.array([0.5, 1.0]) * rect_height * np.sin(mid_pos_rad[i_gene]))
        y_lin = (np.cos(start_pos_rad[i_gene]) +
                 np.array([0.5, 1.0]) * rect_height * np.cos(mid_pos_rad[i_gene]))

        ax.plot(x_lin, y_lin, color='black', linewidth=linewidth)

        # Arrow
        dx_arr = rect_width * np.cos(mid_pos_rad[i_gene]) / 3.0
        dy_arr = - rect_width * np.sin(mid_pos_rad[i_gene]) / 3.0

        ax.arrow(x_lin[1], y_lin[1], dx_arr, dy_arr,
                 linewidth=linewidth, head_width=0.02, color='black')

        ## Print gene ID
        if print_ids and ((id_central is None and i_gene % id_interval == 0) or
                          (id_central is not None and i_gene % id_central == 0) or
                          (i_gene == id_ko)):

            if mid_gene_id:
                x_id = np.sin(mid_pos_rad[i_gene]) - 0.5 * rect_height * np.sin(mid_pos_rad[i_gene])
                y_id = np.cos(mid_pos_rad[i_gene]) - 0.5 * rect_height * np.cos(mid_pos_rad[i_gene])
                top_coef = 0.91
                bot_coef = 0.92
            else:
                x_id = x0
                y_id = y0
                top_coef = 0.915
                bot_coef = 0.93

            if naming_type == 'alpha':
                gene_name = letters[i_gene]
            elif naming_type == 'pos':
                gene_name = i_gene
            elif naming_type == 'id':
                gene_name = gene.id

            if orient_angle[i_gene] < 120 or orient_angle[i_gene] > 240:  # Top part
                ha = 'left'
                if gene.orientation == 1 and (not mid_gene_id):  # Lagging
                    ha = 'right'
                ax.text(x=top_coef*x_id, y=top_coef*y_id, s=gene_name, rotation=orient_angle[i_gene],
                        ha=ha, va='bottom', rotation_mode='anchor', fontsize=text_size)
            else:  # Bottom part
                ha = 'right'
                if gene.orientation == 1 and (not mid_gene_id):  # Lagging
                    ha = 'left'
                ax.text(x=bot_coef*x_id, y=bot_coef*y_id, s=gene_name, rotation=orient_angle[i_gene]+180,
                        ha=ha, va='top', rotation_mode='anchor', fontsize=text_size)

    ## Interaction graph
    if inter_graph is not None:
        radius = 0.75 - 0.5 * rect_height
        _plot_inter_graph(ax, indiv, radius, mid_pos_rad, inter_graph)

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
    elif coloring_type == 'by-id':
        draw_legend = False

    if draw_legend:
        ax.legend(handles=patches, title='Gene type', loc='center', ncol=ncol,
                  handletextpad=0.6,
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

def _plot_supercoiling_ring(fig,
                            data,
                            ring_color_type,
                            shift_rad,
                            show_bar,
                            bar_text,
                            text_size):

    ## Plot local supercoiling along the genome, at the end of the individual's lifecycle
    sc_rect = [0, 0, 1, 1]
    sc_ax = fig.add_axes(sc_rect, projection='polar', frameon=False)
    sc_ax.set_ylim(0, 1)

    if ring_color_type == 'tsc':
        cmap = 'seismic'
        min_sc = -0.15
        max_sc = 0.15

    elif ring_color_type == 'delta':
        # Use one of the white -> color colormaps from matplotlib, but make it start from pure white
        cmap_values = plt.get_cmap('Oranges')(np.linspace(0, 1, 50))
        cmap = mpl.colors.LinearSegmentedColormap.from_list(name='my_cmap', colors=['white', *cmap_values])

        min_sc = 0 # Only absolute values are passed
        max_sc = 0.10

    norm = mpl.colors.Normalize(min_sc, max_sc) # Extremum values for the SC level

    if np.min(data) < min_sc or np.max(data) > max_sc:
        print(f'SC values out of bounds! min {np.min(data)}, max {np.max(data)}', file=sys.stderr)

    # theta values (see
    # https://matplotlib.org/devdocs/gallery/images_contours_and_fields/pcolormesh_grids.html)
    # To have the crisp version: put len(data)+1 in theta and [data] as the 3rd argument of pcolormesh()
    # To have the blurry version: put len(data) in theta and [data, data] ----------------------------
    theta = np.linspace(0, 2 * np.pi, len(data)) - shift_rad
    radius = np.linspace(.6, .72, 2)

    mesh = sc_ax.pcolormesh(theta, radius, [data, data], shading='gouraud',
                            norm=norm, cmap=plt.get_cmap(cmap))
    sc_ax.set_yticklabels([])
    sc_ax.set_xticklabels([])
    #sc_ax.spines['polar'].set_visible(False)
    sc_ax.set_theta_zero_location('N')
    sc_ax.set_theta_direction('clockwise')

    # Color bar for the SC level
    if show_bar:
        height = 0.83
        #          [left,  bottom,     width, height]
        cbar_rect = [-0.15, (1 - height)/2, 1, height]
        cbar_ax = fig.add_axes(cbar_rect, frameon=False)
        cbar_ax.set_axis_off()

        cbar = fig.colorbar(mesh, ax=cbar_ax, pad=0.0, location='left')
        if bar_text is None:
            bar_text = '$\sigma_{TSC}$'
        cbar.set_label(bar_text, fontsize=30)
        if ring_color_type == 'tsc':
            cbar.ax.invert_yaxis()
        cbar.ax.tick_params(labelsize=text_size)


def plot_genome_and_tsc(indiv,
                        sigma,
                        shift=0, # Shift everything by `shift` bp: the position at shift bp is on top
                        ring_data=None, # Optionally replace TSC data with user-provided data
                        ring_color_type='tsc', # 'tsc': red-white-blue, 'delta': white-orange
                        show_bar=False,
                        bar_text=None, # Legend for the ring data color bar
                        inter_graph=None, # Plot an interaction graph inside
                        coloring_type='type', # 'type', 'on-off', 'by-id'
                        naming_type='pos', # 'pos', 'alpha', 'id'
                        hatched_genes=None, # Highlight some genes
                        print_ids=False,
                        mid_gene_id=False,
                        id_interval=5,
                        id_ko=None, # special plotting for KO genes
                        id_central=None, # special plotting for subnetworks
                        show_plot=True, # Disable interactive plotting if we're making a lot of plots
                        plot_name=None):

    # Argument sanity checks
    if coloring_type not in ['type', 'on-off', 'by-id']:
        raise ValueError(f'Unknown coloring type "{coloring_type}"')

    if naming_type not in ['pos', 'alpha', 'id']:
        raise ValueError(f'Unknown naming type "{naming_type}"')
    if naming_type == 'alpha' and indiv.nb_genes > 26:
        raise ValueError(f'Trying to plot with letters on an individual with too many genes ({indiv.nb_genes})')

    text_size = 18

    ## Plot
    fig = plt.figure(figsize=(9,9), dpi=dpi)

    # Plot the genes
    _plot_gene_ring(fig=fig,
                    indiv=indiv,
                    sigma=sigma,
                    shift=shift,
                    inter_graph=inter_graph,
                    coloring_type=coloring_type,
                    naming_type=naming_type,
                    hatched_genes=hatched_genes,
                    print_ids=print_ids,
                    mid_gene_id=mid_gene_id,
                    id_interval=id_interval,
                    id_ko=id_ko,
                    id_central=id_central,
                    text_size=text_size)

    _, genome_length = indiv.compute_gene_positions(include_coding=True)
    if ring_data is None:
        # Compute the local supercoiling level
        n = 1000  # the number of data points
        data_positions = np.linspace(0, genome_length, n, dtype=int)
        data = indiv.compute_final_sc_at(sigma, data_positions) - sigma - indiv.sigma_basal
    else:
        # Reuse user data
        data = ring_data

    # Convert the shift from bp to rad
    shift_rad = shift * 2 * np.pi / genome_length
    _plot_supercoiling_ring(fig=fig,
                            data=data,
                            ring_color_type=ring_color_type,
                            shift_rad=shift_rad,
                            show_bar=show_bar,
                            bar_text=bar_text,
                            text_size=text_size)

    ## Wrapping up
    if plot_name:
        plt.savefig(plot_name, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()

    plt.close()
