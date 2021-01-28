# Synthetic Traveling Snowman Problem

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import seaborn as sns
import random
import math
import copy
from skimage import graph


# set random seed for reproducability
np.random.seed(42)


def make_synthetic_grid(xdim, ydim, mu, sigma):
    """Generate a Random Normal Snow Depth Map"""
    rand_norm_snow = np.random.normal(mu, sigma, (xdim, ydim))
    # turn any negative depths to 0
    rand_norm_snow[rand_norm_snow < 0] = 0
    return rand_norm_snow


def make_destinations(row_ixs, col_ixs):
    """Generate list of destinations (xy coordinates)

    Parameters:
    row_ixs (list): row indicies of destinations
    col_ixs (list): col indicies of destinations
    Returns:
    dsts (list): x-y coordinate pairs of destinations
    """
    dsts = list(itertools.product(row_ixs, col_ixs))
    return dsts


def set_go_nogo_threshold(grid, threshold):
    return (grid >= threshold) * grid

def compute_cost(snow_depth, snow_depth_mask, cost_multiplier):

    snow_cost = np.max(snow_depth) - snow_depth
    # costmax1 = np.max(snow_cost)
    mask = (snow_depth_mask == 0)
    snow_cost[mask] = np.max(snow_cost) * cost_multiplier
    return snow_cost


def init_lcps(nodes):
    """Initialize a dictionary to compute Least Cost Paths"""
    node_names = []
    node_dict = {}
    i = 1

    for n in nodes:
        node_str = 'n' + str(i)
        node_names.append(node_str)
        node_dict[node_str] = {}
        node_dict[node_str]['src'] = n
        i += 1

    for kn in node_dict:
        node_dict[kn]['dsts'] = {}
        node_name_copy = node_names.copy()
        node_name_copy.remove(kn)

        for nn in node_name_copy:
            node_dict[kn]['dsts'][nn] = {}

        node_copy = nodes.copy()
        node_copy.remove(node_dict[kn]['src'])

        for d in node_dict[kn]['dsts']:
            node_dict[kn]['dsts'][d]['coords'] = node_dict[d]['src']
    return node_dict, node_names


def compute_lcps(node_dict, cost_map, snow):
    # for each node compute LCPs to all other nodes
    for k in node_dict:
        lcp_start = node_dict[k]['src']
        for d in node_dict[k]['dsts']:
            lcp_stop = node_dict[k]['dsts'][d]['coords']
            indices, weight = graph.route_through_array(cost_map,
                                                        lcp_start,
                                                        lcp_stop,
                                                        fully_connected=True,
                                                        geometric=True)
            node_dict[k]['dsts'][d]['lcp indices'] = indices
            node_dict[k]['dsts'][d]['lcp cost'] = weight
            indices = np.array(indices).T
            path = np.zeros_like(snow)
            path[indices[0], indices[1]] = 1
            node_dict[k]['dsts'][d]['route'] = path
    return node_dict


def stack_lcps(node_dict, grid_x, grid_y):
    # get the lcp for each other node and stack them together in an array
    for k in node_dict:
        lcp_map = np.zeros([grid_x, grid_y])
        for d in node_dict[k]['dsts']:
            for r in node_dict[k]['dsts'][d]:
                lcp_map += node_dict[k]['dsts'][d]['route']
        node_dict[k]['lcps to all other nodes'] = lcp_map
    return node_dict


def get_all_lcps_all_nodes(node_dict, grid_x, grid_y):
    # get all lcps all nodes
    all_lcps = np.zeros([grid_x, grid_y])
    for k in node_dict:
        all_lcps += node_dict[k]['lcps to all other nodes']
    return all_lcps


def init_distance_matrix(node_dict):
    # insert the zero distance for the lcp for each node to itself
    # that will be the diagonal of our distance matrix
    for k in node_dict:
        node_dict[k]['dsts'][k] = {}
        node_dict[k]['dsts'][k]['lcp cost'] = 0.0

    # initialize the distance matrix
    df = pd.DataFrame(node_names)
    for k in node_dict:
        df[k] = 1
    df.set_index(df[0], inplace=True)
    del df[0]
    return node_dict, df


def fill_distance_matrix(node_dict, node_names, df):
    for k in node_names:
        lcp_cost_list = []
        for n in node_dict[k]['dsts']:
            lcp_cost_list.append(node_dict[k]['dsts'][n]['lcp cost'])
        self_dist_0 = lcp_cost_list.pop()

        ix_of_node = node_names.index(k)
        lcp_cost_list.insert(ix_of_node, self_dist_0)
        df.loc[k] = lcp_cost_list
    df.index.names = ['Node']
    return df


def init_tsp_tour(nodes):
    cities = [list(n) for n in nodes]
    # Create a random tour of the order in which to visit the cities
    tour = random.sample(range(len(cities)), len(cities))
    # But always put our origin (16, 16) at the beginning
    tour.pop(tour.index(0))
    tour.insert(0, 0)
    return cities, tour


def compute_tour_distance(tour_order):

    hops = []
    i = 0
    j = 1
    while j < len(tour_order):
        hops.append(df.iat[tour_order[i], tour_order[j]])
        i += 1
        j += 1
    last_hop_home = df.iat[tour_order[-1], tour_order[0]]
    hops.append(last_hop_home)
    tour_distance = sum(hops)
    return tour_distance.round(3)


def solve_tsp(cities, tour, n_sims=100000):
    for temp in np.logspace(0, 5, n_sims)[::-1]:
        [i, j] = sorted(random.sample(range(1, len(cities)), 2))
        new_tour = tour[:i] + tour[j:j + 1] + \
            tour[i + 1:j] + tour[i:i + 1] + tour[j + 1:]
        old_distances = compute_tour_distance(tour)
        new_distances = compute_tour_distance(new_tour)
        if math.exp((old_distances - new_distances) / temp) > random.random():
            tour = copy.copy(new_tour)
    return tour


def get_tsp_route(tour, node_names, grid_x, grid_y, node_dict):
    tsp_order = [node_names[i] for i in tour]
    tsp_route = np.zeros([grid_x, grid_y])
    tsp_hop_costs = []

    i = 0
    while i < len(tsp_order) - 1:
        #hop_start = node_dict[tsp_order[i]]['src']
        #hop_end = node_dict[tsp_order[i]]['dsts'][tsp_order[i + 1]]['coords']
        hop_route = node_dict[tsp_order[i]]['dsts'][tsp_order[i + 1]]['route']
        hop_cost = node_dict[tsp_order[i]
                             ]['dsts'][tsp_order[i + 1]]['lcp cost']

        tsp_route += (hop_route * (i + 2))
        tsp_hop_costs.append(hop_cost)
        i += 1

    tsp_route += (node_dict[tsp_order[-1]]['dsts']
                  [tsp_order[0]]['route'] * (i + 1))
    tsp_hop_costs.append(node_dict[tsp_order[-1]]
                         ['dsts'][tsp_order[0]]['lcp cost'])

    # np.cumsum(tsp_hop_costs)
    total_cost = np.sum(tsp_hop_costs).round(2)
    return total_cost, tsp_route, tsp_hop_costs

############################
# Plotting Functions Below #
############################


def plot_snow_and_nodes():
    cmap = plt.cm.Blues
    textstr = 'Snow Depth: ' + '$\mu=%.2f,  \sigma$=%.2f' % (mu, sigma)
    plt.figure(figsize=(8, 8))
    plt.title(textstr)
    ax = plt.imshow(snow, vmin=0, vmax=snow.max(),
                    cmap=cmap, interpolation='none')
    plt.colorbar()
    plt.scatter([i[0] for i in nodes], [i[1] for i in nodes],
                marker='o', c='m', s=100, label='Node')
    plt.scatter(origin[0], origin[1],
                marker='o', c='y', s=125, label='Origin')
    plt.legend()


def plot_snow_thresh_and_nodes():
    cmap = plt.cm.Blues
    cmap.set_under('sienna')
    textstr = 'Snow Depth: ' + \
        '$\mu=%.2f,  \sigma$=%.2f, threshold=%.2f' % (mu, sigma, thresh)
    plt.figure(figsize=(8, 8))
    plt.title(textstr)
    ax = plt.imshow(snow_threshed, vmin=0.0001, vmax=snow.max(),
                    cmap=cmap, interpolation='none')
    plt.colorbar()
    plt.scatter([i[0] for i in nodes], [i[1] for i in nodes],
                marker='o', c='m', s=100, label='Node')
    plt.scatter(origin[0], origin[1],
                marker='o', c='y', s=125, label='Origin')
    plt.legend()


def plot_cost_and_nodes():

    cmap = plt.cm.Greens
    cmap.set_over('sienna')
    plt.figure(figsize=(8, 8))
    plt.title('Cost Map')
    plt.imshow(cost_map, vmin=0, vmax=(np.max(cost_map) / cost_multiplier),
               cmap=cmap, interpolation='none')
    plt.colorbar()
    plt.scatter([i[0] for i in nodes], [i[1] for i in nodes],
                marker='o', c='m', s=100, label='Node')
    plt.scatter(origin[0], origin[1],
                marker='o', c='y', s=125, label='Origin')
    plt.legend()


def plot_node_lcps(node_key):
    cmap = plt.cm.gray_r
    cmap.set_under('none')
    plt.figure(figsize=(8, 8))
    plt.imshow(node_dict[node_key]['lcps to all other nodes'],
               vmin=0, vmax=1, cmap=cmap, interpolation='none')
    plt.title(node_key + ": LCPs to all other nodes")
    plt.scatter([i[0] for i in nodes], [i[1] for i in nodes],
                marker='o', c='m', s=100, label='Node')
    plt.scatter(origin[0], origin[1],
                marker='o', c='y', s=125, label='Origin')
    plt.legend()


def plot_all_node_lcps():
    cmap = plt.cm.gray_r
    cmap.set_under('none')
    plt.figure(figsize=(8, 8))
    plt.imshow(all_lcps, vmin=0, vmax=1,
               cmap=cmap, interpolation='none')
    plt.scatter([i[0] for i in nodes], [i[1] for i in nodes],
                marker='o', c='m', s=100, label='Node')
    plt.scatter(origin[0], origin[1],
                marker='o', c='y', s=125, label='Origin')
    plt.legend()


def plot_distance_matrix():
    cmap = plt.cm.Greens
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(df, square=True, cmap=cmap)
    ax.set_xlabel('Node')
    ax.set_title('Distance (Cost of LCP) Matrix Between All Nodes')


def plot_tsp_route():
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    ax1 = axes[0]
    cmap = plt.cm.Reds
    cmap.set_under('none')
    ax1.imshow(tsp_route, vmin=1, vmax=49, cmap=cmap)
    cax = fig.add_axes([0.1, 0.1, 0.4, 0.05])
    cax.set_yticks([])
    cbar = plt.colorbar(mappable=ax1.get_children()[-2],
                        orientation='horizontal',
                        ticks=[1, 10, 20, 30, 40, 49], cax=cax)
    cax.set_xticklabels(['start', '10', '20', '30', '40', 'finish'])
    ax1.set_title('TSP Route of LCPs Between Nodes')
    ax1.scatter([i[0] for i in nodes], [i[1] for i in nodes],
                marker='+', c='k', s=250,
                alpha=0.5, label='Node')
    ax1.scatter(origin[0], origin[1],
                marker='*', c='b', s=300, alpha=0.5,
                label='Origin')
    ax2 = axes[1]
    ax2.plot(np.cumsum(tsp_hop_costs))
    ax2.set_ylabel('Cost')
    ax2.set_xlabel('Hop Number')
    ax2.set_title('TSP Cumulative Cost')


def plot_tsp_on_snow_thresh():
    cmap = plt.cm.Blues
    cmap.set_under('sienna')

    textstr = 'Snow Depth: ' + \
        '$\mu=%.2f,  \sigma$=%.2f, threshold=%.2f' % (mu, sigma, thresh)
    plt.figure(figsize=(8, 8))
    plt.title(textstr)

    ax = plt.imshow(snow_threshed, vmin=0.0001, vmax=snow.max(),
                    cmap=cmap, interpolation='none')

    cmap = plt.cm.gray_r
    cmap.set_under('none')
    plt.imshow(tsp_route, interpolation='none', vmin=1, vmax=2, cmap=cmap)
    plt.scatter([i[0] for i in nodes], [i[1] for i in nodes],
                marker='o', c='m', s=100,
                alpha=0.99, label='Node')
    plt.scatter(origin[0], origin[1],
                marker='o', c='y', s=100, alpha=0.99,
                label='Origin')
    plt.legend()
##########################
# End Plotting Functions #
##########################


if __name__ == '__main__':
    grid_x = 128
    grid_y = 128
    mu = 0.5
    sigma = 0.3
    snow = make_synthetic_grid(grid_x, grid_y, mu=mu, sigma=sigma)
    nodex = list(np.arange(16, 128, 16))
    nodey = list(np.arange(16, 128, 16))
    nodes = make_destinations(nodex, nodey)
    origin = (16, 16)
    thresh = 0.30
    snow_threshed = set_go_nogo_threshold(snow, thresh)
    cost_multiplier = 10.0
    cost_map = compute_cost(snow, snow_threshed, cost_multiplier)
    node_dict, node_names = init_lcps(nodes)
    node_dict = compute_lcps(node_dict, cost_map, snow)
    node_dict = stack_lcps(node_dict, grid_x, grid_y)
    all_lcps = get_all_lcps_all_nodes(node_dict, grid_x, grid_y)
    node_dict, df = init_distance_matrix(node_dict)
    df = fill_distance_matrix(node_dict, node_names, df)
    cities, tour = init_tsp_tour(nodes)
    tour = solve_tsp(cities, tour, 100000)
    total_cost, tsp_route, tsp_hop_costs = get_tsp_route(tour, node_names,
                                                         grid_x, grid_y,
                                                         node_dict)
    plot_snow_and_nodes()
    plt.savefig('figs/synthetic_normal_with_nodes.png',
                bbox_inches='tight', dpi=300)
    plot_snow_thresh_and_nodes()
    plt.savefig('figs/synthetic_normal_masked_with_nodes.png',
                bbox_inches='tight', dpi=300)
    plot_cost_and_nodes()
    plt.savefig('figs/cost_surface_with_nodes.png',
                bbox_inches='tight', dpi=300)
    plot_node_lcps('n1')
    plt.savefig('figs/n1_lcps.png',
                bbox_inches='tight', dpi=300)
    plot_tsp_route()
    plt.savefig('figs/tsp_solution.png',
                bbox_inches='tight', dpi=300)
    plot_tsp_on_snow_thresh()
    plt.savefig('figs/tsp_solution_over_snow_thresh.png',
                bbox_inches='tight', dpi=300)
