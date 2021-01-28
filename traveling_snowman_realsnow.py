import rasterio
import numpy as np
import matplotlib.pyplot as plt
from traveling_snowman_synthetic import *


def read_snow_map(raster):
    src = rasterio.open(raster)
    snow = src.read(1)
    return snow


if __name__ == '__main__':
    grid_x = 512
    grid_y = 512
    snow = read_snow_map('camden.tif')
    nodex = list(np.arange(8, 512, 128))
    nodey = list(np.arange(8, 512, 128))
    nodes = make_destinations(nodex, nodey)
    origin = (8, 8)
    thresh = 0.30
    snow_threshed = set_go_nogo_threshold(snow, thresh)
    cost_multiplier = 10.0
    cost_map = compute_cost(snow, snow_threshed, cost_multiplier)
    node_dict, node_names = init_lcps(nodes)
    node_dict = compute_lcps(node_dict, cost_map, snow)
    node_dict = stack_lcps(node_dict, grid_x, grid_y)
    all_lcps = get_all_lcps_all_nodes(node_dict, grid_x, grid_y)
    node_dict, df = init_distance_matrix(node_dict, node_names)
    df = fill_distance_matrix(node_dict, node_names, df)
    cities, tour = init_tsp_tour(nodes)
    tour = solve_tsp(cities, tour, df, 3)
    total_cost, tsp_route, tsp_hop_costs = get_tsp_route(tour, node_names,
                                                        grid_x, grid_y,
                                                        node_dict)
    # plot_snow_and_nodes(snow, nodes, origin)
    # plt.savefig('figs/snow_with_nodes.png',
    #             bbox_inches='tight', dpi=300)
    # plot_snow_thresh_and_nodes(snow, nodes, origin)
    # plt.savefig('figs/snow_masked_with_nodes.png',
    #             bbox_inches='tight', dpi=300)
    # # plot_cost_and_nodes()
    # plt.savefig('figs/snow_cost_surface_with_nodes.png',
    #             bbox_inches='tight', dpi=300)
    # # plot_node_lcps('n1')
    # plt.savefig('figs/snow_n1_lcps.png',
    #             bbox_inches='tight', dpi=300)
    # plot_tsp_route(tsp_route, nodes, tsp_hop_costs, origin)
    # plt.savefig('figs/snow_tsp_solution.png',
    #             bbox_inches='tight', dpi=300)
    plot_tsp_on_snow_thresh(tsp_route, snow_threshed, snow, nodes, origin)
    plt.savefig('figs/snow_tsp_solution_over_snow_thresh.png',
                bbox_inches='tight', dpi=300)
