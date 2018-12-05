import random as rd
import functools as ft
import itertools as it
import operator as op
import collections as col
import numpy as np
from enum import Enum
import datetime
from decimal import Decimal, getcontext
import pickle
from os import listdir
from os.path import isfile, join
import copy
import cv2


# ---------------------------------------------------------------------------------
# auxiliary functions

class EdgeType(Enum):
    BRIDGE = 0
    LINK = 1


def uint8_array(rows):
    return np.array(rows).astype(np.uint8)


def time_print(msg):
    print('[' + str(datetime.datetime.now()) + ']', msg)


def overlay_classified_edges(image, edge_dictionary, edge_scores):
    image_copy = cv2.cvtColor(copy.deepcopy(image) * 200, cv2.COLOR_GRAY2RGB)

    for edge in edge_dictionary.keys():
        u, v = edge
        # color = (rd.randint(50, 255), rd.randint(50, 255), rd.randint(50, 255))
        if tuple([u, v]) in edge_scores.keys():
            if edge_scores[tuple([u, v])] == EdgeType.LINK:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
        elif tuple([v, u]) in edge_scores.keys():
            if edge_scores[tuple([v, u])] == EdgeType.LINK:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
        else:
            color = (0, 0, 255)
            # color = (0, 0, 127)
        # cv2.circle(image, point, radius, random_color, cv2.FILLED)
        for point in edge_dictionary[edge]:
            image_copy[point] = color

    return image_copy


def overlay_edges(image, edge_list, color=None):
    image_copy = copy.deepcopy(image)
    # random_color = (100, 156, 88)
    if color is None:
        random_color = (rd.randint(50, 255), rd.randint(50, 255), rd.randint(50, 255))
    else:
        random_color = color
    for point in edge_list:
        # cv2.circle(image, point, radius, random_color, cv2.FILLED)
        image_copy[point] = random_color
    return image_copy


def overlay_images(image, mask, vertexes_list=None, radius=1):
    if vertexes_list is None:
        image = image - mask

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        mask[:, :, 0] = 0
        mask[:, :, 1] = 0

        res = cv2.addWeighted(image, 1, mask, 1, 0)

        # cv2.imshow("res", res)
        # cv2.imshow("mask", mask)
        # cv2.waitKey()

        return res
    else:
        # switch row col coordinates
        # this to be used if cv2.circle is used
        # vertexes_list = [(c, r) for r, c in vertexes_list]
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # random_color = (100, 156, 88)
        for point in vertexes_list:
            random_color = (rd.randint(0, 100), rd.randint(100, 200), rd.randint(0, 255))
            # cv2.circle(image, point, radius, random_color, cv2.FILLED)
            image[point] = random_color

        return image


def save_image_like(image_like, pixel_list, name):
    image = np.zeros_like(image_like)
    result = list(filter(lambda px: False if px[0] < 0 or px[0] + 1 > image_like.shape[0] or px[1] < 0 or
                                            px[1] + 1 > image_like.shape[1] else True, pixel_list))
    for x in result:
        image[x] = 255
    res = overlay_images(image_like * 255, image)
    cv2.imwrite(name + ".png", res)


def view_image_like(image_like, pixel_list, name):
    image = np.zeros_like(image_like)
    for x in pixel_list:
        image[x] = 255
    res = overlay_images(image_like * 255, image)
    cv2.imshow(name, res)
    cv2.waitKey()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------------
# returns a median pixel in a collection of pixels
# attempt to estimate the medial pixel as follows
# for each pixel we find the pixels 'above' it and to the 'left' of it -> they increase a counter
# for each same pixel we find the pixels 'below' it and to the 'right of it -> they decrease a counter
# the closer the value to 0 the more 'centered' the pixel is
# if more than one has same minimal value, we return the first one
def median_pixel(pixels):
    before_x = [sum(map(lambda p1: 1 if p1[0] > p2[0] else 0, pixels)) for p2 in pixels]
    after_x = [sum(map(lambda p1: -1 if p1[0] < p2[0] else 0, pixels)) for p2 in pixels]
    before_y = [sum(map(lambda p1: 1 if p1[1] > p2[1] else 0, pixels)) for p2 in pixels]
    after_y = [sum(map(lambda p1: -1 if p1[1] < p2[1] else 0, pixels)) for p2 in pixels]

    median_score = min(list(map(lambda x: abs(sum(x)), zip(before_x, after_x, before_y, after_y))))
    return pixels[np.argmin(median_score)]


# ---------------------------------------------------------------------------------
# 'close' a junction to a rectangle form. if the rectangle of the junction has missing pixels
# we add them as well as long as they are part of the ridge matrix
def close_junctions(junctions_mask, ridge_mask):
    # cv2.imwrite("before.png", overlay_images(ridge_mask * 255, junctions_mask * 255))
    candidates_list = set(map(tuple, np.argwhere(ridge_mask > 0)))
    n_labels, labels = cv2.connectedComponents(junctions_mask, connectivity=8)
    all_points = set()
    for i in range(1, n_labels):
        # for each label we retrieve its list of pixels
        one_junction_pixels = set(map(tuple, np.argwhere(labels == i)))
        # for junctions smaller than 5, we add a radius of 2 to the center pixel to increase the junction size
        if len(one_junction_pixels) < 7:
            median = median_pixel(list(one_junction_pixels))

            one_junction_points = set(it.product(range(median[0] - 2, median[0] + 2),
                                                 range(median[1] - 2, median[1] + 2)))
        # otherwise, we 'close' the junction by adding to the junction non-marked pixels
        # in the bounding square region of the junction pixels
        else:
            # find range = to make it square
            x_values = list(map(lambda pixel: pixel[0], one_junction_pixels))
            y_values = list(map(lambda pixel: pixel[1], one_junction_pixels))
            one_junction_points = set(it.product(range(np.min(x_values), np.max(x_values) + 1),
                                                 range(np.min(y_values), np.max(y_values) + 1)))
        all_points = all_points.union(one_junction_points)

    for point in set.intersection(set(all_points), candidates_list):
        junctions_mask[point] = 1
        candidates_list.remove(point)

    # cv2.imwrite("after.png", overlay_images(ridge_mask*255, junctions_mask*255))
    return junctions_mask


# ---------------------------------------------------------------------------------
# mark junction pixels using kernels and the ridge matrix
def mark_junction_pixels(binary_image):
    def rotate(mat):
        return mat, mat.T, np.flipud(mat), np.fliplr(mat.T)

    junctions = map(uint8_array, [
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
    ])

    rotated_mats = (rotated for mat in junctions for rotated in rotate(mat))

    junctions_mask = ft.reduce(op.or_, map(ft.partial(cv2.erode, binary_image), rotated_mats),
                               np.zeros_like(binary_image))
    # we need to close junctions -> find min_x min_y max_x max_y then add ridge pixels to it
    # cv2.imwrite("before.png", overlay_images(binary_image*255, junctions_mask*255))
    return close_junctions(junctions_mask, binary_image)


# ---------------------------------------------------------------------------------
# extract local maxima pixels
def calculate_local_maxima_mask(image):
    base = list(map(uint8_array, [
        [
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ],
        [
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
        ],
        [
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ],
    ]))

    kernels = [mat for mat in base + [mat.T for mat in base]]

    local_maxima_result = (image > cv2.dilate(image, kernel)
                           for kernel in kernels)

    return ft.reduce(op.or_, local_maxima_result).astype(np.uint8)


# ---------------------------------------------------------------------------------
# vertex extraction
def get_vertexes(ridges_matrix, junction_pixels_mask):

    # label connected components - each connected component is a junction
    n_components, labels = cv2.connectedComponents(junction_pixels_mask, connectivity=8)
    # for each junction we find the vertex pixel - the one with highest magnitude
    vertexes_mask = np.zeros(junction_pixels_mask.shape, np.uint8)
    vertexes_list = []
    vertexes_dictionary = {}
    for i in range(1, n_components):
        # for each label we retrieve its list of pixels
        pixels_list_i = np.argwhere(labels == i)
        # we find the index of the pixel in the list with the highest magnitude
        max_index = np.argmax(list(map(lambda pixel: ridges_matrix[tuple(pixel)], pixels_list_i)))
        # we return its coordinates
        co_ordinates = tuple(pixels_list_i[max_index])
        # modify vertexes list and mask accordingly
        vertexes_list.append(co_ordinates)
        vertexes_dictionary[co_ordinates] = [tuple(x) for x in pixels_list_i]
        vertexes_mask[co_ordinates] = 1

    return vertexes_dictionary, vertexes_list, labels, vertexes_mask


# ---------------------------------------------------------------------------------
# retrieves m_connected pixels that are part of the edge pixels
# to be used for the bfs algorithm
def m_connected_candidates(pixel, edge_pixels):
    def add_offset(offset):
        return tuple(map(op.add, pixel, offset))

    four_connected = list(filter(lambda p: add_offset(p) in edge_pixels, [(1, 0), (0, 1), (-1, 0), (0, -1)]))

    diagonally_connected = list(filter(lambda p: add_offset(p) in edge_pixels,
                                       [(1, 1), (-1, 1), (-1, -1), (1, -1)]))

    uniquely_diagonally_connected = list(filter(lambda p: {(0, p[1]), (p[0], 0)}.isdisjoint(set(four_connected)),
                                                diagonally_connected))

    return [add_offset(offset) for offset in four_connected + uniquely_diagonally_connected]


# ---------------------------------------------------------------------------------
# returns for edge (u,v) its shortest m-connected list of pixels from pixel u to pixel v
def m_edge_bfs(start, end, edge_list):
    visited = set()
    to_visit = col.deque([start])
    edges = col.deque()
    done = False
    while not done and to_visit:
        current = to_visit.popleft()
        visited.add(current)
        candidates = [v for v in m_connected_candidates(current, edge_list)
                      if v not in visited and v not in to_visit]
        for vertex in candidates:
            edges.append([current, vertex])
            to_visit.append(vertex)
            if vertex == end:
                done = True
    # find path from end -> start
    final_edges = [end]
    current = end
    while current != start:
        sub_edges = list(filter(lambda item: item[1] == current, edges))
        one_edge = sub_edges.pop()
        final_edges.append(one_edge[0])
        current = one_edge[0]
    final_edges.append(start)
    return final_edges


# ---------------------------------------------------------------------------------
# some vertexes have one edge only, we remove the edge and the vertex from the graph
def remove_one_edge_vertexes(edge_dictionary):
    changed = False
    # remove all vertexes that have one edge only and their edge as well from the graph
    done = False
    while not done:
        all_vertexes = []
        for element in [item for item in edge_dictionary.keys()]:
            all_vertexes.extend(element)
        counted_edges = col.Counter(all_vertexes)
        if 1 not in counted_edges.values():
            done = True
        else:
            # find the vertex that has one edge
            # find first vertex with two edges
            element = tuple()
            for e in counted_edges:
                if counted_edges[e] == 1:
                    element = e
                    break
            # we find the relevant u for the edge (u,v)
            edge_for_element = [item for item in edge_dictionary.keys() if element in item]
            # we find the group of pixels of the edge, and remove from dictionary
            item_0 = edge_for_element[0]
            edge_dictionary.pop(item_0)
            changed = True

    return changed, edge_dictionary


# ---------------------------------------------------------------------------------
# if a vertex has two edges, then it should not be a vertex in the graph
# both of its edges are merged as a new edge and this vertex is removed
def merge_two_edge_vertexes(edge_dictionary):
    changed = False
    done = False
    while not done:
        all_vertexes = []
        for element in [item for item in edge_dictionary.keys()]:
            all_vertexes.extend(element)
        counted_edges = col.Counter(all_vertexes)
        # we check for each vertex in how many edges it participates in
        # each key has two vertexes of an edge (start, end)
        # for each vertex we count the number of times it is found in the edge list
        # if the vertex v has two exactly, we find the relevant u1 and u2 for the edges (u1,v) and (u2,v)
        if 2 not in counted_edges.values():
            done = True
        else:
            # find first vertex with two edges
            element = tuple()
            for e in counted_edges:
                if counted_edges[e] == 2:
                    element = e
                    break
            # we find the relevant u1 and u2 for the edges (u1,v) and (u2,v)
            edges_for_element = [item for item in edge_dictionary.keys() if element in item]
            flat_edges_for_element = []
            for e in edges_for_element:
                flat_edges_for_element.extend(e)
            # find new edge vertexes
            new_edge_vertexes = tuple([item for item in flat_edges_for_element if item != element])
            # for each edge, we find its group of pixels, and remove from dictionary
            item_0 = edges_for_element[0]
            item_1 = edges_for_element[1]
            first_edge = edge_dictionary.pop(item_0)
            second_edge = edge_dictionary.pop(item_1)
            # combine unique edge pixels together
            # and make v edge pixel and set (u1,u2) the new edge of vertexes u1 and u2
            edge_dictionary[new_edge_vertexes] = list(dict.fromkeys(tuple(first_edge + second_edge + [element])))
            changed = True

    return changed, edge_dictionary


# ---------------------------------------------------------------------------------
# clean graph up by removing 1-edge vertexes, and merging 2-edge vertexes
def clean_graph(edge_dictionary):
    # print(stats)
    time_print('graph cleanup...')
    # for each vertex, if it takes part of two edges exactly, we combine the two edges as one
    # and add the vertex as a new edge pixel
    # done iteratively until no more vertexes found that have two edges exactly
    # before_ridge_mask = cv2.cvtColor(np.zeros_like(ridges_mask), cv2.COLOR_GRAY2RGB)
    # for edge_list in edge_dictionary.values():
    #    before_ridge_mask = overlay_edges(before_ridge_mask, edge_list)
    # cv2.imwrite('graph_edges_result_before.png', before_ridge_mask)
    # merges edges of vertexes that have two edges

    # iteratively apply merging and removal operations
    # once after another, until graph stabilizes
    changed, edge_dictionary = merge_two_edge_vertexes(edge_dictionary)
    changed, edge_dictionary = remove_one_edge_vertexes(edge_dictionary)
    while changed:
        changed, edge_dictionary = merge_two_edge_vertexes(edge_dictionary)
        if changed:
            changed, edge_dictionary = remove_one_edge_vertexes(edge_dictionary)
    # after_ridge_mask = cv2.cvtColor(np.zeros_like(ridges_mask), cv2.COLOR_GRAY2RGB)
    #for edge_list in edge_dictionary.values():
    #    after_ridge_mask = overlay_edges(after_ridge_mask, edge_list)
    # cv2.imshow('result_after', after_ridge_mask)
    # cv2.imwrite('graph_edges_result_after.png', after_ridge_mask)
    # cv2.imwrite('left.png', left)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return edge_dictionary


# ---------------------------------------------------------------------------------
# edge extraction
def get_edges(ridges_mask, junction_pixels_mask, vertexes_list):

    all_junction_pixels_set = set(map(tuple, np.argwhere(junction_pixels_mask != 0)))
    # rgb_ridge_mask = cv2.cvtColor(ridges_mask, cv2.COLOR_GRAY2RGB)
    # original_ridge_mask = cv2.cvtColor(ridges_mask, cv2.COLOR_GRAY2RGB)
    left = cv2.cvtColor(np.zeros_like(ridges_mask), cv2.COLOR_GRAY2RGB)
    # we remove the junctions from the ridge mask
    # this way we disconnect them from each other
    edges_mask = ridges_mask - junction_pixels_mask
    # each connected component is an edge in the graph
    n_edges, labels = cv2.connectedComponents(edges_mask, connectivity=8)
    # cv2.imwrite('0_edges_mask.png', edges_mask*255)
    # add to a dictionary - edge number, and its list of pixels
    edge_dictionary = {}
    edge_set = set()
    # 0 vertexes, 1 vertexes, 2 vertexes, 3 or more
    # stats = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # 8-neighborhood offsets for a pixel
    neighborhood = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    for i in range(1, n_edges):
        # reset junction list
        junction_pixels_set = all_junction_pixels_set
        # for each label we retrieve its list of pixels
        edge_pixels = list(map(tuple, np.argwhere(labels == i)))
        # start_edge_size = len(edge_pixels)

        # iteratively - add in junction pixels that are nearby to the current edge
        # new pixels are added to the edge, then we do the same (increasing edge width/length by 1)
        # new pixels come only from junction pixels
        # this way the edge covers the vertexes as well
        empty = False

        # save_image_like(ridges_mask, edge_pixels, str(i) + "_edge_mask_" + str(j))
        # Note: edges that have one vertex are discarded
        while not empty:
            # add-in candidate junctions

            # calculate values of the pixels using the offsets for all edge_pixels of edge i
            pixels_with_neighbors = set(tuple(map(lambda o: (o[0][0] + o[1][0], o[0][1] + o[1][1]),
                                                  it.product(edge_pixels, neighborhood))))
            # save_image_like(ridges_mask, pixels_with_neighbors, str(i) + '_pixels_with_neighbors')
            # junction pixels are added back
            candidate_edge_pixels = list(set.intersection(pixels_with_neighbors, junction_pixels_set))
            # print('candidate_edge_pixels=', len(candidate_edge_pixels))
            # filter these pixels from junction list
            junction_pixels_set = set(filter(lambda edge_pixel: edge_pixel not in candidate_edge_pixels,
                                             junction_pixels_set))
            # save_image_like(ridges_mask, vertexes_set, str(i) + "_vertexes_set_after_" + str(j))

            # view_image_like(ridges_mask, vertexes_set, "vertexes_set_after")
            # stopping criteria - once we can't add more pixels to the edge from junction pixels, we stop
            # print(len(candidate_edge_pixels))
            if len(candidate_edge_pixels) == 0:
                empty = True
            else:
                edge_pixels += candidate_edge_pixels
            # save_image_like(ridges_mask, edge_pixels, str(i) + "_edge_mask_" + str(j))
        # once this edge is complete - we find its vertexes
        start_end_vertexes = list(set.intersection(set(edge_pixels), set(vertexes_list)))
        # if an edge has more than two vertexes -> it can be split into multiple edges where each edge has two vertexes
        # we do that in disjoint manner for the edges, for each two vertexes we ensure no other vertexes are found
        # in the bounding rectangle then find its pixels
        # we do that for each two vertexes u,v in the vertex list for the original edge
        if len(start_end_vertexes) > 2:
            disjoint_product_vertexes = col.deque()
            for traverse_i in range(0, len(start_end_vertexes)):
                for traverse_j in range(0, len(start_end_vertexes)):
                    if traverse_i > traverse_j:
                        disjoint_product_vertexes.append((start_end_vertexes[traverse_i],
                                                          start_end_vertexes[traverse_j]))
            for v1, v2 in disjoint_product_vertexes:
                    top_x, top_y = tuple(map(min, zip(v1, v2)))
                    bottom_x, bottom_y = tuple(map(max, zip(v1, v2)))
                    has_inside = ft.reduce(op.or_, map(lambda v: top_x < v[0] < bottom_x and top_y < v[1] < bottom_y,
                                                       start_end_vertexes))
                    # if no other vertexes are found inside the bounding rectangle of the two vertexes
                    # then these vertexes are an edge in our graph
                    # we find its edge pixels and add them to the graph
                    if not has_inside:
                        m_adjacent_edge_pixels = m_edge_bfs(v1, v2, edge_pixels)
                        # rgb_ridge_mask = overlay_edges(rgb_ridge_mask, m_adjacent_edge_pixels)
                        edge_dictionary[tuple([v1, v2])] = m_adjacent_edge_pixels
                        # stats[2] += 1
        elif len(start_end_vertexes) == 2:
            # then apply m-adjacency to find the shortest slimmest version of the edge from u to v
            m_adjacent_edge_pixels = m_edge_bfs(start_end_vertexes[0], start_end_vertexes[1], edge_pixels)
            # rgb_ridge_mask = overlay_edges(rgb_ridge_mask, m_adjacent_edge_pixels)
            # add to dictionary instead of edge_pixels
            edge_dictionary[tuple(start_end_vertexes)] = m_adjacent_edge_pixels
            edge_set.update(edge_pixels)
            # stats[2] += 1

    return edge_dictionary


# ---------------------------------------------------------------------------------
# ridge extraction
def ridge_extraction(image_preprocessed):
    # apply distance transform then normalize image for viewing
    dist_transform = cv2.distanceTransform(image_preprocessed, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # normalize distance transform to be of values [0,1]
    normalized_dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    # extract local maxima pixels -- "ridge pixels"
    dist_maxima_mask = calculate_local_maxima_mask(normalized_dist_transform)
    # retrieve the biggest connected component only
    dist_maxima_mask_biggest_component = np.zeros_like(dist_maxima_mask)

    for val in np.unique(dist_maxima_mask)[1:]:
        mask = np.uint8(dist_maxima_mask == val)
        labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        dist_maxima_mask_biggest_component[labels == largest_label] = val
    # extract local maxima pixels magnitude values from the distance transform
    dist_maxima = np.multiply(dist_maxima_mask_biggest_component, dist_transform)

    return dist_maxima_mask_biggest_component, dist_maxima


# ---------------------------------------------------------------------------------
# document pre processing
def pre_process(path):
    # load image as gray-scale,
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # convert to binary using otsu binarization
    image = cv2.threshold(image, 0, 1, cv2.THRESH_OTSU)[1]
    # add white border around image of size 29
    white_border_added = cv2.copyMakeBorder(image, 29, 29, 29, 29, cv2.BORDER_CONSTANT, None, 1)
    # on top of that add black border of size 1
    black_border_added = cv2.copyMakeBorder(white_border_added, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, 0)
    # cv2.imwrite('black_border_added.png', black_border_added*255)
    return black_border_added


# ---------------------------------------------------------------------------------
# calculates angle between three points, result in radians
# using Decimal for increased precision
def calculate_abs_angle(u, v, w):
    getcontext().prec = 28

    u_x, u_y = u
    v_x, v_y = v
    w_x, w_y = w

    x1 = (u_x - v_x).item()
    y1 = (u_y - v_y).item()
    x2 = (w_x - v_x).item()
    y2 = (w_y - v_y).item()

    dot = Decimal(x1 * x2 + y1 * y2)
    norma_1 = Decimal(x1 * x1 + y1 * y1).sqrt()
    norma_2 = Decimal(x2 * x2 + y2 * y2).sqrt()

    val = dot / (norma_1 * norma_2)

    return np.abs(np.arccos(float(val)))


# ---------------------------------------------------------------------------------
# assistance function - calculates for edge (u, v) how close its v junction to the shape of T
def calculate_junction_score(u, v, edges_dictionary, ridges_mask):
    v_x, v_y = v
    edges_of_v = []
    # find all edges in graph that have vertex v as part of it
    candidates = [e for e in edges_dictionary.keys() if v in e]
    edges_pixel_list = dict()
    for candidate in candidates:
        edges_pixel_list[candidate] = edges_dictionary[candidate]

    # for each candidate, we ensure that at least one edge is not v and not u, otherwise it might be u,v !
    for e in candidates:
        # in some cases an edge is a circle, begins and ends at the same vertex
        # these cases need to be ignored
        first, second = e
        if first != v and first != u:
            edges_of_v.append(first)
        if second != v and second != u:
            edges_of_v.append(second)
    # if we have less than two edges for v after filtering, then v does not have two edges
    # this means that we can't calculate its score, we return a failure message
    # print('u,v', u, v)
    # print('edges_of_v', edges_of_v)
    if len(edges_of_v) < 2:
        return None
    # we need to find thw two closest edges to point u from v
    # we calculate the distance between vertex u and each vertex e_i of candidate edge (v, e_i)
    distances_of_v = []
    for edge_of_v in edges_of_v:
        e_x, e_y = edge_of_v
        distances_of_v.append(np.sqrt(np.power((v_x - e_x), 2) + np.power((v_y - e_y), 2)))

    # retrieve two closest vertexes to vertex v
    min_idx = np.argmin(distances_of_v)
    max_idx = np.argmax(distances_of_v)

    w1 = edges_of_v[min_idx]
    distances_of_v[min_idx] = distances_of_v[max_idx] * 2
    min_idx = np.argmin(distances_of_v)

    w2 = edges_of_v[min_idx]
    junction_pixels = dict()
    if tuple([u, v]) in edges_pixel_list.keys():
        junction_pixels[tuple([u, v])] = edges_pixel_list[tuple([u, v])]
    else:
        junction_pixels[tuple([u, v])] = edges_pixel_list[tuple([v, u])]

    if tuple([v, w1]) in edges_pixel_list.keys():
        junction_pixels[tuple([v, w1])] = edges_pixel_list[tuple([v, w1])]
    else:
        junction_pixels[tuple([v, w1])] = edges_pixel_list[tuple([w1, v])]

    if tuple([v, w2]) in edges_pixel_list.keys():
        junction_pixels[tuple([v, w2])] = edges_pixel_list[tuple([v, w2])]
    else:
        junction_pixels[tuple([v, w2])] = edges_pixel_list[tuple([w2, v])]
    # now we have three edges: (u, v), (v, w1), (v, w2)
    # calculate the angles between each two edges
    uv_vw1 = calculate_abs_angle(u, v, w1)
    uv_vw2 = calculate_abs_angle(u, v, w2)
    w1v_vw2 = calculate_abs_angle(w1, v, w2)
    # we find the junction score by finding how far the 2nd cos(degree) is from 0
    scores = []
    bridge_threshold = np.pi / 5.0
    # checks whether an edge is a bridge or a link
    # if its a T junction then the angle between v,w1 and v,w2 is the largest out of three
    # in addition if the two angles between u,v and v,w1 and u,v and v,w2 equals PI then
    # it's a T junction and our edge is a bridge, thus using pi + pi/4 as upper bound and
    # pi - pi/4 as lower bound for junction classification
    if w1v_vw2 >= uv_vw1 and w1v_vw2 >= uv_vw2 \
            and np.pi + bridge_threshold >= uv_vw1 + uv_vw2 >= np.pi - bridge_threshold:
        scores.append(((u, v), EdgeType.BRIDGE, uv_vw1 + uv_vw2))
    else:
        scores.append(((u, v), EdgeType.LINK, max(uv_vw1, uv_vw2)))
    return scores


# ---------------------------------------------------------------------------------
# classify edges, bridge or link
def classify_edges(edges_dictionary, ridges_mask):
    edge_scores = []
    edges_dictionary_keys = edges_dictionary.keys()
    for edge in edges_dictionary_keys:
        u, v = edge
        if u == v:
            continue
        junction_score = calculate_junction_score(u, v, edges_dictionary, ridges_mask)
        if junction_score is not None:
            edge_scores.extend(junction_score)
            junction_score = calculate_junction_score(v, u, edges_dictionary, ridges_mask)
        if junction_score is not None:
            edge_scores.extend(junction_score)

    final_scores = dict()
    # TODO remove collection of edges filtered from edge_scores
    # TODO improves performance, maybe fixes some issues..
    done = False
    while not done:
        edge = edge_scores[0]
        # print('edge_scores=', edge_scores)
        # print('edge=', edge)
        # retrieve all entries with of the edge (u,v) from the scores list
        one_edge_scores = tuple(filter(
            lambda x: edge[0] == x[0] or tuple(reversed(edge[0])) == x[0], edge_scores))
        edge_scores = list(filter(lambda e: e not in one_edge_scores, edge_scores))

        if len(edge_scores) == 0:
            done = True
        # filter out bridge type scores of the edge
        bridge_scores = tuple(map(
            lambda x: x[2], filter(
                lambda x: x[1] == EdgeType.BRIDGE, one_edge_scores)))
        if len(bridge_scores) == 0:
            bridge_score = 0
        else:
            bridge_score = ft.reduce(op.add, bridge_scores) / len(bridge_scores)
        # filter out link type scores of the edge
        link_scores = tuple(map(
            lambda x: x[2], filter(
                lambda x: x[1] == EdgeType.LINK, one_edge_scores)))
        if len(link_scores) == 0:
            link_score = 0
        else:
            link_score = ft.reduce(op.add, link_scores) / len(link_scores)
        # an edge can be classified differently for each of its two vertexes
        # we check each classification value, if the value closer to PI
        # in one of the two classifications - we choose that.
        if np.abs(np.pi - bridge_score) <= np.abs(np.pi - link_score):
            # print('BRIDGE=', bridge_score, ' LINK=', link_score, ' TYPE=', EdgeType.BRIDGE)
            final_scores[edge[0]] = EdgeType.BRIDGE
        else:
            # print('BRIDGE=', bridge_score, ' LINK=', link_score, ' TYPE=', EdgeType.LINK)
            final_scores[edge[0]] = EdgeType.LINK

    return final_scores


# ---------------------------------------------------------------------------------
# main execution function
def execute(input_path):
    # retrieve list of images
    images = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    i = 1
    for image in images:
        file_name = image.split('.')[0]
        print('[' + str(i) + '/' + str(len(images)) + ']', file_name)
        # pre-process image
        time_print('pre-process image...')
        image_preprocessed = pre_process(input_path + image)
        # extract ridges
        time_print('extract ridges...')
        ridges_mask, ridges_matrix = ridge_extraction(image_preprocessed)
        # mark junction pixels
        time_print('mark junction pixels...')
        junction_pixels_mask = mark_junction_pixels(ridges_mask)
        # cv2.imwrite('junction_pixels_mask.png', overlay_images(ridges_mask*255, junction_pixels_mask*255))
        # retrieve vertex pixels
        time_print('retrieve vertex pixels...')
        vertexes_dictionary, vertexes_list, labels, vertex_mask = get_vertexes(ridges_matrix, junction_pixels_mask)
        # save_image_like(ridges_mask, vertexes_list, 'vertex_mask')
        # retrieve edges between two vertexes
        # each edge value is a list of pixels from vertex u to vertex v
        # each edge key is a pair of vertexes (u, v)
        time_print('retrieve edges between two vertexes...')
        edge_dictionary = get_edges(ridges_mask, junction_pixels_mask, vertexes_list)
        time_print('clean graph up...')
        edge_dictionary = clean_graph(edge_dictionary)
        # using each two vertexes of an edge, we classify whether an edge is a brige (between two lines),
        # or a link (part of a line). As a result, we receive a list of edges and their classification
        time_print('classify edges...')
        edge_scores = classify_edges(edge_dictionary, ridges_mask)
        classified_image = overlay_classified_edges(image_preprocessed, edge_dictionary, edge_scores)
        time_print('combine link edges...')
        # cv2.imshow('overlay_classified_edges', classified_image)
        cv2.imwrite(input_path + file_name + '_result.png', classified_image)
        # save the graph in a file
        with open(input_path + file_name + '_graph.pkl', 'wb') as handle:
            pickle.dump(edge_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # save classifications of each edge in a file
        with open(input_path + file_name + '_scores.pkl', 'wb') as handle:
            pickle.dump(edge_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
        i += 1
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        # display
        # overlay_image = overlay_images(ridges_mask * 255, vertex_mask * 255, vertexes_list)
        # cv2.imwrite('overlay_image.png', overlay_image)
        # cv2.imshow('overlay_image', overlay_image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()


execute("./data/")
# execute("_005-1.png")
# execute("0010-1.png")
