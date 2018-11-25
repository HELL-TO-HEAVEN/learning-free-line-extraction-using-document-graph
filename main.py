import random as rd
import functools as ft
import itertools as it
import operator as op
import collections as col
import numpy as np
import sys
import copy
import cv2


# ---------------------------------------------------------------------------------
# auxiliary functions
def uint8_array(rows):
    return np.array(rows).astype(np.uint8)


def overlay_edges(image, edge_list):
    image_copy = copy.deepcopy(image)
    # random_color = (100, 156, 88)
    random_color = (rd.randint(50, 255), rd.randint(50, 255), rd.randint(50, 255))
    for point in edge_list:
        # cv2.circle(image, point, radius, random_color, cv2.FILLED)
        image_copy[point] = random_color
    # print(len(edge_list))
    # cv2.imshow("", image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
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
        # print('current=', current)
        visited.add(current)
        # print('visited=', visited)
        candidates = [v for v in m_connected_candidates(current, edge_list)
                      if v not in visited and v not in to_visit]
        # print('candidates=', candidates)
        for vertex in candidates:
            edges.append([current, vertex])
            to_visit.append(vertex)
            if vertex == end:
                done = True
        # print('to_visit=', to_visit)
    # find path from end -> start
    final_edges = [end]
    current = end
    while current != start:
        sub_edges = list(filter(lambda item: item[1] == current, edges))
        one_edge = sub_edges.pop()
        final_edges.append(one_edge[0])
        current = one_edge[0]
    final_edges.append(start)
    # print('len=', len(final_edges))
    # print('final_edges=', final_edges)
    return final_edges


# ---------------------------------------------------------------------------------
# for each edge (u,v) , and two vertexes, u,v we find the m_adjacent version of the edge
def m_connected_edge(start, end, edge_list):
    m_edge = set()
    visited = set()
    bfs_q = col.deque([start])

    while bfs_q:
        pixel = bfs_q.popleft()
        m_edge.add(pixel)
        if pixel == end:
            return m_edge
        else:
            next_vertexes = [v for v in m_connected_candidates(pixel, edge_list) if v not in visited]
            visited.update(next_vertexes)
            bfs_q.extend(next_vertexes)


# ---------------------------------------------------------------------------------
# edge extraction
def get_edges(ridges_mask, junction_pixels_mask, vertexes_dictionary, vertexes_list):

    all_junction_pixels_set = set(map(tuple, np.argwhere(junction_pixels_mask != 0)))
    rgb_ridge_mask = cv2.cvtColor(ridges_mask, cv2.COLOR_GRAY2RGB)
    original_ridge_mask = cv2.cvtColor(ridges_mask, cv2.COLOR_GRAY2RGB)
    # we remove the junctions from the ridge mask
    # this way we disconnect them from each other
    edges_mask = ridges_mask - junction_pixels_mask
    # each connected component is an edge in the graph
    n_edges, labels = cv2.connectedComponents(edges_mask, connectivity=8)
    cv2.imwrite('0_edges_mask.png', edges_mask*255)
    # add to a dictionary - edge number, and its list of pixels
    edge_dictionary = {}
    # 0 vertexes, 1 vertexes, 2 vertexes, 3 or more
    stats = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(1, n_edges):
        # reset junction list
        junction_pixels_set = all_junction_pixels_set
        # for each label we retrieve its list of pixels
        edge_pixels = list(map(tuple, np.argwhere(labels == i)))

        # iteratively - add in junction pixels that are nearby to the current edge
        # new pixels are added to the edge, then we do the same (increasing edge width/length by 1)
        # new pixels come only from junction pixels
        # this way the edge covers the vertexes as well
        not_empty = True
        j = 0
        # save_image_like(ridges_mask, edge_pixels, str(i) + "_edge_mask_" + str(j))
        while not_empty:

            # add-in candidate junctions
            # 8-neighborhood of a pixel
            neighborhood = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
            # 4-neighborhood of a pixel
            # neighborhood = [(-1, 0), (1, 0), (0, 1), (0, -1)]

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
            j += 1
            # view_image_like(ridges_mask, vertexes_set, "vertexes_set_after")
            # stopping criteria - once we can't add more pixels to the edge from junction pixels, we stop
            # print(len(candidate_edge_pixels))
            if len(candidate_edge_pixels) == 0:
                not_empty = False
            else:
                edge_pixels += candidate_edge_pixels
            # save_image_like(ridges_mask, edge_pixels, str(i) + "_edge_mask_" + str(j))
        # once this edge is complete - we find its vertexes
        start_end_vertexes = list(set.intersection(set(edge_pixels), set(vertexes_list)))
        # TODO this list can be of start_end_vertexes != 2, handle those cases
        # TODO for 1 vertex -> we need to remove? ? ?? ? ? ? ?? ?
        # TODO for 2 vertexes -> good
        # TODO for 3 or more vertexes -> we need to send every u,v where bounding box of u,v has no other vertexes
        # TODO this means they will create edges that do not pass nearby or through any other vertex
        # TODO this effectively will split our edges to better reflect the graph
        if len(start_end_vertexes) >= 2:
            # then apply m-adjacency to find the shortest slimmest version of the edge from u to v
            m_adjacent_edge_pixels = m_edge_bfs(start_end_vertexes[0], start_end_vertexes[1], edge_pixels)
        # add to dictionary instead of edge_pixels
        edge_dictionary[i] = m_adjacent_edge_pixels
    print(stats)
    cv2.imshow('result', rgb_ridge_mask)
    cv2.imwrite('graph_edges_result.png', rgb_ridge_mask)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return edge_dictionary


# ----------------------------------------------------------------------------------
# for each pixel list (of an edge) choose two vertices where u,v in E
# path :u -> pixel list -> v
def create_graph_edges():
    # TODO find all connected components -> each one is an edge
    # TODO find adjacent vertices
    # TODO if a vertex has two edges, then it is not a junction
    # TODO we combine the two edges of the vertex
    # TODO this is done recursively until we are left with vertices that have three edges or more
    # TODO if there are vertices with one edge only - we discard the vertex and its edge
    return


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
    # TODO need to add some comments - what does this do?
    for val in np.unique(dist_maxima_mask)[1:]:
        mask = np.uint8(dist_maxima_mask == val)
        labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        dist_maxima_mask_biggest_component[labels == largest_label] = val
    # extract local maxima pixels magnitude values from the distance transform
    dist_maxima = np.multiply(dist_maxima_mask_biggest_component, dist_transform)

    return dist_maxima_mask_biggest_component, dist_maxima


# ---------------------------------------------------------------------------------
# document pre_processing
def pre_process(path):
    # load image as gray-scale, and convert to binary using otsu binarization
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # TODO do we need binarization? better work on gray-scale unless it does not work well
    image = cv2.threshold(image, 0, 1, cv2.THRESH_OTSU)[1]
    # add white border around image of size 29
    white_border_added = cv2.copyMakeBorder(image, 29, 29, 29, 29, cv2.BORDER_CONSTANT, None, 1)
    # on top of that add black border of size 1
    black_border_added = cv2.copyMakeBorder(white_border_added, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, 0)
    # TODO erode the result to disconnect weakly connected components and removing white noise
    # return cv2.erode(black_border_added, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # TODO above code was removed - seems erode is not good(?) gotta check
    # return non-eroded image - maybe erosion is not good here
    return black_border_added


# ---------------------------------------------------------------------------------
# main function
def run_all(path):
    # pre-process image
    image_preprocessed = pre_process(path)
    # extract ridges
    ridges_mask, ridges_matrix = ridge_extraction(image_preprocessed)
    # mark junction pixels
    junction_pixels_mask = mark_junction_pixels(ridges_mask)
    cv2.imwrite('junction_pixels_mask.png', overlay_images(ridges_mask*255, junction_pixels_mask*255))
    # retrieve vertex pixels
    vertexes_dictionary, vertexes_list, labels, vertex_mask = get_vertexes(ridges_matrix, junction_pixels_mask)
    save_image_like(ridges_mask, vertexes_list, 'vertex_mask')
    # retrieve edges between two vertexes
    # each edge is a series of pixels from vertex u to vertex v
    edges_dictionary = get_edges(ridges_mask, junction_pixels_mask, vertexes_dictionary, vertexes_list)
    for key in edges_dictionary.keys():
        print(key, edges_dictionary[key])

    # display
    overlay_image = overlay_images(ridges_mask * 255, vertex_mask * 255, vertexes_list)
    cv2.imwrite('overlay_image.png', overlay_image)
    cv2.imshow('overlay_image', overlay_image)
    cv2.waitKey()
    cv2.destroyAllWindows()


run_all("part.png")
