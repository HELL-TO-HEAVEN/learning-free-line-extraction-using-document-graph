import random as rd
import functools as ft
import itertools as it
import operator as op
import collections as col

from enum import Enum
import datetime
from decimal import Decimal, getcontext
import pickle
from os import listdir
from os.path import isfile, join
import copy
import numpy as np
import cv2
from skimage import morphology
from skan import csr
from concurrent.futures import ProcessPoolExecutor
from concurrent import futures



def neighbours(x, y, image):
    # "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],     # P2,P3,P4,P5
        img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1]]    # P6,P7,P8,P9


def transitions(neighbors):
    # "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbors + neighbors[0:1]      # P2, P3, ... , P8, P9, P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)


def zhang_suen(image):
    # "the Zhang-Suen Thinning Algorithm"
    Image_Thinned = image.copy()  # deepcopy to protect the original image
    changing1 = changing2 = 1        #  the points to be removed (set as 0)
    while changing1 or changing2:   #  iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape               # x for rows, y for columns
        for x in range(1, rows - 1):                     # No. of  rows
            for y in range(1, columns - 1):            # No. of columns
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1     and    # Condition 0: Point P1 in the object regions
                    2 <= sum(n) <= 4   and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and    # Condition 2: S(P1)=1
                    P2 * P4 * P6 == 0  and    # Condition 3
                    P4 * P6 * P8 == 0):         # Condition 4
                    changing1.append((x,y))
        for x, y in changing1:
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1   and        # Condition 0
                    2 <= sum(n) <= 4  and       # Condition 1
                    transitions(n) == 1 and      # Condition 2
                    P2 * P4 * P8 == 0 and       # Condition 3
                    P2 * P6 * P8 == 0):            # Condition 4
                    changing2.append((x,y))
        for x, y in changing2:
            Image_Thinned[x][y] = 0
    return Image_Thinned


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
            edge_type, score = edge_scores[tuple([u, v])]
            if edge_type == EdgeType.BRIDGE:
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
        r, g, b = image_copy[point]
        if r == 0 and g == 0 and b == 0:
            image_copy[point] = random_color
        else:
            image_copy[point] = (255, 255, 255)
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
    def idx_check(index):
        if index < 0:
            return 0
        else:
            return index

    def erosion(binary_img_matrix, structuring_element_in):
        structuring_element = copy.deepcopy(structuring_element_in)
        offset = np.where(structuring_element == 2)
        structuring_element[offset[0][0], offset[1][0]] = 1
        binary_img_matrix = np.asarray(binary_img_matrix)
        structuring_element = np.asarray(structuring_element)
        ste_shp = structuring_element.shape
        eroded_img = np.zeros_like(binary_img_matrix)
        ste_origin = (int(np.floor((structuring_element.shape[0]) / 2.0)),
                      int(np.floor((structuring_element.shape[1]) / 2.0)))
        offset_row = ste_origin[0] - offset[0][0]
        offset_col = ste_origin[1] - offset[1][0]

        for i in range(ste_origin[0], len(binary_img_matrix) - ste_origin[0]):
            for j in range(ste_origin[1], len(binary_img_matrix[0]) - ste_origin[1]):
                overlap = binary_img_matrix[idx_check(i - ste_origin[0]):i + (ste_shp[0] - ste_origin[0]),
                          idx_check(j - ste_origin[1]):j + (ste_shp[1] - ste_origin[1])]
                shp = overlap.shape
                ste_first_row_idx = int(np.fabs(i - ste_origin[0])) if i - ste_origin[0] < 0 else 0
                ste_first_col_idx = int(np.fabs(j - ste_origin[1])) if j - ste_origin[1] < 0 else 0

                ste_last_row_idx = ste_shp[0] - 1 - (
                            i + (ste_shp[0] - ste_origin[0]) - binary_img_matrix.shape[0]) if i + (
                            ste_shp[0] - ste_origin[0]) > binary_img_matrix.shape[0] else ste_shp[0] - 1
                ste_last_col_idx = ste_shp[1] - 1 - (
                            j + (ste_shp[1] - ste_origin[1]) - binary_img_matrix.shape[1]) if j + (
                            ste_shp[1] - ste_origin[1]) > binary_img_matrix.shape[1] else ste_shp[1] - 1

                if shp[0] != 0 and shp[1] != 0 and np.array_equal(
                        np.logical_and(overlap, structuring_element[ste_first_row_idx:ste_last_row_idx + 1,
                                                ste_first_col_idx:ste_last_col_idx + 1]),
                        structuring_element[ste_first_row_idx:ste_last_row_idx + 1, ste_first_col_idx:ste_last_col_idx + 1]):
                    eroded_img[i - offset_row, j - offset_col] = 1
        return eroded_img

    def rotate(mat):
        return mat, mat.T, np.flipud(mat), np.fliplr(mat.T)

    junctions = map(uint8_array, [
        [
            [1, 1, 2, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],

        ],
        [
            [1, 1, 2, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        [
            [0, 0, 0, 1, 1, 2, 1, 1],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 1, 2, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        [
            [1, 1, 0, 0, 0, 1, 1],
            [0, 1, 1, 0, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ],
        [
            [1, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 2, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
        ],
    ])

    rotated_mats = (rotated for mat in junctions for rotated in rotate(mat))
    junctions_mask = ft.reduce(op.or_, map(ft.partial(erosion, binary_image), rotated_mats),
                               np.zeros_like(binary_image))
    # we need to close junctions -> find min_x min_y max_x max_y then add ridge pixels to it
    # cv2.imwrite("before.png", overlay_images(binary_image*255, junctions_mask*255))
    #
    # return close_junctions(junctions_mask, binary_image)
    return junctions_mask


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
# retrieves connected pixels that are part of the edge pixels
# to be used for the bfs algorithm
# 8 connected neighborhood of a pixel
def connected_candidates(pixel, skeleton):
    def add_offset(offset):
        return tuple(map(op.add, pixel, offset))

    def in_bounds_and_true(p):
        row, col = add_offset(p)
        if row >=0 and \
                col >=0 and \
                row < skeleton.shape[0] and \
                col < skeleton.shape[1] and \
                skeleton[row][col]:
            return True
        else:
            return False

    eight_connected = list(filter(in_bounds_and_true, [(1, 0), (0, 1), (-1, 0), (0, -1),
                                                                      (1, 1), (-1, 1), (-1, -1), (1, -1)]))

    return [add_offset(offset) for offset in eight_connected]



# ---------------------------------------------------------------------------------
# retrieves m_connected pixels that are part of the edge pixels
# to be used for the bfs algorithm
def m_connected_candidates(pixel, edge_pixels):
    def add_offset(offset):
        return tuple(map(op.add, pixel, offset))
    # print('edge_pixels=', edge_pixels)
    four_connected = list(filter(lambda p: add_offset(p) in edge_pixels, [(1, 0), (0, 1), (-1, 0), (0, -1)]))
    # print('four_connected=', four_connected)
    diagonally_connected = list(filter(lambda p: add_offset(p) in edge_pixels,
                                       [(1, 1), (-1, 1), (-1, -1), (1, -1)]))
    # print('diagonally_connected=', diagonally_connected)
    uniquely_diagonally_connected = list(filter(lambda p: {(0, p[1]), (p[0], 0)}.isdisjoint(set(four_connected)),
                                                diagonally_connected))
    # print('uniquely_diagonally_connected=', uniquely_diagonally_connected)
    return [add_offset(offset) for offset in four_connected + uniquely_diagonally_connected]




# ---------------------------------------------------------------------------------
# returns for edge (u,v) its shortest m-connected list of pixels from pixel u to pixel v
def m_edge_bfs(edge, edge_list):
    start, end = edge
    visited = set()
    to_visit = col.deque([start])
    edges = col.deque()
    done = False
    while not done and to_visit:
        current = to_visit.popleft()
        # print('current=', current)
        visited.add(current)
        candidates = [v for v in m_connected_candidates(current, edge_list)
                      if v not in visited and v not in to_visit]
        # print('candidates=', candidates)
        for vertex in candidates:
            edges.append([current, vertex])
            to_visit.append(vertex)
            if vertex == end:
                done = True
                # print('done!')
    # print('start=', start, 'end=', end)
    # print('candidates=', edges)
    # exit()
    # find path from end -> start
    final_edges = [end]
    current = end
    failed = False
    while current != start and not failed:
        sub_edges = list(filter(lambda item: item[1] == current, edges))
        if sub_edges:
            one_edge = sub_edges.pop()
            final_edges.append(one_edge[0])
            current = one_edge[0]
        else:
            failed = True

    final_edges.append(start)
    if failed:
        return edge, []
    else:
        return edge, final_edges


# ---------------------------------------------------------------------------------
# returns for edge (u,v) its shortest connected list of pixels from pixel u to pixel v
def edge_bfs(start, end, skeleton):

    visited = set()
    to_visit = col.deque([start])
    edges = col.deque()
    done = False
    while not done and to_visit:
        current = to_visit.popleft()
        visited.add(current)
        candidates = [v for v in connected_candidates(current, skeleton)
                      if v not in visited and v not in to_visit]
        # print('candidates=', candidates)
        for vertex in candidates:
            edges.append([current, vertex])
            to_visit.append(vertex)
            if vertex == end:
                done = True
    # print('start=', start, 'end=', end)
    # print('candidates=', edges)
    # exit()

    # find path from end -> start
    final_edges = [end]
    current = end
    failed = False
    while current != start and not failed:
        # print('current=', current)
        sub_edges = list(filter(lambda item: item[1] == current, edges))
        # print('sub_edges=', sub_edges)
        if sub_edges:
            one_edge = sub_edges.pop()
            final_edges.append(one_edge[0])
            current = one_edge[0]
        else:
            failed = True

    final_edges.append(start)
    # print('finalEdges=', final_edges)
    # exit()

    if failed:
        print(start, end, 'fail')
        return start, end, []
    else:
        # print(start, end, 'success')
        return start, end, final_edges


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
# merge_overlapped_edges(edge_dictionary)
# TODO merging overlapped edges
def merge_overlapped_edges(edge_dictionary, ridges_mask):
    keys_list = list(edge_dictionary.keys())
    for edge_i in range(0, len(edge_dictionary.keys()) - 1):
        edge_i_key = keys_list[edge_i]
        edge_i_js = []
        edge_i_intersections = []
        edge_i_intersections_length = []

        for edge_j in range(edge_i + 1, len(edge_dictionary.keys()) - 1):
            edge_j_key = keys_list[edge_j]
            edge_i_js.append(edge_j_key)
            intersected_set = set(edge_dictionary[edge_i_key]).intersection(edge_dictionary[edge_j_key])
            edge_i_intersections.append(intersected_set)
            edge_i_intersections_length.append(len(intersected_set))
            # / len( set(edge_dictionary[edge_i_key]).union(edge_dictionary[edge_j_key])))
        max_intersection_index = np.argmax(edge_i_intersections_length)
        print('max_idx=', max_intersection_index, 'max_val=', edge_i_intersections_length[max_intersection_index])

        image = cv2.cvtColor(np.zeros_like(ridges_mask), cv2.COLOR_GRAY2RGB)
        image = draw_edges([edge_i_key], edge_dictionary, image, (255, 0, 0))
        cv2.namedWindow('edge_i')
        cv2.imshow('edge_i', image)
        cv2.imwrite('edge_i.png', image)
        image = cv2.cvtColor(np.zeros_like(ridges_mask), cv2.COLOR_GRAY2RGB)
        print('edge_i_js[max_intersection_index]=', edge_i_js[max_intersection_index]   )
        image = draw_edges([edge_i_js[max_intersection_index]], edge_dictionary, image, (0, 0, 255))
        cv2.namedWindow('edge_j')
        cv2.imwrite('edge_j.png', image)
        cv2.imshow('edge_j', image)
        image = cv2.cvtColor(np.zeros_like(ridges_mask), cv2.COLOR_GRAY2RGB)
        image = overlay_edges(image, edge_i_intersections[max_intersection_index], (0, 255, 0))
        cv2.namedWindow('overlap')
        cv2.imwrite('overlap.png', image)
        cv2.imshow('overlap', image)

        print('edge_i:', edge_dictionary[edge_i_key])
        print('edge_j:', edge_i_js[max_intersection_index])
        print('intersection:', edge_i_intersections[max_intersection_index])

        cv2.waitKey()
        cv2.destroyAllWindows()
        # exit()
    return edge_dictionary


# ---------------------------------------------------------------------------------
# clean graph up by removing 1-edge vertexes, and merging 2-edge vertexes
def clean_graph(edge_dictionary, ridges_mask):
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

    edge_dictionary = merge_overlapped_edges(edge_dictionary, ridges_mask)
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
# get_edges_between_vertexes(edges, degrees)
def get_edges_between_vertexes(edges, degrees):

    image = cv2.cvtColor(np.zeros_like(degrees, np.uint8), cv2.COLOR_GRAY2RGB)
    edge_dictionary = dict()

    all_pixels = np.where(degrees == 255)
    all_pixels = list(zip(all_pixels[0], all_pixels[1]))
    # print(image.shape)

    # for pixel in all_pixels :
    #     image[pixel] = (0, 0, 255)
    colors = []
    time_print('finding edges...')

    pool = ProcessPoolExecutor(max_workers=30)
    # wait_for = [pool.submit(m_edge_bfs, edge, all_pixels) for edge in edges]
    wait_for = [pool.submit(edge_bfs, edge, all_pixels) for edge in edges]
    # results = [f.result() for f in futures.as_completed(wait_for)]
    results = []
    total = len(wait_for)
    i = 1
    for f in futures.as_completed(wait_for):
        result = f.result()
        # print(result)
        if result[1]:
            results.append(result)
            print('[' + str(i) + '/' + str(total) + '] ' + str(len(result[1])))
        else:
            print('[' + str(i) + '/' + str(total) + '] failed!!!!!')
        i += 1

    for result in results:
        edge, edge_list = result
        random_color = (rd.randint(50, 200), rd.randint(50, 200), rd.randint(50, 200))
        while random_color in colors:
            random_color = (rd.randint(50, 200), rd.randint(50, 200), rd.randint(50, 200))
            image = draw_edge(edge_list, image, random_color)
            colors.append(random_color)
    cv2.imwrite('result.png', image)
    cv2.namedWindow('result')
    cv2.imshow('result', image)


    # with Pool(processes=40) as pool:
    #    temp = [pool.apply_async(m_edge_bfs, (edge, all_pixels)) for edge in edges]
    #    print('waiting...')
    #    i = 1
    #    for t in temp:
    #        results = t.get()
    #        print(i, end=' ')
    #        i += 1
    #    print('done!')



    # for edge in edges:
    #    print('[' + str(i) + '/' + str(total) + ']')
    #   i += 1
    #    v, u = edge
    #     m_adjacent_edge_pixels = m_edge_bfs(v, u, all_pixels)
    #     edge_dictionary[edge] = m_adjacent_edge_pixels

    for edge in edge_dictionary.keys():
        random_color = (rd.randint(50, 200), rd.randint(50, 200), rd.randint(50, 200))
        while random_color in colors:
            random_color = (rd.randint(50, 200), rd.randint(50, 200), rd.randint(50, 200))
            image = draw_edge(edge_dictionary[edge], image, random_color)
            colors.append(random_color)

    cv2.namedWindow('result')
    cv2.imshow('result', image)
    # print('len=', len(m_adjacent_edge_pixels), 'edge=', m_adjacent_edge_pixels)
    cv2.waitKey()
    cv2.destroyAllWindows()
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


# -
# All vertexes with one degree (take part of one edge only) - they are removed
# All vertexes with two degree (take part of two edges exactly) - they are merged
# this is done iteratively, until all vertexes have a degree of three or more!
def remove_one_degree_edges(skeleton, iter_index):
    cv2.imwrite('skel_' + str(iter_index) + '.png', skeleton.astype(np.uint8) * 255)

    # TODO add to paper information about this !!! remove redundant edges
    # summarization shows each edge, its start and end pixel, length, distance, etc
    # ALSO!! it has two types of edges, one that has two vertexes in graph
    # others that have one vertex only - those to be removed!
    # TODO THESE ARE EDGE END POINTS - TWO VERTEX COORDINATES FOR EACH EDGE
    branch_data = csr.summarise(skeleton)
    coords_cols = (['img-coord-0-%i' % i for i in [1, 0]] +
                   ['img-coord-1-%i' % i for i in [1, 0]])
    coords = branch_data[coords_cols].values.reshape((-1, 2, 2))
    # TODO Each Vertex to stay in the graph needs to have a degree of two or more
    # TODO Iteratively, we remove those that have less than two degree
    # TODO We stop only when there are no more vertexes left with low degree
    # TODO THEN WE EXTRACT THE EDGES - USING BFS ?! NEED TO FIND A GOOD WAY

    done = False
    changed = False
    while not done:
        flat_coords = [tuple(val) for sublist in coords for val in sublist]
        len_before = len(coords)
        for item in set(flat_coords):
            # print('item=', item, 'count=', flat_coords.count(item))
            # 1 degree vertexes are to be removed from graph
            if flat_coords.count(item) < 2:
                coords = list(filter(lambda x: tuple(x[0]) != item and tuple(x[1]) != item, coords))
            # 2 degree vertexes need their edges to be merged
            if flat_coords.count(item) == 2:
                fc = list(filter(lambda x: tuple(x[0]) == item or tuple(x[1]) == item, coords))
                if len(fc) < 2:
                    continue # TODO CHECK IF THIS IS PROBLIMATIC - ! IF ALL ARE MERGED FOR REAL
                # print('fc=', fc)
                coords = list(filter(lambda x: tuple(x[0]) != item and tuple(x[1]) != item, coords))
                e1_s = fc[0][0]
                e1_e = fc[0][1]
                e2_s = fc[1][0]
                e2_e = fc[1][1]

                # print('e1_s=', e1_s, 'e1_e=', e1_e)
                # print('e2_s=', e2_s, 'e2_e=', e2_e)
                # exit()
                # print(coords)
                if ft.reduce(op.and_, map(lambda e: e[0] == e[1], zip(e1_s, e2_s))):
                    coords.append(np.array([e1_e, e2_e]))
                elif ft.reduce(op.and_, map(lambda e: e[0] == e[1], zip(e1_s, e2_e))):
                    coords.append(np.array([e1_e, e2_s]))
                elif ft.reduce(op.and_, map(lambda e: e[0] == e[1], zip(e1_e, e2_s))):
                    coords.append(np.array([e1_s, e2_e]))
                else:
                    coords.append(np.array([e1_s, e2_s]))
        if len_before == len(coords):
            done = True
        else:
            changed = True
            print('before=', len_before, 'after=', len(coords))

    skel = cv2.cvtColor(skeleton.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)
    # cv2.namedWindow('skeleton')
    cv2.imwrite('skeleton.png', skel)
    # cv2.imshow('skeleton', skel)
    # TODO DISCONNECT EVERY JUNCTION - THIS HELPS BFS CONVERGE FASTER!
    tmp_skel = copy.deepcopy(skeleton)
    for coord in coords:
        start, end = coord
        start = (start[1], start[0])
        end = (end[1], end[0])
        # print(start, end)
        start_neighborhood = connected_candidates(start, skeleton)
        end_neighborhood = connected_candidates(end, skeleton)
        for point in start_neighborhood + end_neighborhood:
            tmp_skel[point] = False
        tmp_skel[start] = False
        tmp_skel[end] = False
    # cv2.namedWindow('skeleton_junctions')
    # cv2.imwrite('skeleton_junctions.png', skel)
    # cv2.imshow('skeleton_junctions', skel)

    # TODO NOW WE EXTRACT EDGES, FIND BFS (SHORTEST PATH) BETWEEN TWO GIVEN VERTEXES
    cv2.imwrite('base_' + str(iter_index) + '.png', tmp_skel.astype(np.uint8) * 255)

    skel = np.zeros_like(skeleton)
    results = []
    for edge in coords:
        start, end = edge
        start = (start[1], start[0])
        end = (end[1], end[0])
        start_neighborhood = connected_candidates(start, skeleton)
        end_neighborhood = connected_candidates(end, skeleton)
        for point in start_neighborhood + end_neighborhood:
            tmp_skel[point] = True
        tmp_skel[start] = True
        tmp_skel[end] = True
        _, _, result = edge_bfs(start, end, tmp_skel)
        start_neighborhood = connected_candidates(start, skeleton)
        end_neighborhood = connected_candidates(end, skeleton)
        for point in start_neighborhood + end_neighborhood:
             tmp_skel[point] = False
        tmp_skel[start] = False
        tmp_skel[end] = False
        results.append((start, end, result))
        for point in result:
            skel[point] = True

    colors = []
    image = cv2.cvtColor(np.zeros_like(skeleton, np.uint8), cv2.COLOR_GRAY2RGB)
    for result in results:
        start, end, edge_list = result
        random_color = (rd.randint(50, 200), rd.randint(50, 200), rd.randint(50, 200))
        while random_color in colors:
            random_color = (rd.randint(50, 200), rd.randint(50, 200), rd.randint(50, 200))
        for point in edge_list:
            image[point] = random_color
        colors.append(random_color)
    cv2.imwrite('iter_' + str(iter_index) + '.png', image)
    return skel, results, changed


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
    # TODO show before and after result
    # cv2.namedWindow('before')
    # cv2.imshow('before', dist_maxima_mask_biggest_component * 255)
    # cv2.imwrite('before_zhang.png', dist_maxima_mask_biggest_component * 255)
    # TODO check which skeletonization is used !!!!! The skeleton is thinned usign this method
    # we extract our own skeleton, here we just use thinning after ridge extraction
    skeleton = morphology.skeletonize(dist_maxima_mask_biggest_component)
    # TODO THIS SKELETONIZATION USES -> [Zha84]	(1, 2) A fast parallel algorithm for thinning digital patterns, T. Y. Zhang and C. Y. Suen, Communications of the ACM, March 1984, Volume 27, Number 3.
    # cv2.imwrite('skeleton.png', dist_maxima_mask_biggest_component.astype(np.uint8) * 255)
    # TODO to add to paper!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # degree has each pixel and # of neighbors, to distinguish junction from non junction
    # so far I think 3+ neighbors are chosen, but begin from highest number of neighbors in greedy manner
    # each time if a pixel chosen as junction, a nearby pixel cannot be chosen as junction even if it fits
    # the minimum number of pixels

    # cv2.imwrite('degrees.png', cv2.normalize(degrees, None, 0, 255, cv2.NORM_MINMAX))
    # TODO add to paper information about this !!! remove redundant edges
    # summarization shows each edge, its start and end pixel, length, distance, etc
    # ALSO!! it has two types of edges, one that has two vertexes in graph
    # others that have one vertex only - those to be removed!
    # TODO THESE ARE EDGE END POINTS - TWO VERTEX COORDINATES FOR EACH EDGE
    # branch_data, g, coords_img, skeleton_ids, num_skeletons = csr.summarise(skeleton)
    # coords_cols = (['img-coord-0-%i' % i for i in [1, 0]] +
    #                ['img-coord-1-%i' % i for i in [1, 0]])
    # coords = branch_data[coords_cols].values.reshape((-1, 2, 2))
    # TODO Each Vertex to stay in the graph needs to have a degree of two or more
    # TODO Iteratively, we remove those that have less than two degree
    # TODO We stop only when there are no more vertexes left with low degree
    # TODO THEN WE EXTRACT THE EDGES - USING BFS ?! NEED TO FIND A GOOD WAY

    changed = True
    results = []
    iter_index = 0
    while changed:
        print('iter', iter_index)
        skeleton, results, changed = remove_one_degree_edges(skeleton, iter_index)
        iter_index += 1
    print('done')
    colors = []
    image = cv2.cvtColor(np.zeros_like(skeleton, np.uint8), cv2.COLOR_GRAY2RGB)
    edge_dictionary = dict()
    for result in results:
        start, end, edge_list = result
        edge_dictionary[(start, end)] = result
        random_color = (rd.randint(50, 200), rd.randint(50, 200), rd.randint(50, 200))
        while random_color in colors:
            random_color = (rd.randint(50, 200), rd.randint(50, 200), rd.randint(50, 200))
        for point in edge_list:
            image[point] = random_color
        colors.append(random_color)
    cv2.imwrite('resultFinal.png', image)
    cv2.namedWindow('resultFinal')
    cv2.imshow('resultFinal', image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return skeleton, edge_dictionary


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
    # angle between u, v and v, w
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
    if norma_1 == 0.0:
        print('norma_1==0->', u, v, w)
        norma_1 = Decimal(0.0001)
    if norma_2 == 0.0:
        print('norma_2==0->', u, v, w)
        norma_2 = Decimal(0.0001)
    val = dot / (norma_1 * norma_2)

    return np.abs(np.arccos(float(val)))


# ---------------------------------------------------------------------------------
# assistance function - receives u,v v,w1 v,w2 - three edges
# calculates the distance from shape T for these three edges
# where u,v is the I in T
def calculate_junction_t_distance(u, v, w1, w2, edges_dictionary):
    v_x, v_y = v
    max_dist = 9
    max_dist_v = [x for x in range(-max_dist, max_dist + 1)]
    max_dist_candidates_x = list(map(lambda x: x + v_x, max_dist_v))
    max_dist_candidates_y = list(map(lambda y: y + v_y, max_dist_v))

    left_column = list(map(lambda e: (v_x - max_dist, e), max_dist_candidates_y))
    right_column = list(map(lambda e: (v_x + max_dist, e), max_dist_candidates_y))
    top_column = list(map(lambda e: (e, v_y - max_dist), max_dist_candidates_x))
    bottom_column = list(map(lambda e: (e, v_y + max_dist), max_dist_candidates_x))

    junction_pixels = dict()
    if tuple([u, v]) in edges_dictionary.keys():
        junction_pixels[tuple([u, v])] = edges_dictionary[tuple([u, v])]
    else:
        junction_pixels[tuple([u, v])] = edges_dictionary[tuple([v, u])]

    if tuple([v, w1]) in edges_dictionary.keys():
        junction_pixels[tuple([v, w1])] = edges_dictionary[tuple([v, w1])]
    else:
        junction_pixels[tuple([v, w1])] = edges_dictionary[tuple([w1, v])]

    if tuple([v, w2]) in edges_dictionary.keys():
        junction_pixels[tuple([v, w2])] = edges_dictionary[tuple([v, w2])]
    else:
        junction_pixels[tuple([v, w2])] = edges_dictionary[tuple([w2, v])]

    w1_in_radius = [i for i in left_column + right_column + top_column + bottom_column if
                    i in junction_pixels[tuple([v, w1])]]
    if len(w1_in_radius) == 0:
        w1_in_radius = [w1]
    w2_in_radius = [i for i in left_column + right_column + top_column + bottom_column if
                    i in junction_pixels[tuple([v, w2])]]
    if len(w2_in_radius) == 0:
        w2_in_radius = [w2]
    u_in_radius = [i for i in left_column + right_column + top_column + bottom_column if
                   i in junction_pixels[tuple([u, v])]]
    if len(u_in_radius) == 0:
        u_in_radius = [u]

    # now we have three edges: (u, v), (v, w1), (v, w2)
    # calculate the angles between each two edges
    uv_vw1 = calculate_abs_angle(u_in_radius[0], v, w1_in_radius[0])
    uv_vw2 = calculate_abs_angle(u_in_radius[0], v, w2_in_radius[0])
    w1v_vw2 = calculate_abs_angle(w1_in_radius[0], v, w2_in_radius[0])

    # distance from t-shape
    return np.abs(np.pi - w1v_vw2) + np.abs(np.pi / 2.0 - uv_vw1) + np.abs(np.pi / 2.0 - uv_vw2)


# ---------------------------------------------------------------------------------
# assistance function - calculates for edge (u, v) how close its v junction to the shape of T
def calculate_junction_score(u, v, edges_dictionary, ridges_mask, window):
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
        # find a pixel of distance of R along the edge of (v, e)
        # this way the angle calculations are more accurate
        # solves the issue with non-straight edges that skew the final angle calculation
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

    # we calculate the angles at pixels of a up to max distance from v
    # otherwise, due to edges not being straight, we might get wrong angles
    # TODO PARAMETER_TO_CALCULATE
    max_dist = 9
    max_dist_v = [x for x in range(-max_dist, max_dist + 1)]
    max_dist_candidates_x = list(map(lambda x: x + v_x, max_dist_v))
    max_dist_candidates_y = list(map(lambda y: y + v_y, max_dist_v))

    left_column = list(map(lambda e: (v_x - max_dist, e), max_dist_candidates_y))
    right_column = list(map(lambda e: (v_x + max_dist, e), max_dist_candidates_y))
    top_column = list(map(lambda e: (e, v_y - max_dist), max_dist_candidates_x))
    bottom_column = list(map(lambda e: (e, v_y + max_dist), max_dist_candidates_x))

    # res = overlay_edges(cv2.cvtColor(np.zeros_like(ridges_mask), cv2.COLOR_GRAY2RGB),
    #                     left_column, (255, 255, 255))
    # res = overlay_edges(res, right_column, (255, 255, 255))
    # res = overlay_edges(res, top_column, (255, 255, 255))
    # res = overlay_edges(res, bottom_column, (255, 255, 255))

    w1_in_radius = [i for i in left_column + right_column + top_column + bottom_column if
                    i in junction_pixels[tuple([v, w1])]]
    if len(w1_in_radius) == 0:
        w1_in_radius = [w1]
    w2_in_radius = [i for i in left_column + right_column + top_column + bottom_column if
                    i in junction_pixels[tuple([v, w2])]]
    if len(w2_in_radius) == 0:
        w2_in_radius = [w2]
    u_in_radius = [i for i in left_column + right_column + top_column + bottom_column if
                   i in junction_pixels[tuple([u, v])]]
    if len(u_in_radius) == 0:
        u_in_radius = [u]
    # now we have three edges: (u, v), (v, w1), (v, w2)
    # calculate the angles between each two edges
    uv_vw1 = calculate_abs_angle(u_in_radius[0], v, w1_in_radius[0])
    uv_vw2 = calculate_abs_angle(u_in_radius[0], v, w2_in_radius[0])
    w1v_vw2 = calculate_abs_angle(w1_in_radius[0], v, w2_in_radius[0])
    # res = overlay_edges(res, junction_pixels[tuple([u, v])], (0, 0, 255))
    # res = overlay_edges(res, junction_pixels[tuple([v, w1])], (0, 255, 0))
    # res = overlay_edges(res, junction_pixels[tuple([v, w2])], (255, 0, 0))
    # cv2.imshow(window, res)
    # cv2.waitKey()
    # distance_from_shape_t = np.abs(np.pi - w1v_vw2) + np.abs(np.pi / 2.0 - uv_vw1) + np.abs(np.pi / 2.0 - uv_vw2)
    # print('u=', u, 'v=', v, 'w1=', w1, 'w2=', w2)
    # print('u_in=', u_in_radius[0], 'v=', v, 'w1_in=', w1_in_radius[0], 'w2_in=', w2_in_radius[0])
    # print('uv_vw1=', uv_vw1, 'uv_vw2=', uv_vw2, 'w1v_vw2=', w1v_vw2)
    # each edge will have two distances [0, 1] from our optimal shape:
    # if u,v is the I part of the T shape -> we have an optimal $bridge$ edge
    # if u,v is the ‾ of ‾‾  of the T (‾|‾ shape) -> we have an optimal $link$ edge
    scores = list()
    scores.append(((u, v), EdgeType.BRIDGE, np.abs(np.pi - w1v_vw2)))

    scores.append(((u, v), EdgeType.LINK,
                   min(np.abs(np.pi / 2.0 - uv_vw1), np.abs(np.pi / 2.0 - uv_vw2))))

    # print(scores)
    # res = overlay_edges(cv2.cvtColor(np.zeros_like(ridges_mask), cv2.COLOR_GRAY2RGB),
    #                     junction_pixels[tuple([u, v])], (255, 255, 255))
    # res = overlay_edges(res, junction_pixels[tuple([u, v])], (255, 255, 255))
    # res = overlay_edges(res, junction_pixels[tuple([v, w1])], (0, 255, 0))
    # res = overlay_edges(res, junction_pixels[tuple([v, w2])], (0, 0, 255))
    # cv2.imshow(window, res)
    # cv2.waitKey()
    return scores


# ---------------------------------------------------------------------------------
# for each vertex, we choose every three edges leaving that vertex and calculate
# their T-distance. The three edges generate a shape, the closer the shape to T
# the smaller the T-distance is.
def calculate_junction_t_distances(vertex_dictionary, edges_dictionary, ridges_mask):

    t_shape_distances = dict()
    for v in vertex_dictionary:

        v_edges = list(filter(lambda uw: uw[0] == v or uw[1] == v, edges_dictionary.keys()))
        if len(v_edges) < 3:
            continue
        vertexes = list(map(lambda uw: uw[0] if uw[1] == v else uw[1], v_edges))
        # print('v=', v)
        for combination in it.combinations(vertexes, 3):
            u, w1, w2 = combination
            # print('combination=', combination)
            # the t-shape calculation is done as follows:
            # (u = B, w1 = L, w2 = L)
            # this is done for three variants for each three edges
            # each edge can be Bridge, then the other two are Links
            t_shape_distances[(u, v, w1, w2)] = calculate_junction_t_distance(u, v, w1, w2, edges_dictionary)
            t_shape_distances[(w1, v, u, w2)] = calculate_junction_t_distance(w1, v, u, w2, edges_dictionary)
            t_shape_distances[(w2, v, w1, u)] = calculate_junction_t_distance(w2, v, w1, u, edges_dictionary)

    return t_shape_distances


# ---------------------------------------------------------------------------------
# classify edges, bridge or link
# TODO OLD WAY - TO BE REMOVED
def classify_edges(edges_dictionary, ridges_mask):
    window = cv2.namedWindow("point")
    edge_scores = dict()
    edges_dictionary_keys = edges_dictionary.keys()
    for edge in edges_dictionary_keys:
        u, v = edge
        if u == v:
            continue
        junction_score_1 = calculate_junction_score(u, v, edges_dictionary, ridges_mask, window)
        junction_score_2 = calculate_junction_score(v, u, edges_dictionary, ridges_mask, window)
        if junction_score_1 is not None and junction_score_2 is not None:
            edge_scores[(u, v)] = (junction_score_1, junction_score_2)

    final_scores = dict()
    for edge in edge_scores.keys():
        junction_score_1, junction_score_2 = edge_scores[edge]

        bridge_1, link_1 = junction_score_1
        bridge_2, link_2 = junction_score_2

        _, _, bridge_score_1 = bridge_1
        _, _, bridge_score_2 = bridge_2
        _, _, link_score_1 = link_1
        _, _, link_score_2 = link_2

        bridge_score = bridge_score_1 + bridge_score_2
        link_score = link_score_1 + link_score_2

        # each edge might have up to two scores: bridge score, link score
        if link_score < bridge_score - np.pi / 6:
            final_scores[edge] = (EdgeType.LINK, link_score)
        else:
            final_scores[edge] = (EdgeType.BRIDGE, bridge_score)

    return final_scores


# ---------------------------------------------------------------------------------
# calculate_angles
def calculate_angles(u, edge, edge_pixels, edge_neighbors, edge_neighbors_pixels, ridges_mask, vertexes_along_edges):
    def calc_angle(in_v, in_u, in_edge_pixels, neighbor, vertexes_along_edges):

        neighbor_edge, neighbor_pixels = neighbor
        w1, w2 = neighbor_edge
        in_w = w1 if in_v == w2 else w2

        v_x, v_y = in_v
        max_dist = 9
        max_dist_v = [x for x in range(-max_dist, max_dist + 1)]
        max_dist_candidates_x = list(map(lambda x: x + v_x, max_dist_v))
        max_dist_candidates_y = list(map(lambda y: y + v_y, max_dist_v))

        left_column = list(map(lambda e: (v_x - max_dist, e), max_dist_candidates_y))
        right_column = list(map(lambda e: (v_x + max_dist, e), max_dist_candidates_y))
        top_column = list(map(lambda e: (e, v_y - max_dist), max_dist_candidates_x))
        bottom_column = list(map(lambda e: (e, v_y + max_dist), max_dist_candidates_x))

        in_w_in_radius = [i for i in left_column + right_column + top_column + bottom_column if i in neighbor_pixels]
        if len(in_w_in_radius) == 0:
            in_w_in_radius = [in_w]

        in_u_in_radius = [i for i in left_column + right_column + top_column + bottom_column if i in in_edge_pixels]
        if len(in_u_in_radius) == 0:
            in_u_in_radius = [in_u]

        # before_ridge_mask = cv2.cvtColor(np.zeros_like(ridges_mask), cv2.COLOR_GRAY2RGB)
        # before_ridge_mask = overlay_edges(before_ridge_mask, neighbor_pixels, (0, 0, 255))
        # before_ridge_mask = overlay_edges(before_ridge_mask, in_edge_pixels, (255, 0, 0))
        # before_ridge_mask = overlay_edges(before_ridge_mask, left_column, (255, 255, 255))
        # before_ridge_mask = overlay_edges(before_ridge_mask, right_column, (255, 255, 255))
        # before_ridge_mask = overlay_edges(before_ridge_mask, top_column, (255, 255, 255))
        # before_ridge_mask = overlay_edges(before_ridge_mask, bottom_column, (255, 255, 255))

        # cv2.namedWindow('after')
        # cv2.imshow('after', before_ridge_mask)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # print('in_u_in_radius=', in_u_in_radius[0], 'in_v=', in_v, 'in_w_in_radius=', in_w_in_radius[0])
        local_angle = calculate_abs_angle(in_u_in_radius[0], in_v, in_w_in_radius[0])
        # global_angle = calculate_abs_angle(in_u, in_v, in_w)
        # print('in_v=', in_v)
        vertexes = [e for e in vertexes_along_edges[(in_v, in_u)] if e != in_v]
        angles = [calculate_abs_angle(elem_u, in_v, in_w) for elem_u in vertexes]
        max_global_angle = np.max(angles)
        return max(local_angle, max_global_angle)

    v1, v2 = edge
    v = v1 if u == v2 else v2

    return [calc_angle(u, v, edge_pixels, neighbor, vertexes_along_edges) for neighbor in
            zip(edge_neighbors, edge_neighbors_pixels)]


# ---------------------------------------------------------------------------------
# get_candidate
def get_candidate(u, v, edge, edge_pixels, edge_dictionary, ridges_mask, vertexes_along_edges):
    edge_neighbors = [edge for edge in edge_dictionary.keys() if u in edge and v not in edge]
    edge_neighbors_pixels = [edge_dictionary[edge] for edge in edge_neighbors]
    # print('u=', u, 'v=', v, 'edge=', edge)
    if edge_neighbors:
        edge_angles = calculate_angles(u, edge, edge_pixels, edge_neighbors, edge_neighbors_pixels, ridges_mask,
                                       vertexes_along_edges)
        max_angle_idx = np.argmax(edge_angles)
        max_angle = edge_angles[max_angle_idx]
        max_angle_neighbor = edge_neighbors[max_angle_idx]
        return max_angle_neighbor, max_angle
    else:
        return None, None


# ---------------------------------------------------------------------------------
# combine edges
def combine_edges(vertexes_list, edge_dictionary, ridges_mask):
    vertexes_along_edges = dict()
    before_ridge_mask = cv2.cvtColor(np.zeros_like(ridges_mask), cv2.COLOR_GRAY2RGB)
    for edge_list in edge_dictionary.values():
        before_ridge_mask = overlay_edges(before_ridge_mask, edge_list)
    # for each edge (u,v ) we find its neighbour edges (u, w) v!=w
    # we find angle between two.
    # then find min angle, then if min angle =< 30degrees we combine together as new edge
    # this is done iteratively over and over until no more edges can be combined
    # we do this in greedy manner ! ! ! ! ! ! ! ! ! ! ! !
    done = False
    while not done:
        candidates = []
        angles = []
        edge_dictionary_iter = copy.deepcopy(edge_dictionary)
        while edge_dictionary_iter:
            edge, edge_pixels = edge_dictionary_iter.popitem()
            u, v = edge
            if u == v:
                continue
            if edge not in vertexes_along_edges.keys():
                vertexes_along_edges[edge] = edge
            # print('vertexes_along_edges=', vertexes_along_edges)
            max_angle_neighbor, max_angle = get_candidate(u, v, edge, edge_pixels, edge_dictionary_iter, ridges_mask,
                                                          vertexes_along_edges)
            if max_angle_neighbor is not None:
                candidates.append((max_angle_neighbor, edge))
                angles.append(max_angle)

        max_angle_index = np.argmax(angles)
        edge_one, edge_two = candidates[max_angle_index]
        threshold = np.pi - np.pi / 3.5
        # print('angles[min_angle_index]=', angles[max_angle_index])
        if angles[max_angle_index] > threshold:
            # we combine those two: edge (u,v) and min_angle_edge (v, w)
            # then remove edge, and min_angle_neighbor from graph
            # which means we need to remove all edges where v is part of them ! ! !
            part_one = edge_dictionary.pop(edge_one)
            part_two = edge_dictionary.pop(edge_two)

            vertexes_of_edge_one = vertexes_along_edges.pop(edge_one) if edge_one in vertexes_along_edges.keys() \
                else edge_one
            vertexes_of_edge_two = vertexes_along_edges.pop(edge_two) if edge_two in vertexes_along_edges.keys() \
                else edge_two

            new_edge = tuple(set(edge_one + edge_two).difference(set(edge_one).intersection(edge_two)))
            vertexes_along_edges[new_edge] = tuple(set(vertexes_of_edge_one + vertexes_of_edge_two))
            # print('edge_one=', edge_one, 'edge_two=', edge_two, 'new_edge=', new_edge)
            new_edge_pixels = list(set(part_one + part_two))
            edge_dictionary[new_edge] = new_edge_pixels
        else:
            done = True

    # after_ridge_mask = cv2.cvtColor(np.zeros_like(ridges_mask), cv2.COLOR_GRAY2RGB)
    # for edge_list in edge_dictionary.values():
    #    after_ridge_mask = overlay_edges(after_ridge_mask, edge_list)
    # cv2.namedWindow('before')
    # cv2.namedWindow('after')
    # cv2.imshow('before', before_ridge_mask)
    # cv2.imshow('after', after_ridge_mask)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return edge_dictionary


# ---------------------------------------------------------------------------------
# draw_graph_edges
def draw_graph_edges(edge_dictionary, ridges_mask, window_name, wait_flag=False):
    after_ridge_mask = cv2.cvtColor(np.zeros_like(ridges_mask), cv2.COLOR_GRAY2RGB)
    for edge_list in edge_dictionary.values():
        colors = []
        random_color = (rd.randint(50, 200), rd.randint(50, 200), rd.randint(50, 200))
        while random_color in colors:
            random_color = (rd.randint(50, 200), rd.randint(50, 200), rd.randint(50, 200))
        after_ridge_mask = overlay_edges(after_ridge_mask, edge_list, random_color)
        colors.append(random_color)
    for two_vertex in edge_dictionary.keys():
        v1, v2 = two_vertex
        after_ridge_mask[v1] = (255, 255, 255)
        after_ridge_mask[v2] = (255, 255, 255)

    cv2.namedWindow(window_name)
    cv2.imshow(window_name, after_ridge_mask)
    cv2.imwrite(window_name + '.png', after_ridge_mask)
    if wait_flag:
        cv2.waitKey()
        cv2.destroyAllWindows()


# -
#
def draw_edge(edge, image, color):
    for point in edge:
        image[point] = color
    return image


# ---------------------------------------------------------------------------------
# draw_edges(edges, edge_dictionary, image, color):
def draw_edges(edges, edge_dictionary, image, color):
    for edge in edges:
        edge_list = edge_dictionary[edge]
        image = overlay_edges(image, edge_list, color)
    return image


# ---------------------------------------------------------------------------------
# get_nearby_pixels
def get_nearby_pixels(u, v, w1, w2, edges_dictionary):
    v_x, v_y = v
    max_dist = 9
    max_dist_v = [x for x in range(-max_dist, max_dist + 1)]
    max_dist_candidates_x = list(map(lambda x: x + v_x, max_dist_v))
    max_dist_candidates_y = list(map(lambda y: y + v_y, max_dist_v))

    left_column = list(map(lambda e: (v_x - max_dist, e), max_dist_candidates_y))
    right_column = list(map(lambda e: (v_x + max_dist, e), max_dist_candidates_y))
    top_column = list(map(lambda e: (e, v_y - max_dist), max_dist_candidates_x))
    bottom_column = list(map(lambda e: (e, v_y + max_dist), max_dist_candidates_x))

    junction_pixels = dict()
    if tuple([u, v]) in edges_dictionary.keys():
        junction_pixels[tuple([u, v])] = edges_dictionary[tuple([u, v])]
    else:
        junction_pixels[tuple([u, v])] = edges_dictionary[tuple([v, u])]

    if tuple([v, w1]) in edges_dictionary.keys():
        junction_pixels[tuple([v, w1])] = edges_dictionary[tuple([v, w1])]
    else:
        junction_pixels[tuple([v, w1])] = edges_dictionary[tuple([w1, v])]

    if tuple([v, w2]) in edges_dictionary.keys():
        junction_pixels[tuple([v, w2])] = edges_dictionary[tuple([v, w2])]
    else:
        junction_pixels[tuple([v, w2])] = edges_dictionary[tuple([w2, v])]

    w1_in_radius = [i for i in left_column + right_column + top_column + bottom_column
                    if i in junction_pixels[(v, w1)]]
    if len(w1_in_radius) == 0:
        w1_in_radius = [w1]

    w2_in_radius = [i for i in left_column + right_column + top_column + bottom_column
                    if i in junction_pixels[(v, w2)]]
    if len(w2_in_radius) == 0:
        w2_in_radius = [w2]

    u_in_radius = [i for i in left_column + right_column + top_column + bottom_column
                   if i in junction_pixels[(u, v)]]
    if len(u_in_radius) == 0:
        u_in_radius = [u]

    return u_in_radius[0], w1_in_radius[0], w2_in_radius[0]


# ---------------------------------------------------------------------------------
# calculate_junctions_t_scores
def calculate_junctions_t_scores(edge_dictionary, ridges_mask):
    # get rid of edges that have one neighbor only
    done = False
    while not done:
        done = True
        for edge in edge_dictionary.keys():
            u, v = edge
            if u == v:
                edge_dictionary.pop(edge)
                done = False
                break

            junction_v_edges = [edge for edge in edge_dictionary
                                if (edge[0] == v and edge[1] != u) or (edge[0] != u and edge[1] == v)]
            if len(junction_v_edges) == 1:
                new_v = junction_v_edges[0][0] if junction_v_edges[0][0] != v else junction_v_edges[0][1]
                new_edge = (u, new_v)
                edge_one = edge_dictionary.pop(edge)
                edge_two = edge_dictionary.pop(junction_v_edges[0])
                new_edge_pixels = list(set(edge_one + edge_two).difference(set(edge_one).intersection(edge_two)))
                # print('edge=', (u, v), 'edges=', junction_v_edges, 'len=', len(junction_v_edges), 'new=', new_edge)
                edge_dictionary[new_edge] = new_edge_pixels
                done = False
                break

            junction_u_edges = [edge for edge in edge_dictionary
                                if (edge[0] != v and edge[1] == u) or (edge[0] == u and edge[1] != v)]
            if len(junction_u_edges) == 1:
                # print('edge=', (v, u), 'edges=', junction_u_edges, 'len=', len(junction_u_edges))
                new_u = junction_u_edges[0][0] if junction_u_edges[0][0] != u else junction_u_edges[0][1]
                new_edge = (new_u, v)
                edge_one = edge_dictionary.pop(edge)
                edge_two = edge_dictionary.pop(junction_u_edges[0])
                new_edge_pixels = list(set(edge_one + edge_two).difference(set(edge_one).intersection(edge_two)))
                edge_dictionary[new_edge] = new_edge_pixels
                done = False
                break
    # calculate for each 3 edges of a junction their T score
    # return a list
    t_scores = dict()
    for edge in edge_dictionary:
        u, v = edge
        junction_v_edges = [edge for edge in edge_dictionary
                            if (edge[0] == v and edge[1] != u) or (edge[0] != u and edge[1] == v)]
        v_edges = [e[0] if e[1] == v else e[1] for e in junction_v_edges]
        # print('edge=', edge)
        # print('junction=', junction_v_edges)
        for combination in it.combinations(v_edges, 2):
            w1, w2 = combination
            # get coordinates in radius 9 - then calculate angle
            # in_u, in_w1, in_w2 = get_nearby_pixels(u, v, w1, w2, edge_dictionary)
            in_u = u
            in_w1 = w1
            in_w2 = w2
            # print('in_u=', in_u, 'in_w1=', in_w1,'v=', v, 'in_w2=', in_w2)
            uv_vw1 = calculate_abs_angle(in_u, v, in_w1)
            uv_vw2 = calculate_abs_angle(in_u, v, in_w2)
            w1v_vw2 = calculate_abs_angle(in_w1, v, in_w2)
            uv_bridge = np.abs(np.pi - w1v_vw2) + np.abs(np.pi / 2.0 - uv_vw1) + np.abs(np.pi / 2.0 - uv_vw2)
            vw1_bridge = np.abs(np.pi - uv_vw1) + np.abs(np.pi / 2.0 - uv_vw2) + np.abs(np.pi / 2.0 - w1v_vw2)
            vw2_bridge = np.abs(np.pi - uv_vw2) + np.abs(np.pi / 2.0 - uv_vw1) + np.abs(np.pi / 2.0 - w1v_vw2)
            t_scores[(u, v, w1, w2)] = [(u, v, uv_bridge), (v, w1, vw1_bridge), (v, w2, vw2_bridge)]

        junction_u_edges = [edge for edge in edge_dictionary
                            if (edge[0] == u and edge[1] != v) or (edge[0] != v and edge[1] == u)]
        u_edges = [e[0] if e[1] == u else e[1] for e in junction_u_edges]
        for combination in it.combinations(u_edges, 2):
            w1, w2 = combination
            vu_uw1 = calculate_abs_angle(v, u, w1)
            vu_uw2 = calculate_abs_angle(v, u, w2)
            w1u_uw2 = calculate_abs_angle(w1, u, w2)
            vu_bridge = np.abs(np.pi - w1u_uw2) + np.abs(np.pi / 2.0 - vu_uw1) + np.abs(np.pi / 2.0 - vu_uw2)
            uw1_bridge = np.abs(np.pi - vu_uw1) + np.abs(np.pi / 2.0 - vu_uw2) + np.abs(np.pi / 2.0 - w1u_uw2)
            uw2_bridge = np.abs(np.pi - vu_uw2) + np.abs(np.pi / 2.0 - vu_uw1) + np.abs(np.pi / 2.0 - w1u_uw2)
            t_scores[(v, u, w1, w2)] = [(v, u, vu_bridge), (u, w1, uw1_bridge), (u, w2, uw2_bridge)]

    # in greedy manner: find junction in t_scores where u,v v,w1 v,w2 has minimum T score
    # mark u,v as Bridge
    # mark v,w1 and v,w2 as Link
    # TODO OPTION 1 - 100% greedy - and remove conflicts on the go
        # remove all u,v from t_scores marked as L
        # remove all v,w1 and v,w2 from t_scores marked as B
    # TODO OPTION 2 - each time check for conflicts, and mark as such
        # add junction to B and L lists
        # check whether new min junction
    bridges = []
    links = []

    while t_scores:
        done = True
        # find minimum score for each junction
        min_t_scores = dict()
        for key in t_scores.keys():
            min_score_index = np.argmin(map(lambda e1, e2, score: score, t_scores[key]))
            min_t_scores[key] = t_scores[key][min_score_index]

        # print(min_t_scores.values())
        # find junction with minimum score of all junctions
        values = [value[2] for value in min_t_scores.values()]
        min_score_index = np.argmin(values)
        min_score_key = list(min_t_scores.keys())[min_score_index]
        min_score = min_t_scores[min_score_key]
        # print('min_score_index=', min_score_index, 'min_score_key=', min_score_key, 'min_score=', min_score)
        # add to bridges - CHECK FOR CONFLICT
        new_bridge = (min_score[0], min_score[1])
        if new_bridge not in edge_dictionary.keys():
            p1, p2 = new_bridge
            new_bridge = (p2, p1)

        # print('t_scores[min_score_key]=', t_scores[min_score_key])
        # add to links - CHECK FOR CONFLICT
        two_links = [item for item in t_scores[min_score_key] if item is not min_score]
        # print('two_links=', two_links)
        new_links = []
        for link in two_links:
            e1, e2, _ = link
            new_link = (e1, e2)
            if new_link not in edge_dictionary.keys():
                new_link = (e2, e1)
            new_links.append(new_link)
        # check for conflicts before adding them
        if new_bridge not in links and set(links).isdisjoint(new_links):
                bridges.append(new_bridge)
                links.extend(new_links)

        # remove minimum t score junction from t_scores
        t_scores.pop(min_score_key)
        # print('B=', bridges, 'L=', links)

    draw_graph_edges(edge_dictionary, ridges_mask, 'before')
    image = cv2.cvtColor(np.zeros_like(ridges_mask), cv2.COLOR_GRAY2RGB)
    image = draw_edges(bridges, edge_dictionary, image, (255, 0, 0))
    image = draw_edges(links, edge_dictionary, image, (0, 255, 0))
    rest = [x for x in edge_dictionary.keys() if x not in set(bridges).union(links)]
    image = draw_edges(rest, edge_dictionary, image, (0, 0, 255))
    cv2.namedWindow('after')
    cv2.imshow('after', image)
    cv2.imwrite('after_with_white.png', image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    exit()
    return 0


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
        time_print('extract ridges, junctions...')
        # ridges_mask, ridges_matrix = ridge_extraction(image_preprocessed)
        skeleton, edge_dictionary = ridge_extraction(image_preprocessed)



        # mark junction pixels
        # time_print('mark junction pixels...')
        # junction_pixels_mask = mark_junction_pixels(ridges_mask)
        # cv2.imwrite('junction_pixels_mask.png', overlay_images(ridges_mask*255, junction_pixels_mask*255))
        # retrieve vertex pixels
        # time_print('retrieve vertex pixels...')
        # vertexes_dictionary, vertexes_list, labels, vertex_mask = get_vertexes(ridges_matrix, junction_pixels_mask)
        # save_image_like(ridges_mask, vertexes_list, 'vertex_mask')
        # retrieve edges between two vertexes
        # each edge value is a list of pixels from vertex u to vertex v
        # each edge key is a pair of vertexes (u, v)
        # time_print('retrieve edges between two vertexes...')
        # edge_dictionary = get_edges_between_vertexes(edges, degrees)
        # edge_dictionary = get_edges(ridges_mask, junction_pixels_mask, vertexes_list)
        # time_print('clean graph up...')
        # edge_dictionary = clean_graph(edge_dictionary, ridges_mask)
        # using each two vertexes of an edge, we classify whether an edge is a brige (between two lines),
        # or a link (part of a line). As a result, we receive a list of edges and their classification
        # time_print('classify edges...')
        # edge_scores = classify_edges(edge_dictionary, ridges_mask)
        # TODO ...
        # calculate for each junction its B L L, L B L, L L B values using distance from T shape
        # in greedy manner -
        #   choose the assignment with minimum value for u,v,w - for all junctions for every combination
        # TODO step 1: for each u,v v,w1 v,w2 JUNCTION -> calculate 3 scores: L L B, L B L, L L B distance from T
        t_scores = calculate_junctions_t_scores(edge_dictionary, skeleton)
        # TODO step 2: visualize result -> for each edge: if all B GREEN, if all L BLUE, mixed RED
        #
        # TODO step 3: some options (depending on result in step 2)
        #       TODO 3.1: for each edge that has no agreement we try to improve by choosing one of the two

        # TODO current work . . . combine edges
        # time_print('calculate vertex T-scores...')
        # calculate_junction_t_distances(vertexes_list, edge_dictionary, ridges_mask)
        # combined_edge_dictionary = combine_edges(vertexes_list, edge_dictionary, ridges_mask)
        # after_ridge_mask = cv2.cvtColor(np.zeros_like(ridges_mask), cv2.COLOR_GRAY2RGB)
        # for edge_list in combined_edge_dictionary.values():
        #    colors = []
        #    random_color = (rd.randint(50, 255), rd.randint(50, 255), rd.randint(50, 255))
        #    while random_color in colors:
        #        random_color = (rd.randint(50, 255), rd.randint(50, 255), rd.randint(50, 255))
        #    after_ridge_mask = overlay_edges(after_ridge_mask, edge_list, random_color)
        #    colors.append(random_color)
        # cv2.imwrite(file_name + '_result.png', after_ridge_mask)
        # cv2.namedWindow('before')
        # cv2.namedWindow('after')
        # cv2.imshow('before', before_ridge_mask)
        # cv2.imshow('after', after_ridge_mask)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        # classified_image = overlay_classified_edges(image_preprocessed, edge_dictionary, edge_scores)
        # time_print('combine link edges...')

        # cv2.imshow('overlay_classified_edges', classified_image)
        # cv2.imwrite(input_path + '/results/' + file_name + '_result.png', classified_image)
        # save the graph in a file
        with open(input_path + '/results/' + file_name + '_graph.pkl', 'wb') as handle:
            pickle.dump(edge_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # save classifications of each edge in a file
        # with open(input_path + '/results/' + file_name + '_scores.pkl', 'wb') as handle:
        #     pickle.dump(edge_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
        i += 1
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        # display
        # overlay_image = overlay_images(ridges_mask * 255, vertex_mask * 255, vertexes_list)
        # cv2.imwrite('overlay_image.png', overlay_image)
        # cv2.imshow('overlay_image', overlay_image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    execute("./data/")
    # execute("_005-1.png")
    # execute("0010-1.png")
