import random as rd
import functools as ft
import operator as op
import numpy as np
import cv2


# ---------------------------------------------------------------------------------
# auxiliary functions
def uint8_array(rows):
    return np.array(rows).astype(np.uint8)


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
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # random_color = (100, 156, 88)
        for point in vertexes_list:
            random_color = (rd.randint(0, 100), rd.randint(100, 200), rd.randint(0, 255))
            # cv2.circle(image, point, radius, random_color, cv2.CV_FILLED)
            image[point] = random_color

        return image


# ----------------------------------
# returns ????
def m_connected(pixel, mask):
    def is_in_image(co_ordinate, shape):
        r, c = co_ordinate
        rows, cols = shape
        return (0 <= r < rows) and (0 <= c < cols)

    def is_in(co_ordinate):
        r, c = co_ordinate
        return is_in_image((r, c), mask.shape) and mask[r, c]

    def add_offset(offset):
        return tuple(map(op.add, pixel, offset))

    four_connected = [offset for offset in [(1, 0), (0, 1), (-1, 0), (0, -1)] if is_in(add_offset(offset))]

    uniquely_diagonally_connected = [(o_r, o_c) for o_r, o_c in [(1, 1), (-1, 1), (-1, -1), (1, -1)]
                                     if {[(0, o_r), (o_c, 0)]}.isdisjoint(four_connected)
                                     and is_in(add_offset((o_r, o_c)))]

    return [add_offset(offset) for offset in four_connected + uniquely_diagonally_connected]


# --------------
#
def find_shortest_paths(vertexes_list, mask):
    # for each u,v in V we find the shortest m-connected path
    # using m-connected ensures no redundant paths are calculated - speeds up process
    return 0


# ---------------------------------------------------------------------------------
# find junction pixels
def rotate(mat):
    return mat, mat.T, np.flipud(mat), np.fliplr(mat.T)


def mark_junction_pixels(binary_image):
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

    junctions_mask = ft.reduce(op.or_, map(ft.partial(cv2.erode, binary_image), rotated_mats), np.zeros_like(binary_image))
    # this is dilated to be used in edge extraction later
    # dilation is used to increase the junction size (to cover the width of the skeleton)
    # otherwise, we don't segment the graph to its edges properly once we remove the junctions
    return cv2.dilate(junctions_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3)))


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
        vertexes_dictionary[co_ordinates] = pixels_list_i
        vertexes_mask[co_ordinates] = 1

    return vertexes_dictionary, vertexes_list, labels, vertexes_mask


# ---------------------------------------------------------------------------------
# edge extraction
def get_edges(ridge_mask, junction_pixels_mask):
    # we remove the junctions from the ridge mask
    # this way we disconnect them from each other
    edges_mask = ridge_mask - junction_pixels_mask
    # each connected component is an edge in the graph
    n_edges, labels = cv2.connectedComponents(edges_mask, connectivity=8)
    # add to a dictionary - edge number, and its list of pixels
    edge_dictionary = {}
    for i in range(1, n_edges):
        # for each label we retrieve its list of pixels
        edge_dictionary[i] = np.argwhere(labels == i)

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
    # TODO do we need binarization? better work on grayscale unless it does not work well
    image = cv2.threshold(image, 0, 1, cv2.THRESH_OTSU)[1]
    # add white border around image of size 29
    white_border_added = cv2.copyMakeBorder(image, 29, 29, 29, 29, cv2.BORDER_CONSTANT, None, 1)
    # on top of that add black border of size 1
    black_border_added = cv2.copyMakeBorder(white_border_added, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, 0)
    # TODO erode the result to disconnect weakly connected components and removing white noise
    # TODO return cv2.erode(black_border_added, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
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
    # retrieve vertex pixels
    vertexes_dictionary, vertexes_list, labels, vertex_mask = get_vertexes(ridges_matrix, junction_pixels_mask)
    # retrieve edges between two vertexes
    edges_dictionary, edges_list = get_edges(ridges_matrix, junction_pixels_mask)

    # display
    overlay_image = overlay_images(ridges_mask * 255, vertex_mask * 255, vertexes_list)
    cv2.imwrite('overlay_image.png', overlay_image)
    cv2.imshow('overlay_image', overlay_image)
    cv2.waitKey()
    cv2.destroyAllWindows()


run_all("part.png")
