import os
import cv2
import math
import copy
import pylab
import shutil
import datetime

import numpy as np
import random as rd
import operator as op
import functools as ft
import itertools as it
import collections as col

from skan import csr
from os import listdir
from scipy import optimize
from scipy import integrate as intg
from skimage import morphology
from os.path import isfile, join
from matplotlib import pyplot as plt
from decimal import Decimal, getcontext

from concurrent import futures
from sklearn.mixture import GaussianMixture
from concurrent.futures import ProcessPoolExecutor


def interpolated_intercepts(x, y1, y2):
    """Find the intercepts of two curves, given by the same x data"""

    def intercept(point1, point2, point3, point4):
        """find the intersection between two lines
        the first line is defined by the line between point1 and point2
        the first line is defined by the line between point3 and point4
        each point is an (x,y) tuple.

        So, for example, you can find the intersection between
        intercept((0,0), (1,1), (0,1), (1,0)) = (0.5, 0.5)

        Returns: the intercept, in (x,y) format
        """

        def line(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0]*p2[1] - p2[0]*p1[1])
            return A, B, -C

        def intersection(L1, L2):
            D  = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]

            x = Dx / D
            y = Dy / D
            return x,y

        L1 = line([point1[0],point1[1]], [point2[0],point2[1]])
        L2 = line([point3[0],point3[1]], [point4[0],point4[1]])

        R = intersection(L1, L2)

        return R

    idxs = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)

    xcs = []
    ycs = []

    for idx in idxs:
        xc, yc = intercept((x[idx], y1[idx]),((x[idx+1], y1[idx+1])), ((x[idx], y2[idx])), ((x[idx+1], y2[idx+1])))
        xcs.append(xc)
        ycs.append(yc)
    return np.array(xcs), np.array(ycs)


# ---------------------------------------------------------------------------------
# draw_edges(edges, edge_dictionary, image, color):
def draw_edges(edges, edge_dictionary, image, color, image_offset_values=None):
    for edge in edges:
        if edge not in edge_dictionary.keys():
            continue
        edge_list = edge_dictionary[edge]
        if image_offset_values is not None:
            offset_edge_list = [(e[0]+image_offset_values[0], e[1]+image_offset_values[1]) for e
                                in edge_list]
        else:
            offset_edge_list = edge_list
        image = overlay_edges(image, offset_edge_list, color)
    return image


# ---------------------------------------------------------------------------------
# auxiliary function to overlay edges on original image
def overlay_edges(image, edge_list, color=None):
    image_copy = copy.deepcopy(image)

    # random_color = (100, 156, 88)
    if color is None:
        random_color = (rd.randint(50, 255), rd.randint(50, 255), rd.randint(50, 255))
    else:
        random_color = color
    for point in edge_list:
        # print('point=', point)
        # cv2.circle(image, point, radius, random_color, cv2.FILLED)
        r, g, b = image_copy[point]
        if r == 0 and g == 0 and b == 0:
            image_copy[point] = random_color
        else:
            image_copy[point] = (0, 255, 255)
    return image_copy


# ---------------------------------------------------------------------------------
# get humanly distinguishable colors
def get_spaced_colors(n):
    max_value = 255**3
    min_value = 150**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(min_value, max_value, interval)]
    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]


# ---------------------------------------------------------------------------------
# draw_graph_edges
def draw_graph_edges(edge_dictionary, ridges_mask, window_name, wait_flag=False, overlay=False, image_offset_values=None, file_name=None):
    if overlay:
        after_ridge_mask = ridges_mask
    else:
        after_ridge_mask = cv2.cvtColor(np.zeros_like(ridges_mask), cv2.COLOR_GRAY2RGB)

    random_colors = get_spaced_colors(len(edge_dictionary.values()) * 2)
    i = 0
    for edge_list in edge_dictionary.values():
        if image_offset_values is not None:
            offset_edge_list = [(e[0]+image_offset_values[0], e[1]+image_offset_values[1]) for e
                                in edge_list]
        else:
            offset_edge_list = edge_list
        after_ridge_mask = overlay_edges(after_ridge_mask, offset_edge_list, random_colors[i])
        i += 1
    for two_vertex in edge_dictionary.keys():
        v1, v2 = two_vertex
        if image_offset_values is not None:
            v1 = (v1[0] + image_offset_values[0], v1[1] + image_offset_values[1])
            v2 = (v2[0] + image_offset_values[0], v2[1] + image_offset_values[1])
        after_ridge_mask[v1] = (255, 255, 255)
        after_ridge_mask[v2] = (255, 255, 255)

    name = './' + window_name + '/' + file_name + '.png'
    cv2.imwrite(name, after_ridge_mask)
    cv2.imwrite('./' + window_name + '/' + file_name + '_inverted.png', 255 - after_ridge_mask)
    if wait_flag:
        cv2.namedWindow(window_name)
        cv2.imshow(window_name, after_ridge_mask)
        cv2.waitKey()
        cv2.destroyAllWindows()

    return name


# ---------------------------------------------------------------------------------
# split touching lines. find mean line width in image
# find connected components
# calculate width (x_max - x_min)
# cluster connected components following width
# find cluster with highest width average
# for each connected component create histogram. each bin is row value count number of pixels
# find three adjacent bins with min average of all, and remove them from image
def split_touching_lines(image, average_width=None):
    def calc_valleys(gauss_n, hist):
        x_s = np.asarray([z for z in range(len(hist))])
        # print('len(good_n)=', len(good_n))
        # average_error = sum(good_n) / len(good_n)
        # print('good_n=', good_n)
        result = list()
        # print('gauss_n=', gauss_n)
        for i_dx in range(0, len(gauss_n) - 1):
            gauss_1 = gauss_n[i_dx]
            gauss_2 = gauss_n[i_dx + 1]
            x_s_i, _ = interpolated_intercepts(x_s, np.asarray([gauss_1(r) for r in x_s]),
                                               np.asarray([gauss_2(r) for r in x_s]))

            # print(len(x_s_i))
            if len(x_s_i) == 2:
                result.append(np.int32(x_s_i[1]))
            elif len(x_s_i) == 1:
                result.append(np.int32(x_s_i[0]))
            # for elem1 in x_s_i:
            #     result.append(np.int32(elem1))
        return result

    def get_n_cut_valleys(hist, n):
        # plt.clf()
        # plt.plot(histogram, color='black', label='hist')
        # plt.xlim([0, len(hist)])
        x_s = np.asarray([z for z in range(len(hist))])

        # n gaussians fitting attempt
        try:
            piece = math.floor(len(hist) / n)
            y_n = [hist[i_dx * piece: (i_dx + 1) * piece] for i_dx in range(0, n)]
            x_n = [np.asarray([z for z in range(len(y_n[y_i_dx]))]) for y_i_dx in range(0, len(y_n))]
            params_n = [optimize.curve_fit(gauss, x_i, y_i, method='trf')[0] for x_i, y_i in zip(x_n, y_n)]

        except RuntimeError:
            # print('could not fit')
            # plt.clf()
            return [None for l in range(n)], [np.inf for l in range(n)]
        except ValueError:
            return [None for l in range(n)], [np.inf for l in range(n)]

        # append shift for mu_i
        mu_n = [params_i[0] for params_i in params_n]
        sigma_n = [params_i[1] for params_i in params_n]
        A_n = [params_i[2] for params_i in params_n]
        shift = math.floor((len(hist) + 1) / n)

        mu_n = [mu_i + i_dx * shift for i_dx, mu_i in enumerate(mu_n)]
        gauss_n = [ft.partial(gauss, mu=mu_i, sigma=sigma_i, A=A_i) for mu_i, sigma_i, A_i in
                   zip(mu_n, sigma_n, A_n)]

        good_n = [np.abs(intg.quad(gauss_i, 0, len(hist), args=())[0] - intg.quad(gauss_i, -np.inf, np.inf, args=())[0])
                  for gauss_i in gauss_n]

        return gauss_n, good_n
        # plt.plot(x_s, gauss(x_s, mu1, sigma1, A1), color='green', lw=3, label='gauss1')
        # plt.plot(x_s, gauss(x_s, mu2, sigma2, A2), color='blue', lw=3, label='gauss2')
        # plt.plot(x_s, gauss(x_s, mu3, sigma3, A3), color='red', lw=3, label='gauss3')
        # plt.plot(x_s, gauss(x_s, mu4, sigma4, A4), color='yellow', lw=3, label='gauss4')
        # plt.plot(x_s, bimodal(x_s, mu1, sigma1, A1, mu2, sigma2, A2), color='red', lw=3, label='bi-modal')
        # plt.legend()
        # plt.show()
        # plt.clf()
        # print('x_s_1=', x_s_1)
        # print('x_s_2=', x_s_2)
        # print('x_s_3=', x_s_3)

    def gauss(x_value, mu, sigma, A):
        return A * pylab.exp(-(x_value - mu) ** 2 / 2 / sigma ** 2)

    def draw_component_on_image(label_i, all_labels, overlay_image, color):
        for index in zip(*np.where(all_labels == label_i)):
            overlay_image[index] = color

    def draw_component(pixels_list, draw_on_image):
        for p in pixels_list:
            draw_on_image[p] = (255, 0, 0)

    def cluster_elements(heights_to_cluster, average_width=None):
        n_clusters = 3
        heights_to_cluster = np.asarray(heights_to_cluster).reshape(-1, 1)
        # k_means = KMeans(n_clusters=n_clusters)
        gmm = GaussianMixture(n_components=n_clusters)
        # k_means.fit(heights_to_cluster)
        gmm.fit(heights_to_cluster)
        # y_k_means = k_means.predict(heights_to_cluster)
        y_k_means = gmm.predict(heights_to_cluster)
        # print('y_k_means=', y_k_means)
        # print('y_k_gmms=', y_k_means_gmm)
        # print('=------------------=')
        clustered_heights = [[] for k in range(n_clusters)]

        for k in range(n_clusters):
            ix = 0
            for y_k in y_k_means:
                if y_k == k:
                    clustered_heights[k].append(heights_to_cluster[ix][0])
                ix += 1

        # for k, cluster_i in enumerate(clustered_heights):
        #     print(cluster_i)
        #     print('average=', k, sum(cluster_i) / len(cluster_i))

        # for k, item in enumerate(clustered_heights):
        #     print('clustered_heights=', k, item)
        clusters = [list(filter(lambda x_0: x_0 == k, y_k_means)) for k in range(n_clusters)]
        cluster_size = [len(cluster) for cluster in clusters]
        # print('cluster_size=', cluster_size)
        cluster_total_widths = [sum(heights_to_cluster[l[0]] for l in enumerate(y_k_means) if l[1] == k) for k in
                                 range(n_clusters)]
        cluster_average_sizes = [pair[0] / pair[1] for pair in zip(cluster_total_widths, cluster_size)]

        print('cluster_sizes=', cluster_size)
        print('cluster_total_widths=', cluster_total_widths)
        print('cluster_average_sizes=', cluster_average_sizes)

        cluster_total = [ft.reduce(lambda x_1, y_1: x_1 + y_1[1] if y_1[0] == k else x_1,
                                   zip(list(y_k_means), heights_to_cluster), 0) for k in range(n_clusters)]
        max_cluster = np.argmax([c[1]/c[0] for c in zip(cluster_size, cluster_total)])
        # print('maxCluster=', max_cluster)
        # print('widths=', cluster_average_sizes)
        ratios = [(cluster_average_sizes[z] / cluster_average_sizes[max_cluster])[0] for z in range(n_clusters)]
        max_average = cluster_average_sizes[np.argmax(ratios)]
        print('first_cluster:', 'index=', np.argmax(ratios), 'size=', cluster_average_sizes[np.argmax(ratios)])
        print('ratios=', ratios)
        first_cluster = ratios[np.argmax(ratios)]
        ratios[np.argmax(ratios)] = -1
        second_cluster = ratios[np.argmax(ratios)]
        ratios[np.argmax(ratios)] = -1
        third_cluster = ratios[np.argmax(ratios)]
        print('second_cluster:', 'index=', np.argmax(ratios), 'size=', cluster_average_sizes[np.argmax(ratios)])
        # print('second_cluster=', second_cluster)
        # print('max_cluster=', max_cluster)
        # print('second_cluster=', np.argmax(ratios))
        print('average_width=', average_width, '2nd_cluster=', cluster_average_sizes[np.argmax(ratios)])
        # if average_width is not None and 0.6 * average_width > cluster_average_sizes[np.argmax(ratios)]:
        #     return y_k_means, None, None
        if average_width is not None:
            # return y_k_means, max_cluster, cluster_average_sizes[np.argmax(ratios)], average_width
            return y_k_means, max_cluster, max_average, average_width

        if first_cluster * 3 > second_cluster and second_cluster * 3 > third_cluster:
            # return y_k_means, max_cluster, cluster_average_sizes[np.argmax(ratios)], None
            return y_k_means, max_cluster, max_average, None
        else:
            return y_k_means, None, None, None

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8, ltype=cv2.CV_32S)
    heights = [component[3] for component in stats[1:]]
    clustered, max_cluster_index, average_width, old_width = cluster_elements(heights, average_width)

    if max_cluster_index is None:
        # print('No touching lines to be split!')
        return image, None, None, 0, None

    # to_view = cv2.cvtColor(np.zeros_like(labels, np.uint8), cv2.COLOR_GRAY2RGB)
    # i = 1
    # for component in clustered:
    #         if component == max_cluster_index:
    #             draw_component_on_image(i, labels, to_view, (0, 0, 255))
    #        i += 1

    to_view = cv2.cvtColor(np.zeros_like(labels, np.uint8), cv2.COLOR_GRAY2RGB)
    i = 1
    for component in clustered:
        if component == 0:
            draw_component_on_image(i, labels, to_view, (0, 0, 255))
        elif component == 1:
            draw_component_on_image(i, labels, to_view, (0, 255, 0))
        elif component == 2:
            draw_component_on_image(i, labels, to_view, (255, 0, 0))
        i += 1
    # print('max_cluster_index=', max_cluster_index)
    before_splitting = copy.deepcopy(to_view)
    i = 0
    total_segmented = 0
    # print('components to be checked: ', len(clustered))
    # print('clustered=', clustered)
    for component in clustered:
        i += 1
        component_image = np.zeros_like(image)
        draw_component_on_image(i, labels, component_image, 255)
        x, y, w, h, = cv2.boundingRect(component_image)
        component_image = component_image[y: y + h, x: x + w]
        # cv2.imwrite(str(i) + '_' + str(component) + '.png', component_image)
        # find bimodal gaussian fit
        # calc area of each gaussian in (0, len(histogram)
        # calc overlapping area
        if component == max_cluster_index:
            # print('INDEX = ', i, )
            component_image = np.zeros_like(image)
            draw_component_on_image(i, labels, component_image, 255)
            x, y, w, h, = cv2.boundingRect(component_image)
            component_image = component_image[y: y + h, x: x + w]
            new_image = cv2.cvtColor(np.zeros_like(component_image), cv2.COLOR_GRAY2RGB)
            new_image[component_image == 255] = (255, 0, 0)
            # cv2.imwrite('component_before.png', new_image)
            component_image = new_image
            component_indexes = list(zip(*np.where(labels == i)))
            # create histogram then split !
            y_indexes = [index[0] for index in component_indexes]
            # print('y_indexes=', y_indexes)
            min_y = min(y_indexes)
            max_y = max(y_indexes)
            j = 0
            histogram = [0 for x in range(min_y, max_y + 1)]
            print('old_width=', old_width, 'component_width=', len(histogram))
            if old_width is not None and old_width > len(histogram):
                continue
            for y_val in range(min_y, max_y + 1):
                histogram[j] = y_indexes.count(y_val)
                j += 1

            candidate_xs = [get_n_cut_valleys(histogram, i) for i in range(2, 10)]
            candidates_xs = [elem[0] for elem in candidate_xs]
            candidates_xs_errors = [elem[1] for elem in candidate_xs]
            # print('candidate_xs[0]=', candidate_xs[0])
            # print('candidate_xs=', candidate_xs)

            candidate_xs = list(filter(lambda elem: elem is not None, candidate_xs))

            all_xs = candidate_xs[0]
            min_average_error = sum(candidate_xs[0][1]) / len(candidate_xs[0][1])
            for candidate in candidate_xs[1:]:
                new_error = sum(candidate[1]) / len(candidate[1])
                if min_average_error > new_error:
                    all_xs = candidate
                    min_average_error = new_error
            if min_average_error == np.inf:
                continue
            all_xs = calc_valleys(all_xs[0], histogram)
            # print('all_xs=', all_xs)

            # print('min_valley=', min_valley)
            # plt.plot(x_s, gauss(x_s, mu1, sigma1, A1), color='green', lw=3, label='gauss1')
            # plt.plot(x_s, gauss(x_s, mu2, sigma2, A2),
            #         color='blue', lw=3, label='gauss2')
            # plt.plot(x_s, bimodal(x_s, mu1, sigma1, A1, mu2, sigma2, A2), color='red', lw=3, label='bi-modal')
            # plt.savefig('plotted_gaussians.png', dpi=1200)

            # cv2.destroyAllWindows()
            # remove
            total_segmented += 1
            range_to_remove = 15
            # this to make sure the distance between the two suggested points is high enough
            # if len(xs) > 1 and xs[0][0] - xs[1][0] < 0.9 * len(histogram) / 2:
            #     xs = xs[1:]
            # TODO Think how to try and split the split images again. maybe another fit can be done?
            # TODO check height of each part in comparison to 2nd cluster size?! !

            for item in all_xs:
                min_valley = np.int32(item[0])
                # print(min_valley)
                if histogram[min_valley] > np.max(histogram) * 0.5:
                    continue
                ranges = [range(-range_to_remove, 0), range(0, range_to_remove)]
                for one_range in ranges:
                    for j in one_range:
                        y_to_remove = min_y + min_valley + j
                        cropped_y_to_remove = min_valley + j
                        indices_to_remove = [index for index in component_indexes if index[0] == y_to_remove]
                        if len(indices_to_remove) == 0:
                            continue
                        for idx in range(component_image.shape[1]):
                            try:
                                if np.any(component_image[cropped_y_to_remove, idx] != 0):
                                    component_image[cropped_y_to_remove, idx] = (255, 255, 255)
                            except IndexError as e:
                                print(str(e))
                        # print(indices_to_remove)
                        most_left = np.min([index[1] for index in indices_to_remove])
                        # print('most_left=', most_left)
                        most_right = np.max([index[1] for index in indices_to_remove])
                        # print('most_right=', most_right)
                        # print('distance to remove=', most_right - most_left)
                        # print('total=', component_image.shape[1])
                        if most_right - most_left > 0.6 * component_image.shape[1]\
                                and len(indices_to_remove) > 0.6 * np.max(histogram):
                            break
                        for index in indices_to_remove:
                            image[index] = 0
                            to_view[index] = (255, 255, 255)
            # cv2.imwrite('after_component_image.png', component_image)
            # plt.legend()
            # plt.show()
            # plt.clf()
    # print('total_segmented=', total_segmented)
    return image, to_view, before_splitting, total_segmented, average_width


# ---------------------------------------------------------------------------------
# document pre processing
def pre_process(path, file_name, str_idx=''):
    def cluster_elements(all_stats):
        max_cluster_threshold = 0.15
        n_clusters = 11
        data_list = [stat[5] for stat in all_stats]
        # print('data_list=', data_list)
        data = np.asarray(data_list).reshape(-1, 1)
        # k_means = KMeans(n_clusters=n_clusters)
        k_means = GaussianMixture(n_components=n_clusters)
        k_means.fit(data)
        y_k_means = k_means.predict(data)

        cluster_size = [len(list(filter(lambda x0: x0 == i, y_k_means))) for i in range(n_clusters)]
        total = sum(cluster_size)

        cluster_total = [ft.reduce(lambda x, y: x + y[1] if y[0] == i else x, zip(list(y_k_means), data_list), 0)
                         for i in range(n_clusters)]
        minimum_cluster = np.argmin([c[1]/c[0] if c[0] > 0 else 9999 for c in zip(cluster_size, cluster_total)])
        minimum_cluster_size = cluster_size.count(minimum_cluster)
        # THIS IS A THRESHOLD
        # print('minimum_cluster_size=', minimum_cluster_size, 'total=', total)
        if minimum_cluster_size / total < max_cluster_threshold:
                return y_k_means, minimum_cluster
        else:
            return y_k_means, None

    # load image as gray-scale,

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    x, y, w, h = cv2.boundingRect(image)
    image = image[y + 1: y + h - 1, x + 1: x + w - 1]

    cv2.imwrite('./' + file_name + '/original_image.png', image)
    cv2.imwrite('./' + file_name + '/original_image_inverted.png', 255 - image)
    # image = cv2.erode(image, np.ones((3, 3), np.uint8), iterations=3)
    # cv2.imwrite('./' + file_name + '/dilated_image.png', image)
    # using gaussian adaptive thresholding
    # image = cv2.adaptiveThreshold(image, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 255, 9)
    image = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    image = 1 - image
    cv2.imwrite('./' + file_name + '/otsu.png', image * 255)
    cv2.imwrite('./' + file_name + '/otsu_inverted.png', 255 - image * 255)
    # convert to binary using otsu binarization
    # image = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8, ltype=cv2.CV_32S)
    stats = np.asarray([np.append(stat[0], stat[1]) for stat in zip(range(num_labels), stats)])

    # removing small artifacts and diactrics, using k-means
    results, min_cluster = cluster_elements(stats)
    if min_cluster is not None:
        index = 0
        time_print(str_idx + 'elements to be deleted: ' + str(np.count_nonzero(results == 0)))
        for clustered in results:
            if clustered == min_cluster:
                # print('deleted:', index, 'size=', stats[index, 5])
                labels[labels == index] = 0
            index += 1

        labels[labels != 0] = 1
        # cv2.namedWindow('image')
        # cv2.imshow('image', image * 255)
        image_no_tiny_elements = op.and_(image, labels.astype(np.uint8))
        cv2.imwrite('./' + file_name + '/image_no_tiny_elements.png', image_no_tiny_elements * 255)
    else:
        time_print(str_idx + 'NO ELEMENTS to be deleted: MIN CLUSTER SIZE =' + str(np.count_nonzero(results == 0)))
        image_no_tiny_elements = image

    # split touching lines
    time_print(str_idx + 'split touching lines ...')

    # all_images = [copy.deepcopy(image_no_tiny_elements) for i in range(1)]
    # all_attempts = [split_touching_lines(img) for img in all_images]
    # all_removals = [all_attempts[i][3] for i in range(1)]
    # image_no_tiny_elements, to_view, before_splitting, total_segmented = all_attempts[np.argmax(all_removals)]
    image_no_tiny_elements, to_view, before_splitting, total_segmented, average_width = \
        split_touching_lines(image_no_tiny_elements)
    if to_view is None:
        time_print(str_idx + 'No touching lines need to be split! 1')
    else:
        time_print(str_idx + 'LINES SPLIT DONE! REMOVED= ' + str(total_segmented))
        cv2.imwrite('./' + file_name + '/before_remove_touching_lines_1.png', before_splitting)
        cv2.imwrite('./' + file_name + '/after_remove_touching_lines_1.png', to_view)
        cv2.imwrite('./' + file_name + '/removed_touching_lines_1.png', image_no_tiny_elements * 255)
        image_no_tiny_elements, to_view, before_splitting, total_segmented, average_width = \
            split_touching_lines(image_no_tiny_elements, average_width)
        if to_view is None:
            time_print(str_idx + 'No touching lines need to be split! 2 ')
        else:
            time_print(str_idx + 'LINES SPLIT DONE! REMOVED= ' + str(total_segmented))
            cv2.imwrite('./' + file_name + '/before_remove_touching_lines_2.png', before_splitting)
            cv2.imwrite('./' + file_name + '/after_remove_touching_lines_2.png', to_view)
            cv2.imwrite('./' + file_name + '/removed_touching_lines_2.png', image_no_tiny_elements * 255)

    # add white border around image of size 29
    white_border_added_image = cv2.copyMakeBorder(image, 39, 39, 39, 39, cv2.BORDER_CONSTANT, None, 0)
    for_view = copy.deepcopy(white_border_added_image)
    white_border_added_image_no_tiny_elements = cv2.copyMakeBorder(image_no_tiny_elements, 39, 39, 39, 39,
                                                                   cv2.BORDER_CONSTANT, None, 0)

    cv2.rectangle(white_border_added_image, (0, 0),
                  (white_border_added_image.shape[1] - 1, white_border_added_image.shape[0] - 1), 1)
    cv2.rectangle(white_border_added_image_no_tiny_elements, (0, 0),
                  (white_border_added_image_no_tiny_elements.shape[1] - 1,
                   white_border_added_image_no_tiny_elements.shape[0] - 1), 1)
    cv2.imwrite('./' + file_name + '/rectangle_white_border_added_image_no_tiny_elements.png',
                white_border_added_image_no_tiny_elements * 255)

    # TODO ANCHORS THAT ARE EDGE OF IMAGE - DUE TO THIS METHOD WE DONT DO THAT INSTEAD WE GIVE RECTANGLE CORNERS
    # excludes = [(0, 0), (skeleton.shape[1], 0), (0, skeleton.shape[0]), (skeleton.shape[1], skeleton.shape[0])]
    # these are the 4 anchors for the text found in the original image
    # anchors = [(x - 20, y - 20), (x + w + 20, y - 20), (x - 20, y + h + 20), (x + w + 20, y + h + 20)]
    # here we take ROI as the image
    # cv2.namedWindow('before')
    # cv2.imshow('before', white_border_added_image_no_tiny_elements * 255)
    x, y, w, h = cv2.boundingRect(white_border_added_image_no_tiny_elements)
    # print(white_border_added_image_no_tiny_elements.shape)
    white_border_added_image_no_tiny_elements = white_border_added_image_no_tiny_elements[y: y + h, x: x + w]
    # print(white_border_added_image_no_tiny_elements.shape)
    # cv2.namedWindow('after')
    # cv2.imshow('after', white_border_added_image_no_tiny_elements * 255)
    # cv2.waitKey()
    #exit()
    # change anchors to 4 corners
    anchors = [(0, 0), (white_border_added_image_no_tiny_elements.shape[1], 0),
               (0, white_border_added_image_no_tiny_elements.shape[0]),
               (white_border_added_image_no_tiny_elements.shape[1], white_border_added_image_no_tiny_elements.shape[0])]
    image_offset_values = (y, x)
    # invert images (now black is black and white is white)
    for_view = 1 - for_view
    black_border_added = 1 - white_border_added_image
    black_border_added_no_tiny_elements = 1 - white_border_added_image_no_tiny_elements

    cv2.imwrite('./' + file_name + '/preprocessed_image.png', black_border_added * 255)
    cv2.imwrite('./' + file_name + '/preprocessed_image_inverted.png', 255 - black_border_added * 255)

    cv2.imwrite('./' + file_name + '/preprocessed_image_no_tiny_elements.png', black_border_added_no_tiny_elements * 255)
    cv2.imwrite('./' + file_name + '/preprocessed_image_no_tiny_elements_inverted.png',
                255 - black_border_added_no_tiny_elements * 255)

    return for_view, black_border_added_no_tiny_elements, anchors, image_offset_values


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
# retrieves connected pixels that are part of the edge pixels
# to be used for the bfs algorithm
# 8 connected neighborhood of a pixel
def connected_candidates(pixel, skeleton):
    def add_offset(offset):
        return tuple(map(op.add, pixel, offset))

    def in_bounds_and_true(p):
        r, c = add_offset(p)
        if 0 <= r < skeleton.shape[0] and 0 <= c < skeleton.shape[1] and skeleton[r][c]:
            return True
        else:
            return False

    eight_connected = list(filter(in_bounds_and_true, [(1, 0), (0, 1), (-1, 0), (0, -1),
                                                                      (1, 1), (-1, 1), (-1, -1), (1, -1)]))

    return [add_offset(offset) for offset in eight_connected]


# ---------------------------------------------------------------------------------
# extract local maxima pixels
def calculate_local_maxima_mask(image):
    def uint8_array(rows):
        return np.array(rows).astype(np.uint8)

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
            [0, 0, 0,  1, 0],
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


def time_print(msg):
    print('[' + str(datetime.datetime.now()) + ']', msg)


# ---------------------------------------------------------------------------------
# All vertexes with one degree (take part of one edge only) - they are removed
# All vertexes with two degree (take part of two edges exactly) - they are merged
# if three edges create a three edged circle: (u,v) (v,w) (w,u), we remove (w,u)
# this is done iteratively, until all vertexes have a degree of three or more!
def prune_graph(skeleton, iter_index, file_name, anchors, idx_str=''):
    def in_bounds(p):
        r, c = p
        if 0 <= r < skeleton.shape[1] and 0 <= c < skeleton.shape[0]:
            return True
        else:
            return False

    def add_range(tup):
        v_x, v_y = tup
        max_dist_v = [x for x in range(-7, 7 + 1)]
        max_dist_candidates_x = list(map(lambda x: x + v_x, max_dist_v))
        max_dist_candidates_y = list(map(lambda y: y + v_y, max_dist_v))
        return [(x, y) for x in max_dist_candidates_x for y in max_dist_candidates_y if in_bounds((x, y))]

    def unique_rows(a):
        a = np.ascontiguousarray(a)
        unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
        return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

    def identical(e1, e2):
        return e1[0] == e2[0] and e1[1] == e2[1]

    cv2.imwrite('./' + file_name + '/skel_' + str(iter_index) + '.png', skeleton.astype(np.uint8) * 255)
    cv2.imwrite('./' + file_name + '/skel_' + str(iter_index) + '_inverted.png', 255 - skeleton.astype(np.uint8) * 255)
    # important! removes pixels due to vertex removal from previous iteration
    skeleton = morphology.skeletonize(skeleton)
    # TODO add to paper information about this !!! remove redundant edges
    # summarization shows each edge, its start and end pixel, length, distance, etc
    # ALSO!! it has two types of edges, one that has two vertexes in graph
    # others that have one vertex only - those to be removed!
    # TODO THESE ARE EDGE END POINTS - TWO VERTEX COORDINATES FOR EACH EDGE
    branch_data = csr.summarise(skeleton)
    coords_cols = (['img-coord-0-%i' % i for i in [1, 0]] +
                   ['img-coord-1-%i' % i for i in [1, 0]])
    # removes duplicate entries! for some reason they are present in the result
    coords = unique_rows(branch_data[coords_cols].values).reshape((-1, 2, 2))

    # TODO Each Vertex to stay in the graph needs to have a degree of two or more
    # TODO Iteratively, we remove those that have less than two degree
    # TODO We stop only when there are no more vertexes left with low degree
    # TODO THEN WE EXTRACT THE EDGES - USING BFS ?! NEED TO FIND A GOOD WAY

    try_again = False

    len_before = len(coords)
    # TODO The 4 corners are excluded from this process. they are needed as anchors.
    # excludes = [(0, 0), (skeleton.shape[1], 0), (0, skeleton.shape[0]), (skeleton.shape[1], skeleton.shape[0])]
    excludes = anchors
    exclude = []
    excluded = []
    for ex in excludes:
        exclude.extend(add_range(ex))

    done = False
    while not done:
        changed = False
        flat_coords = [tuple(val) for sublist in coords for val in sublist]
        unique_flat_coords = list(set(flat_coords))
        current = 0
        while not changed and current < len(unique_flat_coords):
            item = unique_flat_coords[current]
            current += 1
            # print('item=', item, 'count=', flat_coords.count(item))
            # 1 degree vertexes are to be removed from graph
            if flat_coords.count(item) < 2:
                # print('item=', item)
                if item in exclude:
                    excluded.append((item[1], item[0]))
                    continue
                changed = True
                coords = list(filter(lambda x: tuple(x[0]) != item and tuple(x[1]) != item, coords))
                # print('flat_coords.count(item)=', flat_coords.count(item), 'fc=', fc)
            # 2 degree vertexes need their edges to be merged
            elif flat_coords.count(item) == 2:
                changed = True
                fc = list(filter(lambda x: tuple(x[0]) == item or tuple(x[1]) == item, coords))
                # print('flat_coords.count(item)=', flat_coords.count(item), 'fc=', fc)
                if len(fc) != 2:
                    print('item=', item, 'fc=', fc)
                coords = list(filter(lambda x: tuple(x[0]) != item and tuple(x[1]) != item, coords))
                e1_s = fc[0][0]
                e1_e = fc[0][1]
                e2_s = fc[1][0]
                e2_e = fc[1][1]
                if ft.reduce(op.and_, map(lambda e: e[0] == e[1], zip(e1_s, e2_s))) and \
                        not identical(e1_e, e2_e):
                    coords.append(np.array([e1_e, e2_e]))
                elif ft.reduce(op.and_, map(lambda e: e[0] == e[1], zip(e1_s, e2_e))) and \
                        not identical(e1_e, e2_s):
                    coords.append(np.array([e1_e, e2_s]))
                elif ft.reduce(op.and_, map(lambda e: e[0] == e[1], zip(e1_e, e2_s))) and \
                        not identical(e1_s, e2_e):
                    coords.append(np.array([e1_s, e2_e]))
                elif ft.reduce(op.and_, map(lambda e: e[0] == e[1], zip(e1_e, e2_e))) and \
                        not identical(e1_s, e2_s):
                    coords.append(np.array([e1_s, e2_s]))
                else:
                    changed = False
        if not changed:
            done = True
            time_print(idx_str + 'before= ' + str(len_before) + ' after= ' + str(len(coords)))
            try_again = len_before != len(coords)
            # print(list(map(lambda co: (np.abs(co[0][0]-co[1][0]), np.abs(co[0][1]-co[1][1])), coords)))

    skel = cv2.cvtColor(skeleton.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)
    # cv2.namedWindow('skeleton')
    # cv2.imwrite('skeleton.png', skel)
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
    cv2.imwrite('./' + file_name + '/base_' + str(iter_index) + '.png', tmp_skel.astype(np.uint8) * 255)

    # create results, for each edge, we find its corresponding pixels
    # result list contains edge information: (start, end, [pixels])
    results = []
    results_dict = dict()
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
        results_dict[(start, end)] = result

    # filter out circles -> (u,v) (v,w) (w,u), then (w,u) is removed
    # (w,u) is the longest line out of the three in a 3-edge circle
    remove_candidates = set()
    for result in results_dict.keys():
        v, u = result
        candidates_v = [e for e in results_dict.keys() if v in e and u not in e]
        candidates_v_w = [e[0] if e[1] == v else e[1] for e in candidates_v]
        candidates_u = [e for e in results_dict.keys() if u in e and v not in e]
        candidates_u_w = [e[0] if e[1] == u else e[1] for e in candidates_u]
        for vw in candidates_v_w:
            for uw in candidates_u_w:
                if vw == uw:
                    w = vw
                    if (v, u) in results_dict.keys():
                        candidate_vu = (v, u)
                        len_vu = len(results_dict[(v, u)])
                    else:
                        candidate_vu = (u, v)
                        len_vu = len(results_dict[(u, v)])

                    if (w, v) in results_dict.keys():
                        candidate_wv = (w, v)
                        len_wv = len(results_dict[(w, v)])
                    else:
                        candidate_wv = (v, w)
                        len_wv = len(results_dict[(v, w)])

                    if (u, w) in results_dict.keys():
                        candidate_uw = (u, w)
                        len_uw = len(results_dict[(u, w)])
                    else:
                        candidate_uw = (w, u)
                        len_uw = len(results_dict[(w, u)])
                    if len_vu > len_uw and len_vu > len_wv:
                        remove_candidates.add(candidate_vu)
                    elif len_uw > len_vu and len_uw > len_wv:
                        remove_candidates.add(candidate_uw)
                    elif len_wv > len_vu and len_wv > len_vu:
                        remove_candidates.add(candidate_wv)
    # remove all edges that create a 3-edged circle,
    # for each edge the removed edge is the longest of all 3-edges of the circle
    time_print(idx_str + 'removing circles ...')
    remove_items = [(edge[0], edge[1], results_dict.pop(edge)) for edge in remove_candidates]
    # if no edge was removed above, but a circle is removed, we need a new iteration due to changes.
    if remove_items:
        try_again = True
    time_print(idx_str + 'before= ' + str(len(results)) + ' to_remove= ' + str(len(remove_items)))
    results = list(filter(lambda element: element not in remove_items, results))

    # create new skeleton following graph pruning
    skel = np.zeros_like(skeleton)
    for result in results:
        u, v, pixel_list = result
        for point in pixel_list:
            skel[point] = True

    # create result image after iteration is done and store to image for illustration
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
    cv2.imwrite('./' + file_name + '/iter_' + str(iter_index) + '.png', image)
    cv2.imwrite('./' + file_name + '/iter_' + str(iter_index) + '_inverted.png', 255 - image)
    return skel, results, excluded, try_again


# ---------------------------------------------------------------------------------
# ridge extraction
def ridge_extraction(image_preprocessed, file_name, anchors, idx_str=''):
    # apply distance transform then normalize image for viewing
    dist_transform = cv2.distanceTransform(image_preprocessed, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # normalize distance transform to be of values [0,1]
    normalized_dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    cv2.imwrite('./' + file_name + '/normalized_dist_transform.png', normalized_dist_transform * 255)
    cv2.imwrite('./' + file_name + '/normalized_dist_transform_inverted.png', 255 - normalized_dist_transform * 255)
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
    # dist_maxima = np.multiply(dist_maxima_mask_biggest_component, dist_transform)
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

    # print('file_name=', file_name)
    cv2.imwrite('./' + file_name + '/skeleton_original.png', dist_maxima_mask_biggest_component.astype(np.uint8) * 255)
    cv2.imwrite('./' + file_name + '/skeleton_original_inverted.png', 255 - dist_maxima_mask_biggest_component.astype(np.uint8) * 255)
    changed = True
    results = []
    iter_index = 0
    time_print(idx_str + 'pruning redundant edges and circles...')
    excluded = []
    while changed:
        time_print(idx_str + 'iter ' + str(iter_index))
        skeleton, results, excluded, changed = prune_graph(skeleton, iter_index, file_name, anchors, idx_str)
        iter_index += 1
    time_print(idx_str + 'done')

    colors = []
    image = cv2.cvtColor(np.zeros_like(skeleton, np.uint8), cv2.COLOR_GRAY2RGB)
    edge_dictionary = dict()
    for result in results:
        start, end, edge_list = result
        if start == end:  # TODO WHY THE GRAPH HAS THESE EDGES? BUG IN LIBRARY?
            continue
        edge_dictionary[(start, end)] = edge_list
        random_color = (rd.randint(50, 200), rd.randint(50, 200), rd.randint(50, 200))
        while random_color in colors:
            random_color = (rd.randint(50, 200), rd.randint(50, 200), rd.randint(50, 200))
        for point in edge_list:
            image[point] = random_color
        colors.append(random_color)
    cv2.imwrite('./' + file_name + '/skeleton_pruned.png', image)
    cv2.imwrite('./' + file_name + '/skeleton_pruned_inverted.png', 255 - image)

    # cv2.namedWindow('resultFinal')
    # cv2.imshow('resultFinal', image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # get vertexes
    graph_vertexes = list(set([tuple(val) for sublist in edge_dictionary.keys() for val in sublist]))

    return skeleton, edge_dictionary, graph_vertexes, excluded


# -
#
def calculate_abs_angle_difference(u, v, w):
    def find_min_angle(p_u, p_v):
        # \__ u,v v,w1
        x_w_1 = p_v[0] + np.abs(p_v[0] - p_u[0])
        y_w_1 = p_v[1]
        angle_1 = calculate_abs_angle(p_u, p_v, (x_w_1, y_w_1))
        # _\ w2,v v,u
        x_w_2 = p_v[0] - np.abs(p_v[0] - p_u[0])
        y_w_2 = p_v[1]
        angle_2 = calculate_abs_angle(p_u, p_v, (x_w_2, y_w_2))
        # \ u,v
        #  | v,w2
        x_w_3 = p_v[0]
        y_w_3 = p_v[1] + np.abs(p_v[1] - p_u[1])
        angle_3 = calculate_abs_angle(p_u, p_v, (x_w_3, y_w_3))
        # \| u,v v,w3
        x_w_4 = p_v[0]
        y_w_4 = p_v[1] - np.abs(p_v[1] - p_u[1])
        angle_4 = calculate_abs_angle(p_u, p_v, (x_w_4, y_w_4))

        return np.min([angle_1, angle_2, angle_3, angle_4])
    before = calculate_abs_angle(u, v, w)
    after = np.abs(find_min_angle(u, v) - find_min_angle(v, w))
    print('before=', before, 'after=', after)
    return np.pi - after


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


def get_nearby_pixels_two_edges(v, w1, w2, edges_dictionary, max_dist):
    v_x, v_y = v
    max_dist_v = [x for x in range(-max_dist, max_dist + 1)]
    max_dist_candidates_x = list(map(lambda x: x + v_x, max_dist_v))
    max_dist_candidates_y = list(map(lambda y: y + v_y, max_dist_v))

    left_column = list(map(lambda e: (v_x - max_dist, e), max_dist_candidates_y))
    right_column = list(map(lambda e: (v_x + max_dist, e), max_dist_candidates_y))
    top_column = list(map(lambda e: (e, v_y - max_dist), max_dist_candidates_x))
    bottom_column = list(map(lambda e: (e, v_y + max_dist), max_dist_candidates_x))

    junction_pixels = dict()
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

    return w1_in_radius[0], w2_in_radius[0]


# ---------------------------------------------------------------------------------
# get_nearby_pixels
def get_nearby_pixels(u, v, w1, w2, edges_dictionary, max_dist):
    v_x, v_y = v
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
# calculate edge t scores using local region only
#
def calculate_edge_scores_local(u, v, edge_dictionary, t_scores, max_dist):
    # print('u=', u, 'v=', v)
    junction_v_edges = [edge for edge in edge_dictionary
                        if (edge[0] == v and edge[1] != u) or (edge[0] != u and edge[1] == v)]
    # print(junction_v_edges)
    v_edges = [e[0] if e[1] == v else e[1] for e in junction_v_edges]
    for combination in it.combinations(v_edges, 2):
        w1, w2 = combination
        in_u, in_w1, in_w2 = get_nearby_pixels(u, v, w1, w2, edge_dictionary, max_dist=max_dist)
        uv_vw1 = calculate_abs_angle(in_u, v, in_w1)
        uv_vw2 = calculate_abs_angle(in_u, v, in_w2)
        w1v_vw2 = calculate_abs_angle(in_w1, v, in_w2)
        uv_bridge = np.abs(np.pi - w1v_vw2) + np.abs(np.pi / 2.0 - uv_vw1) + np.abs(np.pi / 2.0 - uv_vw2)
        vw1_bridge = np.abs(np.pi - uv_vw1) + np.abs(np.pi / 2.0 - uv_vw2) + np.abs(np.pi / 2.0 - w1v_vw2)
        vw2_bridge = np.abs(np.pi - uv_vw2) + np.abs(np.pi / 2.0 - uv_vw1) + np.abs(np.pi / 2.0 - w1v_vw2)
        t_scores[(u, v, w1, w2)] = [(u, v, uv_bridge), (v, w1, vw1_bridge), (v, w2, vw2_bridge)]


# ---------------------------------------------------------------------------------
# finds "best" out of local region angle and complete edge angle - for each edge
# totals 6 possible combinations
#
def calculate_edge_scores(u, v, edge_dictionary, t_scores, excluded, max_dist):
    junction_v_edges = [edge for edge in edge_dictionary
                        if (edge[0] == v and edge[1] != u) or (edge[0] != u and edge[1] == v)]
    v_edges = [e[0] if e[1] == v else e[1] for e in junction_v_edges]
    # print('edge=', edge)
    # print('junction=', junction_v_edges)
    # For each edge we check u,v v,w1 v,w2
    # other side below...
    for combination in it.combinations(v_edges, 2):
        w1, w2 = combination
        # print(w1)
        # print(excluded)
        if w1 in excluded or w2 in excluded:
            continue
        # get coordinates in radius 9 - then calculate angle
        in_u, in_w1, in_w2 = get_nearby_pixels(u, v, w1, w2, edge_dictionary, max_dist=max_dist)
        # print('in_u=', in_u, 'in_w1=', in_w1,'v=', v, 'in_w2=', in_w2)
        u_s = [u, in_u]
        w1_s = [w1, in_w1]
        w2_s = [w2, in_w2]
        uv_vw1 = [calculate_abs_angle(one_u, v, one_w1) for one_u in u_s for one_w1 in w1_s]
        uv_vw2 = [calculate_abs_angle(one_u, v, one_w2) for one_u in u_s for one_w2 in w2_s]
        w1v_vw2 = [calculate_abs_angle(one_w1, v, one_w2) for one_w1 in w1_s for one_w2 in w2_s]
        uv_bridge = np.min([np.abs(np.pi - one_w1v_vw2) + np.abs(np.pi / 2.0 - one_uv_vw1) +
                            np.abs(np.pi / 2.0 - one_uv_vw2) for one_w1v_vw2 in w1v_vw2
                            for one_uv_vw1 in uv_vw1 for one_uv_vw2 in uv_vw2])
        vw1_bridge = np.min([np.abs(np.pi - one_uv_vw1) + np.abs(np.pi / 2.0 - one_uv_vw2) +
                             np.abs(np.pi / 2.0 - one_w1v_vw2) for one_uv_vw1 in uv_vw1
                             for one_uv_vw2 in uv_vw2 for one_w1v_vw2 in w1v_vw2])
        vw2_bridge = np.min([np.abs(np.pi - one_uv_vw2) + np.abs(np.pi / 2.0 - one_uv_vw1) +
                             np.abs(np.pi / 2.0 - one_w1v_vw2) for one_uv_vw2 in uv_vw2
                             for one_uv_vw1 in uv_vw1 for one_w1v_vw2 in w1v_vw2])
        t_scores[(u, v, w1, w2)] = [(u, v, uv_bridge), (v, w1, vw1_bridge), (v, w2, vw2_bridge)]


# ---------------------------------------------------------------------------------
# calculate_junctions_t_scores
def calculate_junctions_t_scores(edge_dictionary, excluded):
    t_scores = dict()
    for edge in edge_dictionary:
        # new t_scores added to t_scores variable inside calculate_edge_scores
        u, v = edge
        # print(u)
        # print(v)
        # print(excluded)
        if u in excluded or v in excluded:
            continue
        calculate_edge_scores(u, v, edge_dictionary, t_scores, excluded, max_dist=7)
        calculate_edge_scores(v, u, edge_dictionary, t_scores, excluded, max_dist=7)
    # return all possibilities
    return t_scores


# ---------------------------------------------------------------------------------
# calculate minimum l_score for each vertex
def calculate_junctions_l_scores(edge_dictionary, vertexes, excluded, max_dist=7):
    vertexes_l_scores = dict()
    for vertex in vertexes:
        if vertex in excluded:
            # print('vertex=', vertex)
            continue
        # print('v=', vertex)
        # get list of edges that that vertex is part of
        edges_of_vertex = [e for e in edge_dictionary.keys() if e[0] == vertex or e[1] == vertex]
        # print('edges_of_v=', edges_of_vertex)
        junction_l_scores = dict()
        junctions = []
        l_scores = []
        for combination in it.combinations(edges_of_vertex, 2):
            e1, e2 = combination
            e1_e1, e1_e2 = e1
            e2_e1, e2_e2 = e2
            if e1_e1 in excluded or e1_e2 in excluded or e2_e1 in excluded or e2_e2 in excluded:
                # print ('e1_e1=', e1_e1, 'e1_e2=', e2_e2, 'e2_e1=', e2_e1, 'e2_e2=', e2_e2)
                continue
            # print('e1=', e1, 'e2=', e2)
            w1 = e1[0] if e1[0] != vertex else e1[1]
            w2 = e2[0] if e2[0] != vertex else e2[1]
            # print('w1=', w1, 'w2=', w2)

            # get coordinates in radius 9 - then calculate angle
            in_w1, in_w2 = get_nearby_pixels_two_edges(vertex, w1, w2, edge_dictionary, max_dist=max_dist)

            angle_w1v_vw2 = calculate_abs_angle(in_w1, vertex, in_w2)

            edges_of_w1 = [e for e in edge_dictionary.keys() if e[0] == w1 and e[1] != vertex
                           or e[1] == w1 and e[0] != vertex]
            edges_of_w2 = [e for e in edge_dictionary.keys() if e[0] == w2 and e[1] != vertex
                           or e[1] == w2 and e[0] != vertex]
            # print('edges_of_w1=', edges_of_w1)
            # print('edges_of_w2=', edges_of_w2)
            for edge_of_w1 in edges_of_w1:
                w1_e1, w1_e2 = edge_of_w1
                if w1_e1 in excluded or w1_e2 in excluded:
                    # print('w1_e1=', w1_e1, 'w1_e2=', w1_e2)
                    continue
                z1 = edge_of_w1[0] if edge_of_w1[0] != w1 else edge_of_w1[1]
                in_v, in_z1 = get_nearby_pixels_two_edges(w1, vertex, z1, edge_dictionary, max_dist=max_dist)
                # print('z1=', z1)
                angle_z1w1_w1v = calculate_abs_angle(in_z1, w1, in_v)
                for edge_of_w2 in edges_of_w2:
                    w2_e1, w2_e2 = edge_of_w2
                    if w2_e1 in excluded or w2_e2 in excluded:
                        # print('w2_e1=', w2_e1, 'w2_e2=', w2_e2)
                        continue
                    z2 = edge_of_w2[0] if edge_of_w2[0] != w2 else edge_of_w2[1]
                    in_v, in_z2 = get_nearby_pixels_two_edges(w2, vertex, z2, edge_dictionary, max_dist=max_dist)
                    # print('z2=', z2)
                    angle_z2w2_w2v = calculate_abs_angle(in_z2, w2, in_v)
                    junctions.append((z1, w1, vertex, w2, z2))

                    l_scores.append(np.abs(np.pi - angle_w1v_vw2) * 0.5 +
                                    np.abs(np.pi - angle_z1w1_w1v) * 0.25 +
                                    np.abs(np.pi - angle_z2w2_w2v) * 0.25)
                    junction_l_scores[(z1, w1, vertex, w2, z2)] = np.abs(np.pi - angle_w1v_vw2) * 0.5 + \
                                                                  np.abs(np.pi - angle_z1w1_w1v) * 0.25 + \
                                                                  np.abs(np.pi - angle_z2w2_w2v) * 0.25
        # print(junction_l_scores)
        # min_junction = junctions[np.argmin(l_scores)]
        # print('min junction=', min_junction, 'min score=', min_score)
        if l_scores:  # TODO why would this happen
            min_score = np.min(l_scores)
            vertexes_l_scores[vertex] = min_score
    # print(vertexes_l_scores)
    return vertexes_l_scores


# ---------------------------------------------------------------------------------
#
def overlay_and_save(bridges, links, rest, edge_dictionary, image_preprocessed, file_name, score_type):
    image = 1 - image_preprocessed
    image *= 255
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = draw_edges(bridges, edge_dictionary, image, (255, 0, 0))
    image = draw_edges(links, edge_dictionary, image, (0, 255, 0))
    # rest = [x for x in edge_dictionary.keys() if x not in set(bridges).union(links)]
    image = draw_edges(rest, edge_dictionary, image, (0, 0, 255))
    cv2.imwrite('./' + file_name + '/overlayed_classifications_' + score_type + '.png', image)

    return image


# ---------------------------------------------------------------------------------
#
def greedy_classification(t_scores, edge_dictionary, skeleton, file_name, score_type, image_offset_values):
    bridges = set()
    links = set()
    use_later = set()
    index = 1
    while t_scores:
        if index % 500 == 0:
            time_print(len(t_scores))
        # else:
        #     print(len(t_scores), end=' ')
        index += 1
        # find minimum score for each junction
        min_t_scores = dict()
        for key in t_scores.keys():
            min_score_index = np.argmin(map(lambda e_1, e_2, score: score, t_scores[key]))
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

        # add new links to links set
        if new_bridge not in links:
            bridges.add(new_bridge)
            for link in two_links:
                e1, e2, _ = link
                new_link = (e1, e2)
                if new_link not in edge_dictionary.keys():
                    new_link = (e2, e1)
                links.add(new_link)
        # check for conflicts before adding them??
        # add new bridge to bridges set
        # remove minimum t score junction from t_scores
        t_scores.pop(min_score_key)
        # print('B=', bridges, 'L=', links)
    # print()
    skeleton = skeleton.astype(np.uint8)

    # find anchor edges
    anchors = [x for x in edge_dictionary.keys() if x not in set(bridges).union(links)]

    # mark edges connected to anchor points as conflict edges
    for bridge in bridges:
        for coord in [coord for edge in anchors for coord in edge]:
            if coord in bridge:
                links.add(bridge)
                break

    # draw_graph_edges(edge_dictionary, skeleton, 'before')
    image = cv2.cvtColor(np.zeros_like(skeleton), cv2.COLOR_GRAY2RGB)
    image = draw_edges(bridges, edge_dictionary, image, (255, 0, 0))
    image = draw_edges(links, edge_dictionary, image, (0, 255, 0))
    image = draw_edges(anchors, edge_dictionary, image, (0, 0, 255))

    cv2.imwrite('./' + file_name + '/classifications_' + score_type + '.png', image)
    cv2.imwrite('./' + file_name + '/classifications_' + score_type + '_inverted.png', 255 - image)

    # cv2.namedWindow('after')
    # cv2.imshow('after', image)
    # cv2.namedWindow('r')
    # cv2.imshow('r', image_preprocessed)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return bridges, links, anchors, edge_dictionary


# ---------------------------------------------------------------------------------
#
def create_v_scores(t_scores, l_scores):
    def normalise(max_val, min_val, val):
        return (val - min_val) / (max_val - min_val)

    # normalise t_scores [0, 1]
    all_scores = [x[2] for value in t_scores.values() for x in value]
    max_t = np.max(all_scores)
    min_t = np.min(all_scores)

    l_values = list(l_scores.values())
    max_l = np.max(l_values)
    min_l = np.min(l_values)

    # normalise l_scores [0, 1]
    # we add v l_score to the correct places in t_scores dictionary
    v_scores = dict()
    for t_score in t_scores.keys():
        v = t_score[1]
        l_score = l_scores[v]
        scores = t_scores[t_score]
        v_score = list(map(lambda x: (x[0], x[1], normalise(max_t, min_t, x[2]) +
                                      normalise(max_l, min_l, l_score)), scores))
        v_scores[t_score] = v_score

    return v_scores


# ---------------------------------------------------------------------------------
#
def combine_edges(bridges, links, rest, edge_dictionary):
    def merge_group(candidate_edges, edge_dict, adj_list, anchors, threshold=np.pi / 3):
        # we combine two links as one if angle between them is minimum
        done = False
        while not done:
            done = True
            merge_link_1 = []
            merge_link_2 = []
            merge_angles = 0
            # in find the 'best' two links for merge - those that have highest angle of all
            for link in candidate_edges:
                u, v = link
                # print('link=', link)
                neighbors = [l for l in candidate_edges if v in l and u not in l]
                real_neighbors = [e if e not in adj_list.keys() else adj_list[e] for e in neighbors]
                # print('neighbors=', neighbors)
                if real_neighbors:
                    angles = [calculate_abs_angle(u, v, e[0] if e[1] == v else e[1]) for e in real_neighbors]
                    # print('angles=', angles)
                    max_index = np.argmax(angles)
                    max_neighbor = real_neighbors[max_index]
                    is_anchor = False
                    # if we reached anchor points we skip
                    for coord in [coord for e in anchors for coord in e]:
                        if coord in link and coord in max_neighbor:
                            is_anchor = True
                            break
                    if not is_anchor and angles[max_index] > merge_angles:
                        merge_angles = angles[max_index]
                        merge_link_1 = link
                        merge_link_2 = max_neighbor

            # now we merge 'best' candidates together modifying the document graph
            if merge_link_1 and np.abs(np.pi - merge_angles) < threshold:

                done = False
                pixels_1 = edge_dict.pop(merge_link_1)
                pixels_2 = edge_dict.pop(merge_link_2)

                # print('pixels_1=', pixels_1)
                # print('pixels_2=', pixels_2)
                new_edge = (merge_link_1[0], merge_link_2[0] if merge_link_2[0] != merge_link_1[1] else merge_link_2[1])
                edge_dict[new_edge] = list(set(pixels_1 + pixels_2))
                candidate_edges = [e for e in candidate_edges if e != merge_link_1 and e != merge_link_2]
                candidate_edges.append(new_edge)
                # print('link_1=', merge_link_1, 'link_2=', merge_link_2, 'new_edge=', new_edge)

        return candidate_edges, edge_dict, adj_list

    # draw_graph_edges(edge_dictionary, res, 'before', wait_flag=False, overlay=True)

    can_be_both = [e for e in links if e in bridges]
    # bridges = [e for e in bridges if e not in can_be_both]
    definite_links = [e for e in links if e not in bridges]
    adjacency_list = dict()

    image_unmodified = cv2.cvtColor(np.zeros([833, 1172], np.uint8), cv2.COLOR_GRAY2RGB)
    # for edge in edge_dictionary.keys():
    #     edge_list = edge_dictionary[edge]
    #     image_unmodified = overlay_edges(image_unmodified, edge_list, (255, 0, 0))
    # cv2.imwrite('zero.png', image_unmodified)

    # remove bridges from graph that are not marked as conflict
    only_bridges = [bridge for bridge in bridges if bridge not in can_be_both]
    use_later = dict()
    for bridge in only_bridges:
        use_later[bridge] = edge_dictionary.pop(bridge)

    # image_unmodified = cv2.cvtColor(np.zeros([833, 1172], np.uint8), cv2.COLOR_GRAY2RGB)
    # for edge in edge_dictionary.keys():
    #     if edge[0] == edge[1]:
    #         print('[start] ERROR:', edge)
    #     edge_list = edge_dictionary[edge]
    #     image_unmodified = overlay_edges(image_unmodified, edge_list, (255, 0, 0))
    # cv2.imwrite('start.png', image_unmodified)

    # combine edges - first iteration for link edges
    definite_links, edge_dictionary, adjacency_list = merge_group(definite_links, edge_dictionary, adjacency_list, rest)

    both = definite_links + can_be_both
    # combine edges - second iteration, now we include conflict edges
    both, edge_dictionary, adjacency_list = merge_group(both, edge_dictionary, adjacency_list, rest,
                                                        threshold=np.pi / 5)

    # remove bridges from document graph that were not used in conflict stage
    only_bridges = [bridge for bridge in bridges if bridge in edge_dictionary.keys()]
    bridges_dict = dict()
    for bridge in only_bridges:
        bridges_dict[bridge] = edge_dictionary.pop(bridge)

    return bridges_dict, edge_dictionary, use_later


# ---------------------------------------------------------------------------------
# finalize_graph - remove wrong direction edges - and combine edges of two degree vertexes
# text_angle is text direction relative to x axis
def finalize_graph(combined_graph, only_bridges, anchors, threshold=np.pi / 4):
    def calc_angle(edge):
        u_e, v_e = edge
        # print('u=', u, 'v=', v)
        delta_x = v_e[0] - u_e[0]
        delta_y = v_e[1] - u_e[1]
        # print('delta_x=', delta_x, 'delta_y=', delta_y)
        angle = np.arctan2(delta_y, delta_x)
        return np.abs(angle)

    anchors = [(anchor[1], anchor[0]) for anchor in anchors]
    anchor_edges = [edge for edge in combined_graph.keys() if edge[0] in anchors or edge[1] in anchors]

    # flat_anchors = [tuple(val) for sublist in anchor_edges for val in sublist]
    # remove edges that their angle is not within text_angle threshold
    # remove_candidates = [edge for edge in combined_graph.keys() if edge[0] not in flat_anchors
    #                     and edge[1] not in flat_anchors and calc_angle(edge) < threshold]
    # print('remove_candidates=', remove_candidates)
    remove_candidates = [edge for edge in combined_graph.keys() if calc_angle(edge) < threshold]
    for candidate in set(remove_candidates + anchor_edges):
        combined_graph.pop(candidate)

    # merge two degree vertexes
    done = False
    while not done:
        all_vertexes = [coord for edge in combined_graph.keys() for coord in edge]
        vertexes = list(set(all_vertexes))
        done = True
        for vertex in vertexes:
            two_links = [link for link in combined_graph.keys() if vertex in link]
            if len(two_links) == 2:
                link_1, link_2 = two_links
                if link_1[0] == link_2[0]:
                    u = link_1[1]
                    w = link_2[1]
                    v = link_1[0]
                elif link_1[0] == link_2[1]:
                    u = link_1[1]
                    w = link_2[0]
                    v = link_1[0]
                elif link_1[1] == link_2[0]:
                    u = link_1[0]
                    w = link_2[1]
                    v = link_1[1]
                else:
                    u = link_1[0]
                    w = link_2[0]
                    v = link_1[1]
                new_edge = (u, w)
                if np.abs(np.pi - calculate_abs_angle(u, v, w)) < threshold:
                    pixels_1 = combined_graph.pop(link_1)
                    pixels_2 = combined_graph.pop(link_2)
                    combined_graph[new_edge] = list(set(pixels_1 + pixels_2))
                    done = False
                    break

    all_vertexes = [coord for edge in combined_graph.keys() for coord in edge]
    vertexes = list(set(all_vertexes))
    # cv2.namedWindow('edge')
    # for vertex in vertexes:
        # two_links = [link for link in combined_graph.keys() if vertex in link]
        # print(len(two_links), 'vertex=', two_links)
        # image_unmodified = cv2.cvtColor(np.zeros([833, 1172], np.uint8), cv2.COLOR_GRAY2RGB)
        # edge_list = combined_graph[two_links[0]]
        # image_unmodified = overlay_edges(image_unmodified, edge_list, (255, 0, 0))
        # image_unmodified[two_links[0][0]] = (255, 255, 255)
        # image_unmodified[two_links[0][1]] = (255, 255, 255)
        # cv2.imshow('edge', image_unmodified)
        # cv2.waitKey()
    # cv2.destroyAllWindows()

    # add back bridges following this rule: if (u,v) and (w,z) are two edges in combined_graph
    # check if deg(v)=1 and deg(w)=1 and (v,w) \in only_bridges
    # then remove (u,v) remove (w,z) add (u,z) instead
    done = False
    # print('only_bridges_total=', len(only_bridges.keys()))
    # print('combined_graph.keys=', combined_graph.keys())
    # print('------------------------------')
    # print('only_bridges.keys=', only_bridges.keys())
    while not done:
        done = True
        for bridge in only_bridges.keys():
            v, w = bridge
            # print('v=', v, 'w=', w)
            link_1 = [link for link in combined_graph.keys() if v in link]
            link_2 = [link for link in combined_graph.keys() if w in link]
            # print('link_1=', link_1)
            # print('link_2=', link_2)
            if len(link_1) == 1 and len(link_2) == 1:
                # print('adding back bridge!!')
                pixels_1 = combined_graph.pop(link_1[0])
                pixels_2 = combined_graph.pop(link_2[0])
                pixels_3 = only_bridges[bridge]
                new_edge = (link_1[0][0] if link_1[0][0] != v else link_1[0][1],
                            link_2[0][0] if link_2[0][0] != w else link_2[0][1])
                combined_graph[new_edge] = list(set(pixels_1 + pixels_2 + pixels_3))
                done = False
                break

    return combined_graph


# ---------------------------------------------------------------------------------
# main execution function
def execute(input_path, output_path):
    # retrieve list of images
    images = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    i = 1

    for image in images:
        file_name = image.split('.')[0]
        print('[' + str(i) + '/' + str(len(images)) + ']', file_name)
        file_name = output_path + file_name
        if os.path.exists(file_name) and os.path.isdir(file_name):
            shutil.rmtree(file_name)
        os.mkdir(file_name)

        # pre-process image
        time_print('pre-process image...')
        image_view, image_preprocessed, anchors, image_offset_values = pre_process(input_path + image, file_name)
        # create dir for results

        # extract ridges
        time_print('extract ridges, junctions...')
        # ridges_mask, ridges_matrix = ridge_extraction(image_preprocessed)
        skeleton, edge_dictionary, vertexes, excluded = ridge_extraction(image_preprocessed, file_name, anchors)

        # calculate T_scores for V
        time_print('calculating t scores ...')
        t_scores = calculate_junctions_t_scores(edge_dictionary, excluded)
        # calculate l_scores for V
        time_print('calculating l scores ...')
        l_scores = calculate_junctions_l_scores(edge_dictionary, vertexes, excluded)
        # classify using both t_scores and l_scores for v
        v_scores = create_v_scores(t_scores, l_scores)
        time_print('greedy manner labeling ...')
        bridges, links, rest, edge_dictionary = greedy_classification(v_scores, edge_dictionary, skeleton, file_name,
                                                                      'v_scores', image_offset_values)
        # overlay_and_save(bridges, links, rest, edge_dictionary, image_view, file_name, 'v_scores')
        image = 1 - image_view
        image *= 255
        res_no_finalization = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        res_finalization = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        time_print('combining graph edges ...')
        only_bridges, combined_graph, use_later = combine_edges(bridges, links, rest, edge_dictionary)

        draw_graph_edges(combined_graph, res_no_finalization, file_name, wait_flag=False, overlay=True,
                         image_offset_values=image_offset_values, file_name='final_result_no_finalize')

        time_print('finalizing document graph ...')

        finalized_combined_graph = finalize_graph(combined_graph, use_later, anchors)

        name = draw_graph_edges(finalized_combined_graph, res_finalization, file_name, wait_flag=False, overlay=True,
                                image_offset_values=image_offset_values, file_name='final_result_finalize')

        # result = draw_graph_edges(combined_graph, cv2.cvtColor(res, cv2.COLOR_RGB2GRAY), file_name, wait_flag=False,
        # overlay=False)
        # result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        # result[result != 0] = 1
        time_print('SAVED: ' + str(name))
        i += 1


def process_image_parallel(image_data, len_images, input_path, output_path):
    i, image = image_data
    idx_str = '[' + str(i) + '/' + str(len_images) + '] '
    file_name = image.split('.')[0]
    print('[' + str(i) + '/' + str(len_images) + ']', file_name)
    file_name = output_path + file_name
    if os.path.exists(file_name) and os.path.isdir(file_name):
        shutil.rmtree(file_name)
    os.mkdir(file_name)

    # pre-process image
    time_print(idx_str + 'pre-process image...')
    image_view, image_preprocessed, anchors, image_offset_values = pre_process(input_path + image, file_name, idx_str)
    # create dir for results
    # extract ridges
    time_print('[' + str(i) + '/' + str(len_images) + '] extract ridges, junctions...')
    # ridges_mask, ridges_matrix = ridge_extraction(image_preprocessed)
    skeleton, edge_dictionary, graph_vertexes, excluded = ridge_extraction(image_preprocessed, file_name, anchors,
                                                                           idx_str)
    image_preprocessed[image_preprocessed == 1] = 2
    image_preprocessed[image_preprocessed == 0] = 1
    image_preprocessed[image_preprocessed == 2] = 0

    # calculate t_scores for v
    time_print(idx_str + 'calculating t scores ...')
    t_scores = calculate_junctions_t_scores(edge_dictionary, excluded)
    # calculate l_scores for v
    time_print(idx_str + 'calculating l scores ...')
    l_scores = calculate_junctions_l_scores(edge_dictionary, graph_vertexes, excluded)
    # classify using both t_scores and l_scores for v
    v_scores = create_v_scores(t_scores, l_scores)
    time_print(idx_str + 'greedy manner labeling ...')
    bridges, links, rest, edge_dictionary = greedy_classification(v_scores, edge_dictionary, skeleton, file_name,
                                                                  'v_scores', image_offset_values)
    overlay_and_save(bridges, links, rest, edge_dictionary, image_view, file_name, 'v_scores')

    time_print(idx_str + 'combining graph edges ...')
    only_bridges, combined_graph = combine_edges(bridges, links, rest, edge_dictionary)

    image = 1 - image_view
    image *= 255
    res_no_finalization = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    res_finalization = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    draw_graph_edges(combined_graph, res_no_finalization, file_name, wait_flag=False, overlay=True,
                     image_offset_values=image_offset_values, file_name='final_result_no_finalize')
    time_print(idx_str + 'finalizing document graph ...')
    finalized_combined_graph = finalize_graph(combined_graph, only_bridges, anchors)

    name = draw_graph_edges(finalized_combined_graph, res_finalization, file_name, wait_flag=False, overlay=True,
                            image_offset_values=image_offset_values, file_name='final_result_finalize')

    # result = draw_graph_edges(combined_graph, cv2.cvtColor(res, cv2.COLOR_RGB2GRAY), file_name, wait_flag=False,
    # overlay=False)
    # result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    # result[result != 0] = 1
    time_print('SAVED: ' + str(name))

    return i


# ---------------------------------------------------------------------------------
# main execution function
def execute_parallel(input_path, output_path):
    # retrieve list of images
    images = [f for f in listdir(input_path) if isfile(join(input_path, f))]

    pool = ProcessPoolExecutor(max_workers=11)
    wait_for = [pool.submit(process_image_parallel, image, len(images), input_path, output_path) for image in zip(range(1, len(images)), images)]
    # results = [f.result() for f in futures.as_completed(wait_for)]
    i = 0
    total = len(images)
    for f in futures.as_completed(wait_for):
        i += 1
        time_print('[' + str(i) + '/' + str(total) + ']' + str(f.result()) + ' done!')


if __name__ == "__main__":
        # execute_parallel('./data/original/')
        # execute_parallel('./data/')
        # execute_parallel('./CSG18_data/', './CSG18_results/')
        # execute_parallel('./data/', './results/')
        execute('./data/', './results/')
        # execute_parallel('./CSG18_data/', './CSG18_results/')
        # execute_parallel('./CB55_data/', './CB55_results/')
