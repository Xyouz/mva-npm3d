#
#
#      0===================0
#      |    6 Modelling    |
#      0===================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Plane detection by region growing
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Xavier ROYNARD - 19/02/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time


# ----------------------------------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def compute_planarities_and_normals(points, radius):

    normals = points * 0
    planarities = points[:, 0] * 0

    # TODO:

    return planarities, normals


def region_criterion(p1, p2, n1, n2):
    return True


def queue_criterion(p):
    return True


def RegionGrowing(cloud, normals, planarities, radius):

    # TODO:

    N = len(cloud)
    region = np.zeros(N, dtype=bool)

    return region


def multi_RegionGrowing(cloud, normals, planarities, radius, NB_PLANES=2):

    # TODO:

    plane_inds = np.arange(0, 0)
    plane_labels = np.arange(0, 0)
    remaining_inds = np.arange(0, N)

    return plane_inds, remaining_inds, plane_labels


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    N = len(points)

    # Computes normals of the whole cloud
    # ***********************************
    #

    # Parameters for normals computation
    radius = 0.2

    # Computes normals of the whole cloud
    t0 = time.time()
    planarities, normals = compute_planarities_and_normals(points, radius)
    t1 = time.time()
    print('normals and planarities computation done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('../planarities.ply',
              [points, planarities],
              ['x', 'y', 'z', 'planarities'])

    # Find a plane by Region Growing
    # ******************************
    #

    if False:
        # Define parameters of Region Growing
        radius = 0.2

        # Find a plane by Region Growing
        t0 = time.time()
        region = RegionGrowing(points, normals, planarities, radius)
        t1 = time.time()
        print('Region Growing done in {:.3f} seconds'.format(t1 - t0))

        # Get inds from bollean array
        plane_inds = region.nonzero()[0]
        remaining_inds = (1 - region).nonzero()[0]

        # Save the best plane
        write_ply('../best_plane.ply',
                  [points[plane_inds], colors[plane_inds], labels[plane_inds], planarities[plane_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'planarities'])
        write_ply('../remaining_points.ply',
                  [points[remaining_inds], colors[remaining_inds], labels[remaining_inds], planarities[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'planarities'])

    # Find multiple in the cloud
    # ******************************
    #

    if False:
        # Define parameters of multi_RANSAC
        radius = 0.2
        NB_PLANES = 10

        # Recursively find best plane by RANSAC
        t0 = time.time()
        plane_inds, remaining_inds, plane_labels = multi_RegionGrowing(points, normals, planarities, radius, NB_PLANES)
        t1 = time.time()
        print('multi RegionGrowing done in {:.3f} seconds'.format(t1 - t0))

        # Save the best planes and remaining points
        write_ply('../best_planes.ply',
                  [points[plane_inds], colors[plane_inds], plane_labels.astype(np.int32)],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'plane_label'])
        write_ply('../remaining_points_.ply',
                  [points[remaining_inds], colors[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

        print('Done')
