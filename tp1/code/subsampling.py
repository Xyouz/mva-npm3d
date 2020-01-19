#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Subsampling of a point cloud
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KDTree

from itertools import product

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def cloud_decimation(points, colors, labels, factor):

    # YOUR CODE
    decimated_points = points[::factor]
    decimated_colors = colors[::factor]
    decimated_labels = labels[::factor]

    return decimated_points, decimated_colors, decimated_labels


def grid_subsampling(points, voxel_size):

    #   Tips :
    #       > First compute voxel indices in each direction for all the points (can be negative).
    #       > Sum and count the points in each voxel (use a dictionaries with the indices as key).
    #         Remember arrays cannot be dictionary keys, but tuple can.
    #       > Divide the sum by the number of point in each cell.
    #       > Do not forget you need to return a numpy array and not a dictionary.
    #

    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)

    out = {}
    for point in points:
        index = tuple(((point - xyz_min)//voxel_size).astype(int))
        try:
            current_voxel = out[index]
            out[index] = current_voxel.append(point)
        except KeyError:
            out[index] = [point]

    for index in out:
        pass

    # kdtree = KDTree(points, leaf_size=30, metric="chebyshev")

    # linx = np.arange(xyz_min[0], xyz_max[0] + voxel_size/2, voxel_size)
    # liny = np.arange(xyz_min[1], xyz_max[1] + voxel_size/2, voxel_size)
    # linz = np.arange(xyz_min[2], xyz_max[2] + voxel_size/2, voxel_size)

    # subsampled_points = []
    # for x, y, z in product(linx, liny, linz):
    #     neighbors = kdtree.query_radius(np.array([[x,y,z]]), r=voxel_size/2)[0]
    #     if len(neighbors) > 0:
    #         avg = np.mean(points[neighbors],axis=0)
    #         subsampled_points.append(avg)

    return subsampled_points


def grid_subsampling_colors(points, colors, voxel_size):

    # YOUR CODE
    subsampled_points = None
    subsampled_colors = None

    return subsampled_points, subsampled_colors


# ------------------------------------------------------------------------------------------
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
    labels = data['label']    

    # Decimate the point cloud
    # ************************
    #

    # Define the decimation factor
    factor = 300

    # Decimate
    t0 = time.time()
    decimated_points, decimated_colors, decimated_labels = cloud_decimation(points, colors, labels, factor)
    t1 = time.time()
    print('decimation done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('../decimated.ply', [decimated_points, decimated_colors, decimated_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    # Subsample the point cloud on a grid
    # ***********************************
    #

    # Define the size of the grid
    voxel_size = 0.2

    # Subsample
    t0 = time.time()
    subsampled_points = grid_subsampling(points, voxel_size)
    t1 = time.time()
    print('Subsampling done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('../grid_subsampled.ply', [subsampled_points], ['x', 'y', 'z'])
    
    print('Done')
