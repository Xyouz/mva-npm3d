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

    out = {}
    for point in points:
        index = tuple(((point - xyz_min)//voxel_size).astype(int))
        try:
            out[index].append(point)
        except KeyError:
            out[index] = [point]

    subsampled_points = np.zeros((len(out), 3))
    for c, index in enumerate(out):
        position = np.array(out[index]).mean(axis=0)
        subsampled_points[c] = position

    return subsampled_points


def grid_subsampling_colors(points, colors, voxel_size):
    pointcolor = np.hstack((points, colors))
    xyz_min = points.min(axis=0)

    out = {}
    for pc in pointcolor:
        index = tuple(((pc[:3] - xyz_min)//voxel_size).astype(int))
        try:
            out[index].append(pc)
        except KeyError:
            out[index] = [pc]

    subsampled_points = np.zeros((len(out), 3))
    subsampled_colors = np.zeros((len(out), 3))
    for c, index in enumerate(out):
        position_color = np.array(out[index]).mean(axis=0)
        subsampled_points[c] = position_color[:3]
        subsampled_colors[c] = position_color[3:]
    return subsampled_points, subsampled_colors.astype(np.uint8)

def grid_subsampling_labels(points, colors, labels, voxel_size):    
    xyz_min = points.min(axis=0)
    label_min = labels.min()
    label_max = labels.max()

    bin_label = label_binarize(labels, classes = list(range(label_min, label_max + 1)))
    pointcolor = np.hstack((points, colors, bin_label))

    out = {}
    for pc in pointcolor:
        index = tuple(((pc[:3] - xyz_min)//voxel_size).astype(int))
        try:
            out[index].append(pc)
        except KeyError:
            out[index] = [pc]

    subsampled_points = np.zeros((len(out), 3))
    subsampled_colors = np.zeros((len(out), 3))
    subsampled_labels = np.zeros(len(out), dtype=int)
    for c, index in enumerate(out):
        pos_col_lab = np.array(out[index])
        position_color = pos_col_lab[:,:6].mean(axis=0)
        subsampled_points[c] = position_color[:3]
        subsampled_colors[c] = position_color[3:6]
        subsampled_labels[c] = label_min + pos_col_lab[:,6:].sum(axis=0).argmax()
    return subsampled_points, subsampled_colors.astype(np.uint8), subsampled_labels.astype(np.uint32)


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
    
    # Subsample with color
    t0 = time.time()
    subsampled_points, subsampled_colors = grid_subsampling_colors(points, colors, voxel_size)
    t1 = time.time()
    print('Subsampling with color done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('../grid_subsampled_color.ply', [subsampled_points, subsampled_colors], ['x', 'y', 'z', 'red', 'green', 'blue'])
    
    # Subsample with labels
    t0 = time.time()
    subsampled_points, subsampled_colors, subsampled_labels = grid_subsampling_labels(points, colors, labels, voxel_size)
    t1 = time.time()
    print('Subsampling with labels done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('../grid_subsampled_labels.ply', [subsampled_points, subsampled_colors, subsampled_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    
    print('Done')
