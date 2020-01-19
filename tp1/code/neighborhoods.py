#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Third script of the practical session. Neighborhoods in a point cloud
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
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

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


def brute_force_spherical(queries, supports, radius):

    neighborhoods = []
    for q in queries:
        distances = cdist(q[np.newaxis], supports,'sqeuclidean').reshape(-1)
        idx = (distances <= radius)
        neighborhoods.append(supports[idx])

    return neighborhoods


def brute_force_KNN(queries, supports, k):

    neighborhoods = []
    for q in queries:
        distances = cdist(q[np.newaxis], supports,'sqeuclidean').reshape(-1)
        idx = np.argpartition(distances,k)[:k]
        neighborhoods.append(supports[idx])

    return neighborhoods






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

    # Brute force neighborhoods
    # *************************
    #

    # If statement to skip this part if you want
    if True:

        # Define the search parameters
        neighbors_num = 1000
        radius = 0.2
        num_queries = 10

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Search spherical
        t0 = time.time()
        neighborhoods = brute_force_spherical(queries, points, radius)
        t1 = time.time()

        # Search KNN      
        neighborhoods = brute_force_KNN(queries, points, neighbors_num)
        t2 = time.time()

        # Print timing results
        print('{:d} spherical neighborhoods computed in {:.3f} seconds'.format(num_queries, t1 - t0))
        print('{:d} KNN computed in {:.3f} seconds'.format(num_queries, t2 - t1))

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        total_KNN_time = points.shape[0] * (t2 - t1) / num_queries
        print('Computing spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_spherical_time / 3600))
        print('Computing KNN on whole cloud : {:.0f} hours'.format(total_KNN_time / 3600))



    # KDTree neighborhoods
    # ********************
    #

    # If statement to skip this part if wanted
    if True:

        # Define the search parameters
        num_queries = 1000

        leaf_sizes = [1,10,30, 60, 100, 500, 1000]
        timings = []
        for leaf_size in leaf_sizes:
            tree = KDTree(points, leaf_size=leaf_size)
            t1 = time.time()
            tree.query_radius(points[:num_queries],r=0.3)
            t2 = time.time()

            timings.append(t2 - t1)

        plt.semilogx(leaf_sizes, timings)
        plt.title("Time in seconds to do 1000 queries.")
        plt.show()

        # The previous plot suggests to use a leaf_size of 30
        tree = KDTree(points, leaf_size=30)

        radius_list = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]
        timings = []
        for radius in radius_list:
            t1 = time.time()
            tree.query_radius(points[:num_queries], r=radius)
            t2 = time.time()
            timings.append(t2-t1)
        
        plt.plot(radius_list,timings)
        plt.title("Time in seconds to do 1000 queries depending on the radius")
        plt.show()

        total_time = points.shape[0] * timings[1] / num_queries
        print("Estimated time to get all 20cm neighborhoods : {:.0f} minutes".format(total_time / 60))

        