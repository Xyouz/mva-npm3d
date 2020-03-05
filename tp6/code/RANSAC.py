#
#
#      0===================0
#      |    6 Modelling    |
#      0===================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      First script of the practical session. Plane detection by RANSAC
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


def compute_plane(points):
    point = points.mean(axis=0)

    delta = points[1:] - points[0]
    normal = np.cross(delta[0], delta[1])
    normal = normal / np.linalg.norm(normal)
    return point, normal


def in_plane(points, ref_pt, normal, threshold_in=0.1):  
    # TODO: return a boolean mask of points in range
    diff = points - ref_pt.squeeze()
    dist = np.abs(np.sum(diff * normal.squeeze(), axis=1))
    indices = dist < threshold_in

    return indices


def RANSAC(points, NB_RANDOM_DRAWS=100, threshold_in=0.1, P=0.99):
    
    best_ref_pt = np.zeros((3,1))
    best_normal = np.zeros((3,1))
    
    n_point = len(points)

    max_vote = 0

    # The current estimate of the number of draws required to make sure the plane
    # found is the best with probability at least P
    max_draws = NB_RANDOM_DRAWS
    draws = 0

    while draws <= max_draws:
        draws += 1
        points_id = np.random.choice(n_point,size=3)
        ref, norm = compute_plane(points[points_id])

        current_vote = in_plane(points, ref, norm).sum()
        if current_vote > max_vote:
            max_vote = current_vote
            best_ref_pt = ref
            best_normal = norm
            
            max_draws = np.log(1-P)/np.log(1 - (max_vote/len(points))**3)
            max_draws = min(max_draws, NB_RANDOM_DRAWS)
    
    print("Plane found with {} draws".format(draws))
    return best_ref_pt, best_normal


def multi_RANSAC(points, NB_RANDOM_DRAWS=100, threshold_in=0.1, NB_PLANES=2):
    
    plane_inds = np.zeros(len(points), dtype=bool)
    remaining_inds = np.ones(len(points), dtype=bool)
    labels = np.zeros(len(points), dtype=int)

    for label in range(NB_PLANES):
        ind = remaining_inds.nonzero()[0]
        ref, normal = RANSAC(points[ind], NB_RANDOM_DRAWS, threshold_in)
        new_plan = in_plane(points[ind], ref, normal, threshold_in)
        np_ind = new_plan.nonzero()[0]
        plane_inds[ind[np_ind]] = True
        labels[ind[np_ind]] = label
        remaining_inds[ind[np_ind]] = False

    plane_inds = plane_inds.nonzero()[0]
    return plane_inds, remaining_inds.nonzero()[0], labels[plane_inds]


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

    # Computes the plane passing through 3 randomly chosen points
    # ***********************************************************
    #

    if False:

        # Define parameter
        threshold_in = 0.1

        # Take randomly three points
        pts = points[np.random.randint(0, N, size=3)]

        # Computes the plane passing through the 3 points
        t0 = time.time()
        ref_pt, normal = compute_plane(pts)
        t1 = time.time()
        print('plane computation done in {:.3f} seconds'.format(t1 - t0))

        # Find points in the plane and others
        t0 = time.time()
        points_in_plane = in_plane(points, ref_pt, normal, threshold_in)
        t1 = time.time()
        print('plane extraction done in {:.3f} seconds'.format(t1 - t0))
        plane_inds = points_in_plane.nonzero()[0]
        # Save the 3 points and their corresponding plane for verification
        pts_clr = np.zeros_like(pts)
        pts_clr[:, 0] = 1.0
        write_ply('../triplet.ply',
                  [pts, pts_clr],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])
        write_ply('../triplet_plane.ply',
                  [points[plane_inds], colors[plane_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

    # Computes the best plane fitting the point cloud
    # ***********************************
    #

    if True:

        # Define parameters of RANSAC
        NB_RANDOM_DRAWS = 150
        threshold_in = 0.05

        # Find best plane by RANSAC
        t0 = time.time()
        best_ref_pt, best_normal = RANSAC(points, NB_RANDOM_DRAWS, threshold_in)
        t1 = time.time()
        print('RANSAC done in {:.3f} seconds'.format(t1 - t0))

        # Find points in the plane and others
        points_in_plane = in_plane(points, best_ref_pt, best_normal, threshold_in)
        plane_inds = points_in_plane.nonzero()[0]
        remaining_inds = (1-points_in_plane).nonzero()[0]
        
        print("Number of points in the best extracted plane :", len(plane_inds), ", total number of points ", len(points))

        # Save the best extracted plane and remaining points
        write_ply('../best_plane.ply',
                  [points[plane_inds], colors[plane_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])
        write_ply('../remaining_points.ply',
                  [points[remaining_inds], colors[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

    # Find multiple planes in the cloud
    # *********************************
    #

    if True:

        # Define parameters of multi_RANSAC
        NB_RANDOM_DRAWS = 6000
        threshold_in = 0.05
        NB_PLANES = 5

        # Recursively find best plane by RANSAC
        t0 = time.time()
        plane_inds, remaining_inds, plane_labels = multi_RANSAC(points, NB_RANDOM_DRAWS, threshold_in, NB_PLANES)
        t1 = time.time()
        print('\nmulti RANSAC done in {:.3f} seconds'.format(t1 - t0))

        # Save the best planes and remaining points
        write_ply('../best_planes.ply',
                  [points[plane_inds], colors[plane_inds], plane_labels.astype(np.int32)],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'plane_label'])
        write_ply('../remaining_points_.ply',
                  [points[remaining_inds], colors[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

        print('Done')
