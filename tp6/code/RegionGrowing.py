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

def local_PCA(points):

    bar = points.mean(axis=0)

    centered = (points - bar)[:,:,np.newaxis]
    cov = (np.matmul(centered, centered.transpose(0,2,1))).mean(axis=0)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    return eigenvalues, eigenvectors


def compute_planarities_and_normals(points, radius):

    normals = points * 0
    planarities = points[:, 0] * 0

    kdtree = KDTree(points)

    neighborhoods = kdtree.query_radius(points, radius)

    planarities = np.zeros(points.shape[0])
    eigenvectors = np.zeros((points.shape[0], 3, 3))

    for i, ind in enumerate(neighborhoods):
        val, vec = local_PCA(points[ind,:])
        planarities[i] = (val[1] - val[0]) / val[2]
        eigenvectors[i] = vec

    normals = eigenvectors[:,:,0]
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    return planarities, normals


def region_criterion(p1, p2, n1, n2, t1=0.1, t2=5):
    vec = p2 - p1
    dist = np.dot(n1, vec)

    angle = 180 * np.arccos(n1@n2) / np.pi
    
    # Check angle in both directions as normals orientation tends to be unstable
    return dist <= t1 and (angle <= t2 or angle >= 180 - t2)


def queue_criterion(p, t=0.1):
    return p >= t


def RegionGrowing(cloud, normals, planarities, radius):

    # TODO:

    tree = KDTree(cloud)

    N = len(cloud)
    region = np.zeros(N, dtype=bool)

    seed = np.random.choice(N)

    region[seed] = True
    Q = [seed]

    while len(Q) != 0:
        q = Q.pop()
        neighbors = tree.query_radius(cloud[q].reshape(1,3), radius)
        for p in neighbors[0]:
            if region[p]:
                continue
            if region_criterion(cloud[q], cloud[p], normals[q], normals[p]):
                region[p] = True
                if queue_criterion(planarities[p]):
                    Q.append(p)

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
    # print("Don't forget to use the real normals and planarities!!")
    # normals = np.zeros((len(points),3))
    # planarities = np.zeros(len(points))
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

    if True:
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

        labels = np.ones(len(points))
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
