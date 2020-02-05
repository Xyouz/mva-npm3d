#
#
#      0=============================0
#      |    TP4 Point Descriptors    |
#      0=============================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Script of the practical session
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

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time

from tqdm import tqdm

# ------------------------------------------------------------------------------------------
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


def neighborhood_PCA(query_points, cloud_points, radius, use_tqdm=False):

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    
    kdtree = KDTree(cloud_points)

    neighborhoods = kdtree.query_radius(query_points, radius)

    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))

    if use_tqdm:
        disp = tqdm
    else :
        disp = lambda x:x

    for i, ind in disp(enumerate(neighborhoods)):
        val, vec = local_PCA(cloud_points[ind,:])
        all_eigenvalues[i] = val
        all_eigenvectors[i] = vec

    return all_eigenvalues, all_eigenvectors


def compute_features(query_points, cloud_points, radius, use_tqdm=False):

    # Compute the features for all query points in the cloud
    val, vec = neighborhood_PCA(query_points, cloud_points, radius, use_tqdm)
  
    e1 = val[:,2] + 1e-10
    e2 = val[:,1]
    e3 = val[:,0]
    
    L = 1 - e2/e1
    P = (e2 - e3) / e1
    S = e3 / e1
    O = (e1*e2*e3)**(1/3)
    A = 1 - e3/e1
    E = - np.sum(val * np.log(val + 1e-10), axis=1)
    Sigma = val.sum(axis=1)
    C = e3 / Sigma

    ez = np.zeros((3,1))
    ez[2,0] = 1.

    normal = vec[:,:,0]
    V = 2 * np.arcsin(np.abs(normal@ez)) / np.pi

    return V, L, P, S, O, A, E, Sigma, C


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # PCA verification
    # ****************
    #

    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        eigenvalues, eigenvectors = local_PCA(cloud)

        # Print your result
        print(eigenvalues)

        # Expected values :
        #
        #   [lambda_3; lambda_2; lambda_1] = [ 5.25050177 21.7893201  89.58924003]
        #
        #   (the convention is always lambda_1 >= lambda_2 >= lambda_3)
        #

    # Normal computation
    # ******************
    #

    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        query = cloud

        val, vec = neighborhood_PCA(query, cloud, 0.5)

        normal = vec[:,:,0]

        # Naive alignement method
        rectifier = np.sign((normal*[1,0.2,1]).sum(axis=1, keepdims=True))
        normal = normal * rectifier

        # We also save the value of lambda_min
        write_ply('../Lille_small_normal.ply', [query, normal, val[:,0]], ['x', 'y', 'z', 'nx', 'ny', 'nz', 'label'])


    # Features computation
    # ********************
    #

    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        query = cloud#[::1000]

        vert, lin, plan, spher = compute_features(query, cloud, 0.5)
                
        write_ply('../Lille_small_feat.ply', [query, vert, lin, plan, spher], ['x', 'y', 'z', 'vert', 'lin', 'plan', 'spher'])
