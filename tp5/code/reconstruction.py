#
#
#      0===========================================================0
#      |              TP5 Surface Reconstruction                   |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 02/02/2018
#



# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time

from skimage import measure

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Hoppe surface reconstruction
def compute_hoppe(points,normals,volume,number_cells,min_grid,length_cell):
    return

				
# EIMLS surface reconstruction
def compute_eimls(points,normals,volume,number_cells,min_grid,length_cell):
    return
				
				
				
if __name__ == '__main__':

    # Path of the file
    file_path = '../data/sphere_normals.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    normals = np.vstack((data['nx'], data['ny'], data['nz'])).T

	# Compute the min and max of the data points
    min_grid = np.copy(points[0, :])
    max_grid = np.copy(points[0, :])
    for i in range(1,points.shape[0]):
        for j in range(0,3):
            if (points[i,j] < min_grid[j]):
                min_grid[j] = points[i,j]
            if (points[i,j] > max_grid[j]):
                max_grid[j] = points[i,j]
				
	# Increase the bounding box of data points by decreasing min_grid and inscreasing max_grid
    min_grid = min_grid - 0.10*(max_grid-min_grid)
    max_grid = max_grid + 0.10*(max_grid-min_grid)

	# Number_cells is the number of voxels in the grid in x, y, z axis
    number_cells = 10 #100
    length_cell = np.array([(max_grid[0]-min_grid[0])/number_cells,(max_grid[1]-min_grid[1])/number_cells,(max_grid[2]-min_grid[2])/number_cells])
	
	# Create a volume grid to compute the scalar field for surface reconstruction
    volume = np.zeros((number_cells+1,number_cells+1,number_cells+1),dtype = np.float32)

	# Compute the scalar field in the grid
    compute_hoppe(points,normals,volume,number_cells,min_grid,length_cell)
    #compute_eimls(points,normals,volume,number_cells,min_grid,length_cell)

	# Compute the mesh from the scalar field based on marching cubes algorithm
    verts, faces, normals_tri, values_tri = measure.marching_cubes_lewiner(volume, level=0.0, spacing=(length_cell[0],length_cell[1],length_cell[2]))
	
	# Plot the mesh using matplotlib 3D
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)
    ax.set_xlim(0, number_cells*length_cell[0])
    ax.set_ylim(0, number_cells*length_cell[1])
    ax.set_zlim(0, number_cells*length_cell[2])
    plt.axis('off')
    plt.show()
	


