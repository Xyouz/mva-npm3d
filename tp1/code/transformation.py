#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      First script of the practical session. Transformation of a point cloud
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

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#
def find_centroid(data):
    return np.mean(data, axis=0)

def rotate_z(data, angle=-np.pi/2):
    m = np.array([
        [ np.cos(angle) ,np.sin(angle)  , 0],
        [-np.sin(angle)  ,np.cos(angle)  , 0],
        [ 0, 0, 1]
    ])
    return data @ m

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

    # Path of the file
    file_path = '../data/bunny.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate x, y, and z in a (N*3) point matrix
    points = np.vstack((data['x'], data['y'], data['z'])).T

    # Concatenate R, G, and B channels in a (N*3) color matrix
    colors = np.vstack((data['red'], data['green'], data['blue'])).T

    # Get the scalar field which represent density as a vector
    density = data['scalar_density']

    # Transform point cloud
    # *********************
    #
    #   Follow the instructions step by step
    #

    # Center the cloud on its centroid
    centroid = find_centroid(points)
    transformed_points = points - centroid
    
    # Divide its scale by a factor 2
    transformed_points = 0.5 * transformed_points

    # Apply a -90° rotation around z-axis (counterclockwise)
    transformed_points = rotate_z(transformed_points, -np.pi/2)

    # Recenter the cloud at the original position
    transformed_points = transformed_points + centroid

    # Apply a -10cm translation along y-axis
    transformed_points[:,1] += -0.1  # Assuming 1 unit <=> 1m


    # Save point cloud
    # *********************
    #
    #   Save your result file
    #   (See write_ply function)
    #

    # Save point cloud
    write_ply('../little_bunny.ply', [transformed_points, colors, density], ['x', 'y', 'z', 'red', 'green', 'blue', 'density'])

    print('Done')
