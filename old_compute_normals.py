import os
import sys
import numpy as np
sys.path.insert(0, '../utils')
sys.path.insert(0, '../models')
sys.path.insert(0, '../trained_models')
from IPython.display import display
import argparse
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import math
import time
from scipy.stats import multivariate_normal
import pickle



parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='scene10_dyn_train_00_000000', help='full path to input point cloud')
parser.add_argument ('--output_path', type=str, default='../log/outputs/results', help='full path to input point cloud')
parser.add_argument('--gpu_idx', type=int, default=0, help='index of gpu to use, -1 for cpu')
parser.add_argument('--trained_model_path', type=str, default='../trained_models/DeepFit',
                    help='path to trained model')
parser.add_argument('--mode', type=str, default='DeepFit', help='how to compute normals. use: DeepFit | classic')
parser.add_argument('--k_neighbors', type=int, default=256, help='number of neighboring points for each query point')
parser.add_argument('--jet_order', type=int, default=3,
                    help='order of jet to fit: 1-4. if in DeepFit mode, make sure to match training order')
parser.add_argument('--compute_curvatures', type=bool, default=True, help='true | false indicator to compute curvatures')
args = parser.parse_args()

all_values = []
all_dot_values = []
final_cuboids = []
xyz_lst = []
n_lst = []
n_flow = []
xyz_lst_new = []

# def vector_magnitude(x, y):
#     return math.sqrt(x**2 + y**2)

def normal_flow(x,y,z):

    column_vector = np.array([[x], [y]])
    #coloum_vector = np.transpose(coloum_vector)
    squared_norm = np.linalg.norm((column_vector))**2
    result = (-z * (((column_vector)) / squared_norm))
    result = result*166.666666667
    
    return  result
def avg(x):
    squared_sum = np.sum(x**2)
    avg_squared_sum = squared_sum / np.size(x)
    avg = np.sqrt(avg_squared_sum)
    return avg
    

def divide_cuboid(cuboid_dimensions,x,y,z):
    divisions = []
    
    # Get the dimensions of the cuboid
    length, width, height = cuboid_dimensions
    
    # Calculate the dimensions of each divided cuboid
    divided_length = length / 2
    divided_width = width / 2
    divided_height = height / 2
    
    # Generate the divisions
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x_start = i * divided_length+x

                x_end = x_start + divided_length

                y_start = j * divided_width+y
                y_end = y_start + divided_width
                z_start = k * divided_height+z
                z_end = z_start + divided_height
                
                division = {
                    'x': (x_start, x_end),
                    'y': (y_start, y_end),
                    'z': (z_start, z_end)
                }
                divisions.append(division)
    return divisions

# def calculate_normal_vector_pca(points):
#     if len(points) < 3:
#         raise ValueError("You need at least 3 points to calculate a normal vector.")

#     points_array = np.array(points)

#     # Calculate the centroid of the points
#     centroid = np.mean(points_array, axis=0)

#     # Subtract the centroid to center the points
#     centered_points = points_array - centroid

#     # Calculate the covariance matrix
#     covariance_matrix = np.dot(centered_points.T, centered_points)

#     # Perform eigenvalue decomposition on the covariance matrix
#     eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

#     # Find the eigenvector corresponding to the smallest eigenvalue
#     normal_vector = eigenvectors[:, 0]

#     # Normalize the normal vector
#     normalized_normal = normal_vector / np.linalg.norm(normal_vector)

#     return normalized_normal


import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

all_points_x = []

all_points_y = []

all_points_z = []

def generate_random_point_in_circle(radius, center_x, center_y):
    angle = random.uniform(0, 2*math.pi)
    x = center_x + radius * math.cos(angle)
    y = center_y + radius * math.sin(angle)
    return x, y

# def generate_random_points_in_3d_circle(radius, center_x, center_y, center_z, num_points):
#     points = []
#     for _ in range(num_points):
#         x, y = generate_random_point_in_circle(radius, center_x, center_y)
#         z = (center_z)
#         points.append((x, y, z))
#     return points

# # Set the parameters
# radius = 100
# center_x = 50
# center_y = 240
# center_z = 50
# num_points = 300

# # Generate random points
# for i in range (100):

#     random_points = generate_random_points_in_3d_circle(radius, center_x, center_y, center_z, num_points)
#     center_z = center_z+10
#     center_x = center_x+5
    
    
    

#     # Separate the points into x, y, and z coordinates
#     x_coords = [point[0] for point in random_points]
#     y_coords = [point[1] for point in random_points]
#     z_coords = [point[2] for point in random_points]
#     all_points_x.append(x_coords)
#     all_points_y.append(y_coords)
#     all_points_z.append(z_coords)

# # Print the generated points

points_xy = np.load(args.input + '/dataset_events_xy.npy').astype('float32')
points_t = np.load(args.input + '/dataset_events_t.npy').astype('float32')

all_values = []
all_dot_values = []
final_cuboids = []
xyz_lst = []
n_lst = []

idx_frame = 1
ts = points_t[0]

dtt = 0.0256


#@jit(nopython = True)
while ts <= points_t[-1]:
    samples = [0,8,72,584,4680,37448,299492,2396644]





    left_indices = np.searchsorted(points_t, ts)

   
    right_indices = np.searchsorted(points_t, ts + dtt, side='right')


    xyz = np.array([points_xy[left_indices:right_indices, 0], points_xy[left_indices:right_indices, 1], 10000*points_t[left_indices:right_indices]]).T
    
    points = np.array(xyz)
    X, Y, Z = xyz[:,0],xyz[:,1],xyz[:,2]
    cuboid = [640,480,10000]


    all_cuboids = []
    final_cuboids = []
    first_cuboid = { 'cuboid' : cuboid,
            'x': [0, 640],
            'y': [0, 480],
            'z': [0, 256]

    }
    all_cuboids.append(first_cuboid)


    

    for i in range (100000000000):
      
        
        
        if all_cuboids == []:
           
            break
        first_dict = all_cuboids[0]

        divisions = divide_cuboid(first_dict['cuboid'],first_dict['x'][0],first_dict['y'][0],first_dict['z'][0])
        # print(divisions)
        res_x = list(map(itemgetter('x'), divisions))
        res_y = list(map(itemgetter('y'), divisions))
        res_z = list(map(itemgetter('z'), divisions))
        # new_res_x = np.ravel(res_x)
        # set_x = set(new_res_x)
        # print(set_x)

    

        if i == samples[0]:
           
            cuboid[0] = cuboid[0]/2
            cuboid[1] = cuboid[1]/2
            cuboid[2] = cuboid[2]/2
            samples.pop(0)
        





        for j in range (8):
           
            
            x_start = res_x[j][0]
            x_end = res_x[j][1]
            y_start = res_y[j][0]
            y_end = res_y[j][1]
            z_start = res_z[j][0]
            z_end = res_z[j][1]

            x_values = points[:, 0]  
            y_values = points[:, 1]  
            z_values = points[:, 2]  

            x_mask = np.logical_and(x_start <= x_values, x_values <= x_end)
            y_mask = np.logical_and(y_start<= y_values, y_values <= y_end)
            z_mask = np.logical_and(Z[0] + z_start <= z_values, z_values <= Z[0] + z_end)

            combined_mask = np.logical_and(np.logical_and(x_mask, y_mask), z_mask)
            selected_points = points[combined_mask]
            num_of_points = np.shape(selected_points)
            f_points = selected_points
            
            
            if num_of_points[0]<3:
               
                continue
            selected_points = np.array(selected_points)
            


            # Calculate the centroid of the points
            centroid = np.mean(selected_points, axis=0)
            selected_points -= centroid

            
            

            selected_points = np.array(selected_points)
            # Compute the covariance matrix
            covariance_matrix = np.cov(selected_points.T)
            # Perform eigenvalue decomposition on the covariance matrix
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
         

            # Find the eigenvector corresponding to the smallest eigenvalue
            min_eigenvalue_index = np.argmin(eigenvalues) 
            normal_vector = eigenvectors[:, min_eigenvalue_index]
            
            
            dot_product = np.dot(selected_points, normal_vector)
            av_pro = avg(dot_product)
            normal__flow = normal_flow(normal_vector[0],normal_vector[1],normal_vector[2])
            normal__flow = [item for sublist in normal__flow for item in sublist]

           
            

            if av_pro<3:
    
                    sss = np.shape(f_points)
                    for n in range (sss[0]):
                        xyz_lst.append([f_points[n][0],f_points[n][1]])

                        xyz_lst_new.append(f_points[n])
                        n_lst.append(normal_vector)
                        n_flow.append(normal__flow)
        
            else:
               
                small_cuboids= { 'cuboid' : cuboid,
                        'x': [x_start],
                        'y': [y_start],
                        'z': [z_start]
                    }
                all_cuboids.append(small_cuboids)
        
      

        all_cuboids.pop(0)
    
    


    xyz_lst = np.array(xyz_lst)
    n_lst = np.array(n_lst)
    n_flow = np.array(n_flow)
    xyz_lst_new = np.array(xyz_lst_new)
    
    
    # plt.plot(all_values,all_dot_values)
    # plt.show()
    normals_output_file_name = os.path.join(args.input, 'normals', os.path.splitext(os.path.basename(args.input))[0] + '_'+str(idx_frame)+'.normals')
    xyz_output_file_name = os.path.join(args.input, 'xyz', os.path.splitext(os.path.basename(args.input))[0] + '_'+str(idx_frame)+'.xyz')
    normal_flow_file_name = os.path.join(args.input, 'nf', os.path.splitext(os.path.basename(args.input))[0] + '_'+str(idx_frame)+'.nf')
    new_xyz_file_name = os.path.join(args.input, 'nf', os.path.splitext(os.path.basename(args.input))[0] + '_'+str(idx_frame)+'.xyz')

    np.savetxt(xyz_output_file_name, xyz_lst_new, delimiter=' ')
    np.savetxt(normals_output_file_name, n_lst, delimiter=' ')


    with open(new_xyz_file_name, 'wb') as file:
        pickle.dump(xyz_lst, file)
    with open(normal_flow_file_name, 'wb') as file:
        pickle.dump(n_flow, file)
    print(n_flow)
    print("files created")

    xyz_lst = []
    n_lst = []
    all_dot_values = []
    all_values = []
    xyz_lst_new = []
    n_flow = []

    idx_frame+=1
    ts += dtt