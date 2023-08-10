import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' # Attempt to disable OpenBLAS multithreading used by NumPy, it makes the script 17% slower
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
import time
import pickle
import math
from numba import jit


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='scene9_dyn_train_02_000000', help='full path to input point cloud')
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
all_values = []
all_dot_values = []
final_cuboids = []
xyz_lst = []
n_lst = []
all_times = []
n_flow = []
xyz_lst_new = []

@jit(nopython=True)
def sort_events(points, x_bin, y_bin, z_bin):
    x_mask = np.logical_and(x_bin[0] <= points[:, 0], points[:, 0] < x_bin[1])
    y_mask = np.logical_and(y_bin[0] <= points[:, 1], points[:, 1] < y_bin[1])
    z_mask = np.logical_and(z_bin[0] <= points[:, 2], points[:, 2] < z_bin[1])
    
    group0 = x_mask & y_mask & z_mask
    group1 = x_mask & y_mask & ~z_mask
    group2 = x_mask & ~y_mask & z_mask
    group3 = x_mask & ~y_mask & ~z_mask
    group4 = ~x_mask & y_mask & z_mask
    group5 = ~x_mask & y_mask & ~z_mask
    group6 = ~x_mask & ~y_mask & z_mask
    group7 = ~x_mask & ~y_mask & ~z_mask


    groups = [group0, group1, group2, group3, group4, group5, group6, group7]

    #groups = np.logical_and(np.logical_and(x_mask, y_mask), z_mask)
    #groups = [points[cuboid_s==i] for i in range(8)]
    return groups

# def vector_magnitude(x, y):
#     return math.sqrt(x**2 + y**2)

def normal_flow(x,y,z):

    column_vector = np.array([[x], [y]])
    #coloum_vector = np.transpose(coloum_vector)
    squared_norm = np.linalg.norm((column_vector))**2
    result = (-z * (((column_vector)) / squared_norm))
    result = result*0.016666667
    # mag = vector_magnitude(result[0],result[1])
    # result = result/mag
    
    return  result

def avg(x):
    squared_sum = np.sum(x**2)
    avg_squared_sum = squared_sum / np.size(x)
    avg = np.sqrt(avg_squared_sum)
    return avg
    

def divide_cuboid(x,y,z):
    divisions = []
    
    # Get the dimensions of the cuboid    
    # Calculate the dimensions of each divided cuboid
    divided_length = (x[1]-x[0]) / 2
    divided_width  = (y[1]-y[0]) / 2
    divided_height = (z[1]-z[0]) / 2

    # Generate the divisions
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x_start = x[0] +  i    * divided_length
                x_end   = x[0] + (i+1) * divided_length

                y_start = y[0] +  j    * divided_width
                y_end   = y[0] + (j+1) * divided_width

                z_start = z[0] +  k    * divided_height
                z_end   = z[0] + (k+1) * divided_height
                
                division = {
                    'x': (x_start, x_end),
                    'y': (y_start, y_end),
                    'z': (z_start, z_end)
                }
                divisions.append(division)
                
    return divisions

points_xy = np.load(args.input + '/dataset_events_xy.npy').astype('float32')
points_t = np.load(args.input + '/dataset_events_t.npy').astype('float32')
print(args.input)


ts = points_t[0]
dtt = 0.0256
idx_frame = 1

while ts <= points_t[-1]:

    # samples = [0,8,72,584,4680,37448,299492,2396644]





    left_indices = np.searchsorted(points_t, ts)

   
    right_indices = np.searchsorted(points_t, ts + dtt, side='right')


    xyz = np.array([points_xy[left_indices:right_indices, 0], points_xy[left_indices:right_indices, 1], 10000*points_t[left_indices:right_indices]]).T
    
    points = np.array(xyz)
    points[:, 2] -= np.min(points[:, 2])

    X, Y, Z = xyz[:,0],xyz[:,1],xyz[:,2]
   
    #cuboid = [640,480,256]

    assert np.all(points[:, 2] >= 0)
    assert np.all(points[:, 2] < 256)

    all_cuboids = []
    final_cuboids = []
    first_cuboid = {
            'points': points,
            'x': [0, 640],
            'y': [0, 480],
            'z': [0, 256]

    }

    all_cuboids.append(first_cuboid)

    # i = -1
    num_cubes_done = 0
    num_cubes = 1
    while True:
        # i+=1
      
        if all_cuboids == []:
           
            break
        first_dict = all_cuboids[0]

        # print('hi')
        # print(first_dict['x'],first_dict['y'],first_dict['z'])
        divisions = divide_cuboid(first_dict['x'],first_dict['y'],first_dict['z'])
        num_cubes += len(divisions)
        
        # print(divisions)
        # for d in divisions:
        #     assert d['z'][1] <= 256

        res_x = list(map(itemgetter('x'), divisions))
        res_y = list(map(itemgetter('y'), divisions))
        res_z = list(map(itemgetter('z'), divisions))
        

        # if i == samples[0]:
           
        #     cuboid[0] = cuboid[0]/2
        #     cuboid[1] = cuboid[1]/2
        #     cuboid[2] = cuboid[2]/2
        #     samples.pop(0)

       

       
     
        # new_res_x = np.ravel(res_x)
        # x_bin = sorted(set(new_res_x))
        # x_bin = np.copy(x_bin)

        # new_res_y = np.ravel(res_y)
        # y_bin = sorted(set(new_res_y))
        # y_bin = np.copy(y_bin)

        # new_res_z = np.ravel(res_z)
        # z_bin = sorted(set(new_res_z))
        # z_bin = np.copy(z_bin)
       

        groups = sort_events(first_dict['points'], divisions[0]['x'], divisions[0]['y'], divisions[0]['z'])

        # print([np.sum(g) for g in groups])

        start2 = time.time()
        for j in range (8):
            
            x_start = res_x[j][0]
            x_end = res_x[j][1]
            y_start = res_y[j][0]
            y_end = res_y[j][1]
            z_start = res_z[j][0]
            z_end = res_z[j][1]
           


           
            another_selected_points = groups[j]
            selected_points = first_dict['points'][another_selected_points]
            points_to_pass = np.copy(selected_points)


            start001 = time.time()



            num_of_points = np.shape(selected_points)
            
            # another_num_of_points = np.shape(another_selected_points)
            # print(num_of_points)
            # print("another points")
            # print(another_num_of_points)           
            
            f_points = selected_points
          
            # print(num_of_points)

            if num_of_points[0]<3:
                num_cubes_done += 1
                continue
            selected_points = np.array(selected_points)
            

            centroid = np.mean(selected_points, axis=0)
        
            selected_points -= centroid

            
            selected_points = np.array(selected_points)

            covariance_matrix = np.cov(selected_points.T)

            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
         
            min_eigenvalue_index = np.argmin(eigenvalues) 
            normal_vector = eigenvectors[:, min_eigenvalue_index]


            dot_product = np.dot(selected_points, normal_vector)
            av_pro = avg(dot_product)
            
            normal__flow = normal_flow(normal_vector[0],normal_vector[1],normal_vector[2])
            normal__flow = [item for sublist in normal__flow for item in sublist]
            
           
            

            if av_pro<3:
                    num_cubes_done+=1
                    sss = np.shape(f_points)
                    for n in range (sss[0]):
                        xyz_lst.append([f_points[n][0],f_points[n][1]])
                        xyz_lst_new.append(f_points[n])
                        n_lst.append(normal_vector)
                        n_flow.append(normal__flow)

                    
        
            else:
                small_cuboids= {
                        'points': points_to_pass,
                        'x': [x_start, x_end],
                        'y': [y_start, y_end],
                        'z': [z_start, z_end]
                    }
                all_cuboids.append(small_cuboids)
        
        all_cuboids.pop(0)
        start3 = time.time()
        
       
        
       
        # print((start3-start2)*1000,(start1-start0)*1000/100)
        # print(sum(all_times)*1000)


    
    xyz_lst = np.array(xyz_lst)
    n_lst = np.array(n_lst)
    n_flow = np.array(n_flow)
    xyz_lst_new = np.array(xyz_lst_new)
    

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

    print("files created")

    xyz_lst = []
    n_lst = []
    n_flow = []
    all_dot_values = []
    all_values = []
    xyz_lst_new = []
    print(idx_frame)

    idx_frame+=1
    ts += dtt
    