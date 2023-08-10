import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import pickle   
import math
from PIL import Image
import matplotlib.pyplot as plt
from itertools import chain




series = []
indexs = []
differences = []

class NormalFlowSupervisedDataset(Dataset):
    def __init__(self, sequence_paths, network="NormalFlow", target="NormalFlow", transform=None, ):
        # List of sequences that will be used
        self.sequence_paths = sequence_paths

        self.transform = transform
        self.network = network
        self.target = target
        

        # Find out the total number of samples
        self.num_samples = 0
        self.sequence_start_index = np.zeros((len(self.sequence_paths),), dtype=np.int64)
        for i, sequence_path in enumerate(self.sequence_paths):
            nf_path = os.path.join(sequence_path, 'dataset_normal_flow.npz')
            nf_data = np.load(nf_path)

            self.sequence_start_index[i] = self.num_samples
            self.num_samples += nf_data['num_of_samples']

        self.opened_in_out_pairs = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Pre-open all the npz pairs to get a 50% boost in single thread 25% in multithread
        # Use this hack to pre-open them to because NumPy does not support sharing npz objects
        # across processes
        if self.opened_in_out_pairs is None:
            self.opened_in_out_pairs = {}
            for sequence_path in self.sequence_paths:
                sequence = os.path.split(sequence_path)[-1]
                nf_path = os.path.join(sequence_path, 'dataset_normal_flow.npz')
                self.opened_in_out_pairs[sequence] = np.load(nf_path)

        seq_index = np.searchsorted(self.sequence_start_index, idx, side='right') - 1
        sequence_path = self.sequence_paths[seq_index]
        sequence = os.path.split(sequence_path)[-1]

        in_out_pairs = self.opened_in_out_pairs[sequence]
        

        frame_idx = in_out_pairs['index_to_frame_id'][idx - self.sequence_start_index[seq_index]]
        rjust_frame_idx = str(frame_idx).rjust(10, '0')
       
        
        # X is input, Y is output (with mask)
        X_pca  = in_out_pairs[f'normal_flow_pca_{rjust_frame_idx}']
        X_ec   = in_out_pairs[f'normal_flow_event_count_{rjust_frame_idx}']
        X_ts   = in_out_pairs[f'normal_flow_event_timestamp_{rjust_frame_idx}']

        if self.network == "NormalFlow":
            X = np.dstack((X_pca, X_ec))     
        elif self.network == "OpticalFlow":
            X = np.dstack((X_ec, X_ts))
        else:
            raise ValueError(f"Incorrect Network Name {self.network}, Use NormalFlow or OpticalFlow")
            
        if self.target == "NormalFlow":
            Y      = in_out_pairs[f'normal_flow_gt_{rjust_frame_idx}']
            Y_mask = in_out_pairs[f'normal_flow_gt_mask_{rjust_frame_idx}']
        elif self.target == "OpticalFlow":
            Y      = in_out_pairs[f'optical_flow_gt_{rjust_frame_idx}']
            Y_mask = in_out_pairs[f'optical_flow_gt_mask_{rjust_frame_idx}']
        elif self.target == "BGR":
            Y      = in_out_pairs[f'bgr_image_{rjust_frame_idx}']
            Y_mask = in_out_pairs[f'bgr_image_mask_{rjust_frame_idx}']
        else:
            raise ValueError(f"Incorrect Target Name {self.target}, Use NormalFlow or OpticalFlow")

        X      = torch.tensor(np.nan_to_num(X)).permute(2, 0, 1)
        Y      = torch.tensor(np.nan_to_num(Y)).permute(2, 0, 1)
        Y_mask = torch.tensor(Y_mask)

        if self.transform:
            X = self.transform(X)
            Y = self.transform(Y)
            Y_mask = self.transform(Y_mask)
       

        return X, Y, Y_mask
    
def visualize_optical_flow(flowin):
    flow=np.ma.array(flowin, mask=np.isnan(flowin))
    theta = np.mod(np.arctan2(flow[:, :, 0], flow[:, :, 1]) + 2*np.pi, 2*np.pi)

    flow_norms = np.linalg.norm(flow, axis=2)
    flow_norms_normalized = flow_norms / np.max(np.ma.array(flow_norms, mask=np.isnan(flow_norms)))

    flow_hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    flow_hsv[:, :, 0] = 179 * theta / (2*np.pi)
    flow_hsv[:, :, 1] = 255 * (flow_norms_normalized >0 )
    flow_hsv[:, :, 2] = 255 * (flow_norms_normalized > 0) 
 
    flow_hsv[np.logical_and(np.isnan(theta), np.isnan(flow_norms_normalized)), :] = 0

    return flow_hsv

# This accepts a list of arrows while the function in evimo_flow.py uses a grid
def draw_flow_arrows(img, xx, yy, dx, dy, p_skip=15, num_arrows=None, mag_scale=1.0, color=(0, 0, 0)):
    if num_arrows:
        p_skip = max(1, int(xx.shape[0] / num_arrows))
    

    xx     = xx[::p_skip].flatten()
    yy     = yy[::p_skip].flatten()
    flow_x = dx[::p_skip].flatten()
    flow_y = dy[::p_skip].flatten()
    flow_x = np.where(np.isnan(flow_x), 0, flow_x)
    flow_y = np.where(np.isnan(flow_y), 0, flow_y)
    


    
         
    

    for x, y, u, v in zip(xx, yy, flow_x, flow_y):
        cv2.arrowedLine(img,
                        (int(x), int(y)),
                        (int(x+mag_scale*u), int(y+mag_scale*v)),
                        color,
                        tipLength=0.2)
        
def divide_and_return(image_width, image_height, x_coordinates, y_coordinates, dx, dy, num_rows, num_columns):
    region_width = image_width / num_columns
    region_height = image_height / num_rows

    regions_x = [[] for _ in range(num_rows * num_columns)]
    regions_y = [[] for _ in range(num_rows * num_columns)]
    regions_dx = [[] for _ in range(num_rows * num_columns)]
    regions_dy = [[] for _ in range(num_rows * num_columns)]

    for x, y, dx_val, dy_val in zip(x_coordinates, y_coordinates, dx, dy):
        column_index = int(x / region_width)
        row_index = int(y / region_height)
        
        if 0 <= column_index < num_columns and 0 <= row_index < num_rows:
            region_idx = row_index * num_columns + column_index
            if len(regions_x[region_idx]) < 500:
                regions_x[region_idx].append(x)
                regions_y[region_idx].append(y)
                regions_dx[region_idx].append(dx_val)
                regions_dy[region_idx].append(dy_val)

    return regions_x, regions_y, regions_dx, regions_dy
        
def draw_flow_arrowsss(img, xx, yy, dx, dy, p_skip=15, num_arrows=None, mag_scale=1.0, color=(0, 0, 0)):
    if num_arrows:
        p_skip = max(1, int(xx.shape[0] / num_arrows))

    

    xx     = xx[::p_skip].flatten()
    yy     = yy[::p_skip].flatten()
    flow_x = dx[::p_skip].flatten()
    flow_y = dy[::p_skip].flatten()
    flow_x = np.where(np.isnan(flow_x), 0, flow_x)
    flow_y = np.where(np.isnan(flow_y), 0, flow_y)
    
    image_width = 640
    image_height = 480
    num_rows = 4
    num_columns = 5

    xx, yy, flow_x, flow_y = divide_and_return(image_width, image_height, xx, yy, flow_x, flow_y, num_rows, num_columns)
    xx = list(chain.from_iterable(xx))
    yy = list(chain.from_iterable(yy))
    flow_x = list(chain.from_iterable(flow_x))
    flow_y = list(chain.from_iterable(flow_y))

    



    

    for x, y, u, v in zip(xx, yy, flow_x, flow_y):
        # if (x - int(x+mag_scale*u)>1) or  (y-int(y+mag_scale*v)>1):
        #     continue

        if x>0:

            tip_x =int(x+(mag_scale*u))
        if x<0:

            tip_x =int(x-(mag_scale*u))

        if y>0:

            tip_y =int(y+(mag_scale*v))
        if y<0:

            tip_y =int(y-(mag_scale*v))

        print(tip_x)
        
        

        dx = tip_x - x
        dy = tip_y - y

        length = math.sqrt(dx**2 + dy**2)
     
        if length>7:
            continue
        # length = math.sqrt((tip_x - x)**2 + (tip_y - y)**2)
        # if length>5:
        #     vector = np.array([tip_x- x, tip_y - y])
        #     normalized_vector = (vector / length)*7

        #     tip_x = x + (normalized_vector[0]) 
        #     tip_y = y + (normalized_vector[1]) 
        

        cv2.arrowedLine(img,
                        (int(x), int(y)),
                        (int(tip_x), int(tip_y)),
                        color,
                        tipLength=0.2)
        
        
def visualize_train_layer(train_layer):
    nf = train_layer[:, :, 0:2]
    ec = train_layer[:, :, 2]
    return visualize_target_layer(ec, nf)

def visualize_target_layer(ec, nf, reconstructed=None):
    

    ec = (128*(ec > 0)).astype(np.uint8)

    ogxx, ogyy, __ = np.nonzero(nf)
    oguuvv = nf[ogxx, ogyy,:]
    oguu, ogvv = oguuvv[:,0], oguuvv[:,1]

    pixel_array = np.dstack((ogxx,ogyy,oguu,ogvv))
    height, width = 480, 640
   
    image = np.zeros((height, width, 2), dtype=np.float32)

# Fill the image with the pixel values
    for pixel_data in pixel_array[0]:
        x, y, r, g = pixel_data
        x, y = int(x), int(y) 
        image[x,y] = [r, g]

    flow_hsv = visualize_optical_flow(image)
    flow_bgr = cv2.cvtColor(flow_hsv, cv2.COLOR_HSV2BGR)

    # cv2.imshow('gt_flow', flow_bgr) 
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ec_rgb = np.dstack((ec, ec, ec)) 

    draw_flow_arrows(ec_rgb, ogyy, ogxx, oguu, ogvv, 
        p_skip=20, mag_scale=5.0, color=(0,255,0))


    return ec_rgb,flow_bgr
def vector_magnitude(x, y):
        return math.sqrt(x**2 + y**2)

def visualize_target_layer_nf(ec,xy,nf,reconstructed=None):
    

    ec = (128*(ec > 0)).astype(np.uint8)

    ogxx_a, ogyy_a = xy[:,1], xy[:,0]
    
    oguu_a, ogvv_a = nf[:,0],nf[:,1]


    # for i in range (len (oguu_a)):
    #     mag = vector_magnitude(oguu_a[i],ogvv_a[i])
    #     oguu_a[i],ogvv_a[i] = oguu_a[i]/mag, ogvv_a[i]/mag

    pixel_array_a = np.dstack((ogxx_a,ogyy_a,oguu_a,ogvv_a))

    
    height, width = 480, 640
   
    image_a = np.zeros((height, width, 2), dtype=np.float32)

# Fill the image with the pixel values
    for pixel_data_a in pixel_array_a[0]:
        x, y, r, g = pixel_data_a
       


        x, y = int(x), int(y) 
        
        if x == 480:
            continue
        if y == 640:
            continue
        image_a[x,y] = [r, g]
     

    flow_hsv_a = visualize_optical_flow(image_a)
    flow_bgr_a = cv2.cvtColor(flow_hsv_a, cv2.COLOR_HSV2BGR)


    ec_rgb_a = np.dstack((ec, ec, ec))
   


    draw_flow_arrowsss(ec_rgb_a, ogyy_a, ogxx_a, oguu_a, ogvv_a,
        p_skip=20  , mag_scale=5.0, color=(0,255,0))
    
    



    return ec_rgb_a,flow_bgr_a


def visualize_pytorch_layers(train_layer, gtnf,xy,nf , gtnf_mask, reconstructed=None):
    train_layer = train_layer[0, :, :].permute(1,2,0).numpy()
    gtnf        = gtnf       [0, :, :].permute(1,2,0).numpy()        
    gtnf_mask   = gtnf_mask  [0, :, :].numpy()

    if reconstructed is not None:
        reconstructed = reconstructed[0, :, :].permute(1,2,0).numpy

    assert train_layer.dtype == np.float32
    assert gtnf.dtype == np.float32
    assert gtnf_mask.dtype == bool
    assert gtnf.max() <= 640
    assert gtnf.min() >= -640



    gtnf[gtnf_mask==False, :] = 0

    vis_gt,img = visualize_target_layer(train_layer[:, :, 2], gtnf, reconstructed=reconstructed)
    vis_train,img_a = visualize_target_layer_nf(train_layer[:, :, 2], xy,nf,reconstructed=reconstructed)

   
   
    return vis_train, vis_gt ,img,img_a

def find_closest_number_index(input_number, number_array):
    absolute_difference = np.abs(np.array(number_array) - input_number)
    closest_index = np.argwhere(absolute_difference == np.min(absolute_difference))[0][0]
    return closest_index

def visualize_layers(t_file):
   
    frames = 1  
    frame = 1    
    counter = 1
    folder = "scene10_dyn_train_00_000000"   
    time = 0.0256
    l = DataLoader(d, batch_size=1, shuffle=False, num_workers=10)
    
    
    counter  = 0
    for i in range(372):
        time_seq = time*frame
        series.append(time_seq)
        frame = frame+1
        

    
    for i in (t_file):
        index = find_closest_number_index(i, series)
        indexs.append(index)
    
    

   
    # result = [t_file[index] for index in indexs]
    # difference = np.diff(result)
    # print(np.shape(indexs))
    # plt.plot(indexs)
    # plt.show()


    frame_width = 1280
    frame_height = 960
    output_video_path = '/home/enigma/prg/DeepFit/tutorial/scene10_dyn_train_00_000000/output_video_2.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
    out = cv2.VideoWriter(output_video_path, fourcc, 10, (frame_width, frame_height))    
    for train_layer, gtnf, gtnf_mask in l:

        
        if indexs[1]-indexs[0]==0 or indexs[1]-indexs[0]==1:
            if indexs[0] == 0:
        
                xy = f"/home/enigma/prg/DeepFit/tutorial/{folder}/nf/{folder}_{indexs[0]+1}.xyz"
                nf = f"/home/enigma/prg/DeepFit/tutorial/{folder}/nf/{folder}_{indexs[0]+1}.nf"
               
            else:
                xy = f"/home/enigma/prg/DeepFit/tutorial/{folder}/nf/{folder}_{indexs[0]}.xyz"
                nf = f"/home/enigma/prg/DeepFit/tutorial/{folder}/nf/{folder}_{indexs[0]}.nf"
        
                
            
            
            with open(xy, 'rb') as file:
                xy_data = pickle.load(file)
            with open(nf, 'rb') as file:
                nf_data = pickle.load(file)
        
        


            
            vis_train, vis_gt,img,img_a = visualize_pytorch_layers(train_layer, gtnf,xy_data,nf_data ,gtnf_mask)
            h_stack = np.hstack((img, img_a))
            hh_stack = np.hstack((vis_gt, vis_train))
            f_img = np.vstack((h_stack,hh_stack))

            output_path1 = f'/home/enigma/prg/DeepFit/tutorial/{folder}/GT_color/{frames}.jpg'
            output_path2 = f'/home/enigma/prg/DeepFit/tutorial/{folder}/color/{frames}.jpg'
            output_path3 = f'/home/enigma/prg/DeepFit/tutorial/{folder}/GT_normal_flow/{frames}.jpg'
            output_path4 = f'/home/enigma/prg/DeepFit/tutorial/{folder}/normal_flow/{frames}.jpg'




# Save the image to the specified directory
            cv2.imwrite(output_path1, img)
            cv2.imwrite(output_path2, img_a)
            cv2.imwrite(output_path3, vis_gt)
            cv2.imwrite(output_path4, vis_train)

            cv2.imshow('GT , Normal Flow', f_img)
            cv2.waitKey(0)

            indexs.pop(0)
            frames = frames+1
            out.write(f_img)
          
        else :
        
            cc = indexs[0]
       
            
            for i in range(indexs[1]-indexs[0]):

                if indexs[0] == 0:
            
                    xy = f"/home/enigma/prg/DeepFit/tutorial/{folder}/nf/{folder}_{cc+1}.xyz"
                    nf = f"/home/enigma/prg/DeepFit/tutorial/{folder}/nf/{folder}_{cc+1}.nf"
                else:
                 
                    xy = f"/home/enigma/prg/DeepFit/tutorial/{folder}/nf/{folder}_{cc}.xyz"
                  
                    nf = f"/home/enigma/prg/DeepFit/tutorial/{folder}/nf/{folder}_{cc}.nf"
                   
                    
                with open(xy, 'rb') as file:
                    xy_data = pickle.load(file)
                with open(nf, 'rb') as file:
                    nf_data = pickle.load(file)
            
            


                
                vis_train, vis_gt,img,img_a = visualize_pytorch_layers(train_layer, gtnf,xy_data,nf_data ,gtnf_mask)
                h_stack = np.hstack((img, img_a))
                hh_stack = np.hstack((vis_gt, vis_train))
                f_img = np.vstack((h_stack,hh_stack))

                output_path1 = f'/home/enigma/prg/DeepFit/tutorial/{folder}/GT_color/{frames}.jpg'
                output_path2 = f'/home/enigma/prg/DeepFit/tutorial/{folder}/color/{frames}.jpg'
                output_path3 = f'/home/enigma/prg/DeepFit/tutorial/{folder}/GT_normal_flow/{frames}.jpg'
                output_path4 = f'/home/enigma/prg/DeepFit/tutorial/{folder}/normal_flow/{frames}.jpg'




   
   
                cv2.imwrite(output_path1, img)
                cv2.imwrite(output_path2, img_a)
                cv2.imwrite(output_path3, vis_gt)
                cv2.imwrite(output_path4, vis_train)

                cv2.imshow('GT , Normal Flow', f_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                
                frames = frames+1
                cc = cc+1
                out.write(f_img)
                
            indexs.pop(0) 


    out.release()
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    import glob
    from tqdm import tqdm
    from torch.utils.data import DataLoader  
    folder = "scene10_dyn_train_00_000000"         

    sequence_paths = sorted(list(glob.glob('/home/enigma/prg/DeepFit/tutorial/*')))
    assert len(sequence_paths) > 0
    def subtract_elements_by_first_element(input_list):
        if len(input_list) == 0:
            return []  # Return an empty list if the input list is empty
        
        first_element = input_list[0]
        result = [element - first_element for element in input_list]
        return result
    

    sequence_paths = [f'/home/enigma/prg/DeepFit/tutorial/{folder}/']
    t_file = f"/home/enigma/prg/DeepFit/tutorial/scene10_dyn_train_00_000000/dataset_normal_flow/t.npy"
    d = NormalFlowSupervisedDataset(sequence_paths)
    t_file = np.load(t_file)
    t_file = subtract_elements_by_first_element(t_file)


    visualize_layers(t_file)