#This script is to find the top 4 images that contribute the most to the model training.
#The images are selected based on the number of input data points they have.
#The images with the most input data points are selected as the top 4 images.
#The length scales are varied from 0.1 to 10 to find the best length scale for the model.

import numpy as np
import torch
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.distributions import MultitaskMultivariateNormal
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def load_points3D(file_path):
    points3d_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#') or len(line.strip()) == 0:
                continue
            parts = line.split()
            point_id = int(parts[0])
            x, y, z = map(float, parts[1:4])

            r,g,b = np.array([int(round(float(c))) for c in parts[4:7]])
            points3d_dict[point_id] = [x, y, z, r / 255.0, g / 255.0, b / 255.0]
    return points3d_dict

file_path_points3d = '/home/staff/zhihao/Downloads/3dgs/mogp/gp_evaluation/mipnerf360/360_v2/flowers/sparse/0/points3D.txt'
points3d_dict = load_points3D(file_path_points3d)


#
def parse_images_file(file_path, points3d_dict):
    valid_data = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith('#') or not lines[i].strip():
                i += 1
                continue

            image_data = lines[i].strip().split()
            image_name = image_data[9]
            i += 1
            keypoints_data = lines[i].strip().split()
            points2d = []
            k = 0
            while k < len(keypoints_data):
                x, y = map(float, keypoints_data[k:k+2])
                point3d_id = int(keypoints_data[k+2])
                if point3d_id != -1 and point3d_id in points3d_dict:
                    points2d.append((x, y) + tuple(points3d_dict[point3d_id]))
                k += 3
            if points2d:
                valid_data[image_name] = points2d
            i += 1

    return valid_data


file_path_images = '/home/staff/zhihao/Downloads/3dgs/mogp/gp_evaluation/mipnerf360/360_v2/flowers/sparse/0/images.txt'
valid_data = parse_images_file(file_path_images, points3d_dict)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D



def generate_test_data(valid_data, depth_file_path):
    data_by_image = {}
    depth_images = np.load(depth_file_path)
    print(depth_images.shape)
    image_indices = {name: idx for idx, name in enumerate(sorted(valid_data.keys()))}
    #from 4 directions
    movements = {
        'left': (-50, 0),
        #'right': (1, 0),
        ##'up': (0, -1),
       # 'down': (0, 1)
    }
    for image_name, data_points in valid_data.items():
        input_data = []
        output_data = []
        test_data = []

        current_depth_image = depth_images[image_indices[image_name]].T
        image_height, image_width = current_depth_image.shape


        for point in data_points:
            x, y = int(point[0]), int(point[1])
            original_depth = current_depth_image[x, y]   # Normalize depth
            input_data.append([x, y, original_depth])
            output_data.append(point[2:])

            # Generate test data around the point
            for direction, (dx, dy) in movements.items():
                new_x, new_y = x + dx, y + dy
                # Check image boundaries

                if 0 <= new_x < image_width and 0 <= new_y < image_height:
                    new_depth = current_depth_image[new_y, new_x]
                    test_data.append([new_x, new_y, new_depth])
        input_data = np.array(input_data, dtype=float)
        test_data = np.array(test_data,dtype=float)
        input_data[:, 0] /= image_width  # Normalize x to [0, 1]
        input_data[:, 1] /= image_height  # Normalize y to [0, 1]
        test_data[:, 0] /= image_width
        test_data[:, 1] /= image_height
        # Store normalized data by image name
        data_by_image[image_name] = {
            'input': input_data,
            'output': np.array(output_data, dtype=float),
            'test': np.array(test_data, dtype=float)  # Store test data
        }

    return data_by_image

depth_file_path = '/home/staff/zhihao/Downloads/3dgs/mogp/gp_evaluation/mipnerf360/360_v2/flowers/depth/f.npy'

data_by_image = generate_test_data(valid_data,depth_file_path)

#top_images = sorted(data_by_image.items(), key=lambda x: len(x[1]['input']), reverse=True)[-4:]
top_images = sorted(data_by_image.items(), key=lambda x: len(x[1]['input']), reverse=True)[:4]
top_image_names = [image[0] for image in top_images]
print("Top 4 images based on input data size:", top_image_names)
print(len(np.load(depth_file_path)))