import numpy as np
from sklearn.preprocessing import MinMaxScaler


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def find_max_point_id(file_path):
    max_point_id = 0
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            point_id = int(parts[0])
            max_point_id = max(max_point_id, point_id)
    return max_point_id

def load_points3D(file_path):
    points3d_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#') or len(line.strip()) == 0:
                continue
            parts = line.split()
            point_id = int(parts[0])
            x, y, z = map(float, parts[1:4])
            #r, g, b = map(int, parts[4:7])
            r,g,b = np.array([int(round(float(c))) for c in parts[4:7]])
            points3d_dict[point_id] = [x, y, z, r / 255.0, g / 255.0, b / 255.0]
    return points3d_dict

file_path_points3d = '/home/staff/zhihao/Downloads/3dgs/mogp/gp_evaluation/nerf_sythetic/ship/sparse/0/points3D.txt'
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


file_path_images = '/home/staff/zhihao/Downloads/3dgs/mogp/gp_evaluation/nerf_sythetic/ship/sparse/0/images.txt'
valid_data = parse_images_file(file_path_images, points3d_dict)

depth_file_path = '/home/staff/zhihao/Downloads/3dgs/mogp/gp_evaluation/nerf_sythetic/ship/depth/m.npy'
def generate_test_data(valid_data, depth_file_path, radius_factor=0.2, num_samples=10):
    """
    Generate test data adaptively around training data points using dynamic movements.

    Parameters:
    - valid_data (dict): Dictionary of image names and training points.
    - depth_file_path (str): Path to the depth images (NumPy file).
    - radius_factor (float): Fraction of the image size used to define movement radius.
    - num_samples (int): Number of dynamic movements (directions) to sample around each point.

    Returns:
    - data_by_image (dict): Dictionary containing input, output, and test data for each image.
    """
    # Load depth images and precompute image dimensions
    depth_images = np.load(depth_file_path)
    image_indices = {name: idx for idx, name in enumerate(sorted(valid_data.keys()))}

    data_by_image = {}

    for image_name, data_points in valid_data.items():
        # Pre-fetch depth image and dimensions
        current_depth_image = depth_images[image_indices[image_name]]
        image_height, image_width = current_depth_image.shape

        # Calculate adaptive radius based on image dimensions
        adaptive_radius = int(radius_factor * min(image_height, image_width))

        # Generate dynamic movements using polar coordinates
        angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        movements = np.array([
            (int(adaptive_radius * np.cos(angle)), int(adaptive_radius * np.sin(angle)))
            for angle in angles
        ])

        for image_name, data_points in valid_data.items():
            input_data = []
            output_data = []
            test_data = []

            current_depth_image = depth_images[image_indices[image_name]]
            image_height, image_width = current_depth_image.shape
            for point in data_points:
                x, y = int(point[0]), int(point[1])
                if x < 0 or x >= image_width or y < 0 or y >= image_height:
                    # Handle out-of-bounds case, skip or adjust
                    continue
                original_depth = current_depth_image[y, x]
                input_data.append([x, y, original_depth])
                output_data.append(point[2:])

                # Generate test data around the point
                for dx, dy in movements:

                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < image_width and 0 <= new_y < image_height:
                        new_depth = current_depth_image[new_y, new_x]
                        test_data.append([new_x, new_y, new_depth])

            input_data = np.array(input_data, dtype=float)
            test_data = np.array(test_data, dtype=float)
            input_data[:, 0] /= image_width  # Normalize x to [0, 1]
            input_data[:, 1] /= image_height  # Normalize y to [0, 1]
            test_data[:, 0] /= image_width
            test_data[:, 1] /= image_height
            data_by_image[image_name] = {
                'input': input_data,
                'output': np.array(output_data, dtype=float),
                'test': test_data  # Store test data
            }

        return data_by_image



data_by_image_new = generate_test_data(valid_data, depth_file_path)

# Preprocess the track information from images.txt
def preprocess_tracks(images_txt_path):
    """Preprocess images.txt to map point3D IDs to track information."""
    tracks = {}
    with open(images_txt_path, 'r') as file:
        lines = file.readlines()
        for image_idx, line in enumerate(lines):
            if line.startswith('#') or not line.strip():
                continue
            if line.strip().endswith('.jpg') or line.strip().endswith('.png'):
                image_id = int(line.split()[0])  # IMAGE_ID is the first column
                points_line = lines[image_idx + 1].strip().split()
                for idx in range(0, len(points_line), 3):
                    x, y, p3d_id = map(float, points_line[idx:idx + 3])
                    p3d_id = int(p3d_id)
                    if p3d_id not in tracks:
                        tracks[p3d_id] = []
                    point2d_idx = idx // 3  # POINT2D_IDX is the index in the 2D points list
                    tracks[p3d_id].append((image_id, point2d_idx))
    return tracks


# Count the number of 3D points in the existing points3D.txt file
def count_3d_points(file_path):
    count = 0
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#') or not line.strip():
                continue
            count += 1
    return count


# Write the points3D.txt file in bulk for better performance
def write_points3D_txt_optimized(points3D_file_path, final_points, tracks, starting_point_id):
    """Write points3D.txt in bulk for better performance."""
    point_id = starting_point_id
    lines = []

    for x, y, z, r, g, b in final_points:
        # Get TRACK[] information
        track_info = tracks.get(point_id, [])
        track_info_str = ' '.join(f"{image_id} {point2d_idx}" for image_id, point2d_idx in track_info)

        # Construct the line for this point
        line = f"{point_id} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0.2 {track_info_str}\n"
        lines.append(line)
        point_id += 1

    # Write all lines to the file in one operation
    with open(points3D_file_path, 'a') as file:
        file.writelines(lines)


# Main script
if __name__ == "__main__":


    data_by_image_new = generate_test_data(valid_data, depth_file_path)
    all_output_data = []
    for k in data_by_image_new.keys():  # Loop through all keys (images)
        all_output_data.append(data_by_image_new[k]['output'])  # Collect outputs
    all_output_data = np.vstack(all_output_data)  # Stack into a single array
    scaler_output = MinMaxScaler()
    scaler_output.fit(all_output_data)
    input_data_normalized = data_by_image_new['000072.png']['input']

    # Load predicted data
    predicted_var = np.load('/home/staff/zhihao/Downloads/3dgs/mogp/gp_evaluation/nerf_sythetic/ship/gp/72test_var.npy')[0]
    predicted_variance = np.array(predicted_var).reshape(-1, 6)

    # Compute the 60th percentile threshold for variance
    r_var = predicted_variance[:, 3]
    g_var = predicted_variance[:, 4]
    b_var = predicted_variance[:, 5]
    rgb_mean = (r_var + g_var + b_var) / 3
    threshold = np.percentile(rgb_mean, 50)
    filtered_indices = rgb_mean <= threshold
    filtered_variance = predicted_variance[filtered_indices]


    # Load the means data
    pre_points = np.load('/home/staff/zhihao/Downloads/3dgs/mogp/gp_evaluation/nerf_sythetic/ship/gp/72mean.npy')[0]
    pre_points = np.array(pre_points).reshape(-1,6)  # Reshape if needed, assuming the mean has the same structure as variance
    filtered_means = pre_points[filtered_indices]
    #filtered_means = pre_points
    pre_points = filtered_means
    pre_points_normalized = scaler_output.inverse_transform(pre_points)

    original_pre_points = pre_points_normalized
    x, y, z = original_pre_points[:, 0], original_pre_points[:, 1], original_pre_points[:, 2]
    r = (original_pre_points[:, 3] * 255).astype(int)
    g = (original_pre_points[:, 4] * 255).astype(int)
    b = (original_pre_points[:, 5] * 255).astype(int)
    pre_final_points = np.column_stack((x, y, z, r, g, b))

    print(pre_final_points)
    # Paths
    images_txt_path = '/home/staff/zhihao/Downloads/3dgs/mogp/gp_evaluation/nerf_sythetic/ship/sparse/0/images.txt'
    points3D_file_path = "/home/staff/zhihao/Downloads/3dgs/mogp/gp_evaluation/nerf_sythetic/ship/sparse/0/points3D.txt"
    max_point_id = find_max_point_id(points3D_file_path)
    # Get starting point ID
    num_3d_points = max_point_id
    starting_point_id = num_3d_points + 1

    # Preprocess tracks
    tracks = preprocess_tracks(images_txt_path)

    # Write points3D.txtpython

    write_points3D_txt_optimized(
        points3D_file_path,
        pre_final_points,
        tracks,
        starting_point_id
    )