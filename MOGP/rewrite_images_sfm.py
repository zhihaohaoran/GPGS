import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import re
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

def count_3d_points(file_path):
    count = 0
    with open(file_path, 'r') as file:
        for line in file:
            # Skip comment lines and empty lines
            if line.startswith('#') or not line.strip():
                continue
            count += 1
    return count
file_path_points3d = '/home/staff/zhihao/Downloads/3dgs/mogp/gp_evaluation/nerf_sythetic/ship/sparse/0/points3D.txt'
num_3d_points = count_3d_points(file_path_points3d)
print("Number of 3D points:", num_3d_points)

max_point_id = find_max_point_id(file_path_points3d)
print("Maximum point ID:", max_point_id)

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
            r, g, b = map(lambda v: int(float(v)), parts[4:7])

            points3d_dict[point_id] = [x, y, z, r / 255.0, g / 255.0, b / 255.0]
    return points3d_dict

file_path_points3d = '/home/staff/zhihao/Downloads/3dgs/mogp/gp_evaluation/nerf_sythetic/ship/sparse/0/points3D.txt'
points3d_dict = load_points3D(file_path_points3d)

def update_images_txt_optimized(images_txt_path, predictions):
    # Read the entire file content
    with open(images_txt_path, 'r') as file:
        lines = file.readlines()

    image_indices = {}
    for i, line in enumerate(lines):
        match = re.match(r"^\d+\s.*\s(?P<name>.+\.(jpg|png))$", line)
        if match:
            image_name = match.group("name")
            image_indices[image_name] = i

    # Construct updated lines in memory
    for image_name, points in predictions.items():
        if image_name in image_indices:
            points_line_index = image_indices[image_name] + 1
            points_line = lines[points_line_index].strip()
            points_line += ''.join(f" {x} {y} {points3D_ID}" for x, y, points3D_ID in points)
            lines[points_line_index] = points_line + "\n"

    # Write all lines back to the file in one operation
    with open(images_txt_path, 'w') as file:
        file.writelines(lines)
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


def generate_test_data(valid_data, depth_file_path, radius_factor=0.2, num_samples=10):
    """
    Generate test data adaptively around training data points using dynamic movements.6
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

depth_file_path = '/home/staff/zhihao/Downloads/3dgs/mogp/gp_evaluation/nerf_sythetic/ship/depth/m.npy'

data_by_image_new = generate_test_data(valid_data,depth_file_path)
test_data_normalized_new = data_by_image_new['000072.png']['test']
predicted_var = np.load('/home/staff/zhihao/Downloads/3dgs/mogp/gp_evaluation/nerf_sythetic/ship/gp/72test_var.npy')[0]
predicted_variance = np.array(predicted_var).reshape(-1, 6)

r_var = predicted_variance[:, 3]
g_var = predicted_variance[:, 4]
b_var = predicted_variance[:, 5]
rgb_mean = (r_var + g_var + b_var) / 3
threshold = np.percentile(rgb_mean, 50)
filtered_indices = rgb_mean <= threshold

predict_mean = np.load('/home/staff/zhihao/Downloads/3dgs/mogp/gp_evaluation/nerf_sythetic/ship/gp/72mean.npy')[0]
predict_mean = np.array(predict_mean).reshape(-1, 6)
filtered_means = predict_mean[filtered_indices]
filtered_xyz_rgb_mean = filtered_means

print("Shape of test_data_normalized_new:", test_data_normalized_new.shape)
print("Length of filtered_indices:", len(filtered_indices))


test_data_normalized_new = test_data_normalized_new[filtered_indices]

test_data_normalized_new = test_data_normalized_new
print(len(test_data_normalized_new))

def recover_test_data(test_data_normalized, image_width, image_height, starting_point_id):
    # Revert normalization for x and y
    test_data_recovered = test_data_normalized.copy()
    test_data_recovered[:, 0] *= image_width  # Denormalize x
    test_data_recovered[:, 1] *= image_height  # Denormalize y
    test_data_recovered[:, 0] = np.round(test_data_recovered[:, 0]).astype(int)
    test_data_recovered[:, 1] = np.round(test_data_recovered[:, 1]).astype(int)

    # points3d_id from start_point to end_point(shape-1)
    point_ids = np.arange(starting_point_id, starting_point_id + test_data_recovered.shape[0])

    # Combine x, y, and point IDs
    recovered_data = np.column_stack((
        test_data_recovered[:, 0],  # x
        test_data_recovered[:, 1],  # y
        point_ids                  # points3D_ID
    ))

    return recovered_data

# Example usage
image_width = 800
image_height = 800
starting_point_id = max_point_id + 1
print("Starting point ID:", starting_point_id)
test_data_recovered = recover_test_data(
    test_data_normalized_new,
    image_width,
    image_height,
    starting_point_id
)
test_data_recovered = test_data_recovered.astype(int)
#print(test_data_recovered)
predictions = {
    "000072.png": [(int(row[0]), int(row[1]), int(row[2])) for row in test_data_recovered]
}

images_txt_path = "/home/staff/zhihao/Downloads/3dgs/mogp/gp_evaluation/nerf_sythetic/ship/sparse/0/images.txt"

update_images_txt_optimized(images_txt_path, predictions)