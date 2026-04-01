from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from lab_utils.visualization import plot_feature_vector, show_image_gallery
LABELS = ('cat', 'dog')
LABEL_TO_INDEX = {'cat': 0, 'dog': 1}
IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
SEED = 1234

def label_from_path(path: Path) -> str:
    label = path.parent.name
    if label not in LABEL_TO_INDEX:
        raise ValueError(f'Unexpected label folder: {path}')
    return label

def load_preview_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))

def list_image_paths(label: str) -> list[Path]:
    label_dir = DATA_ROOT / label
    paths = []
    for pattern in IMAGE_EXTENSIONS:
        paths.extend(label_dir.glob(pattern))
    return sorted(paths)

def shuffled_paths(paths: list[Path], seed_offset: int=0) -> list[Path]:
    rng = np.random.default_rng(SEED + seed_offset)
    indices = rng.permutation(len(paths))
    return [paths[int(idx)] for idx in indices]

def sample_paths(paths: list[Path], count: int, seed_offset: int) -> list[Path]:
    ordered = shuffled_paths(paths, seed_offset=seed_offset)
    return ordered[:min(count, len(ordered))]

def sample_per_class(paths: list[Path], n_per_class: int, seed_offset: int=0) -> list[Path]:
    sampled = []
    for label_index, label in enumerate(LABELS):
        label_paths = [path for path in paths if label_from_path(path) == label]
        sampled.extend(sample_paths(label_paths, n_per_class, seed_offset + 50 * label_index))
    return sampled

def split_train_test(paths: list[Path], train_ratio: float=0.7, seed_offset: int=0):
    shuffled = shuffled_paths(paths, seed_offset)
    split_idx = int(len(shuffled) * train_ratio)
    return (shuffled[:split_idx], shuffled[split_idx:])
from pathlib import Path
from PIL import Image
import numpy as np

def load_image_np(path: Path) -> np.ndarray:
    image = Image.open(path)
    np_image = np.array(image.convert('RGB'))
    return np_image

def center_crop(image: np.ndarray, crop_size: int=48) -> np.ndarray:
    h = image.shape[0]
    w = image.shape[1]
    if h < crop_size or w < crop_size:
        raise ValueError(f'Image too small! Got {h}x{w}, need at least {crop_size}x{crop_size}.')
    top_left_h_start = (h - crop_size) // 2
    top_left_w_start = (w - crop_size) // 2
    return image[top_left_h_start:top_left_h_start + crop_size, top_left_w_start:top_left_w_start + crop_size]

def flip_horizontal(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image[:, ::-1]
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f'Expected an H x W x 3 RGB image, got shape {image.shape}.')
    return image[:, ::-1, :]

def normalize_01(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0

def show_histograms(uint8_img, float_img):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(uint8_img.ravel(), bins=50)
    plt.title('Before (uint8: 0–255)')
    plt.subplot(1, 2, 2)
    plt.hist(float_img.ravel(), bins=50)
    plt.title('After (float: 0–1)')
    plt.tight_layout()
    plt.show()

def rgb_to_gray(image_float: np.ndarray) -> np.ndarray:
    greyscale_weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    image_float64 = image_float.astype(np.float64)
    return np.dot(image_float64, greyscale_weights)

def channel_summary(image_float: np.ndarray) -> tuple[np.ndarray, int]:
    color_channel_means = image_float.mean(axis=(0, 1))
    red_mean = color_channel_means[0]
    green_mean = color_channel_means[1]
    blue_mean = color_channel_means[2]
    brightest_channel = np.argmax(color_channel_means)
    return (color_channel_means, brightest_channel)

def convolve2d_matmul(image_gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    input_height = image_gray.shape[0]
    input_width = image_gray.shape[1]
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1
    output_image = np.zeros((output_height, output_width))
    flattened_kernel = kernel.ravel()
    for y in range(output_height):
        for x in range(output_width):
            patch = image_gray[y:y + kernel_height, x:x + kernel_width]
            flattened_patch = patch.ravel()
            response = flattened_patch @ flattened_kernel
            output_image[y, x] = response
    return output_image

def flatten_image(image: np.ndarray) -> np.ndarray:
    return image.ravel()
FEATURE_NAMES = ['mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b', 'brightest_channel', 'edge_mean', 'edge_std', 'row_std_mean']

def extract_features(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    cropped = center_crop(image, crop_size=48)
    image_float = normalize_01(cropped)
    gray = rgb_to_gray(image_float)
    channel_means, brightest_channel = channel_summary(image_float)
    channel_stds = image_float.std(axis=(0, 1)).astype(np.float32)
    filtered = convolve2d_matmul(gray, kernel)
    row_std_profile = np.apply_along_axis(np.std, 1, gray)
    output = np.concatenate([channel_means.astype(np.float32), channel_stds, np.array([brightest_channel], dtype=np.float32), np.array([filtered.mean(), filtered.std()], dtype=np.float32), np.array([row_std_profile.mean()], dtype=np.float32)])
    return output

def build_feature_matrix(paths: list[Path], kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features_list = []
    labels_list = []
    for path in paths:
        image = load_image_np(path)
        features = extract_features(image, kernel)
        label_str = label_from_path(path)
        label_idx = LABEL_TO_INDEX[label_str]
        features_list.append(features)
        labels_list.append(label_idx)
    return (np.array(features_list), np.array(labels_list))
