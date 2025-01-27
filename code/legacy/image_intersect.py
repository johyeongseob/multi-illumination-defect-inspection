import os


def get_image_filenames(folder_path):
    """Returns a set of image file names in the given folder."""
    return set(os.listdir(folder_path))


def find_common_images(folder1, folder2):
    """Returns a set of common image file names between two folders."""
    folder1_images = get_image_filenames(folder1)
    folder2_images = get_image_filenames(folder2)

    common_images = folder1_images.intersection(folder2_images)

    return common_images


# Example folder paths
folder1_path = '../data_set/test/NG1/1'
folder2_path = '../data_set/test/NG1/1'

# Find common image file names
common_images = find_common_images(folder1_path, folder2_path)

if common_images:
    print(f"Common image file names in both folders: {common_images}")
else:
    print("No common image file names between the two folders.")
