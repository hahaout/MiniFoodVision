# Download data
from pathlib import Path
import torch as T
import torchvision
import random
import shutil


def download_food101():
    """
    This function allows you to download Food101 from Pytorch

    Args:
    None

    Returns:
    train_data, test_data, class_names

    Example:
    train_data, test_data = download_food101()
    """
    image_path = Path("data/")
    image_data = image_path / "Food"

    if image_data.is_dir():
        print(f"File existed. Skipping Download...")

    else:
        print(f"Data file is not founded. Start Downloading...")
        torchvision.datasets.Food101(root=image_data,
                                                split="train",
                                                download=True)
        torchvision.datasets.Food101(root=image_data,
                                                split="test",
                                                download=True)
        print(f"Download complete")


import shutil
import random
from pathlib import Path

def split_train_and_test(target_classes, amount_to_get: int):
    """ 
    This function allows you to split Food101 into a smaller subset
    and creates folders in your data directory for storage.

    Args:
        target_classes: The list of target classes you want to split.
        amount_to_get: User can choose what amount they want to split from the original.

    Returns:
        None
    """
    # Setup data paths
    image_path = Path("data/")
    data_path = image_path / "Food" / "food-101" / "images"

    # Create function to separate a random amount of data
    def get_subset(image_path=image_path,
                   data_splits=["train", "test"], 
                   target_classes=target_classes,
                   amount=0.1,
                   seed=42):
        random.seed(seed)
        label_splits = {}
        
        # Get labels
        for data_split in data_splits:
            print(f"[INFO] Creating image split for: {data_split}...")
            label_path = image_path / "Food" / "food-101" / "meta" / f"{data_split}.txt"
            with open(label_path, "r") as f:
                labels = [line.strip("\n") for line in f.readlines() if line.split("/")[0] in target_classes] 
            
            # Get random subset of target class image IDs
            number_to_sample = round(amount * len(labels))
            print(f"[INFO] Getting random subset of {number_to_sample} images for {data_split}...")
            sampled_images = random.sample(labels, k=number_to_sample)
            
            # Apply full paths
            image_paths = [Path(str(data_path / sample_image) + ".jpg") for sample_image in sampled_images]
            label_splits[data_split] = image_paths
        return label_splits
    
    # Get the subsets
    label_splits = get_subset(amount=amount_to_get)

    # Create target directory path
    target_dir_name = "data/" + f"food_{str(int(amount_to_get * 100))}_percent"
    print(f"Creating directory: '{target_dir_name}'")

    # Setup the directories
    target_dir = Path(target_dir_name)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy files to new directories
    for image_split in label_splits.keys():
        for image_path in label_splits[image_split]:
            # Extract class name from the image path
            class_name = image_path.parent.name 
            # Create class subdirectory within train/test
            dest_dir = target_dir / image_split / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)  # Create if it doesn't exist
            print(f"[INFO] Copying {image_path} to {dest_dir}...")
            shutil.copy2(image_path, dest_dir)