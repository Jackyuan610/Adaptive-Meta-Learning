#!/usr/bin/env python3
"""
Dataset setup script for continual few-shot learning experiments.

Downloads and prepares Mini-ImageNet and Omniglot datasets for the project.
"""

import os
import requests
import zipfile
import tarfile
from pathlib import Path
import shutil
from tqdm import tqdm
import argparse


def download_file(url: str, filename: str, chunk_size: int = 8192):
    """
    Download file with progress bar.
    
    Args:
        url: URL to download from
        filename: Local filename to save as
        chunk_size: Size of chunks to download
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            pbar.update(size)


def setup_mini_imagenet(data_dir: str = "data"):
    """
    Download and setup Mini-ImageNet dataset.
    
    Args:
        data_dir: Directory to store datasets
    """
    print("Setting up Mini-ImageNet dataset...")
    
    # Create data directory
    mini_imagenet_dir = os.path.join(data_dir, "mini-imagenet")
    os.makedirs(mini_imagenet_dir, exist_ok=True)
    
    # Mini-ImageNet download URLs
    urls = {
        'train': 'https://github.com/yaoyao-liu/mini-imagenet-tools/raw/master/mini-imagenet/train.zip',
        'val': 'https://github.com/yaoyao-liu/mini-imagenet-tools/raw/master/mini-imagenet/val.zip',
        'test': 'https://github.com/yaoyao-liu/mini-imagenet-tools/raw/master/mini-imagenet/test.zip'
    }
    
    for split, url in urls.items():
        print(f"Downloading {split} split...")
        zip_path = os.path.join(mini_imagenet_dir, f"{split}.zip")
        
        # Download if not exists
        if not os.path.exists(zip_path):
            download_file(url, zip_path)
        
        # Extract if not already extracted
        extract_dir = os.path.join(mini_imagenet_dir, split)
        if not os.path.exists(extract_dir):
            print(f"Extracting {split} split...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(mini_imagenet_dir)
            
            # Rename if necessary
            if os.path.exists(os.path.join(mini_imagenet_dir, f"{split}_images")):
                os.rename(
                    os.path.join(mini_imagenet_dir, f"{split}_images"),
                    extract_dir
                )
    
    print("Mini-ImageNet setup completed!")


def setup_omniglot(data_dir: str = "data"):
    """
    Download and setup Omniglot dataset.
    
    Args:
        data_dir: Directory to store datasets
    """
    print("Setting up Omniglot dataset...")
    
    # Create data directory
    omniglot_dir = os.path.join(data_dir, "omniglot")
    os.makedirs(omniglot_dir, exist_ok=True)
    
    # Omniglot download URL
    url = "https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip"
    zip_path = os.path.join(omniglot_dir, "images_background.zip")
    
    # Download if not exists
    if not os.path.exists(zip_path):
        print("Downloading Omniglot background images...")
        download_file(url, zip_path)
    
    # Extract if not already extracted
    extract_dir = os.path.join(omniglot_dir, "images_background")
    if not os.path.exists(extract_dir):
        print("Extracting Omniglot dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(omniglot_dir)
    
    # Download evaluation images
    eval_url = "https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip"
    eval_zip_path = os.path.join(omniglot_dir, "images_evaluation.zip")
    
    if not os.path.exists(eval_zip_path):
        print("Downloading Omniglot evaluation images...")
        download_file(eval_url, eval_zip_path)
    
    eval_extract_dir = os.path.join(omniglot_dir, "images_evaluation")
    if not os.path.exists(eval_extract_dir):
        print("Extracting Omniglot evaluation dataset...")
        with zipfile.ZipFile(eval_zip_path, 'r') as zip_ref:
            zip_ref.extractall(omniglot_dir)
    
    print("Omniglot setup completed!")


def setup_meta_dataset(data_dir: str = "data"):
    """
    Setup instructions for Meta-Dataset.
    
    Args:
        data_dir: Directory to store datasets
    """
    print("Meta-Dataset setup instructions:")
    print("1. Download Meta-Dataset from: https://github.com/google-research/meta-dataset")
    print("2. Follow the installation instructions in the repository")
    print("3. Set the METADATASET_ROOT environment variable")
    print("4. Run the data processing scripts")
    print("\nFor this project, you can use a simplified version:")
    
    # Create simplified meta-dataset structure
    meta_dataset_dir = os.path.join(data_dir, "meta-dataset")
    os.makedirs(meta_dataset_dir, exist_ok=True)
    
    # Create placeholder files
    datasets = ['aircraft', 'cu_birds', 'dtd', 'fungi', 'ilsvrc_2012', 'omniglot', 'quickdraw', 'vgg_flower']
    
    for dataset in datasets:
        dataset_dir = os.path.join(meta_dataset_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create placeholder file
        placeholder_file = os.path.join(dataset_dir, "README.txt")
        if not os.path.exists(placeholder_file):
            with open(placeholder_file, 'w') as f:
                f.write(f"Placeholder for {dataset} dataset.\n")
                f.write("Please download the full Meta-Dataset for complete functionality.\n")
    
    print("Simplified Meta-Dataset structure created.")


def verify_datasets(data_dir: str = "data"):
    """
    Verify that datasets are properly set up.
    
    Args:
        data_dir: Directory containing datasets
    """
    print("Verifying dataset setup...")
    
    # Check Mini-ImageNet
    mini_imagenet_dir = os.path.join(data_dir, "mini-imagenet")
    if os.path.exists(mini_imagenet_dir):
        splits = ['train', 'val', 'test']
        for split in splits:
            split_dir = os.path.join(mini_imagenet_dir, split)
            if os.path.exists(split_dir):
                num_classes = len([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
                print(f"Mini-ImageNet {split}: {num_classes} classes")
            else:
                print(f"Mini-ImageNet {split}: NOT FOUND")
    else:
        print("Mini-ImageNet: NOT FOUND")
    
    # Check Omniglot
    omniglot_dir = os.path.join(data_dir, "omniglot")
    if os.path.exists(omniglot_dir):
        background_dir = os.path.join(omniglot_dir, "images_background")
        eval_dir = os.path.join(omniglot_dir, "images_evaluation")
        
        if os.path.exists(background_dir):
            num_alphabets = len([d for d in os.listdir(background_dir) if os.path.isdir(os.path.join(background_dir, d))])
            print(f"Omniglot background: {num_alphabets} alphabets")
        else:
            print("Omniglot background: NOT FOUND")
            
        if os.path.exists(eval_dir):
            num_alphabets = len([d for d in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, d))])
            print(f"Omniglot evaluation: {num_alphabets} alphabets")
        else:
            print("Omniglot evaluation: NOT FOUND")
    else:
        print("Omniglot: NOT FOUND")
    
    # Check Meta-Dataset
    meta_dataset_dir = os.path.join(data_dir, "meta-dataset")
    if os.path.exists(meta_dataset_dir):
        datasets = [d for d in os.listdir(meta_dataset_dir) if os.path.isdir(os.path.join(meta_dataset_dir, d))]
        print(f"Meta-Dataset: {len(datasets)} datasets found")
    else:
        print("Meta-Dataset: NOT FOUND")


def main():
    """Main function to setup all datasets."""
    parser = argparse.ArgumentParser(description="Setup datasets for continual few-shot learning")
    parser.add_argument("--data_dir", default="data", help="Directory to store datasets")
    parser.add_argument("--datasets", nargs="+", choices=["mini-imagenet", "omniglot", "meta-dataset", "all"],
                       default=["all"], help="Datasets to setup")
    parser.add_argument("--verify", action="store_true", help="Verify dataset setup")
    
    args = parser.parse_args()
    
    # Create data directory
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Setup datasets
    if "all" in args.datasets or "mini-imagenet" in args.datasets:
        setup_mini_imagenet(args.data_dir)
    
    if "all" in args.datasets or "omniglot" in args.datasets:
        setup_omniglot(args.data_dir)
    
    if "all" in args.datasets or "meta-dataset" in args.datasets:
        setup_meta_dataset(args.data_dir)
    
    # Verify setup
    if args.verify:
        verify_datasets(args.data_dir)
    
    print("Dataset setup completed!")


if __name__ == "__main__":
    main() 