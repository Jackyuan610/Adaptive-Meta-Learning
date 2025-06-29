import os
import csv
import shutil
from tqdm import tqdm

# 源图片目录
IMAGES_DIR = 'images'
# 目标 split 目录
SPLITS = ['train', 'val', 'test']
CSV_FILES = {
    'train': 'train.csv',
    'val': 'val.csv',
    'test': 'test.csv',
}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_csv(csv_path):
    """解析 csv，返回 [(filename, label)] 列表"""
    items = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            filename, label = row[0], row[1]
            items.append((filename, label))
    return items

def main():
    for split in SPLITS:
        csv_file = CSV_FILES[split]
        split_dir = split
        ensure_dir(split_dir)
        items = parse_csv(csv_file)
        print(f"Processing {split}: {len(items)} images")
        for filename, label in tqdm(items):
            label_dir = os.path.join(split_dir, label)
            ensure_dir(label_dir)
            src_path = os.path.join(IMAGES_DIR, filename)
            dst_path = os.path.join(label_dir, filename)
            if not os.path.exists(src_path):
                print(f"Warning: {src_path} not found!")
                continue
            shutil.copyfile(src_path, dst_path)
    print("All splits prepared!")

if __name__ == '__main__':
    main() 