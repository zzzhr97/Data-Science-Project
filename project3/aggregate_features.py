import numpy as np
import os
from tqdm import tqdm

def count_files_in_directory(directory, pbar):
    count = 0
    for dir in os.listdir(directory):
        dir_path = os.path.join(directory, dir)
        if os.path.isdir(dir_path) and 'features' in dir_path:
            for file in os.listdir(dir_path):
                count += 1
                pbar.update(1)
    return count

def aggregate(total_path):
    with tqdm(desc='Counting files...', unit=' samples') as pbar:
        total = count_files_in_directory(total_path, pbar)
    dir_finished = 0
    with tqdm(total=total, desc=f'Processing {dir_finished+1}-th dir', unit=' samples') as pbar:
        for dir in os.listdir(total_path):
            dir_path = os.path.join(total_path, dir)
            if os.path.isdir(dir_path) and 'features' in dir_path:
                save_path = dir_path + '.npy'
                save_data = []
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    save_data.append(np.load(file_path))
                    pbar.update(1)
                save_data = np.stack(save_data, axis=0)
                np.save(save_path, save_data)
                dir_finished += 1
                pbar.set_description(f'Processing {dir_finished+1}-th dir')

def sample(path):
    x = np.load(path)
    print(x.shape)

def main():
    # total_path = './dataset/Animals_with_Attributes2/Extracted_Features/sift1000'
    total_path = './dataset/Animals_with_Attributes2/Extracted_Features/ss'
    aggregate(total_path)

if __name__ == '__main__':
    main()