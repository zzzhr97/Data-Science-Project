
import os
import numpy as np
import time
from tqdm import tqdm

total = 37322
record_path = './dataset/Animals_with_Attributes2/Extracted_Features/ss/record'
record_list = np.array([0 for i in range(total)])

def get_file_record():
    files = os.listdir(record_path)
    for file in files:
        record_list[int(file.split('.')[0])] = 1
    return np.sum(record_list)

def run_record():
    finished = 0
    with tqdm(total=total, desc='Selective Searching...', unit=' samples') as pbar:
        while finished < total:
            cur_finished = get_file_record()
            pbar.update(cur_finished - finished)
            finished = cur_finished

            time.sleep(1)


if __name__ == '__main__':
    run_record()