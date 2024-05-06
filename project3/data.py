from tqdm import tqdm
import numpy as np
import cv2
import json
import torch
from typing import Union, Tuple
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import os
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing

class FeatureDataset(Dataset):
    def __init__(self, args):
        self.args = args
        base_path = os.path.join(args.abs_path, "Animals_with_Attributes2", "Features", "ResNet101")
        self.filename_path = os.path.join(base_path, "AwA2-filenames.txt")
        self.label_path = os.path.join(base_path, "AwA2-labels.txt")
        self.feature_path = os.path.join(args.abs_path, "Animals_with_Attributes2", "Extracted_Features", 
                args.extract_method, f"{args.encoding_method}_features{args.ec_n_clusters}.npy")
        print(f"Features from: [{args.extract_method}] [{args.encoding_method}{args.ec_n_clusters}.npy]")

        self.load_data()
        self.transfer_data()

    def load_data(self):
        self.load_filenames()
        self.load_labels()
        self.load_features()

    def load_filenames(self):
        self.filenames = []
        with open(self.filename_path, 'r') as f:
            for line in tqdm(f, desc='Loading filenames', unit=' samples'):
                self.filenames.append(line.strip())

    def load_labels(self):
        self.unique_classes = set()
        self.labels = []
        with open(self.label_path, 'r') as f:
            for line in tqdm(f, desc='Loading labels', unit=' samples'):
                self.labels.append(int(line))
                self.unique_classes.add(int(line))
        self.unique_classes = list(self.unique_classes)

    def load_features(self):
        print(f"Loading features from {self.feature_path}...", end=' ')
        self.features = np.load(self.feature_path)
        print(f"Done. Feature shape: {self.features.shape}")

    def transfer_data(self):
        self.num_samples = len(self.filenames)
        self.num_classes = len(self.unique_classes)
        self.filenames = np.array(self.filenames)
        self.labels = np.array(self.labels)
        print('Num samples:', self.num_samples)
        print('Num classes:', self.num_classes)

    def split_train_test_data(self):
        print("Splitting data...", end=' ', flush=True)
        # flush stdout
        (   
            self.filenames_train, self.filenames_test,
            self.features_train, self.features_test,
            self.labels_train, self.labels_test
        ) = train_test_split(
            self.filenames, self.features, self.labels, 
            test_size=0.4, 
            random_state=self.args.seed, stratify=self.labels
        )
        print("Done.")

    def get_all_data(self):
        self.split_train_test_data()
        return (
            self.filenames_train, self.filenames_test,
            self.features_train, self.features_test,
            self.labels_train, self.labels_test
        )
    
    def dimension_reduction(self, dim, **kwargs):
        print("Dimension reduction...", end=' ')
        pca = PCA(n_components=dim, **kwargs)
        self.features = pca.fit_transform(self.features)
        print(f"Done.\nReduced feature dimension: {self.features.shape}")

    def normalization(self, **kwargs):
        print("L2 Norm.")
        self.features = preprocessing.normalize(self.features, norm='l2')

    def __getitem__(self, ids, mode='train'):
        """
        Get data by ids.
        - idx: int or list of int
        - mode: 'train', 'test', 'all'
        """
        if mode == 'train':
            filenames = self.filenames_train[ids]
            features = self.features_train[ids]
            labels = self.labels_train[ids]
        elif mode == 'test':
            filenames = self.filenames_test[ids]
            features = self.features_test[ids]
            labels = self.labels_test[ids]
        elif mode == 'all':
            filenames = self.filenames[ids]
            features = self.features[ids]
            labels = self.labels[ids]
        else:
            raise ValueError("Invalid mode.")
        return filenames, features, labels
    
    def __len__(self) -> int:
        return self.num_samples
    
class ImageDataset(Dataset):
    def __init__(self, args, mode='total', transform=None):
        self.args = args
        self.mode = mode
        self.transform = transform
        base_path = os.path.join(args.abs_path, "Animals_with_Attributes2", "Features", "ResNet101")
        self.filename_path = os.path.join(base_path, "AwA2-filenames.txt")
        self.label_path = os.path.join(base_path, "AwA2-labels.txt")
        self.image_path = os.path.join(args.abs_path, "Animals_with_Attributes2", "JPEGImages")

        self.load_data()
        self.transfer_data()

        self.filenames_train, self.filenames_test, self.labels_train, self.labels_test = train_test_split(
            self.filenames, self.labels, test_size=0.4, random_state=args.seed, stratify=self.labels
        )

    def load_data(self):
        self.load_filenames()
        self.load_labels()

    def load_filenames(self):
        self.filenames = []
        with open(self.filename_path, 'r') as f:
            for line in tqdm(f, desc='Loading filenames', unit=' samples'):
                self.filenames.append(line.strip())

    def load_labels(self):
        self.unique_classes = set()
        self.labels = []
        with open(self.label_path, 'r') as f:
            for line in tqdm(f, desc='Loading labels', unit=' samples'):
                self.labels.append(int(line))
                self.unique_classes.add(int(line))
        self.unique_classes = list(self.unique_classes)

    def transfer_data(self):
        self.num_samples = len(self.filenames)
        self.num_classes = len(self.unique_classes)
        self.filenames = np.array(self.filenames)
        self.labels = np.array(self.labels)
        print('Num samples:', self.num_samples)
        print('Num classes:', self.num_classes)

    def __getitem__(self, id: int) -> Tuple[str, np.ndarray, int]:
        if self.mode == 'total':
            filenames = self.filenames
            labels = self.labels
        elif self.mode == 'train':
            filenames = self.filenames_train
            labels = self.labels_train
        elif self.mode == 'test':
            filenames = self.filenames_test
            labels = self.labels_test
        animal_name = filenames[id].split('_')[0]
        image = cv2.imread(os.path.join(self.image_path, animal_name, filenames[id]))

        if self.transform is not None:
            image = self.transform(image)

        return filenames[id], image, labels[id]-1
    
    def __len__(self) -> int:
        if self.mode == 'total':
            return self.num_samples
        elif self.mode == 'train':
            return len(self.filenames_train)
        elif self.mode == 'test':
            return len(self.filenames_test)
        assert False
    
class DescriptorDataset(Dataset):
    def __init__(self, args):
        self.args = args
        base_path = os.path.join(args.abs_path, "Animals_with_Attributes2", "Features", "ResNet101")
        self.filename_path = os.path.join(base_path, "AwA2-filenames.txt")
        self.label_path = os.path.join(base_path, "AwA2-labels.txt")
        self.des_path = os.path.join(args.abs_path, "Animals_with_Attributes2", "Extracted_Features", 
            args.extract_method, "descriptors")

        self.load_data()
        self.transfer_data()
        self.split_data_by_class()

    def load_data(self):
        self.load_filenames()
        self.load_labels()

    def load_filenames(self):
        self.filenames = []
        with open(self.filename_path, 'r') as f:
            for line in tqdm(f, desc='Loading filenames', unit=' samples'):
                self.filenames.append(line.strip())

    def load_labels(self):
        self.unique_classes = set()
        self.labels = []
        with open(self.label_path, 'r') as f:
            for line in tqdm(f, desc='Loading labels', unit=' samples'):
                self.labels.append(int(line))
                self.unique_classes.add(int(line))
        self.unique_classes = list(self.unique_classes)

    def transfer_data(self):
        self.num_samples = len(self.filenames)
        self.num_classes = len(self.unique_classes)
        self.filenames = np.array(self.filenames)
        self.labels = np.array(self.labels)
        print('Num samples:', self.num_samples)
        print('Num classes:', self.num_classes)

    def split_data_by_class(self):
        self.filename_by_class = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.filename_by_class[i] = self.filenames[self.labels == (i+1)]

    def __getitem__(self, id):
        filename = self.filenames[id]
        des = np.load(os.path.join(self.des_path, filename.split('.')[0] + '.npy'), allow_pickle=True)
        return filename, des, self.labels[id]
    
    def __len__(self) -> int:
        return self.num_samples

if __name__ == '__main__':
    from param import parse_args
    args = parse_args()
    args.extract_method = 'ss'
    data = DescriptorDataset(args)

    shape_cnt = 0
    for filename, des, label in tqdm(data):
        shape_cnt += des.shape[0]
    print("average number of des for each image:", shape_cnt / len(data))