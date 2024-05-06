from sys import stdout
import numpy as np
from tqdm import tqdm
import time
import os
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from param import parse_args
from data import FeatureDataset

class Trainer(object):
    def __init__(self):
        self.args = parse_args()
        self.data = FeatureDataset(self.args)
        self.num_samples = len(self.data)
        self.num_classes = self.data.num_classes
        self.feature_dim = self.data.features.shape[1]

        self.result_path = os.path.join('results', self.args.extract_method,
            f'{self.args.encoding_method}_results{self.args.ec_n_clusters}.txt')
        os.makedirs(os.path.dirname(self.result_path), exist_ok=True)
    
    def train_test_with_dim_reduction(self):
        self.data.normalization()   
        if self.feature_dim > self.args.max_dim:
            self.data.dimension_reduction(self.args.max_dim, random_state=self.args.seed)

        (
            self.filenames_train, self.filenames_test,
            self.features_train, self.features_test,
            self.labels_train, self.labels_test
        ) = self.data.get_all_data()

        for kernel in self.args.kernel:
            for C in self.args.C:
                self.train_test(kernel, C)

    def train_test(self, kernel, C):
        print(f"Training with [kernel: {kernel}] [C: {C}] [Max Dim: {self.args.max_dim}]...", end=' ', flush=True)
        start_time = time.time()
        clf = svm.SVC(C=C, kernel=kernel)
        clf.fit(self.features_train, self.labels_train)
        train_time = time.time() - start_time
        print("Done.")

        print("Testing...", end=' ', flush=True)
        result = self.evaluate(clf)
        result['train_time'] = train_time
        print("Done.")

        with open(f'{self.result_path}', 'a') as f:
            string = f"[kernel: {kernel}] [C: {C}] [Max Dim: {self.args.max_dim}]\n" + \
                f"\tacc: {result['acc']:.4f}, f1: {result['f1']:.4f}, recall: {result['recall']:.4f}, " + \
                f"train time: {result['train_time']:.4f}, test time: {result['test_time']:.4f}"
            f.write(string + '\n')
            print(string)

    def evaluate(self, model):
        start_time = time.time()
        label_pred = model.predict(self.features_test)
        predict_time = time.time() - start_time

        acc = accuracy_score(self.labels_test, label_pred)
        f1 = f1_score(self.labels_test, label_pred, average='weighted')
        recall = recall_score(self.labels_test, label_pred, average='weighted')
        assert np.sum(label_pred == self.labels_test) / len(self.labels_test) == acc

        return {
            "acc": acc,
            "f1": f1,
            "recall": recall,
            "label_pred": label_pred,
            "test_time": predict_time
        }
    
    def save_model(self):
        pass
    
    def load_model(self):
        pass
    
    def save_results(self):
        pass
    
    def load_results(self):
        pass
    
    def __call__(self):
        pass