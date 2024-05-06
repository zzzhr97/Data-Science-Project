import numpy as np
from tqdm import tqdm
import os
import time
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from param import parse_args
from data import DescriptorDataset

class BOW(object):
    def __init__(self, args, des_data):
        self.args = args
        print("Fitting...")
        self.cluster_model = KMeans(n_clusters=self.args.ec_n_clusters, 
            verbose=self.args.ec_verbose, random_state=self.args.seed)
        self.cluster_model.fit(des_data)
        print("Done.")

    def __call__(self, dess):
        pred_ys = self.cluster_model.predict(dess)
        bow_histogram = np.bincount(pred_ys, minlength=self.args.ec_n_clusters)
        return bow_histogram

class VLAD(object):
    def __init__(self, args, des_data):
        self.args = args
        print("Fitting...")
        self.cluster_model = KMeans(n_clusters=self.args.ec_n_clusters, 
            verbose=self.args.ec_verbose, random_state=self.args.seed)
        self.cluster_model.fit(des_data)
        print("Done.")

    def __call__(self, dess):
        pred_ys = self.cluster_model.predict(dess)
        cluster_centers = self.cluster_model.cluster_centers_
        vlad_feature = np.zeros((self.args.ec_n_clusters, dess.shape[1]))
        offsets = dess - cluster_centers[pred_ys]
        for i in range(self.args.ec_n_clusters):
            cluster_offsets = offsets[pred_ys == i]
            if len(cluster_offsets) > 0:
                vlad_feature[i] = np.sum(offsets[pred_ys == i], axis=0)
        return vlad_feature.flatten()
    
class FisherVector(object):
    def __init__(self, args, des_data):
        self.args = args
        print("Fitting...")
        self.cluster_model = GaussianMixture(n_components=self.args.ec_n_clusters, 
            verbose=self.args.ec_verbose, random_state=self.args.seed,
            covariance_type=self.args.ec_cotype)
        self.cluster_model.fit(des_data)
        print("Done.")

    def __call__(self, dess):
        labels = self.cluster_model.predict(dess)
        gamma = self.cluster_model.predict_proba(dess)  # n_des, n_clusters
        means = self.cluster_model.means_
        covs = self.cluster_model.covariances_
        sqrt_weights = np.sqrt(self.cluster_model.weights_)
        n_des, feature_dim = dess.shape
        eps = 1e-6

        # (x_j - mu_k) / sigma_k
        # n_des, feature_dim
        decentralized_dess = (dess - means[labels]) / (np.sqrt(covs[labels]) + eps)

        F_mu = np.zeros((self.args.ec_n_clusters, feature_dim))
        F_sigma = np.zeros((self.args.ec_n_clusters, feature_dim))
        for i in range(self.args.ec_n_clusters):
            F_mu[i] = np.sum(gamma[:, i][:, None] * decentralized_dess, axis=0) / sqrt_weights[i] / n_des
            F_sigma[i] = np.sum(gamma[:, i][:, None] * (decentralized_dess ** 2 - 1), axis=0) / sqrt_weights[i] / n_des / np.sqrt(2)
        fisher_vector = np.concatenate([F_mu.flatten(), F_sigma.flatten()])
        assert len(fisher_vector.shape) == 1
        return fisher_vector
    
def get_encoder(args, des_data):
    if args.encoding_method == 'bow':
        return BOW(args, des_data)
    elif args.encoding_method == 'vlad':
        return VLAD(args, des_data)
    elif args.encoding_method == 'fisher':
        return FisherVector(args, des_data)
    else:
        raise ValueError("Invalid encoding method.")
    
def main():
    args = parse_args()
    data = DescriptorDataset(args)
    filenames = []
    training_dess = []
    save_dir = os.path.join(args.abs_path, "Animals_with_Attributes2", "Extracted_Features",
        args.extract_method, f"{args.encoding_method}_features{args.ec_n_clusters}")
    os.makedirs(save_dir, exist_ok=True)

    idxes_origin = [np.where(data.labels == (i+1))[0] for i in range(50)]
    print([len(idx) for idx in idxes_origin])
    idxes_len = [min(len(idx), args.ec_n_samples_per_class) for idx in idxes_origin]
    idxes = [np.random.choice(idx, len) for idx, len in zip(idxes_origin, idxes_len)]
    idxes = np.concatenate(idxes, axis=0)
    print(idxes.shape)

    with tqdm(total=idxes.shape[0], desc='Loading data', unit=' samples') as pbar:
        for i in range(idxes.shape[0]):
            filename, des, label = data[i]
            filenames.append(filename)
            training_dess.append(des)
            assert len(des.shape) == 2
            pbar.update(1)
    training_dess = np.concatenate(training_dess, axis=0)
    print(training_dess.shape)

    start_time = time.time()
    encoder = get_encoder(args, training_dess)
    print(f"Training time: {time.time() - start_time:.2f}s")
    with open(save_dir + '.txt', 'w') as f:
        print(f"Train - {time.time() - start_time:.4f}, ", end='', file=f)

    start_time = time.time()
    num_samples = len(data) if args.ec_num_samples < 0 or args.ec_num_samples > len(data) else args.ec_num_samples
    with tqdm(total=num_samples, desc='Encoding data', unit=' samples') as pbar:
        for i in range(num_samples):
            filename, des, label = data[i]
            feature = encoder(des)
            np.save(os.path.join(save_dir, filename.split('.')[0] + '.npy'), feature)
            del des, feature    
            pbar.update(1)
    with open(save_dir + '.txt', 'a') as f:
        print(f"Encode - {time.time() - start_time:.4f}", file=f)

if __name__ == '__main__':
    main()
    # args = parse_args()
    # data = DescriptorDataset(args)
    # for i in range(9054, 9055):
    #     filename, des, label = data[i]
    #     print(filename)
    #     print(des)
    #     print(label)