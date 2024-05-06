import cv2
import selectivesearch
import os
import copy
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image

from param import parse_args
from data import ImageDataset
from train import ResNet50

class SS(object):
    def __init__(self, args):
        self.args = args
        self.setup_args()
        self.setup_model()
        self.setup_path()

    def setup_args(self):
        self.scale = self.args.ss_scale
        self.sigma = self.args.ss_sigma
        self.min_size = self.args.ss_min_size
        print("scale: {}, sigma: {}, min_size: {}".format(
            self.scale, self.sigma, self.min_size))

    def setup_path(self):
        save_path = f"dataset/Animals_with_Attributes2/Extracted_Features/ss"
        self.image_savepath = os.path.join(save_path, "show_images")
        os.makedirs(self.image_savepath, exist_ok=True)
        self.des_savepath = os.path.join(save_path, "descriptors")
        os.makedirs(self.des_savepath, exist_ok=True)
        self.record_savepath = os.path.join(save_path, "record")
        os.makedirs(self.record_savepath, exist_ok=True)

    def setup_model(self):
        self.device = torch.device(f'cuda:{self.args.ss_device}' if self.args.ss_device >= 0 else 'cpu')
        self.model = torch.load('resnet50_best.pth', map_location=self.device)
        self.model.eval()
        self.model.to(self.device)

    def _to_tensor(self, image: np.ndarray, proposals: list) -> torch.Tensor:
        proposals_torch_list = []
        _resize = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        for x1, y1, x2, y2 in proposals:
            sub_image = image[y1:y2, x1:x2]
            sub_image = _resize(sub_image)
            proposals_torch_list.append(sub_image)
        return torch.stack(proposals_torch_list, dim=0)

    def __call__(self, idx, image, filename):
        # extract proposals
        img_lbl, regions = selectivesearch.selective_search(image, scale=self.scale, 
            sigma=self.sigma, min_size=self.min_size)  # (left, top, width, height)
        proposals = []
        for region in regions:
            l, t, w, h = region['rect']
            if w == 0 or h == 0:
                continue
            proposals.append([l, t, l+w, t+h])

        # visualize
        if self.args.ss_show_images:
            image_annot = copy.deepcopy(image)
            for x1, y1, x2, y2 in proposals:
                image_annot = cv2.rectangle(image_annot, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite(os.path.join(self.image_savepath, filename.split('.')[0] + '.png'), image_annot)

        # convert proposals to batch and extract descriptor features
        batch_proposals = self._to_tensor(image, proposals).to(self.device)
        with torch.no_grad():
            des, _ = self.model(batch_proposals)
            des = des.cpu().numpy()
        np.save(os.path.join(self.des_savepath, filename.split('.')[0] + '.npy'), des)

        # record
        with open(os.path.join(self.record_savepath, str(idx) + '.txt'), 'w') as f:
            f.write(f"FILE: {filename}\n")
            try:
                f.write(f"DES_SHAPE: {des.shape}\n")
            except:
                f.write(f"DES_SHAPE: None\n")

def main():
    args = parse_args()
    data = ImageDataset(args)

    ss = SS(args)

    num_images = len(data) if args.ss_num_images < 0 or args.ss_num_images > len(data) else args.ss_num_images
    index = list(range(len(data)))
    random.shuffle(index)
    print(f"Extracting {num_images} Selective Search features.")
    with tqdm(total=num_images, desc='Extracting Selective Search features', unit=' images') as pbar:
        for i in range(
            args.ss_image_start, 
            min(args.ss_image_start + num_images, len(data))
        ):
            filename, image, label = data[index[i]]
            filename = filename.split('.')[0]
            ss(i, image, filename)
            pbar.update(1)
            pbar.set_description_str(f"Processing {i}")

if __name__ == '__main__':
    main()