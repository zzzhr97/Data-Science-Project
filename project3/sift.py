import cv2
import os
import random
from tqdm import tqdm
import numpy as np

from param import parse_args
from data import ImageDataset

class SIFT(object):
    def __init__(self, args):
        self.args = args
        print("Number of points:", args.sift_num_points)
        self.sift_processor = cv2.SIFT_create(
            nfeatures=args.sift_num_points,
            sigma=1.0,
            contrastThreshold=0.03,
            edgeThreshold=15
        )
        save_path = f"dataset/Animals_with_Attributes2/Extracted_Features/sift{args.sift_num_points}"
        self.image_savepath = os.path.join(save_path, "show_images")
        os.makedirs(self.image_savepath, exist_ok=True)
        self.des_savepath = os.path.join(save_path, "descriptors")
        os.makedirs(self.des_savepath, exist_ok=True)

    def __call__(self, image, filename):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift_processor.detectAndCompute(gray_image, None)

        # save to numpy file
        np.save(os.path.join(self.des_savepath, filename + '.npy'), des)

        # show images with keypoints
        if self.args.sift_show_images:
            image_show = cv2.drawKeypoints(image, kp, image, color=(255, 0, 255))
            cv2.imwrite(os.path.join(self.image_savepath, filename + '.png'), image_show)
        return des
    
def main():
    args = parse_args()
    data = ImageDataset(args)

    sift = SIFT(args)

    num_images = len(data) if args.sift_num_images < 0 or args.sift_num_images > len(data) else args.sift_num_images
    if args.sift_shuffle:
        random.seed(42)
        index = random.sample(range(len(data)), num_images)
    else:
        index = range(num_images)
    print(f"Extracting {num_images} SIFT features.")
    with tqdm(total=num_images, desc='Extracting SIFT features', unit=' images') as pbar:
        for i in range(num_images):
        #for i in range(9054, 9055):
            filename, image, label = data[index[i]]
            filename = filename.split('.')[0]
            des = sift(image, filename)
            pbar.update(1)
            pbar.set_description_str(f"Processing {index[i]}")

if __name__ == '__main__':
    main()