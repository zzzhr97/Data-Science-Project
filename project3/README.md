# Project 3: Feature Encoding for Image Classification

## Task
See [Task Requirements](./Task-Requirements.pdf).

## Run
### Download Dataset
Download the total dataset to `./dataset/` from [AwA2 dataset](https://cvml.ist.ac.at/AwA2/), the path will look like `./dataset/Animals_with_Attributes2/JPEGImages/....`.

### Extracting Local Descriptors
#### SIFT
Modify [`scripts/sift.sh`](./scripts/sift.sh) with reference to [`param.py`](./param.py) and [`sift.py`](./sift.py), and run:
```shell
bash scripts/sift.sh
```  

#### Selective Search with Neural Network Encoder
See [`train.py`](./train.py) and modify the parameters in [`scripts/train.sh`](./scripts/train.sh).
Run the following command to train a `ResNet50` model encoder.
```shell
bash scripts/train.sh
```

Then, use selective search and encoder to obtain local descriptors of each image. 
Before run the command, modify `scripts/ss-gpu*.sh` with reference to [`ss.py`](./ss.py).
```shell
bash scripts/ss-gpu-1.sh
bash scripts/ss-gpu-2.sh
bash scripts/ss-gpu-3.sh
bash scripts/ss-gpu-4.sh
bash scripts/ss-gpu-5.sh
```

You can run the following command to monitor the process.
```shell
python ss_record.py
```

### Encoding Local Descriptors
Use BOW, VLAD and Fisher Vector to encode the features from local descriptors of each image.
Modify `scripts/ec_*.sh` with reference to [`param.py`](./param.py) and [`encoder.py`](./encoder.py), and run:
```shell
bash scripts/ec_bow.sh
bash scripts/ec_vlad.sh
bash scripts/ec_fisher.sh
```

Then, to speed up the process of the SVM classfication, run the following command to aggregate feature files.
```shell
python aggregate_features.py
```

### SVM Classification
Use SVM for classification. See [`trainer.py`](./trainer.py) and [`param.py`](./param.py) and modify `scripts/run_*.sh`.
Run the command:
```shell
python scripts/run_bow.sh
python scripts/run_vlad.sh
python scripts/run_fisher.sh
```