# MTC-LoGANv2-CNN
This is a repository for scripts used in the manuscript "Deep Semi-Supervised Learning Using Generative Adversarial Networks for Automated Seismic Facies Classification of Mass Transport Complex .", submitted to Computers and Geosciences.

The framework proposed in this paper contains two major componements, known as the LoGANv2 for the augmentation of mass transport complex and CNN for the classification of Mass Transport Complex facies. 

## LoGANv2
Implementation of LoGANv2 is based on the open-source code published by [Oeldorf and Spanakis, 2019](https://arxiv.org/abs/1909.09974). 

The github repository for LoGANv2 is listed as https://github.com/cedricoeldorf/ConditionalStyleGAN. 

### Set-up
```
To run the code, an NVIDIA QUADRO P5000 video card with 8GB video memory is required.

Software development environment should be any Python integrated development environment used on an NVIDIA video card.

Programming language: Python 3.6.

Tensorflow-gpu == 1.15.1

Tensorflow == 1.19
```
### Data Preparation
To run LoGANv2, each image in the dataset must have the exact same format in terms of size, extension, colour space and bit depth. To eliminate any irregular images and convert dataset into Tensorflow record files, run dataset_tool.py as

```
# Pickle path = '../data/mypickle.pickle'
mypickle = {"Filenames": list_of_file_paths, "Labels": class_condition_labels}
```
The script is run from the terminal and takes the paths to your images and the path of your TF-record directory as flags
```
python dataset_tool.py create_from_images dataset/logos ../data/my_images
```

### Training

There are serval steps involved in the training of LoGANv2

- In train.py line 37:
```
desc += '-logos'; dataset = EasyDict(tfrecord_dir='logos', resolution=128);
```
- In ./training/dataset.py line 49:
```
self.tfrecord_dir = 'dataset/logos'
```

#### Step 1: Locate path to TF records (e.g. dataset)


### Evaluating the Network

## CNN

### Data Preparation
### Training

### Evaluating the Network
