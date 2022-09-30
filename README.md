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

Programming language: Python 3.6.13

Tensorflow-gpu == 1.15.1

Tensorflow == 1.19

numpy == 1.19.2

scipy == 1.5.2

torch == 1.10

seaborn == 0.11.2
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

#### Step 1: Locate path to TF records (e.g. dataset)

- In train.py line 37:
```
desc += '-logos'; dataset = EasyDict(tfrecord_dir='logos', resolution=128);
```
- In dataset.py line 49:
```
self.tfrecord_dir = 'dataset/logos'
```
##### Step 2: Set number of class-conditions

- In networks_stylegan.py line 388 & line 569, change 
```
label_size = 10
```
#### Step 3: Set hyper-parameters for networks and other indications for the training loop

- Starting at line 112 in training_loop.py
```
G_smoothing_kimg        = 10.0,     # Half-life of the running average of generator weights.
D_repeats               = 2,        # How many times the discriminator is trained per G iteration.
minibatch_repeats       = 1,        # Number of minibatches to run before adjusting training parameters.
reset_opt_for_new_lod   = True,     # Reset optimizer internal state (e.g. Adam moments) when new layers are introduced?
total_kimg              = 20000,    # Total length of the training, measured in thousands of real images.
mirror_augment          = True,     # Enable mirror augment?
drange_net              = [-1,1],   # Dynamic range used when feeding image data to the networks.
```
- Starting at line 384 in networks_stylegan.py:
```
dlatent_size            = 128,          # Disentangled latent (W) dimensionality.
mapping_layers          = 8,            # Number of mapping layers.
mapping_fmaps           = 128,          # Number of activations in the mapping layers.
mapping_lrmul           = 0.01,         # Learning rate multiplier for the mapping layers.
mapping_nonlinearity    = 'lrelu',      # Activation function: 'relu', 'lrelu'.
use_wscale              = True,         # Enable equalized learning rate?
normalize_latents       = True,         # Normalize latent vectors (Z) before feeding them to the mapping layers?
```
- Starting at line 384 in networks_stylegan.py:
```
resolution          = 128,          # Output resolution.
fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
fmap_max            = 128,          # Maximum number of feature maps in any layer.
use_styles          = True,         # Enable style inputs?
const_input_layer   = True,         # First layer is a learned constant?
use_noise           = True,         # Enable noise inputs?
randomize_noise     = True,         # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu'
use_wscale          = True,         # Enable equalized learning rate?
use_pixel_norm      = False,        # Enable pixelwise feature vector normalization?
use_instance_norm   = True,         # Enable instance normalization?
dtype               = 'float32',    # Data type to use for activations and outputs.
fused_scale         = 'auto',       # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically.
blur_filter         = [1,2,1],      # Low-pass filter to apply when resampling activations. None = no filtering.
```
- Starting at line 384 in networks_stylegan.py:
```
fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
fmap_max            = 128,          # Maximum number of feature maps in any layer.
nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu',
use_wscale          = True,         # Enable equalized learning rate?
mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
mbstd_num_features  = 1,            # Number of features for the minibatch standard deviation layer.
fused_scale         = 'auto',       # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically.
blur_filter         = [1,2,1],      # Low-pass filter to apply when resampling activations. None = no filtering.
```

#### Step 4: Initialize training

Initialize training of architecture by running:

```
python train.py
```

### Evaluating the Network
In order to evaluate the network, select evaluation tasks in line 80 of run_metrics.py and insert relevant network pickle path:

```
tasks += [EasyDict(run_func_name='run_metrics.run_pickle', network_pkl='./results/pickle.pkl', dataset_args=EasyDict(tfrecord_dir='logos', shuffle_mb=0), mirror_augment=True)]
```

## CNN
The CNN devleoped in this work is based on the AlexNet proposed by [Krizhevsky et al, 2012](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

### Set-up
```
To run the code, an NVIDIA QUADRO P5000 video card with 8GB video memory is required.

Software development environment should be any Python integrated development environment used on an NVIDIA video card.

Programming language: Python 3.6.13

Tensorflow-gpu == 1.15.1

Tensorflow == 1.19

numpy == 1.19.2

scipy == 1.5.2

torch == 1.10

seaborn == 0.11.2
```
### Data Preparation

The training data for CNN contains both real seismic facies for mass transport complex and synthetic faceies generated by LoGANv2. The tetsing data contains a seperate set of data that the model has not seen before. 

### Training
There are serval steps involved in the training of LoGANv2

#### Step 1: Locate path to combined training set

- In train.py line 43:
```
path_train = \Training\Trial_number
```
- In dataset.py line 49:
```
self.tfrecord_dir = 'dataset/logos'

### Evaluating the Network
