# Image Captioning
By: Dana Rip

## Introduction
In this git we introduce a comparison for image captioning: we implement a transformer and a lstm
with attention. Both architectures are based on encoder of ResNet50. 

## Files
- Under `source/tst` there are several scripts that run the pipline. These are the entry point scripts for operating the comparison.
  - `source/tst/tst_lstm_short.py` - runs a relatively short test to examine the lstm architecture
  - `source/tst/tst_transformer_short.py` - runs a relatively short test to examine the transformer architecture

Under the folder `source/` the main code can be found:
- `ImageCaptioning.py` - This is the main file to run the pipeline itself. If receives a long list of parameters in the `single_run` function.
- `run_experiments.py` - run several experiments together, i.e., sweep through parameters in order to generate
this work graphs. This is also can be an entry point to run this repo.
- `decoding_utils.py` - do a greedy decoding of the transformer. 
Specifically, at each round predicts the next word of the transformer. Predicts the most probable
word.
- `data_preprocessing.py` - holds transformations for data collection, class of Vocabulary to
build the dataset vocabulary. FlickrDataset, a class to prcess the Flickr dataset into DataLoader.
CapsCollate to handle the padding of the sentences in the dataset.
- `networks_lstm.py` - holds all the networks related to the LSTM and soft attention mechanism. 
In addition contains an Encoder based on pretrained ResNet50 from `torchvision` package.
- `networks_transformer.py` - contains the decoder only for transformer based caption generator.
The encoder for the transformer is based on the same encoder as the lstm from `netowrks_lstm.py`.
- `split_dataset.py` - given Flickr8K images and captions, split to train, validation, test sets.
- `show_attention_lstm.py` - show the lstm attention mechanism for the vision part.
- `compute_bleu.py` - this function goes over the test set and compute the BLEU score.
- `utils.py` - small scipts to support other functions like transforming from indices to words, 
from words to indices, etc. 

## How to run from command line
`pyhton -m source.ImageCaptioning`

## Generated folder structure (not included in this git)
The flickr8k dataset need to be in the folder `data/flickr8k/Images/`

The dataset can be downloaded from https://www.kaggle.com/datasets/shadabhussain/flickr8k or from 
https://www.kaggle.com/datasets/adityajn105/flickr8k. The images should be extracted to `data/flickr8k/Images/`.

The following files are not necessarily are given in this git. For example, the images are just to 
big to be included in this git.

```
data/                     # folder that contains the data
  Flickr8kTestImages/     # Created by split_dataset.py-  contains the test images.
  Flickr8kTrainImages/    # Created by split_dataset.py
  Flickr8kValidationImages/     # Created by split_dataset.py
  Images/                 # Download to here all the Flickr8k dataset images. The split will be based on this folder
  captions.txt            # Original captions.txt file
  captions_test.txt       # Creates by split_dataset.py. Contains captions of test.
  captions_train.txt      # Creates by split_dataset.py. Contains captions of train.
  captions_validation.txt     # Creates by split_dataset.py. Contains captions of validation.
  Flickr_8k.testImages.txt    # Input file. The list of test images
  Flickr_8k.trainImages.txt   # Input file. The list of train images
  Flickr_8k.validationImages.txt   # Input file. The list of validation images  
models/                   # will contain the models saves of the ImageCaption.py
long_experiment_results/  # Will contain long experiments results.
tensorboard/              # Will contain the tensorboard graphs and images.
```

## Attribution
Parts of the code here are based mainly on two other repositories:
For the lstm part we based our code on https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning, 
but the given code over there was changed significantly for this comparison project. For the transformer 
part, we based our code on https://github.com/senadkurtisi/pytorch-image-captioning although
also in here the code was changed significantly for our purposes.

