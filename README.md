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
- `run_experiments.py` - run long experiments, i.e., sweep through parameters in order to generate
the work graphs.
- `split_dataset.py` - given Flickr8K images and captions, split to train, validation, test sets.
- `show_attention_lstm.py` - show the lstm attention mechanism for the vision part.
- `compute_bleu.py` - this function goes over the test set and compute the BLEU score.
- `utils.py` - small scipts to support other functions like transforming from indices to words, 
from words to indices, etc. 

## Attribution
Parts of the code here are based mainly on two other repositories:
For the lstm part we based our code on https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning, 
but the given code over there was changed significantly for this comparison project. For the transformer 
part, we based our code on https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning although
also in here the code was changed significantly for our purposes.

