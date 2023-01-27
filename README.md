# Image Captioning
By: Dana Rip

## Introduction
In this git we introduce a comparison for image captioning: we implement a transformer and a lstm
with attention. Both architectures are based on encoder of ResNet50. 

## Files
- Under `source/tst` there are several scripts that run the pipline. These are the entry point scripts for operating the comparison.
  - `source/tst/tst_lstm_short.py` - runs a relatively short test to examine the lstm architecture
  - `source/tst/tst_transformer_short.py` - runs a relatively short test to examine the transformer architecture
- `ImageCaptioning.py` - This is the main file to run the pipeline itself. If receives a long list of parameters in the `single_run` function.