import os
import string
import numpy as np
from PIL import Image
from pickle import dump, load
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
import torchvision.datasets
import torch.utils.data
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# https://data-flair.training/blogs/python-based-project-image-caption-generator-cnn/

# calculate maximum length of descriptions
def max_length(descriptions):
    desc_list = dict_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)


# load doc file
def read_file(filename):
    # Opening the file as read only
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


# get all images captions
def all_img_captions(filename):
    file = read_file(filename)
    captions = file.split('\n')
    descriptions = {}
    for caption in captions[:-1]:
        img, caption = caption.split('\t')
        if img[:-2] not in descriptions:
            descriptions[img[:-2]] = [caption]
        else:
            descriptions[img[:-2]].append(caption)
    return descriptions


# clean txt: convert to lowercase. remove punctuations and words containing numbers.
def clean_text(captions):
    table = str.maketrans('', '', string.punctuation)
    for img, caps in captions.items():
        for i, img_caption in enumerate(caps):
            img_caption.replace("-", " ")
            desc = img_caption.split()
            # lowercase
            desc = [word.lower() for word in desc]
            # remove punctuation
            desc = [word.translate(table) for word in desc]
            # remove hanging 's and a
            desc = [word for word in desc if (len(word) > 1)]
            # remove tokens with numbers
            desc = [word for word in desc if (word.isalpha())]
            # convert back to string
            img_caption = ' '.join(desc)
            captions[img][i] = img_caption
    return captions


# build vocabulary of all unique words
def build_vocabulary(descriptions):
    vocab = set()
    for key in descriptions.keys():
        [vocab.update(d.split()) for d in descriptions[key]]
    return vocab


# save descriptions
def write_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + '\t' + desc)
    data = "\n".join(lines)
    file = open(filename, "w")
    file.write(data)
    file.close()


# transfer learning
# use a pre-trained model (xception) that has been already trained on large dataset (imagenet) and extract the features
# (2048 features). remove the last classification layer and get the features vector.
# note: xception takes 299*299*3 image size as input
def extract_features(model, directory, flickr8k_images_name, image_size=(299, 299), batch_size=2):
    #drip # load a pretrained model
    #drip model = timm.create_model('xception', pretrained=True, )
    #drip model.eval()

    #drip # remove the last layer
    #drip model = torch.nn.Sequential(*(list(model.children())[:-1]))

    # load and preprocess image
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()])

    #drip flickr8k_images_name = []
    flickr8k_images_features = []
    flickr8k_images = []

    #drip for img in os.listdir(directory):
    for img in flickr8k_images_name:
        filename = directory + "/" + img
        image = Image.open(filename).convert('RGB')
        tensor = transform(image).unsqueeze(0)  # transform and add batch dimension

        # model prediction
        with torch.no_grad():
            score = model(tensor)
            features = torch.nn.functional.softmax(score[0], dim=0)

        #drip flickr8k_images_name.append(img)
        flickr8k_images_features.append(features)
        flickr8k_images.append(tensor)
    #drip return flickr8k_images_name, flickr8k_images_features, flickr8k_images
    return flickr8k_images_features, flickr8k_images



# load text file. return the list of image names.
def load_images_name(filename):
    file = read_file(filename)
    images_name = file.split("\n")[:-1]
    return images_name

# create a dictionary that contains captions for each photo from the list of photos.
def load_clean_descriptions(filename, photos):
    file = read_file(filename)
    descriptions = {}
    for line in file.split("\n"):
        words = line.split()
        if len(words)<1:
            continue
        image, image_caption = words[0], words[1:]
        if image in photos:
            if image not in descriptions:
                descriptions[image] = []
            desc = '<start> ' + " ".join(image_caption) + ' <end>'
            descriptions[image].append(desc)
    return descriptions

# returns the dictionary for image names and their feature vector (extracted from xception model)
def load_features(img_names):
    all_features = load(open("features.p", "rb"))
    features = {k: all_features[k] for k in img_names}
    return features

# convert dictionary to list
def dict_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

# creating vocabulary
# this will vectorise text corpus. # each integer will represent token in dictionary
def create_vocab(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, desc_list), specials=['<unk>', '<start>', '<end>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab


# pointers
root_dir = "C:/deep_learning_046211/ImageCaptionGenerator"
dataset_text_dir = root_dir + "/Flickr8k_text"
dataset_images_dir = root_dir + "/drip_Flickr8k_Dataset"

# load a pretrained model
model = timm.create_model('xception', pretrained=True, )
model.eval()
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # remove the last layer

# load captions, map into img/description dictionary, clean captions and build vocabulary
descriptions = all_img_captions(dataset_text_dir + "/Flickr8k.token.txt")
clean_descriptions = clean_text(descriptions)
write_descriptions(clean_descriptions, "descriptions.txt")
vocabulary = build_vocabulary(clean_descriptions)
max_length = max_length(descriptions)

#drip # extract features vector for all images
#drip images_name, images_features, images = extract_features(dataset_images_dir)
#drip dump(images_features, open("features.p", "wb"))

#drip # extract features vector for all images
#drip images_name, images_features, images = extract_features(dataset_images_dir)
#drip dump(images_features, open("features.p", "wb"))


'''
# dbg
# display first images from the train set
transform2pil = transforms.ToPILImage()
fig, ax = plt.subplots(1, len(images), figsize=(12, 2.5))
for i in range(len(images)):
    ax[i].imshow(transform2pil(images[i].squeeze()))
    ax[i].set_title(images_name[i])
    ax[i].axis('off')
plt.show(block=True)
'''

# load dataset for training

# extract features vector for train data
train_images_name = load_images_name(dataset_text_dir + "/drip_Flickr_8k.trainImages.txt")
train_features, train_images = extract_features(model, dataset_images_dir, train_images_name)
train_descriptions = load_clean_descriptions("descriptions.txt", train_images_name)
dump(train_features, open("train_features.p", "wb"))

# tokenizing the vocabulary

# give each word an index, and store that into tokenizer.p pickle file
vocab = create_vocab(train_descriptions)
dump(vocab, open('tokenizer.p', 'wb'))

