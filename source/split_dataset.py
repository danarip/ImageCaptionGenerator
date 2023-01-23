"""
This script takes the whole 8k Flickr pictures and split them to train, validation, test sets.
"""
import os
import shutil

from root import cwd


def file2set(fullpath):
    files = readfile2list(fullpath)
    return set(files)


def readfile2list(fullpath):
    f = open(fullpath, "r")
    s = f.read().strip()
    f.close()
    files = [x.strip() for x in s.split("\n")]
    return files


def savestring2file(s, fullpath):
    text_file = open(fullpath, "w")
    n = text_file.write(s)
    text_file.close()


def split_dataset():
    # Input
    train_file_fullpath = f"{cwd}/data/flickr8k/Flickr_8k.trainImages.txt"
    validation_file_fullpath = f"{cwd}/data/flickr8k/Flickr_8k.validationImages.txt"
    test_file_fullpath = f"{cwd}/data/flickr8k/Flickr_8k.testImages.txt"
    all_images_folder = f"{cwd}/data/flickr8k/Images"
    captions_file = f"{cwd}/data/flickr8k/captions.txt"

    # Outputs
    target_train = f"{cwd}/data/flickr8k/Flickr8kTrainImages/"
    target_validation = f"{cwd}/data/flickr8k/Flickr8kValidationImages/"
    target_test = f"{cwd}/data/flickr8k/Flickr8kTestImages/"
    captions_train_file = f"{cwd}/data/flickr8k/captions_train.txt"
    captions_validation_file = f"{cwd}/data/flickr8k/captions_validation.txt"
    captions_test_file = f"{cwd}/data/flickr8k/captions_test.txt"

    # Script...
    captions_lines = readfile2list(captions_file)
    captions_dict = dict()
    for line in captions_lines[1:]:
        terms = line.split(",")
        if terms[0] not in captions_dict:
            captions_dict[terms[0]] = list()
        captions_dict[terms[0]].append(line)

    train_set = file2set(train_file_fullpath)
    print(f"train set {len(train_set)}")
    validation_set = file2set(validation_file_fullpath)
    print(f"validation set {len(validation_set)}")
    test_set = file2set(test_file_fullpath)
    print(f"test set {len(test_set)}")

    captions_train = [captions_lines[0]]
    captions_validation = [captions_lines[0]]
    captions_test = [captions_lines[0]]

    for filename in os.listdir(all_images_folder):
        if not filename.endswith("jpg"):
            continue
        if filename in train_set:
            shutil.copy(f"{all_images_folder}/{filename}", f"{target_train}")
            captions_train.append("\n".join(captions_dict[filename]))
        if filename in validation_set:
            shutil.copy(f"{all_images_folder}/{filename}", f"{target_validation}")
            captions_validation.append("\n".join(captions_dict[filename]))
        if filename in test_set:
            shutil.copy(f"{all_images_folder}/{filename}", f"{target_test}")
            captions_test.append("\n".join(captions_dict[filename]))

    savestring2file("\n".join(captions_train), captions_train_file)
    savestring2file("\n".join(captions_validation), captions_validation_file)
    savestring2file("\n".join(captions_test), captions_test_file)


if __name__ == "__main__":
    split_dataset()
