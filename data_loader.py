import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary

import pandas as pd

class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab, dictionary):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
        """
        # Read the dataset
        self.data = pd.read_csv(json, header=0,encoding = 'unicode_escape',error_bad_lines=False)
        self.ids = list(range(len(self.data)))
        self.vocab = vocab
        # All the keywords present
        dictionary = pd.read_csv(dictionary, header=0,encoding = 'unicode_escape',error_bad_lines=False)
        self.dictionary = list(dictionary['keys'])

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        data = self.data
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = data.iloc[ann_id]['val']

        array = torch.zeros((len(self.dictionary)))
        for val in self.data.iloc[index]['tk'].split():
            array[self.dictionary.index(val)] = 1

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return array, target

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - array: torch tensor of shape (len(dictionary)).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        array: torch tensor of shape (batch_size, len(dictionary)).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    array, captions = zip(*data)

    # Merge arrays (from tuple of 3D tensor to 4D tensor).
    array = torch.stack(array, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return array, targets, lengths

def get_loader(root, json, vocab, dictionary, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                        json=json,
                        vocab=vocab,
                        dictionary=dictionary)

    # Data loader for COCO dataset
    # This will return (array, captions, lengths) for each iteration.
    # array: a tensor of shape (batch_size, len(dictionary)).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader