## This Repo is a simple implementation of how to build custom dataset in pytorch:
1. Create a class that inherits from torch.utils.data.Dataset
2. Implement the following methods:
    - __init__
    - __len__ (returns the length of the dataset)
    - __getitem__ (returns a img and label from the dataset)
3. Create a dataset that has an annotation file (e.g. csv, json, txt) with the following format:
    - img_path, label
    - img_path, label
    - img_path, label
    - ...\
4. See how to further reduce boilerplate code by using pytorch lightning's DataModule:https://lightning.ai/docs/pytorch/stable/data/datamodule.html

## Example
See `CustomDataset.py` for the implementation of the dataset class and also for an example of how to use the dataset.
