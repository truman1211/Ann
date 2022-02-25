import torch


class dataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        length = len(self.data)
        return length

    def __getitem__(self, item):

        x = self.data[item]

        return x


