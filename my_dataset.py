from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, data_q, target):
        self.data_q = data_q
        self.target = target

    def __getitem__(self, index):
        if self.target is None:
            return self.data_q[index]
        return self.data_q[index], self.target[index]

    def __len__(self):
        return len(self.target)


def get_dataloader(dataset, batch_size=4, shuffle=True, drop_last=False):
    dataset_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                drop_last=drop_last)

    return dataset_loader
