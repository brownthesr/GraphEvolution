from torch_geometric.data import Dataset

class SISDataset(Dataset):
    def __init__(self, root, sequences,transform=None):
        super(SISDataset, self).__init__(root,transform=transform)
        self.sequences = sequences

    def len(self):
        return len(self.sequences)

    def get(self, idx):
        return self.sequences[idx]