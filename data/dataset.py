from torch_geometric.data import Dataset
import torch
class SISDataset(Dataset):
    def __init__(self, root,transform=None):
        super(SISDataset, self).__init__()
        self.sequences = torch.load(root)

    def len(self):
        return len(self.sequences)

    def get(self, idx):
        return self.sequences[idx]