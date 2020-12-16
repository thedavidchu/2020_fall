import torch.utils.data as data
import torch

class AdultDataset(data.Dataset):

    def __init__(self, X, y):
        pass
        ######

        # 4.1 YOUR CODE HERE

        ######
        super(AdultDataset, self).__init__()
        self.data = X
        self.label = y
        

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        pass
        ######

        # 4.1 YOUR CODE HERE

        ######
        # if torch.is_tensor(index):
        #     index = index.tolist()

        return self.X[index], self.y[index]
