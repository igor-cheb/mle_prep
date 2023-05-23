import torch

class Model(torch.nn.Module):
    
    def __init__(self, all_cols):
        super().__init__()
        self.all_cols = all_cols
        
        self.nll_loss = torch.nn.NLLLoss()

        self.lin_1 = torch.nn.Linear(len(self.all_cols), 128)
        self.gelu_1 = torch.nn.GELU()

        self.lin_2 = torch.nn.Linear(128, 2)
        self.m = torch.nn.LogSoftmax(dim=1)

        self.optimiser = torch.optim.Adam(self.parameters(), lr=1e-5)
        
    def forward(self, x):
        x = self.lin_1(x)
        x = self.gelu_1(x)
        x = self.lin_2(x)
        return self.m(x)

    def fit(self, X, y):
        y_pred = self.forward(X)
        loss = self.nll_loss(y_pred, y)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss