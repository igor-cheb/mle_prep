import torch
import pandas as pd
import numpy as np
from src.constants import NUM_FEATS, BATCH_SIZE, STEPS_IN_EPOCH, NUM_EPOCHS

class Model(torch.nn.Module):
    """Class for MLP neural network"""

    def __init__(self):
        super().__init__()
        self.all_losses = []
        
        self.nll_loss = torch.nn.NLLLoss()

        self.lin_1 = torch.nn.Linear(NUM_FEATS, 5)
        self.gelu_1 = torch.nn.GELU()

        self.lin_2 = torch.nn.Linear(5, 2)
        self.m = torch.nn.LogSoftmax(dim=1)

        self.optimiser = torch.optim.SGD(self.parameters(), lr=1e-5)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser, step_size=100, gamma=0.7) #0.6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sequential application of NN layers to the passed matrix"""
        x = self.lin_1(x)
        x = self.gelu_1(x)
        x = self.lin_2(x)
        return self.m(x)

    def predict(self, X: np.ndarray) -> torch.Tensor:
        """Function applies the NN layers and converts the ouput to class prediction"""
        return self.forward(torch.tensor(X, dtype=torch.float32)).argmax(dim=1).detach().numpy()
    
    def score(self, X: np.ndarray, y: pd.Series) -> np.ndarray:
        """Function makes a prediction and scores it with negative loglikelihood"""
        return self.nll_loss(self.forward(torch.tensor(X, dtype=torch.float32)), 
                             torch.tensor(y.to_numpy(), dtype=torch.long)).detach().numpy()
    
    def _select_batch(self, X: np.ndarray, y: np.ndarray, 
                      batch_size: int=BATCH_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Function that samples equal number of samples of each label from passed df 
        and returns X and y for train
        """
        ix = np.append(np.random.choice(np.where(y == 0)[0], int(batch_size/2)), 
                       np.random.choice(np.where(y == 1)[0], int(batch_size/2)))
        return torch.tensor(X[ix], dtype=torch.float32), torch.tensor(y[ix], dtype=torch.long)

    def _train_epoch(self, X: np.ndarray, y: pd.Series, steps: int=STEPS_IN_EPOCH):
        """Function that iterates through data and trains the model for 1 epoch"""
        epoch_loss = []
        for _ in range(steps):
            x_batch, y_batch = self._select_batch(X, y.to_numpy()) 
            y_pred = self.forward(x_batch)

            loss = self.nll_loss(y_pred, y_batch)
            epoch_loss.append(loss.detach().numpy())

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

        self.all_losses.append(sum(epoch_loss)/len(epoch_loss))

    def fit(self, X: np.ndarray, y: pd.Series):
        """Complete training of the neural network"""
        for epoch in range(NUM_EPOCHS):
            self._train_epoch(X=X, y=y)
            self.lr_scheduler.step()