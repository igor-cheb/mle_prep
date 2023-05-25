import torch
import numpy as np
from src.train_sampler_class import TrainSampler

class RNN(torch.nn.Module):
    def __init__(self, vocab_len: int, embedding_dim: int, 
                 hidden_size: int, word_sampler: TrainSampler):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.emb = torch.nn.Embedding(num_embeddings=vocab_len, embedding_dim=embedding_dim)
        self.lstm = torch.nn.LSTMCell(embedding_dim, self.hidden_size)
        self.lin = torch.nn.Linear(hidden_size, vocab_len)

        self.word_sampler = word_sampler
        self.optimiser = torch.optim.Adam(self.parameters())

    def forward(self, char: torch.Tensor, hidden_state: torch.Tensor, cell_state: torch.Tensor):
        """Applies all the network layers to the passed character encoded as a number"""
        embedding = self.emb(char)
        hidden_state, cell_state = self.lstm(embedding, (hidden_state, cell_state))
        output = self.lin(hidden_state)
        return output, hidden_state, cell_state
    
    def _init_zero_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Initiates dummy hidden and cell states for an lstm cell"""
        zero_hidden_state = torch.zeros(self.hidden_size)
        zero_cell_state = torch.zeros(self.hidden_size)
        return zero_hidden_state, zero_cell_state
    
    def _train_epoch(self, rnn_input: torch.Tensor, 
                     target: torch.Tensor) -> np.ndarray:
        loss = 0
        hidden_state, cell_state = self._init_zero_state()
        for char, tar in zip(rnn_input, target):
            output, hidden_state, cell_state = self.forward(char=char,
                                                            hidden_state=hidden_state,
                                                            cell_state=cell_state)
            loss += torch.nn.functional.cross_entropy(output, tar.long())
        
        epoch_loss = loss / target.shape[0] 
        self.optimiser.zero_grad()
        epoch_loss.backward()
        self.optimiser.step()
        return epoch_loss.detach().numpy()
    
    def fit(self, num_epochs):
        self.train_losses = []
        for epoch in range(num_epochs):
            rnn_input, target = self.word_sampler.sample_train()
            epoch_loss = self._train_epoch(rnn_input, target)
            self.train_losses.append(epoch_loss)
