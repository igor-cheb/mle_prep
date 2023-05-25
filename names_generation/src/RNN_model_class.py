import torch
import numpy as np
from src.train_sampler_class import TrainSampler
from src.utilities import vocabulary

class RNN(torch.nn.Module):
    """Class for RNN model"""
    def __init__(self, 
                 embedding_dim: int, 
                 hidden_size: int, 
                 word_sampler: TrainSampler,
                 vocabulary: list=vocabulary):
        super().__init__()
        self.vocabulary = vocabulary

        self.hidden_size = hidden_size
        self.emb = torch.nn.Embedding(num_embeddings=len(self.vocabulary), embedding_dim=embedding_dim)
        self.lstm = torch.nn.LSTMCell(embedding_dim, self.hidden_size)
        self.lin = torch.nn.Linear(hidden_size, len(self.vocabulary))
        self.softmax = torch.nn.Softmax(dim=0)

        self.word_sampler = word_sampler
        self.optimiser = torch.optim.Adam(self.parameters())

    def forward(self, 
                char: torch.Tensor, 
                hidden_state: torch.Tensor, 
                cell_state: torch.Tensor):
        """Applies all the network layers to the passed character encoded as a number"""
        embedding = self.emb(char)
        hidden_state, cell_state = self.lstm(embedding, (hidden_state, cell_state))
        output = self.lin(hidden_state)
        return output, hidden_state, cell_state
    
    def _init_zero_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Initiates dummy hidden and cell states for an rnn cell"""
        zero_hidden_state = torch.zeros(self.hidden_size)
        zero_cell_state = torch.zeros(self.hidden_size)
        return zero_hidden_state, zero_cell_state
    
    def _train_epoch(self, 
                     rnn_input: torch.Tensor, 
                     target: torch.Tensor) -> np.ndarray:
        """Trains 1 epoch based on passed 1 training sample"""
        loss = 0
        hidden_state, cell_state = self._init_zero_state()
        for char, tar in zip(rnn_input, target):
            output, hidden_state, cell_state = self.forward(char=char,
                                                            hidden_state=hidden_state,
                                                            cell_state=cell_state)
            loss += torch.nn.functional.cross_entropy(output, tar.long())
        
        epoch_loss = loss / target.shape[0] # averaging loss for the epoch
        self.optimiser.zero_grad()
        epoch_loss.backward()
        self.optimiser.step()
        return epoch_loss.detach().numpy()  # outputting epoch loss for logging
    
    def fit(self, num_epochs: int):
        """Full training of the rnn for the passed number of epochs"""
        self.train_losses = [] # keeping the list of losses for logging
        for epoch in range(num_epochs):
            rnn_input, target = self.word_sampler.sample_train()
            epoch_loss = self._train_epoch(rnn_input, target)
            self.train_losses.append(epoch_loss)

    def predict(self, names_num: int) -> list:
        """Generates passed number of names"""
        names = []
        for i in range(names_num):
            name = []
            char_ix = torch.tensor(1).long()
            hidden_state, cell_state = self._init_zero_state()
            cnt=0
            while char_ix != torch.tensor(0).long():  # each name is generated until '.' symbol in rnn output
                output, hidden_state, cell_state = self.forward(char=char_ix,
                                                                hidden_state=hidden_state,
                                                                cell_state=cell_state)
                probs = self.softmax(output)
                char_ix = torch.multinomial(probs, 1).squeeze()
                letter = self.vocabulary[char_ix]
                name.append(letter)
                cnt+=1
                if cnt>10: break
            names.append(''.join(name)[:-1])
        return names