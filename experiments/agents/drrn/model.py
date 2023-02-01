import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from .utils.pad_sequences import pad_sequences
from .utils.memory import State

from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.state.device

class DRRN(torch.nn.Module):
    """
        Deep Reinforcement Relevance Network - He et al. '16

    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(DRRN, self).__init__()
        self.embedding    = nn.Embedding(vocab_size, embedding_dim)
        self.obs_encoder  = nn.GRU(embedding_dim, hidden_dim)
        # self.look_encoder = nn.GRU(embedding_dim, hidden_dim)
        # self.inv_encoder  = nn.GRU(embedding_dim, hidden_dim)
        self.act_encoder  = nn.GRU(embedding_dim, hidden_dim)
        self.hidden       = nn.Linear(2*hidden_dim, hidden_dim)
        self.act_scorer   = nn.Linear(hidden_dim, 1)


    def packed_rnn(self, x, rnn):
        """ Runs the provided rnn on the input x. Takes care of packing/unpacking.

            x: list of unpadded input sequences
            Returns a tensor of size: len(x) x hidden_dim
        """
        lengths = torch.tensor([len(n) for n in x], dtype=torch.long, device=device)
        # Sort this batch in descending order by seq length
        lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)
        padded_x = pad_sequences(x)
        x_tt = torch.from_numpy(padded_x).type(torch.long).to(device)
        x_tt = x_tt.index_select(0, idx_sort)
        # Run the embedding layer
        embed = self.embedding(x_tt).permute(1,0,2) # Time x Batch x EncDim
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embed, lengths.cpu())
        # Run the RNN
        out, _ = rnn(packed)
        # Unpack
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        # Get the last step of each sequence
        idx = (lengths-1).view(-1,1).expand(len(lengths), out.size(2)).unsqueeze(0)
        out = out.gather(0, idx).squeeze(0)
        # Unsort
        out = out.index_select(0, idx_unsort)
        return out


    def forward(self, state_batch, act_batch):
        """
            Batched forward pass.
            obs_id_batch: iterable of unpadded sequence ids
            act_batch: iterable of lists of unpadded admissible command ids

            Returns a tuple of tensors containing q-values for each item in the batch
        """
        # Zip the state_batch into an easy access format
        state = State(*zip(*state_batch))
        # This is number of admissible commands in each element of the batch
        act_sizes = [len(a) for a in act_batch]
        # Combine next actions into one long list
        act_batch = list(itertools.chain.from_iterable(act_batch))
        act_out = self.packed_rnn(act_batch, self.act_encoder)
        # Encode the various aspects of the state
        obs_out = self.packed_rnn(state.obs, self.obs_encoder)
        # look_out = self.packed_rnn(state.description, self.look_encoder)
        # inv_out = self.packed_rnn(state.inventory, self.inv_encoder)
        # state_out = torch.cat((obs_out, look_out, inv_out), dim=1)
        state_out = obs_out
        # Expand the state to match the batches of actions
        state_out = torch.cat([state_out[i].repeat(j,1) for i,j in enumerate(act_sizes)], dim=0)
        z = torch.cat((state_out, act_out), dim=1) # Concat along hidden_dim
        z = F.relu(self.hidden(z))
        act_values = self.act_scorer(z).squeeze(-1)
        # Split up the q-values by batch
        return act_values.split(act_sizes)