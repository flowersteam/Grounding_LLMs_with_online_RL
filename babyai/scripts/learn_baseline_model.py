"""
A reimplmentation of the LEARN model (Goyal et al., 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.1)


class LEARNBaselineModel(nn.Module):

    def __init__(self, obs_space, arch="learn", lang_model="gru", instr_dim=128, action_dim=128, hidden_dim=128, dropout=0):
        super().__init__()
        
        self.arch = arch
        self.lang_model = lang_model
        self.instr_dim = instr_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        if self.lang_model in ['gru']:
            self.word_embedding = nn.Embedding(obs_space["instr"], self.instr_dim)
            gru_dim = self.instr_dim
            self.instr_rnn = nn.GRU(
                self.instr_dim, gru_dim, num_layers=2,
                batch_first=True,
                bidirectional=False
            )

        action_input_sizes = [obs_space['num_actions'], self.hidden_dim, self.hidden_dim]
        action_output_sizes = [self.hidden_dim, self.hidden_dim, self.action_dim]
        self.action_mlp = self.mlp(action_input_sizes, action_output_sizes, dropout=dropout)

        cls_input_sizes = [self.action_dim + self.instr_dim, self.hidden_dim, self.hidden_dim]
        cls_output_sizes = [self.hidden_dim, self.hidden_dim, 2]
        self.classification_mlp = self.mlp(cls_input_sizes, cls_output_sizes, dropout=dropout)
        
        self.apply(initialize_parameters)
    
    def mlp(self, in_dim, out_dim, dropout=0, n_layers=3):
        layers = []
        for l in range(n_layers - 1):
            layers.extend([nn.Linear(in_dim[l], out_dim[l]),
                           nn.ReLU(),
                           nn.BatchNorm1d(out_dim[l]),
                           nn.Dropout(dropout)])
        layers.extend([nn.Linear(in_dim[-1], out_dim[-1])])
        return nn.Sequential(*layers)    

    def forward(self, missions, action_frequencies):
        action_enc = self.action_mlp(action_frequencies)
        text_enc = self._get_instr_embedding(missions)
        action_text = torch.cat((action_enc, text_enc,), dim=-1)
        
        logits = self.classification_mlp(action_text)

        preds = torch.argmax(logits, axis=-1)
        return preds, logits

    def _get_instr_embedding(self, instr):
        lengths = (instr != 0).sum(1).long()
        if self.lang_model == 'gru':
            out, _ = self.instr_rnn(self.word_embedding(instr))
            hidden = out[range(len(lengths)), lengths-1, :]
            return hidden