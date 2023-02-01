"""
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class SubtaskPredictionModel(nn.Module):

    def __init__(self, obs_space, arch="siamese", lang_model="gru", instr_dim=128):
        super().__init__()
        
        self.arch = arch
        self.lang_model = lang_model
        self.instr_dim = instr_dim

        if self.lang_model in ['gru']:
            self.word_embedding = nn.Embedding(obs_space["instr"], self.instr_dim)
            gru_dim = self.instr_dim
            self.instr_rnn = nn.GRU(
                self.instr_dim, gru_dim, batch_first=True,
                bidirectional=False
            )

        self.fc1 = nn.Linear(self.instr_dim, self.instr_dim // 2)
        self.fc2 = nn.Linear(self.instr_dim, self.instr_dim // 2)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(self.instr_dim, self.instr_dim // 2)
        self.fc4 = nn.Linear(self.instr_dim // 2, 1)
        
        self.sigmoid = nn.Sigmoid()
        
        self.apply(initialize_parameters)
    
    def forward(self, missions, subtasks):
        if self.arch == "siamese":
            mission_embedding = self._get_instr_embedding(missions)
            subtask_embedding = self._get_instr_embedding(subtasks)

            mission_embedding = self.dropout1(self.fc1(mission_embedding))
            subtask_embedding = self.dropout2(self.fc2(subtask_embedding))

            both_embeddings = torch.cat((mission_embedding, subtask_embedding), dim=-1)
            both_embeddings = self.fc3(both_embeddings)

            logits = self.fc4(both_embeddings)
            preds = self.sigmoid(logits).squeeze(-1)
        elif self.arch == "siamese-l1":
            mission_embedding = self._get_instr_embedding(missions)
            subtask_embedding = self._get_instr_embedding(subtasks)
            
            mission_embedding = self.fc1(mission_embedding)
            subtask_embedding = self.fc2(subtask_embedding)

            dist = torch.norm(mission_embedding - subtask_embedding, p=1, dim=1)
            
            preds = torch.exp(-dist)
            
        return preds

    def _get_instr_embedding(self, instr):
        lengths = (instr != 0).sum(1).long()
        if self.lang_model == 'gru':
            out, _ = self.instr_rnn(self.word_embedding(instr))
            hidden = out[range(len(lengths)), lengths-1, :]
            return hidden