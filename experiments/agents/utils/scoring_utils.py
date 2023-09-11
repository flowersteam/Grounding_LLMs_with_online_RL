import numpy as np
import torch
import torch.nn.functional as F

def scores_stacking(scores):
    scores_sizes = [len(_score) for _score in scores]
    if len(np.unique(scores_sizes)) > 1:
        max_action_space_size = max(scores_sizes)
        stacked_scores = torch.stack([
            F.pad(
                _score,
                (0, max_action_space_size - len(_score)),
                "constant", -torch.inf)
            for _score in scores])
    else:
        stacked_scores = torch.stack(scores)

    return stacked_scores