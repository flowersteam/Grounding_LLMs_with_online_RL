"""
General class for handling instructions provided by demonstrations.
"""

import pickle
import os
import numpy as np
from babyai import utils
# from transformers import BertTokenizer, BertModel
from torch.nn import CosineSimilarity
import torch
import logging
logging.getLogger("transformers").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class InstructionHandler:
    
    def __init__(self, demos=None, missions=None, load_bert=False, save_path=None):
        self.missions = []
        if missions is not None:
            self.missions = missions
        else:
            self.missions = [demo[1] for demo in demos]
        self.missions = np.array(sorted(list(set(self.missions))))
        if load_bert:
            # Deprecated
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.model = BertModel.from_pretrained("bert-base-uncased")
            self.cos = CosineSimilarity(dim=1, eps=1e-6)
            self.build_compounds(os.path.join(save_path, "compound_embeddings.pkl"))
    
    def D_l_size(self):
        return len(self.missions)

    def save(self, item, path):
        utils.create_folders_if_necessary(path)
        with open(path, 'wb') as f:
            pickle.dump(item, f)
    
    def load(self, path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

    def build_compounds(self, path):
        # Deprecated
        logger.info("loading BERT embeddings")
        self.compound_missions = []
        for i in range(len(self.missions)):
            for j in range(len(self.missions)):
                compound = f"{self.missions[i]} and {self.missions[j]}"
                self.compound_missions.append(compound)
        self.compound_missions = np.array(self.compound_missions)
        self.compound_embeddings = self.load(path)
        if self.compound_embeddings is None:
            compound_encodings = self.tokenizer.batch_encode_plus(self.compound_missions, pad_to_max_length=True, return_tensors="pt")
            self.compound_embeddings = self.model(compound_encodings['input_ids'])[0][:, 0, :]
            self.save(self.compound_embeddings, path)
        self.compound_embeddings_norm = self.compound_embeddings/self.compound_embeddings.norm(dim=1)[:, None]
        logger.info("loaded BERT embeddings")

    def get_instruction(self, index):
        return self.missions[index]
    
    def get_index(self, instruction):
        i, = np.where(self.missions == instruction)
        if len(i) > 0:
            return i[0]
        else:
            return None

    def get_random_instruction(self):
        return np.random.choice(self.missions)

    def get_oracle_stack(self, mission, strict=True, unlock=False):
        """ Get the oracle decomposition, only for validation purposes,
            using handcrafted rules.
        """
        words = mission.split(" ")
        if strict:
            if "then" in mission or "after you" in mission:
                if "then" in mission:
                    split = mission.split(", then ")
                elif "after you" in mission:
                    split = mission.split(" after you ")
                subs = []
                for part in split:
                    words = part.split(" ")
                    if part.startswith("pick up "):
                        assert len(words) == 5
                        sub1 = "go to {}".format(" ".join(words[2:5]))
                        subs.extend([sub1])
                    elif part.startswith("open "):
                        assert len(words) == 4
                        sub1 = "go to the {}".format(" ".join(words[2:4]))
                        subs.extend([sub1])
                    elif part.startswith("put "):
                        assert len(words) == 9
                        sub1 = "go to {}".format(" ".join(words[1:4]))
                        subs.extend([sub1])
                return list(set(subs))
            elif mission.startswith("pick up "):
                # pick up the red ball
                assert len(words) == 5
                sub1 = "go to {}".format(" ".join(words[2:5]))
                return [sub1]
            elif mission.startswith("open "):
                # open the red door
                if len(words) == 4:
                    sub1 = "go to the {}".format(" ".join(words[2:4]))
                    return [sub1]
                # open the red door and pick up a green ball
                elif len(words) == 10:
                    sub1 = "go to the {}".format(" ".join(words[2:4]))
                    sub2 = "open the {}".format(" ".join(words[2:4]))
                    sub3 = "go to {}".format(" ".join(words[7:10]))
                    sub4 = "go to the {}".format(" ".join(words[8:10]))
                    return [sub1, sub2, sub3, sub4]
            elif mission.startswith("put "):
                # put the red ball next to the green box
                assert len(words) == 9
                sub1 = "go to {}".format(" ".join(words[1:4]))
                return [sub1]
            elif mission.startswith("move "):
                # move the red block next to the green block
                assert len(words) == 9
                sub1 = "go to {}".format(" ".join(words[1:4]))
                return [sub1]
            else:
                raise NotImplementedError
        else:
            if "then" in mission or "after you" in mission:
                if "then" in mission:
                    split = mission.split(", then ")
                elif "after you" in mission:
                    split = mission.split(" after you ")
                subs = []
                for part in split:
                    words = part.split(" ")
                    if part.startswith("pick up "):
                        assert len(words) == 5
                        sub1 = "go to a {}".format(" ".join(words[3:5]))
                        sub2 = "go to the {}".format(" ".join(words[3:5]))
                        subs.extend([sub1, sub2])
                    elif part.startswith("open "):
                        assert len(words) == 4
                        sub1 = "go to the {}".format(" ".join(words[2:4]))
                        subs.extend([sub1])
                    elif part.startswith("put "):
                        assert len(words) == 9
                        sub1 = "go to a {}".format(" ".join(words[2:4]))
                        sub2 = "go to the {}".format(" ".join(words[2:4]))
                        subs.extend([sub1, sub2])
                return list(set(subs))
            elif mission.startswith("pick up "):
                # pick up the red ball
                assert len(words) == 5
                sub1 = "go to a {}".format(" ".join(words[3:5]))
                sub2 = "go to the {}".format(" ".join(words[3:5]))
                return [sub1, sub2]
            elif mission.startswith("open "):
                # open the red door
                if len(words) == 4:
                    if unlock:
                        sub1 = "pick up a {} key".format(words[2])
                        sub2 = "pick up the {} key".format(words[2])
                        return [sub1, sub2]
                    else:
                        sub1 = "go to the {}".format(" ".join(words[2:4]))
                        return [sub1]
                # open the red door and pick up a green ball
                elif len(words) == 10:
                    sub1 = "go to the {}".format(" ".join(words[2:4]))
                    sub2 = "open the {}".format(" ".join(words[2:4]))
                    sub3 = "go to a {}".format(" ".join(words[8:10]))
                    sub4 = "go to the {}".format(" ".join(words[8:10]))
                    return [sub1, sub2, sub3, sub4]
            elif mission.startswith("put "):
                # put the red ball next to the green box
                assert len(words) == 9
                sub1 = "go to a {}".format(" ".join(words[2:4]))
                sub2 = "go to the {}".format(" ".join(words[2:4]))
                return [sub1, sub2]
            elif mission.startswith("move "):
                # move the red block next to the green block
                assert len(words) == 9
                sub1 = "go to {}".format(" ".join(words[1:4]))
                return [sub1]
            else:
                raise NotImplementedError


    def get_projection_stack(self, mission):
        # Deprecated
        mission_encoding = self.tokenizer.batch_encode_plus([mission], pad_to_max_length=True, return_tensors="pt")
        mission_embedding = self.model(mission_encoding['input_ids'])[0][:, 0, :]
        similarities = self.cos(self.compound_embeddings, mission_embedding)
        best = self.compound_missions[similarities.argmax()]
        return self.get_oracle_stack(best, 'BabyAI-GoToLocal2-v0')


    def get_projection_stacks(self, missions):
        # Deprecated
        mission_encodings = self.tokenizer.batch_encode_plus(missions, pad_to_max_length=True, return_tensors="pt")
        mission_embeddings = self.model(mission_encodings['input_ids'])[0][:, 0, :]
        mission_embeddings_norm = mission_embeddings / mission_embeddings.norm(dim=1)[:, None]
        similarities = torch.mm(self.compound_embeddings_norm, mission_embeddings_norm.T)
        bests = self.compound_missions[similarities.argmax(dim=0)]
        return [self.get_oracle_stack(best, 'BabyAI-GoToLocal2-v0') for best in bests]