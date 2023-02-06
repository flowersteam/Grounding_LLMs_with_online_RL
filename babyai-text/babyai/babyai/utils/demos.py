import os
import pickle
import numpy as np

from .. import utils
from gym_minigrid.minigrid import MiniGridEnv
import blosc


def get_demos_path(demos=None, env=None, origin=None, valid=False):
    valid_suff = '_valid' if valid else ''
    demos_path = (demos + valid_suff
                  if demos
                  else env + "_" + origin + valid_suff) + '.pkl'
    return os.path.join(utils.storage_dir(), 'demos', demos_path)

def get_demos_QG_path(demos=None, env=None, origin=None, valid=False):
    valid_suff = '_valid' if valid else ''
    demos_path = (demos + valid_suff
                  if demos
                  else env + "_" + origin + valid_suff) + '_QG.pkl'
    return os.path.join(utils.storage_dir(), 'demos', demos_path)

def get_demos_QG_voc_path(demos=None, env=None, origin=None, valid=False):
    valid_suff = '_valid' if valid else ''
    demos_path = (demos + valid_suff
                  if demos
                  else env + "_" + origin + valid_suff) + '_QG_vocab.pkl'
    return os.path.join(utils.storage_dir(), 'demos', demos_path)

def load_demos(path, raise_not_found=True):
    try:
        return pickle.load(open(path, "rb"))
    except FileNotFoundError:
        if raise_not_found:
            raise FileNotFoundError("No demos found at {}".format(path))
        else:
            return []

def load_voc(path, raise_not_found=True):
    try:
        return pickle.load(open(path, "rb"))
    except FileNotFoundError:
        if raise_not_found:
            raise FileNotFoundError("No vo found at {}".format(path))
        else:
            return []

def save_demos(demos, path):
    utils.create_folders_if_necessary(path)
    pickle.dump(demos, open(path, "wb"), protocol=4)


def synthesize_demos(demos):
    print('{} demonstrations saved'.format(len(demos)))
    num_frames_per_episode = [len(demo[2]) for demo in demos]
    if len(demos) > 0:
        print('Demo num frames: {}'.format(num_frames_per_episode))


def transform_demos(demos):
    '''
    takes as input a list of demonstrations in the format generated with `make_agent_demos` or `make_human_demos`
    i.e. each demo is a tuple (mission, blosc.pack_array(np.array(images)), directions, actions)
    returns demos as a list of lists. Each demo is a list of (obs, action, done) tuples
    '''
    new_demos = []
    for demo in demos:
        new_demo = []

        mission = demo[0]
        all_images = demo[1]
        directions = demo[2]
        actions = demo[3]

        all_images = blosc.unpack_array(all_images)
        n_observations = all_images.shape[0]
        assert len(directions) == len(actions) == n_observations, "error transforming demos"
        for i in range(n_observations):
            obs = {'image': all_images[i],
                   'direction': directions[i],
                   'mission': mission}
            action = actions[i]
            done = i == n_observations - 1
            new_demo.append((obs, action, done))
        new_demos.append(new_demo)
    return new_demos


def transform_demos_imitation(demos, include_done=False):
    new_demos = []
    for demo in demos:
        new_demo = []

        mission = demo[1]
        all_images = demo[2]
        directions = demo[3]
        actions = demo[4]

        all_images = blosc.unpack_array(all_images)
        n_observations = all_images.shape[0]
        if not include_done:
            n_observations -= 1
        else:
            raise NotImplementedError()
        for i in range(n_observations):
            obs = {'image': all_images[i],
                   'direction': directions[i],
                   'mission': mission}
            action = actions[i]
            done = i == n_observations - 1
            new_demo.append((obs, action, done))
        new_demos.append(new_demo)
    return new_demos


def transform_demos_imitation_done_classifier(demos, oversample=20):
    new_demos = []
    all_missions = set([demo[1] for demo in demos])
    for i, demo in enumerate(demos):
        mission = demo[1]
        images = demo[2]
        images = blosc.unpack_array(images)
        for i in range(oversample):
            pos_image = images[1]
            obs = {'image': pos_image,
                   'mission': mission}
            action = 1
            done = True
            new_demo = [(obs, action, done)]
            new_demos.append(new_demo)
        for i in range(oversample // 2):
            neg_image = images[0]
            obs = {'image': neg_image,
                   'mission': mission}
            action = 0
            done = True
            new_demo = [(obs, action, done)]
            new_demos.append(new_demo)
        replace = False
        neg_missions_pop = list(all_missions - {mission})
        if oversample // 2 > len(neg_missions_pop):
            replace = True
        neg_missions = np.random.choice(neg_missions_pop, oversample // 2, replace=replace)
        for i in range(oversample // 2):
            obs = {'image': pos_image,
                   'mission': neg_missions[i]}
            action = 0
            done = True
            new_demo = [(obs, action, done)]
            new_demos.append(new_demo)
    return new_demos


def transform_demos_imitation_done_classifier_cont(demos, oversample=20):
    new_demos = []
    all_missions = set([demo[1] for demo in demos])
    for i, demo in enumerate(demos):
        mission = demo[1]
        obss = demo[2]
        for i in range(oversample):
            pos_obs = obss[1]
            action = 1
            done = True
            new_demo = [(pos_obs, action, done)]
            new_demos.append(new_demo)
        for i in range(oversample // 2):
            neg_obs = obss[0]
            action = 0
            done = True
            new_demo = [(neg_obs, action, done)]
            new_demos.append(new_demo)
        replace = False
        neg_missions_pop = list(all_missions - {mission})
        if oversample // 2 > len(neg_missions_pop):
            replace = True
        neg_missions = np.random.choice(neg_missions_pop, oversample // 2, replace=replace)
        for i in range(oversample // 2):
            obs = {'pm_position': pos_obs['pm_position'],
                   'objects': pos_obs['objects'],
                   'mission': neg_missions[i]}
            action = 0
            done = True
            new_demo = [(obs, action, done)]
            new_demos.append(new_demo)
    return new_demos


def transform_demos_learn(demos):
    missions = []
    action_freqs = []
    labels = []

    new_demos = []
    for i, demo in enumerate(demos):
        task_id = demo[0]
        mission = demo[1]
        actions = demo[4]
        if len(actions) < 5:
            continue
        while True:
            r, s = np.random.choice(len(actions)), np.random.choice(len(actions)+1)
            r, s = min(r, s), max(r, s)
            if s - r >= 5:
                break
        action_freq = np.bincount(actions[r:s] + [MiniGridEnv.Actions.done])
        action_freq[-1] -= 1
        action_freq = action_freq / np.sum(action_freq)
        
        missions.append(mission)
        action_freqs.append(action_freq)
        labels.append(1)

        if np.random.random() < 0.5:
            while True:
                mission_alt = demos[np.random.choice(len(demos))][1]
                if mission_alt != mission:
                    break
            missions.append(mission_alt)
            action_freqs.append(action_freq)
            labels.append(0)
        else:
            action_freq_alt = np.random.random(len(MiniGridEnv.Actions))
            action_freq_alt = action_freq_alt / np.sum(action_freq_alt)
            missions.append(mission)
            action_freqs.append(action_freq_alt)
            labels.append(0)
    return list(zip(missions, action_freqs, labels))


def transform_demos_subtasks_cross(demos, instr_handler, n=20):

    examples = []

    for i, demo in enumerate(demos):
        mission = demo[0]
        all_subtasks = demo[1]
        all_subtasks_flat = [s for ts in all_subtasks for s in ts[1]]
        pos_subtasks = set(all_subtasks_flat)
        neg_subtasks = set(range(instr_handler.D_l_size())) - pos_subtasks

        if len(pos_subtasks) > 0:
            for s in np.random.choice(list(pos_subtasks), n):
                examples.append([mission, instr_handler.get_instruction(s), 1])
        
        if len(neg_subtasks) > 0:
            for s in np.random.choice(list(neg_subtasks), n):
                examples.append([mission, instr_handler.get_instruction(s), 0])

    return examples


def transform_demos_subtasks_cross_ones(demos, instr_handler):

    examples = []

    for i, demo in enumerate(demos):
        mission = demo[0]
        all_subtasks = set(range(instr_handler.D_l_size()))
        for s in all_subtasks:
            examples.append([mission, instr_handler.get_instruction(s), 1])
    
    return examples
