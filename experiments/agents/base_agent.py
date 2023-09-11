from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, envs):
        self.env = envs
        self.dict_translation_actions = {'turn left': "tourner à gauche",
                                         "turn right": "tourner à droite",
                                         "go forward": "aller tout droit",
                                         "pick up": "attraper",
                                         "drop": "lâcher",
                                         "toggle": "basculer",
                                         "eat": "manger",
                                         "dance": "dancer",
                                         "sleep": "dormir",
                                         "do nothing": "ne rien faire",
                                         "cut": "couper",
                                         "think": "penser"}

    @abstractmethod
    def generate_trajectories(self, dict_modifier, n_tests, language='english'):
        raise NotImplementedError()

    @abstractmethod
    def update_parameters(self):
        raise NotImplementedError()

    def generate_prompt(self, goal, subgoals, deque_obs, deque_actions):
        ldo = len(deque_obs)
        lda = len(deque_actions)

        head_prompt = "Possible action of the agent:"
        for sg in subgoals:
            head_prompt += " {},".format(sg)
        head_prompt = head_prompt[:-1]

        g = " \n Goal of the agent: {}".format(goal)
        obs = ""
        for i in range(ldo):
            obs += " \n Observation {}: ".format(i)
            for d_obs in deque_obs[i]:
                obs += "{}, ".format(d_obs)
            obs += "\n Action {}: ".format(i)
            if i < lda:
                obs += "{}".format(deque_actions[i])
        return head_prompt + g + obs

    def generate_prompt_french(self, goal, subgoals, deque_obs, deque_actions):
        ldo = len(deque_obs)
        lda = len(deque_actions)
        head_prompt = "Actions possibles pour l'agent:"
        for sg in subgoals:
            head_prompt += " {},".format(sg)
        head_prompt = head_prompt[:-1]

        # translate goal in French
        dict_translation_det = {"the": "la",
                           'a': 'une'}
        dict_translation_names = {"box": "boîte",
                             "ball": "balle",
                             "key": "clef"}
        dict_translation_adjs = {'red': 'rouge',
                            'green': 'verte',
                            'blue': 'bleue',
                            'purple': 'violette',
                            'yellow': 'jaune',
                            'grey': 'grise'}

        det = ''
        name = ''
        adj = ''

        for k in dict_translation_det.keys():
            if k in goal:
                det = dict_translation_det[k]
        for k in dict_translation_names.keys():
            if k in goal:
                name = dict_translation_names[k]
        for k in dict_translation_adjs.keys():
            if k in goal:
                adj = dict_translation_adjs[k]
        translation_goal = 'aller à ' + det + ' ' + name + ' ' + adj

        g = " \n But de l'agent: {}".format(translation_goal)
        obs = ""
        for i in range(ldo):
            obs += " \n Observation {}: ".format(i)
            for d_obs in deque_obs[i]:
                obs += "{}, ".format(d_obs)
            obs += "\n Action {}: ".format(i)
            if i < lda:
                obs += "{}".format(deque_actions[i])
        return head_prompt + g + obs

    def prompt_modifier(self, prompt: str, dict_changes: dict) -> str:
        """use a dictionary of equivalence to modify the prompt accordingly
        ex:
        prompt= 'green box red box', dict_changes={'box':'tree'}
        promp_modifier(prompt, dict_changes)='green tree red tree' """

        for key, value in dict_changes.items():
            prompt = prompt.replace(key, value)
        return prompt

