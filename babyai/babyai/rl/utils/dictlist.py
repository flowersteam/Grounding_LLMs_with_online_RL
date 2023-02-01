import random


class DictList(dict):
    """A dictionnary of lists of same size. Dictionnary items can be
    accessed using `.` notation and list items using `[]` notation.

    Example:
        >>> d = DictList({"a": [[1, 2], [3, 4]], "b": [[5], [6]]})
        >>> d.a
        [[1, 2], [3, 4]]
        >>> d[0]
        DictList({"a": [1, 2], "b": [5]})
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __len__(self):
        return len(next(iter(dict.values(self))))

    def __getitem__(self, index):
        return DictList({key: value[index] for key, value in dict.items(self)})

    def __setitem__(self, index, d):
        for key, value in d.items():
            dict.__getitem__(self, key)[index] = value

    def shuffle_lists_same_order(self):
        """
        return the dictionnary with each list of the dictionnary shuffled such that:
        list_1[i]=list_2[i]=list_1[i_shuffle]=list_2[i_shuffle]

        Example:
            >>> d = DictList({"a":[1, 2, 3], "b":[4, 5, 6]})
            >>> d.shuffle_lists_same_order()
            DictList({"a":[3, 1, 2], "b":[6, 4, 5]})
        """
        keys = list(dict.keys(self))
        len_keys = len(keys)
        map_list = list(zip(*[v for v in dict.values(self)]))
        random.shuffle(map_list)
        l = list(zip(*map_list))
        return DictList({keys[i]: list(l[i]) for i in range(len_keys)})
