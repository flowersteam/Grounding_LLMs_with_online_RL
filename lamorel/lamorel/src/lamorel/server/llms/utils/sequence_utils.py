from functools import reduce

concat = lambda x, y: x + y
concat_all = lambda xs: reduce(concat, xs, [])

def flatten_list(seqs):
    flattened, list_map = [], []
    for i, seq in enumerate(seqs):
        flattened += seq
        list_map += [len(seq)]

    return flattened, list_map