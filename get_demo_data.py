import sys
import jsonlines
from collections import Counter

def get_parent(node):
    current = node
    to_update = []
    while parent[current] != current:
        to_update.append(current)
        current = parent[current]

    for up in to_update:
        parent[up] = current
    return current


def to_dict(row):
    items = row.split('COL')
    res = {}
    for item in items:
        if 'VAL' not in item:
            continue
        attr, value = item.split('VAL')
        res[attr.strip()] = value.strip()
    return res


if __name__ == '__main__':
    fn = sys.argv[1]
    out_fn = sys.argv[2]
    lines = open(fn).readlines() # + open('valid.txt').readlines() + open('train.txt').readlines()
    parent = {}

    for line in lines:
        t1, t2, lbl = line.split('\t')
        if t1 not in parent:
            parent[t1] = t1
        if t2 not in parent:
            parent[t2] = t2

        if '1' in lbl:
            p1 = get_parent(t1)
            p2 = get_parent(t2)
            parent[p1] = p2

    p2idx = {}
    pcnt = Counter()
    for item in parent:
        p = get_parent(item)
        if p not in p2idx:
            p2idx[p] = len(p2idx)
        pcnt[p2idx[p]] += 1

    frequent = pcnt.most_common(20)
    frequent = set([p[0] for p in frequent])

    with jsonlines.open(out_fn, mode='w') as writer:
        for item in parent:
            p = get_parent(item)
            if p2idx[p] in frequent:
                output = to_dict(item)
                output['cluster_id'] = p2idx[p]
                writer.write(output)
