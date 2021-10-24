import re
import torch
import random

pat = re.compile('.*/([^_]+)_.+.jpg$')

categories_simple = ['1','10','11','12','13','14','15','16','17','18','19','2','20','3','4','5','6','7','8','9','x1','x2','x3','25-1','25-2','0','empty']
field_rotation = ['20','1','18','4','13','6','10','15','2','17','3','19','7','16','8','11','14','9','12','5']

def _label_pos(s):
    if s is None: return torch.tensor([-1., -1, -1, -1, -1, -1])
    result = []
    for v in s.split('#'):
        if v == 'empty':
            result.append(0.)
            result.append(0.)
        else:
            result.append(float(v) / 2 + 0.5) 
    return torch.tensor(result) * 448

def _label_field(i, x, rotation):
    if x[0] in ['empty', '0']: return [f'{i}-z{x[0]}']
    if x[0] == '25': return [f'{i}-z{x[0]}-{x[1]}']
    idx_before_rotation = field_rotation.index(x[0])
    idx_after_rotation = (idx_before_rotation + 2 * rotation) % 20
    field_after_rotation = field_rotation[idx_after_rotation]
    return [f'{i}-{field_after_rotation}', f'{i}-x{x[1]}']

def _label_fields(c, rotation):
    c = c.split('#')
    c = [x.split('-') for x in c]
    if (len(c) > 3): print(c)
    l = [_label_field(i, x, rotation) for i, x in enumerate(c)]
    return [item for sublist in l for item in sublist]

def label_field(filename_and_rotation):
    match = pat.match(str(filename_and_rotation[0].as_posix()))
    if match is None:
        return None
    c = match[1].split('$')[0]
    return _label_fields(c, filename_and_rotation[1])

def random_labels():
    labels = []
    for i in range(0,3):
        ring = random.randint(20, 26)
        if ring > 22:
            labels.append(f'{i}-z{categories_simple[ring]}')
        else:
            slice = random.randint(0, 19)
            labels.append(f'{i}-{categories_simple[slice]}')
            labels.append(f'{i}-{categories_simple[ring]}')
    return labels

def label_field_random_empty(filename_and_rotation):
    labels = label_field(filename_and_rotation)
    if labels[0] == '0-zempty':
        return random_labels()
    return labels


def label_field_and_pos(fname):
    match = pat.match(str(fname.as_posix()))
    if match is None:
        return None
    split = match[1].split('$')
    fields = _label_fields(split[0])
    pos = _label_pos(split[1] if len(split) > 1 else None)
    return(fields, pos)
