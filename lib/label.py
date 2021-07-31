import re
import torch

pat = re.compile('.*/([^_]+)_.+.jpg$')

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

def _label_field(i, x):
    if x[0] in ['empty', '0']: return [f'{i}-z{x[0]}']
    if x[0] == '25': return [f'{i}-z{x[0]}-{x[1]}']
    return [f'{i}-{x[0]}', f'{i}-x{x[1]}']

def _label_fields(c):
    c = c.split('#')
    c = [x.split('-') for x in c]
    if (len(c) > 3): print(c)
    l = [_label_field(i, x) for i, x in enumerate(c)]
    return [item for sublist in l for item in sublist]

def label_field(fname):
    match = pat.match(str(fname.as_posix()))
    if match is None:
        return None
    c = match[1].split('$')[0]
    return _label_fields(c)

def label_field_and_pos(fname):
    match = pat.match(str(fname.as_posix()))
    if match is None:
        return None
    split = match[1].split('$')
    fields = _label_fields(split[0])
    pos = _label_pos(split[1] if len(split) > 1 else None)
    return(fields, pos)