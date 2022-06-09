import torch

def load_pkl(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def save_pt(model, path, size=(448, 796)):
    model.eval().cpu()
    example = torch.rand(1, 3, *size)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(path)