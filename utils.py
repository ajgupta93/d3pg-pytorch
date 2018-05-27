import torch
from torch.autograd import Variable

def to_numpy(var):
    return var.cpu().data.numpy() if use_cuda else var.data.numpy()

def to_tensor(x, dtype, volatile=False, requires_grad=True):
    x = torch.from_numpy(x).float()
    x = Variable(x, requires_grad=requires_grad).type(dtype=dtype)
    return x

def weightSync(target_model, source_model, tau = 0.001):
    for parameter_target, parameter_source in zip(target_model.parameters(), source_model.parameters()):
        parameter_target.data.copy_((1 - tau) * parameter_target.data + tau * parameter_source.data)
