import torch

def weighted_BCELoss(output, target, weights=[1,1]):
    '''
    Custom function to compute weighted BCELoss for your class.

    Parameters
    ----------
    output : torch.Tensor
        torch.Tensor of variable shape, resulted from your model

    target : torch.Tensor
        torch.Tensor with your class input, matches output' shape.

    weights : array, list or tensor
        len(2) array, list or tensor with the weights for your class, the second position defines weight for your class 1.

    Returns
    -------
    torch.Tensor with negative mean loss
    '''
    output = torch.clamp(output,min=1e-7,max=1-1e-7)
    loss = weights[1] * (target * torch.log(output)) + \
           weights[0] * ((1 - target) * torch.log(1 - output))
    return torch.neg(torch.mean(loss))
