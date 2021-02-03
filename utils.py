import numpy as np
import torch

def get_limits(points):
    min_max = torch.tensor([])
    for row in torch.transpose(points, 0, 1):
        min_val = torch.min(row)  
        max_val = torch.max(row)
        padding = (max_val - min_val) / 100
        min_max = torch.cat((min_max, torch.tensor([[min_val - padding, max_val + padding]])))
    
    return min_max

def uniform_prior_samples(limits, size=10000):
    prior_samples = []
    for limit in limits:
        low_lim = limit[0].detach().numpy()
        high_lim = limit[1].detach().numpy()
        prior_samples.append(np.random.uniform(low_lim, high_lim, size=size))
    
    prior_samples = np.array(prior_samples)
    return np.transpose(prior_samples)