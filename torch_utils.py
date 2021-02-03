import torch


def get_limits(points):
    min_max = torch.tensor([])
    for row in torch.transpose(points, 0, 1):
        min_val = torch.min(row)  
        max_val = torch.max(row)
        padding = (max_val - min_val) / 100
        min_max = torch.cat((min_max, torch.tensor([[min_val - padding, max_val + padding]])))

    return min_max
