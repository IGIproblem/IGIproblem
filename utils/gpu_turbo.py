import torch
import random

def gpu_turbo(eps):
    a = torch.rand(6000, 6000)
    a = a.cuda()
    while True:
        if random.random() < eps:
            b = torch.sin(a)

if __name__ == '__main__':
    gpu_turbo(0.1)

