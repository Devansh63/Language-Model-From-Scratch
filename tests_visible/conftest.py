import random

import torch


def pytest_runtest_setup(item):
    random.seed(0)
    torch.manual_seed(0)
