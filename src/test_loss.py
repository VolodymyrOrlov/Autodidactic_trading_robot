import numpy as np


def test_loss_function(actual, observed):

    def recall(signum):
        return np.sum(actual[observed[observed == signum].index] == signum) / len(observed[observed == signum])

    return 3 / (1 / recall(1) + 1 / recall(0) + 1 / recall(2))