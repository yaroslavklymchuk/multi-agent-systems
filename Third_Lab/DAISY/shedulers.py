import math


def get_explore_rate(t, min_explore_rate, decay_rate):
    return max(min_explore_rate, min(0.8, 1.0 - math.log10((t+1) / decay_rate)))


def get_learning_rate(t, min_learning_rate, decay_rate):
    return max(min_learning_rate, min(0.8, 1.0 - math.log10((t+1) / decay_rate)))
