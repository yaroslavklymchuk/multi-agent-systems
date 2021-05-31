import math


def get_epsilon_rate(t, min_eps_rate, decay_rate):
    return max(min_eps_rate, min(0.8, 1.0 - math.log10((t+1) / decay_rate)))


def get_learning_rate(t, min_learning_rate, decay_rate):
    return max(min_learning_rate, min(0.8, 1.0 - math.log10((t+1) / decay_rate)))
