import math


def smoothing_sheduler(rate, smoothing_coefficient):
    return rate * smoothing_coefficient


def smoothed_decay_based_sheduler(t, rate, decay):
    return rate / (1 + t * decay)


def smoothed_decay_based_sheduler_v2(t, rate, decay):
    return max(rate, min(0.8, 1.0 - math.log10((t + 1) / decay)))


SHEDULERS_MAPPING = {'smoothing_sheduler': smoothing_sheduler,
                     'smoothed_decay_based_sheduler': smoothed_decay_based_sheduler,
                     'smoothed_decay_based_sheduler_v2': smoothed_decay_based_sheduler_v2
                     }

#
# def get_explore_rate(t, min_explore_rate, decay_rate):
#     return max(min_explore_rate, min(0.8, 1.0 - math.log10((t+1) / decay_rate)))
#
#
# def get_learning_rate(t, min_learning_rate, decay_rate):
#     return max(min_learning_rate, min(0.8, 1.0 - math.log10((t+1) / decay_rate)))
