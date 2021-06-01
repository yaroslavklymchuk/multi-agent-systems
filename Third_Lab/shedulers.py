import math


def time_based_sheduler(rate, smoothing_coefficient):
    return rate * smoothing_coefficient


def smoothed_time_based_sheduler(t, rate, decay):
    return rate / (1 + t * decay)


def smoothed_time_based_sheduler_v2(t, rate, decay):
    return max(rate, min(0.8, 1.0 - math.log10((t + 1) / decay)))


SHEDULERS_MAPPING = {'time_based_sheduler': time_based_sheduler,
                     'smoothed_time_based_sheduler': smoothed_time_based_sheduler,
                     'smoothed_time_based_sheduler_v2': smoothed_time_based_sheduler_v2
                     }

# def get_epsilon_rate(t, min_eps_rate, decay_rate):
#     return max(min_eps_rate, min(0.8, 1.0 - math.log10((t+1) / decay_rate)))
#
#
# def get_learning_rate(t, min_learning_rate, decay_rate):
#     return max(min_learning_rate, min(0.8, 1.0 - math.log10((t+1) / decay_rate)))
