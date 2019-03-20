import numpy as np
import pandas as pd


def make_aggregator(func, *args):
    def aggr(case):
        return func(case, *args)
    return aggr


def case_duration(case, time_feature):
    return int((pd.to_datetime(case.iloc[-1][time_feature])
                - pd.to_datetime(case.iloc[0][time_feature])).value / 10e8)


def sum_by_event(case, event_feature, event_name, feature):
    return np.sum(case[case[event_feature] == event_name][feature])


def cat_prob_by_event(case, event_feature, event_name, data, feature):
    feature_value_in_case_by_event = list(case[case[event_feature] == event_name][feature])
    if len(feature_value_in_case_by_event) != 0:
        case_feature = feature_value_in_case_by_event[0]
    else:
        case_feature = list(case[feature])[0]
    count = len(data[data[feature] == case_feature])
    return count / data.shape[0]
