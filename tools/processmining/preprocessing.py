import os
import shutil
import sys
import logging
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm_notebook
from copy import copy
from collections import defaultdict
LOGLEVEL = "DEBUG"
logging.basicConfig(level=logging.DEBUG)


def df2cases(data, case_feature, event_feature):
    """
    Function description

    Parameters
    ----------
    data : pd.DataFrame
        description
    var2: type
        description

    Returns
    -------
    cases : list
        description

    Example:
    --------
    >> a = func(x)
    """
    case_ids = np.unique(data[case_feature])
    cases = []
    for case_id in case_ids:
        case = list(data[data[case_feature] == case_id][event_feature])
        cases.append(case)
    return cases


def case_encoder(case, event2letter):
    return "".join([event2letter[event] for event in case])


def case_decoder(case, letter2event):
    return [letter2event[letter] for letter in case]


def get_event2letter(events):
    size = len(events)
    letters = [chr(i) for i in range(ord("A"), ord("A") + 26)] + [chr(i) for i in range(ord("a"), ord("a") + size - 26)]
    event2letter = dict(zip(events, letters))
    letter2event = dict(zip(letters, events))
    return event2letter, letter2event


def get_event2index(data, event_feature):
    events = np.unique(data[event_feature])
    return dict([(event, i) for i, event in enumerate(events)])


def df2paths(data, case_feature, event_feature, decode=True):
    events = np.unique(data[event_feature])
    event2letter, letter2event = get_event2letter(events)
    case_ids = np.unique(data[case_feature])
    paths = list()
    for case_id in tqdm_notebook(case_ids, desc="cases"):
        case = data[data[case_feature] == case_id][event_feature]
        if case_encoder(case, event2letter) not in paths:
            paths.append(case_encoder(case, event2letter))

    if decode:
        decoded_paths = []
        for path in paths:
            decoded_paths.append(case_decoder(path, letter2event))
        paths = decoded_paths

    return paths, event2letter, letter2event


def events_set(data, event2index):
    vectors = []
    size = len(event2index)
    for case in data:
        desc = np.zeros(size)
        for event in case:
            desc[event2index[event]] = 1
        vectors.append(desc)
    return vectors


def find_cases_by_eventset(data, event2index, eventset):
    size = len(event2index)
    result = []
    for case in data:
        desc = np.zeros(size)
        for event in case:
            desc[event2index[event]] = 1
        if np.array_equal(desc, eventset):
            result.append(case)
    return result


def make_case_df(data, case_feature, feature_aggrs):
    """
    Function description

    Parameters
    ----------
    data: pandas.DataFrame
        Log of events with contex
    case_feature:
        Name of case id dataframe column
    features_aggrs: dict
        dict of kind (feature: aggregator)
        aggregator is a function that has a case (pandas df row) as input
        and int, float or string as output
        see processmining.aggregators

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with new features collected with feature aggregators

    Example:
    --------
    >> aggrs = {
    >>    "Duration": case_duration,
    >>    "Material": cat_prob_by_event,
    >> }
    >> df = make_case_df(data, "CASE", aggrs)
    """
    case_ids = np.unique(data[case_feature])
    features = defaultdict(list)
    for case_id in case_ids:
        case = data[data[case_feature] == case_id]
        for feature, aggr in feature_aggrs.items():
            features[feature].append(aggr(case))
    return pd.DataFrame(features)


def read_csv(filename,
             case_atr_name="CASE",
             event_atr_name="SIMPLIFIED EVENT",
             clean=True,
             *args):

    csv = pd.read_csv(
        filename,
        usecols=[case_atr_name, event_atr_name],
        *args
    )

    log = Log()

    log.case_ids = np.unique(csv[case_atr_name])
    log.events = np.unique(csv[event_atr_name])
    logging.info("Reading log")
    for case_id in log.case_ids:
        case = list(csv[csv[case_atr_name] == case_id][event_atr_name])
        if clean:
            case = log._clean_case(case)
        log.append(case)
    return log


def _get_beta_cases(path_counter, cases_num, beta=0.99):
    path_counter.sort(key=lambda x: len(x[1]), reverse=True)
    cases = []
    paths = []
    betas = []
    cur_cases_num = 0
    for path in path_counter:
        if float(cur_cases_num) / cases_num >= beta:
            break
        cur_cases_num += len(path[1])
        cases = cases + path[1]
        paths.append(path[0])
        betas.append(float(len(path[1])) / cases_num)
    return cases, paths, betas


def _get_top_cases(path_counter, cases_num, top=5):
    path_counter.sort(key=lambda x: len(x[1]), reverse=True)
    top_paths = path_counter[:top]
    paths = [path[0] for path in top_paths]
    betas = [float(len(path[1])) / cases_num for path in top_paths]
    cases = []
    for path in top_paths:
        cases += path[1]
    return cases, paths, betas


def count_pathes(log):
    letter2event = log.letter2event
    event2letter = log.event2letter
    path_counter = {}
    cases = []
    for i, events in enumerate(log):
        lettered_events = "".join([event2letter[event] for event in events])
        cases.append((i, lettered_events))
    for case in cases:
        events = case[1]
        case_id = case[0]
        if events in path_counter.keys():
            path_counter[events].append(case_id)
        else:
            path_counter[events] = [case_id]
    return path_counter


def prune(log, coef=0.8, top=None):
    logging.info("Pruning...")
    if log.events is None:
        logging.info("No events list, running log.get_events")
        log.get_events
    size = len(log.events)
    log.make_dicts()
    path_counter = list(count_pathes(log).items())
    cases_num = len(log)

    if top is None:
        pruned_cases, paths, betas = _get_beta_cases(path_counter, cases_num, coef)
    else:
        pruned_cases, paths, betas = _get_top_cases(path_counter, cases_num, top)
    new_log = copy(Log(log[i] for i in pruned_cases))
    new_log.letter2event = copy(log.letter2event)
    new_log.event2letter = copy(log.event2letter)

    new_events = set()
    for case in new_log:
        new_events = new_events.union(set(case))
    new_log.events = new_events
    return new_log, paths, betas


def _clean_case(case):
    prev_event = case[0]
    new_case = [prev_event]
    for event in case:
        if event != prev_event:
            new_case.append(event)
        prev_event = event
    return new_case


def clean(log, inplace=True):
    logging.info("Cleaning...")
    if not inplace:
        log = copy(log)
    for i in range(len(log)):
        log[i] = _clean_case(log[i])
    return log


class Log(list):
        def __init__(self, *args, events=None):
            self.events = events
            self.letter2event = None
            self.event2letter = None
            super().__init__(*args)

        def make_dicts(self):
            if self.events is not None:
                size = len(self.events)
                letters = [chr(i) for i in range(ord("A"), ord("A") + 26)] + \
                    [chr(i) for i in range(ord("a"), ord("a") + size - 26)]
                self.letter2event = dict(zip(letters, self.events))
                self.event2letter = dict(zip(self.events, letters))
            else:
                logging.info("No event list fo this log")

        def get_events(self):
            if self.events is None:
                events = set()
                for case in self:
                    events = events.union(set(case))
                self.events = events
            else:
                logging.log("Events set already constructed")

        @classmethod
        def from_csv(cls, filename,
                     case_atr="CASE",
                     event_atr="SIMPLIFIED EVENT",
                     date_atr="SYS DATE",
                     do_sort=False):
            csv = pd.read_csv(
                filename,
                usecols=[case_atr, event_atr, date_atr],
            )

            if do_sort:
                csv.sort_values(by=date_atr, inplace=True)

            cases = []
            case_ids = np.unique(csv[case_atr])
            events = np.unique(csv[event_atr])
            logging.info("Reading...")
            for case_id in case_ids:
                case = list(csv[csv[case_atr] == case_id][event_atr])
                cases.append(case)
            return cls(cases, events=events)
