import os
import shutil
import sys
import logging
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from time import time
import graphviz as gv
from tqdm import tqdm_notebook
from copy import copy
LOGLEVEL = "DEBUG"
logging.basicConfig(level=logging.DEBUG)


class AlphaMiner():

    def __init__(self):
        self.zero_cycles = None
        self.model = None
        self.weighted_model = None

    def apply(self, log, output_file=None):

        tl = set()  # all task list
        df = []  # direct following tasks
        cs = []  # causalities tasks
        ncs = []  # non-causality tasks
        par = []  # parallel tasks
        xl = []
        yl = []
        ti = []
        to = []

        self.tl, self.df, self.cs, self.ncs, self.par = self._build_ordering_relations(log)
        self.xl, self.yl, self.ti, self.to = self._make_sets(
            log,
            self.tl,
            self.df,
            self.cs,
            self.ncs)
        logging.debug("Building Petri net")
        self._build_petrinet()

    def _build_ordering_relations(self, log):
        tl = set([item for sub in log for item in sub])
        logging.debug("Got task list")
        df = self.get_direct_followers(log)

        logging.debug("Got direct follows")
        cs = self.get_causalities(tl, df)
        logging.debug("Got causaieties")
        cycles = self.get_one_cycles(log)
        logging.debug("Got cycles")
        cs += cycles

        ncs = self.get_no_causalities(tl, df)
        logging.debug("Got not causaieties")
        par = self.get_parallels(tl, df)
        logging.debug("Got parallels")

        return tl, df, cs, ncs, par

    def _make_sets(self, log, tl, df, cs, ncs):
        xl = self.make_xl_set(tl, df, cs, ncs)
        logging.debug("Got Xl")
        yl = self.make_yl_set(xl)
        ti = self.make_ti_set(log)
        to = self.make_to_set(log)
        return xl, yl, ti, to

    def get_one_cycles(self, log):
        cycles = []
        for case in log:
            prev_event = None
            prev_prev_event = None
            for event in case:
                if prev_prev_event == event:
                    if (prev_prev_event, prev_event) not in cycles:
                        cycles.append((prev_prev_event, prev_event))
                        cycles.append((prev_event, prev_prev_event))
                prev_prev_event = prev_event
                prev_event = event
        return cycles

    @staticmethod
    def get_direct_followers(log):
        df = []
        for trace in log:
            for index, event in enumerate(trace):
                # print index, event
                if index != len(trace) - 1:
                    if (event, trace[index + 1]) not in df:
                        df.append((event, trace[index + 1]))
        return df

    @staticmethod
    def get_causalities(all_tasks, direct_followers):
        cs = []  # causalities
        for event in all_tasks:
            for event2 in all_tasks:
                if (event, event2) not in cs:
                    if (event, event2) in direct_followers and \
                       (event2, event) not in direct_followers:
                        cs.append((event, event2))
        return cs

    @staticmethod
    def get_no_causalities(all_tasks, direct_followers):
        ncs = []  # no causalities
        for event in all_tasks:
            for event2 in all_tasks:
                if (event, event2) not in ncs:
                    if (event, event2) not in direct_followers and \
                       (event2, event) not in direct_followers:
                        ncs.append((event, event2))
        return ncs

    @staticmethod
    def get_parallels(all_tasks, direct_followers):
        par = []  # parallel tasks
        for event in all_tasks:
            for event2 in all_tasks:
                if (event, event2) not in par:
                    if (event, event2) in direct_followers and \
                       (event2, event) in direct_followers:
                        par.append((event, event2))
        return par

    @staticmethod
    def check_set(A, ncs):
        for event in A:
            for event2 in A:
                if (event, event2) not in ncs:
                    return False
        return True

    @staticmethod
    def check_outsets(A, B, cs):
        for event in A:
            for event2 in B:
                if (event, event2) not in cs:
                    return False
        return True

    def make_xl_set(self, all_tasks, direct_followers, causalities, no_causalities):
        import itertools
        xl = set()
        subsets = set()
        for i in range(1, len(all_tasks)):
            for s in itertools.combinations(all_tasks, i):
                subsets.add(s)
        logging.debug("Got subsets")
        logging.info("Processing " + str(len(subsets)) + " subset of " + str(len(all_tasks)) + " events" )
        for a in tqdm_notebook(subsets):
            reta = self.check_set(a, no_causalities)
            for b in subsets:
                retb = self.check_set(b, no_causalities)
                if reta and retb and \
                   self.check_outsets(a, b, causalities):
                    xl.add((a, b))
        return xl

    @staticmethod
    def make_yl_set(xl):
        import copy
        yl = copy.deepcopy(xl)
        for a in xl:
            A = a[0]
            B = a[1]
            for b in xl:

                if set(A).issubset(b[0]) and set(B).issubset(b[1]):
                    if a != b:
                        yl.discard(a)
        return yl

    # Ti is the set of all tasks which occur trace-initially
    @staticmethod
    def make_ti_set(log):
        ti = set()
        [ti.add(event[0]) for event in log]
        return ti

    # To is the set of all tasks which occur trace-terminally
    @staticmethod
    def make_to_set(log):
        to = set()
        [to.add(event[-1]) for event in log]
        return to

    def _build_petrinet_gv(self, yl, ti, to, output_file):
        pn = gv.Digraph(format='png')
        pn.attr(rankdir='LR')  # left to righ layout - default is top down
        pn.node('start', shape='circle', color="lightslateblue", style='filled')
        pn.node('end', shape='circle', color="lightslateblue", style='filled')
        for place_index, elem in enumerate(yl):
            place = "p" + str(place_index)
            for i in elem[0]:
                pn.edge(i,  place)
                pn.node(i, label=i, shape='box', color='lightblue2', style='filled')
                pn.node(place, shape='circle', color="lightslateblue", style='filled')
            for i in elem[1]:
                pn.edge(place, i)
                pn.node(i, label=i, shape='box', color='lightblue2', style='filled')
        for i in ti:
            pn.edge('start', i)
        for o in to:
            pn.edge(o, 'end')
        pn.render(output_file)
        return pn

    def _build_petrinet(self):
        self.model = nx.DiGraph()
        for j, elem in enumerate(self.yl):
            place = "p" + str(j)
            for i in elem[0]:
                self.model.add_edge(i, place)
            for i in elem[1]:
                self.model.add_edge(place, i)
            for i in self.ti:
                self.model.add_edge('start', i)
            for o in self.to:
                self.model.add_edge(o, 'end')

    def draw(self, zero_cycles=None, output_file=None):
        if self.model is not None and output_file is not None:
            # nx.draw(self.model)
            # plt.savefig('model.png')
            return self._build_petrinet_gv(self.yl, self.ti, self.to, output_file)
        else:
            logging.info("Vertex sets are not defined. Build model by applying log first")
        # if output_file is not None:
        #     plt.savefig(output_file)

    def get_model(self):
        if self.model is not None:
            return self.model
        else:
            logging.info("Model has not been built yet. Run apply to build")

    def get_weighted_model(self):
        if self.weighted_model is not None:
            return self.model
        else:
            logging.info("Model has not been weight yet. Run replay to weight")

    @staticmethod
    def _remove_tokens(model):
        new_model = model.copy()
        nodes_names = new_model.nodes()
        for node_name in nodes_names:
            node = model.node[node_name]
            if "Token" in node.keys():
                del new_model.node[node_name]["Token"]
        return new_model

    def replay(self, log):
        model = self.model.copy()
        start = "start"
        end = "end"

        for case in log:
            model.node[start]["Token"] = True
            model.node[end]["Token"] = False

            for event in case:
                in_edges = model.in_edges(event)
                out_edges = model.out_edges(event)
                from_places = [edge[0] for edge in in_edges]
                to_places = [edge[1] for edge in out_edges]
                logging.debug(from_places)
                logging.debug(to_places)
                no_tokens_flag = True
                for place_name in from_places:
                    place = model.node[place_name]
                    if place.get("Token", False):
                        no_tokens_flag = False
                        model.node[place_name]["Token"] = False
                        model.edge[place_name][event]["weight"] = model.edge[place_name][event].get("weight", 0) + 1
                        break
                if no_tokens_flag:
                    logging.debug("No tokens to activate transition")

                for place_name in to_places:
                    place = model.node[place_name]
                    if place.get("Token", False):
                        logging.debug("Error! Bad transition (" + event + "," + place_name  + ")")
                    else:
                        model.node[place_name]["Token"] = True
                        model.edge[event][place_name]["weight"] = model.edge[event][place_name].get("weight", 0) + 1
            self.weighted_model = model
