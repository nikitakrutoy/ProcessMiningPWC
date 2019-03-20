import processmining as pm
import logging
from time import time

start = time()
log = pm.preprocessing.read_csv("/Users/nikitakrutoy/Projects/ProcessMiningPWC/data/final_cut.csv")
logging.debug("Log has been read in {time}s".format(time=(time() - start)))


am = pm.miners.AlphaMiner()

start = time()
am.apply(log)
logging.debug("Model has been built in {time}s".format(time=(time() - start)))

am.draw("model")

start = time()
am.replay(log)
logging.debug("Model has been weighted in {time}s".format(time=(time() - start)))

weighted_model = am.weighted_model


def get_weighted_edges(G):
    edges = []
    for n, nbrs in G.adjacency_iter():
        for nbr, eattr in nbrs.items():
            data = eattr['weight']
            edges.append((n, nbr, data))
    return edges


def get_adjacency_list(model):
    adj_list = {}
    for node_name in model.nodes():
        adj_list[node_name] = (model.edge[node_name])
    return adj_list


print(get_adjacency_list(weighted_model))
