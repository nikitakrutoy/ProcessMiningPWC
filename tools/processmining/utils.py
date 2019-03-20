from collections import defaultdict
import numpy as np
import graphviz as gv
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from .preprocessing import case_encoder


colors = ["#AD1B02", "#D85604", "#FFFFFF", "#E88D14", "#F3BE26", ]
cmap_name = 'pwc'
cm = LinearSegmentedColormap.from_list(
        cmap_name, colors, N=10)

def get_classes_paths(tree, node, path, classes):

    r = tree.children_right[node]
    l = tree.children_left[node]

    result = defaultdict(list)
    reuslt_r = defaultdict(list)
    result_l = defaultdict(list)

    if tree.feature[node] < 0:
        path[node] = 2
        cls = classes[np.argmax(tree.value[node])]
        result[cls].append(path)
    else:
        if r > 0:
            path_r = np.copy(path)
            path_r[node] = 1
            result_r = get_classes_paths(tree, r, path_r, classes)
        if l > 0:
            path_l = np.copy(path)
            path_l[node] = -1
            result_l = get_classes_paths(tree, l, path_l, classes)

        for cls in classes:
            result[cls] = result_r[cls] + result_l[cls]

    return result


def decode(centroids, events):
    results = []
    for centroid in centroids:
        results.append([events[i] for i, flag in enumerate(centroid) if flag == 1])
    return results


def draw_path(path, event2letter=None):
    graph = gv.Digraph(format="png")
    graph.node_attr.update(color='lightblue2', style='filled')
    if event2letter is not None:
        graph.attr(label=case_encoder(path, event2letter), labelloc="top")
    for i in range(len(path) - 1):
        node1 = path[i]
        node2 = path[i + 1]
        graph.node(node1, label=path[i], shape="box",)
        graph.node(node2, label=path[i + 1], shape="box",)
        graph.edge(node1, node2)
    return graph


def draw_descs(descs):
    d = defaultdict(list)

    thres = set()
    for cluster in descs:
        for type in cluster:
            for desc in type[0]:
                    thres.add((desc[0], desc[2]))

    types = []
    for i, cluster in enumerate(descs):
        for j, type in enumerate(cluster):
            types.append("Cluster " + str(i) + ' Type ' + str(j))

    features = []
    for thr in thres:
        features.append(thr[0] + " V " + "%.2f" % thr[1])

    for cluster in descs:
        for type in cluster:
            for feature in features:
                flag = True
                for desc in type[0]:
                    value = desc[1]
                    current_feature = desc[0] + " V " + "%.2f" % desc[2]
                    if current_feature == feature:
                        flag = False
                        d[feature].append(value)
                if flag:
                    d[feature].append(0)
    df = pd.DataFrame(d)
    df["Types"] = types
    df.set_index("Types", inplace=True)
    plt.figure(figsize=(20, 10))
    font = {'size': 14}
    plt.rc('font', **font)
    plt.xticks(rotation=50)
    sns.heatmap(df.transpose(), cmap="bwr")


def draw_centroids(centroids):
    paths = centroids
    cluster1 = paths[0]
    cluster2 = paths[1]

    cluster1 = [
        cluster1[3],
        cluster1[2],
        cluster1[1],
        cluster1[4],
        cluster1[0],
    ]

    cluster2 = [
        cluster2[6],
        cluster2[0],
        cluster2[8],
        cluster2[3],
        cluster2[7],
        cluster2[4],
        cluster2[5],
        cluster2[2],
        cluster2[1],
    ]

    graph = gv.Digraph(format="png")

    with graph.subgraph(name="cluster_1") as c:
        c.attr('node', fillcolor="darkolivegreen2")
        for i in range(len(cluster1) - 1):
            node1 = str(i)
            node2 = str(i + 1)
            c.node(node1, label=cluster1[i], shape="box",)
            c.node(node2, label=cluster1[i + 1], shape="box",)
            c.edge(node1, node2)
        label = '''
        Кластер 1
        Размер: 259
        Внутрикластерное расстояние: 2.02
        '''
        c.attr(label=label, color="white")


    with graph.subgraph(name="cluster_2") as c:
        for i in range(len(cluster2) - 1):
            node1 = str(i + len(cluster1))
            node2 = str(i + 1 + len(cluster1))
            if cluster2[i] in cluster1:
                color1 = "darkolivegreen2"
            else:
                color1 = "indianred1"
            if cluster2[i + 1] in cluster1:
                color2 = "darkolivegreen2"
            else:
                color2 = "indianred1"
            c.node(node1, label=cluster2[i], shape="box", fillcolor=color1)
            c.node(node2, label=cluster2[i + 1], shape="box", fillcolor=color2)
            c.edge(node1, node2)
        label = '''
        Кластер 2
        Размер: 231
        Внутрикластерное расстояние: 1.82
        '''
        c.attr(label=label, color="white")
    graph.attr(label="Межкластрное расстояние: 6.83", rankdir="LR")
    return graph
