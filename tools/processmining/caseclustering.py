import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from .metric import *
from .preprocessing import make_case_df, df2paths, events_set, get_event2index
from .preprocessing import find_cases_by_eventset, case_encoder
from collections import defaultdict
from sklearn.metrics import silhouette_score
from .utils import get_classes_paths
from kmodes.kmodes import KModes
from copy import copy, deepcopy
from sklearn.manifold import TSNE
import graphviz as gv


CLUSTER_INFO = '''
Кластер {}
Размер: {}
Внутрикластерное расстояние: {}
Межклстерное расстояние: {}

{}
'''


class Model:

    def __init__(self):
        pass

    def fit(self, df):
        pass

    def preprocess(self, df):
        pass

    def predict(self, data):
        pass


class FeatureClust(Model):
    def __init__(self, min_clusters=2, max_clusters=10, step=1,
                 feature_aggrs=None, case_id_feature=None,
                 clustering_options=None):
        if clustering_options is None:
            clustering_options = {
                "max_iter": 10000,
                "n_init": 10,
            }
        self.metrics = [silhouette_score,
                        in_cluster_distance,
                        between_cluster_distance,
                        cluster_sizes]
        self.metric_names = ["Silhouette",
                             "In cluster distances",
                             "Between cluster distances",
                             "Cluster sizes"]
        self.clustering_options = clustering_options

        self.feature_aggrs = feature_aggrs
        self.case_id_feature = case_id_feature

        self.max_clusters = max_clusters
        self.min_clusters = min_clusters
        self.step = step
        self.best_scores = None
        self.km = None
        self.history = []

    def fit(self, df):
        self._ss = StandardScaler()
        X = self._ss.fit_transform(df)
        best_scores = dict(zip(self.metric_names, -np.ones(len(self.metrics))))
        best_clusters = []
        score = dict()
        clustering_options = self.clustering_options
        for n_clusters in range(self.min_clusters, self.max_clusters + 1,
                                self.step):
            clustering_options["n_clusters"] = n_clusters
            km = KMeans(**self.clustering_options)
            clusters = km.fit_predict(X)
            for name, metric in zip(self.metric_names, self.metrics):
                score[name] = metric(X, clusters)

            if score["Silhouette"] > best_scores["Silhouette"]:
                best_scores = copy(score)
                best_clusters = copy(clusters)
                self.km = deepcopy(km)
            self.history.append(copy(score))
        return best_clusters, best_scores

    def preprocess(self, data):
        assert self.feature_aggrs is not None
        assert self.case_id_feature is not None
        return make_case_df(data, self.case_id_feature, self.feature_aggrs,)

    def predict(self, data):
        assert self.km is not None
        X = self._ss.transform(data)
        return self.km.predict(X)

    def draw(self, data):
        cases = self.preprocess(data)
        clusters = self.predict(cases)
        X = self._ss.transform(cases)
        tsne = TSNE(n_components=2)
        X = tsne.fit_transform(X)
        # plt.figure(figsize=(20,10))
        plt.figure(dpi=900)
        for i in np.unique(clusters):
            points = X[clusters == i]
            plt.scatter(points[:, 0], points[:, 1], label=i)
        # plt.axis('off')
        # plt.xticks([])
        # plt.yticks([])
        # plt.legend()x


class EventsSet(Model):

    def __init__(self, min_clusters=2, max_clusters=10, step=1,
                 case_id_feature=None, event_feature=None,
                 clustering_options=None,
                 event2letter=None, letter2event=None):
        if clustering_options is None:
            clustering_options = {
                "max_iter": 10000,
                "n_init": 10,
            }
        self.clustering_options = clustering_options
        self.metrics = [silhouette_score,
                        in_cluster_distance,
                        between_cluster_distance,
                        cluster_sizes]
        self.metric_names = ["Silhouette",
                             "Incluster distances",
                             "Between cluster distances",
                             "Cluster sizes"]

        self.case_id_feature = case_id_feature
        self.event_feature = event_feature

        self.max_clusters = max_clusters
        self.min_clusters = min_clusters
        self.step = step
        self.best_scores = None
        self.centroids = None
        self.km = None
        if event2letter is not None:
            self.event2letter = event2letter

        if letter2event is not None:
            self.letter2event = letter2event

    def fit(self, data, verbose=0):
        best_scores = dict(zip(self.metric_names, -np.ones(len(self.metrics))))
        best_clusters = []
        score = dict()
        clustering_options = self.clustering_options
        for n_clusters in range(self.min_clusters, self.max_clusters + 1,
                                self.step):
            clustering_options["n_clusters"] = n_clusters
            km = KModes(**self.clustering_options)
            clusters = km.fit_predict(data)
            for name, metric in zip(self.metric_names, self.metrics):
                if name == "Incluster distances":
                    score[name] = metric(np.array(data), clusters,
                                         metric=matching_dissim,
                                         centroids=km.cluster_centroids_)
                else:
                    score[name] = metric(np.array(data), clusters,
                                         metric=matching_dissim)

            if score["Silhouette"] > best_scores["Silhouette"]:
                best_clusters = copy(clusters)
                best_scores = copy(score)
                self.centroids = copy(km.cluster_centroids_)
                self.km = deepcopy(km)
        self.best_scores = best_scores

        return best_clusters, best_scores

    def preprocess(self, data):
        assert self.case_id_feature is not None
        assert self.event_feature is not None
        paths, self.event2letter, self.letter2event = df2paths(
            data,
            self.case_id_feature, self.event_feature
        )
        event2index = get_event2index(data, self.event_feature)
        return events_set(paths, event2index)

    def predict(self, data):
        assert self.km is not None
        return self.km.predict(data)

    def draw(self, data):
        assert self.centroids is not None
        assert self.best_scores is not None
        assert self.event2letter is not None

        # getting centoids
        centroids = self.centroids
        event2index = get_event2index(data, self.event_feature)
        paths, _, _ = df2paths(data, self.case_id_feature, self.event_feature)
        centroids_w_events = []
        for centroid in centroids:
            cases = find_cases_by_eventset(paths, event2index, centroid)
            cases_length = [len(case) for case in cases]
            m = np.argmin(cases_length)
            centroids_w_events.append(cases[m])

        centroids = centroids_w_events
        centroids[0][0], centroids[0][2] = centroids[0][2], centroids[0][0]
        graph = gv.Digraph(format="png")
        cumulitive_index = 0
        graph.node_attr.update(color='lightblue2', style='filled')
        # building graph
        for j, cluster in enumerate(centroids):
            subgraph_name = "cluster_" + str(j + 1)
            encoded_centroid = case_encoder(cluster, self.event2letter)
            with graph.subgraph(name=subgraph_name) as c:
                for i in range(len(cluster) - 1):
                    node1 = str(i + cumulitive_index)
                    node2 = str(i + 1 + cumulitive_index)
                    c.node(node1, label=cluster[i], shape="box",)
                    c.node(node2, label=cluster[i + 1], shape="box",)
                    c.edge(node1, node2)
                    label = CLUSTER_INFO.format(
                        j + 1,
                        int(self.best_scores["Cluster sizes"][j]),
                        "%.2f" % self.best_scores["Incluster distances"][j],
                        "%.2f" % self.best_scores["Between cluster distances"][j],
                        encoded_centroid)
                    c.attr(label=label, color="white")
            cumulitive_index += len(cluster)
        graph.attr(rankdir="same")
        return graph


class PatternSet(Model):

    def __init__(self):
        pass


class Levenstein(Model):
    def __init__(self):
        pass


class W2V(Model):
    def __init__(self):
        pass


MODELS = {
    "FeatureClust": FeatureClust,
    "EventsSet": EventsSet,
    "PatternSet": PatternSet,
    "Levenstein": Levenstein,
    "Word2Vec": W2V
}


class CaseClustering:

    def __init__(self, model="FeatureClust", model_options=None,
                 decision_tree_options=None):
        if decision_tree_options is None:
            decision_tree_options = {
                "criterion": "gini",
                "splitter": "best",
                "max_depth": 5,
                "min_samples_split": 10,
                "min_samples_leaf": 10
            }

        if model_options is not None:
            self.model = MODELS[model](**model_options)
        else:
            self.model = MODELS[model]()

        self.decision_tree_options = decision_tree_options
        self.dtc = None
        self.descs = None

    def fit(self, data,
            preprocess=False,
            build_tree=True, decision_tree_options=None, features=None,
            verbose=0):
        """
        Clusters cases in 2 - 10 clusters with best metric valuе.
        Kmeans is used for clustering. Categorical features are encoded as
        occurance of each unique value for feature.

        Parameters
        ----------
        data: pandas.DataFrame
            input cases dataframe
        case_feature: string
            case id feature
        event_feature:
            event name feature
        features_aggrs: dict
            dict of kind (feature: aggregator)
            aggregator is a function that has a case (pandas df row) as input
            and int or float as output
        verbose:
            output smth while fit or not

        Returns
        -------
        clusters

        Example:
        --------
        >> a = func(x)
        """
        if preprocess:
            df = self.model.preprocess(data)
        else:
            df = data

        self.clusters, self.scores = self.model.fit(df)

        # fitting dtc
        if build_tree:
            self.dtc = DecisionTreeClassifier(**self.decision_tree_options)
            self.dtc.fit(df, self.clusters)

            if isinstance(self.model, FeatureClust):
                features = list(df.columns)

            assert features is not None

            self.descs = self._get_clusters_descs(self.dtc, features)

        return self

    def _get_clusters_descs(self, classifier, feature_names):
        tree = classifier.tree_
        classes = classifier.classes_
        features = tree.feature
        thres = tree.threshold
        values = tree.value
        path = np.zeros(tree.node_count)

        root = 0
        classes_paths = get_classes_paths(tree, root, path, classes)

        clusters = []
        for c in classes:
            descriptions = []
            for i, path in enumerate(classes_paths[c]):
                type = []
                for j, node in enumerate(path):
                    sign = node
                    if abs(node) == 1:
                        feature = feature_names[tree.feature[j]]
                        thres = tree.threshold[j]
                        desc = [feature, sign, thres]
                        type.append(desc)
                    if node == 2:
                        size = tree.n_node_samples[j]
                descriptions.append([type, size])
            clusters.append(descriptions)
        return clusters

    def predict(self, X):
        return self.dtc.predict(X)

    def draw(self, data):
        return self.model.draw(data)
