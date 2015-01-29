from spyre import server

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class SimpleSineApp(server.App):
    title = "Homework 4 App"

    inputs = []
    input_sel = {}
    input_sel['input_type'] = 'text'
    input_sel['variable_name'] = 'clusters'
    input_sel['value'] = 2
    input_sel['action_id'] = 'update_data'
    inputs.append(input_sel)

    controls = []
    control_sel = {}
    control_sel['control_type'] = 'hidden'
    control_sel['label'] = 'Generate Cluster Plots'
    control_sel['control_id'] = 'update_data'
    controls.append(control_sel)

    tabs = ["Plot", "Table"]

    outputs = []
    output_sel = {}
    output_sel['output_type'] = 'plot'
    output_sel['output_id'] = 'plot'
    output_sel['control_id'] = 'update_data'
    output_sel['tab'] = 'Plot'
    output_sel['on_page_load'] = False
    outputs.append(output_sel)

    output_sel = {}
    output_sel['output_type'] = 'table'
    output_sel['output_id'] = 'table_id'
    output_sel['control_id'] = 'update_data'
    output_sel['tab'] = 'Table'
    output_sel['on_page_load'] = False
    outputs.append(output_sel)

    def _get_wine_features(self, df, scale_features=True, pca_components=0):
        feature_cols = [col for col in df.columns if col != 'Wine']
        X = df[feature_cols]
        # scaling
        if scale_features:
            scale = StandardScaler()
            X = scale.fit_transform(X)
        # pca
        if pca_components:
            pca = PCA(n_components=pca_components)
            X = pca.fit_transform(X)
        return X  # return the features array

    def getData(self, params):
        df = pd.read_csv('../data/wine.csv')
        return df

    def getPlot(self, params):
        clusters = float(params['clusters'])
        df = self.getData(params)
        X = self._get_wine_features(df, scale_features=True, pca_components=5)
        kmeans = KMeans(n_clusters=clusters, init='random', max_iter = 300, random_state=1)
        Y_hat_kmeans = kmeans.fit(X).labels_
        fig, ax = plt.subplots(1, 1)
        ax.scatter(X[:, 0], X[:, 1], c=Y_hat_kmeans)
        return fig


app = SimpleSineApp()
app.launch()
