import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans


class PopulationCluster:
    """ 种群聚类 """

    def __init__(self, n_clusters: int = 3):
        self.cluster_model = KMeans(n_clusters=n_clusters)
        # self.cluster_model = DBSCAN()
    
    def cluster(self, pop):
        self.cluster_model.fit(pop)
        labels = self.cluster_model.labels_
        return labels

    def plot_cluster(self, pop, labels):
        """ 可视化聚类结果 """
        colors = ['red', 'blue', 'yellow']

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        for k, label in enumerate(np.sort(np.unique(labels))):
            index = [True if labels[i]==label else False for i in range(len(labels))]
            ax.scatter(pop.iloc[index, 0], pop.iloc[index, 1], pop.iloc[index, 2], 
                       c=colors[k], s=100, alpha=0.7, linewidth=0.7, edgecolors='black', label=f'Cluster #{label}')

        ax.view_init(elev=20, azim=300)  # elev仰角, azim方位角
        ax.set_xlabel(pop.columns[0])  # x轴名称: RMSE
        ax.set_ylabel(pop.columns[1])  # y轴名称: Diversity
        ax.set_zlabel(pop.columns[2])  # z轴名称: Complexity

        plt.tick_params(labelsize=8)  # 统一坐标轴字体大小
        plt.title('Cluster visualization')
        plt.legend()
        plt.show()

