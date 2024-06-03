# This is the baseline code for merging the fragment tracklet using clustering based on appearance
from sklearn.cluster import KMeans

class postprocess:
    def __init__(self, number_of_people, cluster_method):
        self.n = number_of_people
        if cluster_method == 'kmeans':
            self.cluster_method = KMeans(n_clusters=self.n, random_state=0)
        else:
            raise NotImplementedError

    def run(self,features):

        print('Start Clustering')

        self.cluster_method.fit(features)

        print('Finish Clustering')

        return self.cluster_method.labels_