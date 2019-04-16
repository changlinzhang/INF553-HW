import sys
import numpy as np
import heapq
# from Queue import PriorityQueue

class Data:
    dim = 0

    def __init__(self, index=-1, features=np.array([]), label=None):
        self.i = index
        self.features = features
        self.label = label
        self.predict = None


class Cluster:
    n = 0
    alpha = -1
    deleted_set = set()

    def __init__(self, datas=np.array([]), reps=np.array([]), centroid=None):
        # use numpy array
        self.datas = datas
        self.reps = reps
        self.centroid = centroid

    def shrink2centroid(self, p):
        pp = Data()
        for i in range(Data.dim):
            f = p.features[i]
            pp.features = f + alpha * (self.centroid.features[i] - f)
        return pp


def data_dist(p1, p2):
    return np.sum(np.square(p1.features - p2.features))


def merge(u, v):
    w = Cluster()

    w.datas = np.append(u.datas, v.datas)

    w.centroid = Data()
    u_num = len(u.datas)
    v_num = len(v.datas)
    for i in range(Data.dim):
        f_i = (u.centroid.features[i] * u_num + v.centroid.features[i] * v_num) / (u_num + v_num)
        w.centroid.features = np.append(w.centroid.features, f_i)

    if len(w.datas) <= Cluster.n:
        for p in w.datas:
            pp = w.shrink2centroid(p)
            w.reps = np.append(w.reps, pp)
        return w

    #else
    tmpSet = set()
    for i in range(Cluster.n):
        max_dist = 0
        for p in w.datas:
            if i == 0:
                min_dist = data_dist(p, w.centroid)
            else:
                dists = map(lambda q: data_dist(p, q), list(tmpSet))
                min_dist = min(dists)
            if min_dist >= max_dist:
                max_dist = min_dist
                max_data = p
        tmpSet.add(p)

    for p in tmpSet:
        pp = w.shrink2centroid(p)
        w.reps = np.append(w.reps, pp)

    return w


def cluster(samples, k):
    clusters = initial2clusters(samples)

    if clusters <= k:
        return clusters

    clusters_set = set(clusters)
    pairs = pairwise_distance(clusters)
    # key = lambda pair: pair[2]
    heapq.heapify(pairs)
    times = len(clusters) - k
    while times > 0:
        while True:
            # d, u, v = heapq.heappop(pairs)
            temp = heapq.heappop(pairs)
            print(temp)
            d, u, v = temp
            if u in clusters_set and v in clusters_set:
                break
        w = merge(u, v)
        times -= 1
        clusters_set.remove(u)
        clusters_set.remove(v)
        clusters_set.add(w)
        for c in clusters:
            if c in clusters_set:
                d = cluster_dist(w, c)
                heapq.heappush(pairs, [d, w, c])
    return clusters_set


def initial2clusters(samples):
    clusters = [Cluster(np.array([s]), np.array([s]), s) for s in samples]
    return clusters


def cluster_dist(c1, c2):
    min_d = sys.float_info.max
    for p1 in c1.datas:
        for p2 in c2.datas:
            d = data_dist(p1, p2)
            if d < min_d:
                min_d = d
    return min_d


def pairwise_distance(clusters):
    pairs = []
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            dist = cluster_dist(clusters[i], clusters[j])
            pairs.append([dist, clusters[i], clusters[j]])
    return pairs


def read_sample(sample_file):
    file = open(sample_file, 'r')
    content = file.readlines()[0]
    content = content[1:len(content)-1]
    samples_indexs = content.split(',')
    samples_indexs = [int(s) for s in samples_indexs]
    file.close()
    return samples_indexs


def read_dataset(dataset_file):
    file = open(dataset_file, 'r')
    content = file.readlines()
    datas = []
    for i in range(len(content)):
        features = content[i].split(',')
        label = features[-1]
        features = features[:-1]
        features = [float(f) for f in features]
        data = Data(i, np.array(features), label)
        datas.append(data)
    Data.dim = len(datas[0].features)
    file.close()
    return datas


def sample(samples_indexs, datas):
    samples = []
    for i in samples_indexs:
        samples.append(datas[i])
    return samples


if __name__ == "__main__":
    k = int(sys.argv[1])
    sample_file = sys.argv[2]
    dataset_file = sys.argv[3]
    n = int(sys.argv[4])
    alpha = float(sys.argv[5])

    Cluster.n = n
    Cluster.alpha = alpha

    samples_indexs = read_sample(sample_file)
    datas = read_dataset(dataset_file)
    samples = sample(samples_indexs, datas)

    clusters_set = cluster(samples, k)
