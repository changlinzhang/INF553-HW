import sys
import numpy as np
import heapq


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
            pp.features = np.append(pp.features, f + alpha * (self.centroid.features[i] - f))
        pp.i = p.i
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
    return w


def find_repre(w):
    if len(w.datas) <= Cluster.n:
        for p in w.datas:
            pp = w.shrink2centroid(p)
            w.reps = np.append(w.reps, pp)
        return

    #else
    tmpSet = set()
    for i in range(Cluster.n):
        max_dist = 0
        for p in w.datas:
            if i == 0:
                min_dist = data_dist(p, w.centroid)
            else:
                dists = map(lambda q: data_dist(p, q), list(tmpSet))
                # print(dists)
                min_dist = min(dists)
            if min_dist >= max_dist:
                max_dist = min_dist
                max_data = p
        tmpSet.add(max_data)

    for p in tmpSet:
        pp = w.shrink2centroid(p)
        w.reps = np.append(w.reps, pp)

    return


def cluster(samples, k):
    clusters = initial2clusters(samples)

    if len(clusters) <= k:
        return clusters

    clusters_set = set(clusters)
    pairs = pairwise_distance(clusters)
    # key = lambda pair: pair[2]
    heapq.heapify(pairs)
    times = len(clusters) - k
    while times > 0:
        while True:
            d, u, v = heapq.heappop(pairs)
            if u in clusters_set and v in clusters_set:
                break
        w = merge(u, v)
        times -= 1
        clusters_set.remove(u)
        clusters_set.remove(v)
        for c in clusters_set:
            d = cluster_dist(w, c)
            heapq.heappush(pairs, [d, w, c])
        clusters_set.add(w)
    return clusters_set


def initial2clusters(samples):
    clusters = []
    for s in samples:
        c = Cluster()
        c.datas = np.append(c.datas, s)
        c.centroid = s
        # print(s.features)
        p = c.shrink2centroid(s)
        # print(p.features)
        c.reps = np.append(c.reps, p)
        clusters.append(c)
    return clusters


def cluster_dist(c1, c2):
    dist = np.sum(np.square(c1.centroid.features - c2.centroid.features))
    return dist


def data2cluster_dist(data, c):
    min_d = sys.float_info.max
    for p in c.reps:
        d = data_dist(data, p)
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
    content = file.readlines()
    samples_indexs = [int(line) for line in content]
    file.close()
    return samples_indexs


def read_dataset(dataset_file):
    file = open(dataset_file, 'r')
    content = file.readlines()
    datas = []
    for i in range(len(content)):
        features = content[i].split(',')
        label = features[-1].strip()
        features = features[:-1]
        features = [float(f) for f in features]
        data = Data(i, np.array(features), label)
        datas.append(data)
    Data.dim = len(datas[0].features)
    file.close()
    return datas


# def gene_test_file(dataset_file, samples_indexs):
#     testf = open('test_input.txt', 'w')
#     file = open(dataset_file, 'r')
#     content = file.readlines()
#     for i in samples_indexs:
#         testf.write(content[i])
#     testf.close()
#     file.close()


def sort_cluster(clusters_list):
    # sort inner reps
    for c in clusters_list:
        c.reps = sorted(c.reps, key=lambda rep: rep.i)
        c.datas = sorted(c.datas, key=lambda data: data.i)
    # sort clusters
    clusters_list = sorted(clusters_list, key=lambda c: c.reps[0].i)
    return clusters_list


def label_samples(clusters_list):
    for i in range(len(clusters_list)):
        for d in clusters_list[i].datas:
            d.predict = i


def predict(clusters_list, datas):
    for i in range(len(datas)):
        data = datas[i]
        if data.predict >= 0:
            continue
        if not data.predict:
            min_dist = sys.float_info.max
            predict = -1
            for i in range(len(clusters_list)):
                c = clusters_list[i]
                dist = data2cluster_dist(data, c)
                if dist < min_dist:
                    min_dist = dist
                    predict = i
            data.predict = predict
            clusters_list[predict].datas = np.append(clusters_list[predict].datas, data)


def sample(samples_indexs, datas):
    samples = []
    for i in samples_indexs:
        samples.append(datas[i])
    return samples


def eval(clusters_list, datas):
    truth_pairs = []
    predict_pairs = []

    for c in clusters_list:
        datas_tmp = c.datas
        for i in range(len(datas_tmp)):
            for j in range(i+1, len(datas_tmp)):
                predict_pairs.append((datas_tmp[i].i, datas_tmp[j].i))

    true_cluster = {}
    for d in datas:
        if d.label not in true_cluster:
            true_cluster[d.label] = []
        true_cluster[d.label].append(d.i)

    for value in true_cluster.values():
        datas = sorted(value)
        for i in range(len(datas)):
            for j in range(i+1, len(datas)):
                truth_pairs.append((datas[i], datas[j]))

    intersection = set(truth_pairs) & set(predict_pairs)
    precision = 1.0*len(intersection)/len(predict_pairs)
    recall = 1.0*len(intersection)/len(truth_pairs)
    return precision, recall


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
    ori_datas = datas
    samples = sample(samples_indexs, datas)

    #samples clustering results
    clusters_set = cluster(samples, k)
    for w in clusters_set:
        find_repre(w)
    clusters_list = (list(clusters_set))
    clusters_list = sort_cluster(clusters_list)
    label_samples(clusters_list)

    predict(clusters_list, datas)
    for i in range(len(clusters_list)):
        pre_s = 'Cluster ' + str(i) + ':'
        c = clusters_list[i]
        indexs = []
        datas_tmp = c.datas
        for d in datas_tmp:
            indexs.append(d.i)
        print(pre_s + str(indexs))

    precision, recall = eval(clusters_list, ori_datas)
    print('Precision = %.12f, recall = %0.12f' % (precision, recall))

