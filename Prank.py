# 导入包
import random
import math
import numpy as np
import time
import csv
from tqdm import tqdm
from scipy.sparse import csc_matrix, linalg, eye
from copy import deepcopy


# 定义装饰器，监控运行时间
def timmer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        stop_time = time.time()
        print('Func %s, run time: %s' % (func.__name__, stop_time - start_time))
        return res
    return wrapper


class Dataset():

    def __init__(self, fp):
        # fp: data file path
        self.data = self.loadData(fp)

    @timmer
    def loadData(self, fp):
        data = []
        for l in open(fp):
            data.append(tuple(map(int, l.strip().split('::')[:2])))
        return data

    @timmer
    def splitData(self, M, k, seed=1):
        '''
        :params: data, 加载的所有(user, item)数据条目
        :params: M, 划分的数目，最后需要取M折的平均
        :params: k, 本次是第几次划分，k~[0, M)
        :params: seed, random的种子数，对于不同的k应设置成一样的
        :return: train, test
        '''
        train, test = [], []
        random.seed(seed)
        for user, item in self.data:
            # 这里与书中的不一致，本人认为取M-1较为合理，因randint是左右都覆盖的
            if random.randint(0, M - 1) == k:
                test.append((user, item))
            else:
                train.append((user, item))

        # 处理成字典的形式，user->set(items)
        def convert_dict(data):
            data_dict = {}
            for user, item in data:
                if user not in data_dict:
                    data_dict[user] = set()
                data_dict[user].add(item)
            data_dict = {k: list(data_dict[k]) for k in data_dict}
            return data_dict

        return convert_dict(train), convert_dict(test)


class Metric():

    def __init__(self, train, test, GetRecommendation):
        '''
        :params: train, 训练数据
        :params: test, 测试数据
        :params: GetRecommendation, 为某个用户获取推荐物品的接口函数
        '''
        self.train = train
        self.test = test
        self.GetRecommendation = GetRecommendation
        self.recs = self.getRec()

    # 为test中的每个用户进行推荐
    def getRec(self):
        recs = {}
        for user in self.test:
            rank = self.GetRecommendation(user)
            recs[user] = rank#
        global top_tu
        top_tu = recs
        return recs

    # 定义精确率指标计算方式
    def precision(self):
        all, hit = 0, 0
        for user in self.test:
            test_items = set(self.test[user])
            rank = self.recs[user]
            for item, score in rank:
                if item in test_items:
                    hit += 1
            all += len(rank)
        return round(hit / all * 100, 2)

    # 定义召回率指标计算方式
    def recall(self):
        all, hit = 0, 0
        for user in self.test:
            test_items = set(self.test[user])
            rank = self.recs[user]
            for item, score in rank:
                if item in test_items:
                    hit += 1
            all += len(test_items)
        return round(hit / all * 100, 2)

    # 定义覆盖率指标计算方式
    def coverage(self):
        all_item, recom_item = set(), set()
        for user in self.test:
            for item in self.train[user]:
                all_item.add(item)
            rank = self.recs[user]
            for item, score in rank:
                recom_item.add(item)
        return round(len(recom_item) / len(all_item) * 100, 2)

    # 定义新颖度指标计算方式
    def popularity(self):
        # 计算物品的流行度
        item_pop = {}
        for user in self.train:
            for item in self.train[user]:
                if item not in item_pop:
                    item_pop[item] = 0
                item_pop[item] += 1

        num, pop = 0, 0
        for user in self.test:
            rank = self.recs[user]
            for item, score in rank:
                # 取对数，防止因长尾问题带来的被流行物品所主导
                pop += math.log(1 + item_pop[item])
                num += 1
        return round(pop / num, 6)

    def eval(self):
        metric = {'Precision': self.precision(),
                  'Recall': self.recall(),
                  'Coverage': self.coverage(),
                  'Popularity': self.popularity()}
        print('Metric:', metric)
        return metric


def PersonalRank(train, alpha, N):
    '''
    :params: train, 训练数据
    :params: alpha, 继续随机游走的概率
    :params: N, 推荐TopN物品的个数
    :return: GetRecommendation, 获取推荐结果的接口
    '''

    # 构建索引
    items = []
    for user in train:
        items.extend(train[user])
    id2item = list(set(items))
    users = {u: i for i, u in enumerate(train.keys())}
    items = {u: i + len(users) for i, u in enumerate(id2item)}

    # 计算转移矩阵（注意！！！要按照出度进行归一化）
    item_user = {}
    for user in train:
        for item in train[user]:
            if item not in item_user:
                item_user[item] = []
            item_user[item].append(user)

    data, row, col = [], [], []
    for u in train:
        for v in train[u]:
            data.append(1 / len(train[u]))
            row.append(users[u])
            col.append(items[v])
    for u in item_user:
        for v in item_user[u]:
            data.append(1 / len(item_user[u]))
            row.append(items[u])
            col.append(users[v])

    M = csc_matrix((data, (row, col)), shape=(len(data), len(data)))

    # 获取接口函数
    def GetRecommendation(user):
        seen_items = set(train[user])
        # 解矩阵方程 r = (1-a)r0 + a(M.T)r
        r0 = [0] * len(data)
        r0[users[user]] = 1
        r0 = csc_matrix(r0)
        r = (1 - alpha) * linalg.inv(eye(len(data)) - alpha * M.T) * r0
        r = r.T.toarray()[0][len(users):]
        idx = np.argsort(-r)[:N]
        recs = [(id2item[ii], r[ii]) for ii in idx]
        return recs

    return GetRecommendation


class Experiment():

    def __init__(self, M, N, alpha, fp='./data/ratings.dat'):#该路径'./data/ratings.dat'
        '''
        :params: M, 进行多少次实验
        :params: N, TopN推荐物品的个数
        :params: alpha, 继续随机游走的概率
        :params: fp, 数据文件路径
        '''
        self.M = M
        self.N = N
        self.alpha = alpha
        self.fp = fp
        self.alg = PersonalRank

    # 定义单次实验
    @timmer
    def worker(self, train, test):
        '''
        :params: train, 训练数据集
        :params: test, 测试数据集
        :return: 各指标的值
        '''
        getRecommendation = self.alg(train, self.alpha, self.N)
        metric = Metric(train, test, getRecommendation)
        return metric.eval()

    # 多次实验取平均
    @timmer
    def run(self):
        metrics = {'Precision': 0, 'Recall': 0,
                   'Coverage': 0, 'Popularity': 0}
        dataset = Dataset(self.fp)
        for ii in range(self.M):
            train, test = dataset.splitData(self.M, ii)
            print('Experiment {}:'.format(ii))
            metric = self.worker(train, test)
            metrics = {k: metrics[k] + metric[k] for k in metrics}
        metrics = {k: metrics[k] / self.M for k in metrics}
        print('Average Result (M={}, N={}, ratio={}): {}'.format( \
            self.M, self.N, self.ratio, metrics))

# PersonalRank实验(笔记本跑的太慢，这里没贴实验结果)
M, N, alpha = 2, 10, 0.8
exp = Experiment(M, N, alpha)
exp.run()