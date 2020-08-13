import json

import jieba
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi

from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch





datapath='D:\\temp\\data mining\\Tweets.txt' #Tweets数据集路径

def jieba_tokenize(text):
    return jieba.lcut(text)

'''preprocess函数：生成tf_idf矩阵;label_list'''
def preproccess(datapath):

    text_list=[] #存储文本
    label_list=[] #存储label

    with open(datapath, "r") as f:
        for line in f:
            itemdic = json.loads(line) #line为str类型，itemdic为字典
            text_list.append(itemdic["text"])
            label_list.append(itemdic["cluster"])
    # print textVector,labelVector
    tfidf_vectorizer = TfidfVectorizer(tokenizer=jieba_tokenize, lowercase=True)
    '''
    tokenizer: 指定分词函数
    lowercase: 在分词之前将所有的文本转换成小写
    '''
    # 需要进行聚类的文本集
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_list)
    return tfidf_matrix,label_list

'''K-Means算法：两个参数-tf_idf矩阵、K值'''
def kmeans(k,tfidf_matrix):
    km_cluster = KMeans(n_clusters=k, max_iter=300, n_init=40,
                    init='k-means++',n_jobs=-1)
    '''
    n_clusters: 指定K的值
    max_iter: 对于单次初始值计算的最大迭代次数
    n_init: 重新选择初始值的次数
    init: 制定初始值选择的算法
    n_jobs: 进程个数，为-1的时候是指默认跑满CPU
    注意，这个对于单个初始值的计算始终只会使用单进程计算，
    并行计算只是针对与不同初始值的计算。比如n_init=10，n_jobs=40,
    服务器上面有20个CPU可以开40个进程，最终只会开10个进程
    '''
    #返回各自文本的所被分配到的类索引
    result = km_cluster.fit_predict(tfidf_matrix)
    return result

'''计算预测准确率'''
def evaluate(judge_list,real_list):
    successRate=nmi(real_list,judge_list,average_method='arithmetic')
    return successRate

'''AP近邻传播聚类算法:只有一个参数-tfidf矩阵'''
def affinity_propagation(tfidf_matrix):
    ap_cluster = AffinityPropagation()
    '''
    damping: 衰减系数，默认为 0.5
    convergence_iter: 迭代次后聚类中心没有变化，算法结束，默认为15
    max_iter: 最大迭代次数，默认200
    copy: 是否在元数据上进行计算，默认True，在复制后的数据上进行计算
    preference: S的对角线上的值
    affinity: S矩阵（相似度），默认为euclidean（欧氏距离）矩阵
    '''
    result = ap_cluster.fit_predict(tfidf_matrix)
    return result

'''MeanShift算法：只有一个参数-tfidf矩阵'''
def meanshift(tfidf_matrix):
    ms = MeanShift(bandwidth=0.6, bin_seeding=True)
    result_ms = ms.fit_predict(tfidf_matrix.toarray())
    # print result_ms
    return result_ms

'''spectral_clustering算法'''
def spectral_clustering(tfidf_matrix, num_clusters):
    sc_cluster = SpectralClustering(affinity='rbf', n_clusters=num_clusters)
    result = sc_cluster.fit_predict(tfidf_matrix)
    return result

'''agglomerative_clustering算法'''
def agglomerative_clustering(tfidf_matrix, num_clusters):
    ac = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
    result = ac.fit_predict(tfidf_matrix.todense())
    return result

'''DBSCAN算法'''
def dbscan(tfidf_matrix):
    db = DBSCAN(eps=0.7, min_samples=1)
    result = db.fit_predict(tfidf_matrix)
    return result

'''GaussianMixture算法'''
def GaussianMixture(tfidf_matrix, num_clusters):
    gm = GaussianMixture(n_components=num_clusters, covariance_type='diag', max_iter=15, random_state=0)
    result = gm.fit_predict(tfidf_matrix)
    return result





if __name__ == '__main__':
    tfidf_matrix, real_list = preproccess(datapath)
    total_clusters=84
    '''K-Means算法'''
    judge_list = kmeans(total_clusters, tfidf_matrix)
    rate = evaluate(judge_list, real_list)
    print('使用kmeans算法进行聚类，准确率为：'+str(rate))
    '''AP近邻传播聚类算法'''
    judge_list=affinity_propagation(tfidf_matrix)
    rate = evaluate(judge_list, real_list)
    print('使用AP近邻传播聚类算法进行聚类，准确率为：' + str(rate))
    '''MeanShift算法'''
    judge_list = meanshift(tfidf_matrix)
    rate = evaluate(judge_list, real_list)
    print('使用MeanShift算法进行聚类，准确率为：' + str(rate))
    '''spectral_clustering算法'''
    judge_list = spectral_clustering(tfidf_matrix,total_clusters)
    rate = evaluate(judge_list, real_list)
    print('使用spectral_clustering算法进行聚类，准确率为：' + str(rate))
    '''agglomerative_clustering算法'''
    judge_list = agglomerative_clustering(tfidf_matrix, total_clusters)
    rate = evaluate(judge_list, real_list)
    print('使用agglomerative_clustering算法进行聚类，准确率为：' + str(rate))
    '''DBSCAN算法'''
    judge_list = dbscan(tfidf_matrix)
    rate = evaluate(judge_list, real_list)
    print('使用DBSCAN算法进行聚类，准确率为：' + str(rate))
    '''GaussianMixture算法'''
    judge_list = agglomerative_clustering(tfidf_matrix,total_clusters)
    rate = evaluate(judge_list, real_list)
    print('使用GaussianMixture算法进行聚类，准确率为：' + str(rate))


