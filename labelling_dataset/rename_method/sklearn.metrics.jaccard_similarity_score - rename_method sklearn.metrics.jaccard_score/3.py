# -*- coding: utf-8 -*-

from sklearn import metrics
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import hamming_loss
from sklearn.metrics import zero_one_loss
from sklearn.metrics import jaccard_similarity_score


def evaluation(y_pred, y_prob, y_true):
    coverage = coverage_error(y_true, y_prob)
    hamming = hamming_loss(y_true, y_pred)
    ranking_loss = label_ranking_loss(y_true, y_prob)

    f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
    f1_micro = metrics.f1_score(y_true, y_pred, average='micro')

    acc = 0
    for i in range(y_true.shape[0]):
        acc += jaccard_similarity_score(y_true.iloc[i, :], y_pred.iloc[i, :])  # jaccard_similarity_score
    acc = acc / y_true.shape[0]

    zero_one = zero_one_loss(y_true, y_pred)  # 0-1 error

    performance = {"coverage_error": coverage,
                   "ranking_loss": ranking_loss,
                   "hamming_loss": hamming,
                   "f1_macro": f1_macro,
                   "f1_micro": f1_micro,
                   "Jaccard_Index": acc,
                   "zero_one_error": zero_one}
    return performance


if __name__ == "__main__":
    pass

