import document
import query 
from LambdaRankHW import *
import numpy
import math

def read_queries(fold) :
    train_queries = query.load_queries("HP2003\Fold" + str(fold) + "\\train.txt", 64)
    test_queries = query.load_queries("HP2003\Fold" + str(fold) + "\\test.txt", 64)
    val_queries = query.load_queries("HP2003\Fold" + str(fold) + "\\vali.txt", 64)
    return train_queries, val_queries,test_queries


def get_NDCG(scores, labels, k):
    indexes = np.arange(len(scores))

    scores = zip(scores, indexes)
    scores = sorted(scores, key=lambda x: -x[0])

    # DCG @ k
    sum = 0
    for r in range(k):
        #         print(scores[r][1], labels[scores[r][1]])
        val = np.power(2, labels[scores[r][1]]) - 1
        val /= math.log(r + 1 + 1, 2)
        sum += val

    # NDCG
    sum2 = 0
    labels = sorted(labels, key=lambda x: -x)
    for r in range(k):
        val = np.power(2, labels[r]) - 1
        val /= math.log(r + 1 + 1, 2)
        sum2 += val

    if sum2 == 0:
        return 0

    NDCG = sum / sum2
    return NDCG

def score_queries(ranker, queries, k):
    average_ndcg = 0
    ctr = 0
    for query in queries:
        scores = ranker.score(query)
        labels = query.get_labels()
        NDCG = get_NDCG(scores, labels, min(len(scores), k))
        if NDCG == -1 :
            continue
        ctr += 1
        average_ndcg += NDCG
        print(query.get_qid(), NDCG)

    print("Final NDCG:", average_ndcg / ctr)
    return average_ndcg / ctr


# def filter_queries(queries):
#

        
# queries_train, queries_val,queries_test = read_queries(1)
#
# ranker = LambdaRankHW(64)
# ranker.train_with_queries(queries_val,1)
# score_queries(ranker, queries_val)

# labels=[0,1,0,2,1]
# scores=[0.3, 1, 0.5, 0, 0]
#
# ranking = range(len(labels))
# lambdas = np.zeros(len(ranking) ** 2).reshape((len(ranking), len(ranking)))
#
# #
# for r1 in ranking:
#     for r2 in ranking:
#         s = 0
#         if labels[r1] > labels[r2]:
#             s = 1
#         elif labels[r1] < labels[r2]:
#             s = -1
#
#         lambdas[r1, r2] = 0.5 * (1 - s) - 1.0 / (1 + np.exp(scores[r1] - scores[r2]))
#         lambdas[r2, r1] = 0.5 * (1 + s) - 1.0 / (1 + np.exp(scores[r2] - scores[r1]))
#
# aggregated_l = []
# for r1 in ranking:
#     new_lam = 0
#     for r2 in ranking:
#         if labels[r1] > labels[r2]:
#             new_lam += lambdas[r1, r2]
#         elif labels[r1] < labels[r2]:
#             new_lam -= lambdas[r1, r2]
#     aggregated_l.append(new_lam)
#
# print(aggregated_l)
