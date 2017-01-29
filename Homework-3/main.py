import document
import query 
from LambdaRankHW import *
import numpy
import math



def read_queries(fold) 
    train_queries = query.load_queries("HP2003\Fold" + str(fold) + "\\train.txt", 64)
    test_queries = query.load_queries("HP2003\Fold" + str(fold) + "\\test.txt", 64)
    val_queries = query.load_queries("HP2003\Fold" + str(fold) + "\\vali.txt", 64)
    return train_queries, val_queries,test_queries


def get_NDCG(scores, labels, k):
    indexes = numpy.arange(len(scores))
    
    scores = zip(scores, indexes)
    scores = sorted(scores, key = lambda x: -x[0])
    
    
#     print(query.get_qid(),":", len(labels))
#     print(query.get_qid(), ":", len(scores))
    
    # DCG @ k
    sum = 0
    for r in range(k):
#         print(scores[r][1], labels[scores[r][1]])
        val =  numpy.power(2, labels[scores[r][1]]) - 1
        val /= math.log(r + 1+1,2)
        sum += val
        
#     print("labels")
    #NDCG
    sum2 = 0
    labels = sorted(labels,  key = lambda x: -x)
    for r in range(k):
#         print(labels[r])
        val =  numpy.power(2, labels[r]) - 1
        val /= math.log(r + 1+1, 2)
        sum2 += val
    
    if sum2 == 0:
        return 0
        
#     print(sum,sum2)
    
    NDCG = sum / sum2
    return NDCG


        
train_queries, val_queries,test_queries = read_queries(1)

ranker = LambdaRankHW(64)
ranker.train_with_queries(train_queries,10)


average_ndcg = 0

for query in val_queries:
    scores = ranker.score(query)
    labels = query.get_labels()
    NDCG = get_NDCG(scores,labels, min(len(scores), 10))
    average_ndcg += NDCG
    print(query.get_qid(),NDCG)
    
print("Final NDCG:", average_ndcg/len(val_queries))
        