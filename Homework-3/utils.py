import document
import query 
import numpy
import math

def read_queries(fold) :
    train_queries = query.load_queries("HP2003\Fold" + str(fold) + "\\train.txt", 64)
    val_queries = query.load_queries("HP2003\Fold" + str(fold) + "\\vali.txt", 64)
    test_queries = query.load_queries("HP2003\Fold" + str(fold) + "\\test.txt", 64)
    return train_queries, val_queries,test_queries

def has_relevant_document(labels):
    for label in labels:
        if label != 0:
            return True
    return False


def get_NDCG(scores, labels, k):
    #print(scores)
    #print(labels)
    indexes = numpy.arange(len(scores))

    scores = zip(scores, indexes)
    scores = sorted(scores, key = lambda x: -x[0])
    
    # DCG @ k
    sum = 0
    for r in range(k):
        val =  numpy.power(2, labels[scores[r][1]]) - 1
        val /= math.log(r+1 + 1,2)
        sum += val
    
    #Max DCG (Normalize)
    sum2 = 0
    labels = sorted(labels,  key = lambda x: -x)
    for r in range(k):
        val =  numpy.power(2, labels[r]) - 1
        val /= math.log(r+1 + 1, 2)
        sum2 += val
    
    if sum2 == 0:
        return 1
        
    NDCG = sum / sum2
    return NDCG

def score_queries(ranker, queries):
    average_ndcg = 0
    for query in queries:
        scores = ranker.score(query)
        labels = query.get_labels()
        NDCG = get_NDCG(scores, labels, min(len(scores), 10))
        average_ndcg += NDCG
        print(query.get_qid(), NDCG)

    print("Final NDCG:", average_ndcg / len(queries))

	
def evaluate(ranker, queries):
    ndcgs = {}
    
    null_queries = 0
    average_ndcg = 0
    for query in queries:
        labels = query.get_labels()
        if not has_relevant_document(labels):
            null_queries += 1
            ndcgs[query.get_qid()] = 0
            continue

        scores = ranker.score(query).flatten()
        labels = query.get_labels()
        NDCG = get_NDCG(scores,labels, min(len(labels), 10))
        ndcgs[query.get_qid()] = NDCG
        
        average_ndcg += NDCG
    average_ndcg = average_ndcg/(len(queries) - null_queries) 
    ndcgs['all'] = average_ndcg
    return ndcgs
        
        
		

def Crossfold_validation(rankerClass, folds, epochs):
    all_folds_ndcgs = []
    for fold in folds:
        print "-------------------------------------------------"

        #load queries
        train_queries = fold[0]
        val_queries = fold[1]
        test_queries = fold[2]

        #store NDCG after each epoch for this fold
        fold_ndcgs = []

        #train ranker
        ranker = rankerClass(64)
        for epoch in range(epochs):
            ranker.train_with_queries(train_queries,1)
            ndcgs = utils.evaluate(ranker,val_queries)
            fold_ndcgs.append(ndcgs['all'])
            print "Epoch %d NDCG: %f"%(epoch + 1, ndcgs['all'])

        print fold_ndcgs

        all_folds_ndcgs.append(fold_ndcgs)


    ndcgs_per_epoch = numpy.mean(all_folds_ndcgs,axis = 0)
    return ndcgs_per_epoch
    
    
    
        