import document
import query 
import numpy
import math
from scipy import stats
import itertools

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
    # print(len(scores), len(labels), k)
    indexes = numpy.arange(len(scores))

    scores = zip(scores, indexes)
    scores = scores[:len(labels)]
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
        
        
		

def crossfold_validation(rankerClass, folds, epochs):
    all_folds_ndcgs = []
    cnt=0
    for fold in folds:
        cnt+=1
        print "------------------Fold %d-------------------------------" %cnt

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
            ndcgs = evaluate(ranker,val_queries)
            fold_ndcgs.append(ndcgs['all'])
            print "Epoch %d NDCG: %f"%(epoch + 1, ndcgs['all'])

        print fold_ndcgs

        all_folds_ndcgs.append(fold_ndcgs)


    ndcgs_per_epoch = numpy.mean(all_folds_ndcgs,axis = 0)
    return ndcgs_per_epoch

def evaluate_all(RankerClass, folds, epochs):
    all_folds_ndcgs = []
    for fold in folds:
        #load queries
        train_queries = fold[0]
        val_queries = fold[1]
        test_queries = fold[2]

        #train ranker
        ranker = RankerClass(64)
        for epoch in range(epochs):
            ranker.train_with_queries(train_queries,1)
            ranker.train_with_queries(val_queries,1)
            print "Epoch %d "%(epoch + 1)

        ndcgs = evaluate(ranker, test_queries)

        all_folds_ndcgs.append(ndcgs)
    
    return all_folds_ndcgs

def get_eval_list(folds, eval_dicts):
    l = []
    c =0
    for _, _, test in folds:
        for queryID in test.keys():
            if queryID != 'all':
                labels = test[queryID].get_labels()
                if has_relevant_document(labels):
                    l.append(eval_dicts[c][queryID])
        c+=1
    #for dictionary in eval_dicts:
    #    l.extend(dictionary.values())
    return l

def compute_significance(results, alpha = 0.05):
    ranker_names = list(results.keys())
    
    pairwise = list(itertools.combinations(ranker_names, 2))
         
    p_values = []
    
    for pair in pairwise:
        a = results[pair[0]]
        b = results[pair[1]]
        p_value = stats.ttest_rel(a,b)[1]
        p_values.append((pair,p_value))
        
#     #Bonferroni correction
    alpha_c = alpha / len(pairwise)
    print("Bonferroni corrected alpha:", alpha_c)
    
    
    reject = []
    for p_value in p_values:
        reject.append((p_value[0], p_value[1] < alpha_c))
    
    
    return p_values, reject