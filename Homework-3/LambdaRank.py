__author__ = 'agrotov'

import itertools
import numpy as np
import lasagne
import theano
import theano.tensor as T
import time
from itertools import count
import query
import math
import sys

NUM_EPOCHS = 500

BATCH_SIZE = 1000
NUM_HIDDEN_UNITS = 200
LEARNING_RATE = 0.00005
MOMENTUM = 0.95

# TODO: Implement the lambda loss function
def lambda_loss(output, lambdas):
    return np.multiply(output, lambdas)

def get_max_DCG(labels, k):
    sum2 = 0
    labels = sorted(labels, key=lambda x: -x)
    for r in range(k):
        val = np.power(2, labels[r]) - 1
        val /= math.log(r + 1 + 1, 2)
        sum2 += val

    if sum2 == 0:
        return 0

    return sum2


class LambdaRank:


    NUM_INSTANCES = count()

    def __init__(self, feature_count):
        self.feature_count = feature_count
        self.output_layer = self.build_model(feature_count,1,BATCH_SIZE)
        self.iter_funcs = self.create_functions(self.output_layer)


    # train_queries are what load_queries returns - implemented in query.py
    def train_with_queries(self, train_queries, num_epochs):
        try:
            now = time.time()
            for epoch in self.train(train_queries):
                if epoch['number'] % 1 == 0:
                    print("Epoch {} of {} took {:.3f}s".format(
                    epoch['number'], num_epochs, time.time() - now))
                    print("training loss:\t\t{:.6f}\n".format(epoch['train_loss']))
                    now = time.time()
                if epoch['number'] >= num_epochs:
                    break
        except KeyboardInterrupt:
            pass

    def score(self, query):
        feature_vectors = query.get_feature_vectors()
        scores = self.iter_funcs['out'](feature_vectors)
        return scores


    def build_model(self,input_dim, output_dim,
                    batch_size=BATCH_SIZE):
        """Create a symbolic representation of a neural network with `intput_dim`
        input nodes, `output_dim` output nodes and `num_hidden_units` per hidden
        layer.

        The training function of this model must have a mini-batch size of
        `batch_size`.

        A theano expression which represents such a network is returned.
        """
        print "input_dim",input_dim, "output_dim",output_dim
        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, input_dim),
        )

        l_hidden = lasagne.layers.DenseLayer(
            l_in,
            num_units=200,
            nonlinearity=lasagne.nonlinearities.tanh,
        )


        l_out = lasagne.layers.DenseLayer(
            l_hidden,
            num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.linear,
        )

        return l_out

    # Create functions to be used by Theano for scoring and training
    def create_functions(self, output_layer,
                          X_tensor_type=T.matrix,
                          batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE, momentum=MOMENTUM, L1_reg=0.0000005, L2_reg=0.000003):
        """
            Create functions for training, validation and testing to iterate one
            epoch.
        """
        X_batch = X_tensor_type('x')
        y_batch = T.fvector('y')

        output_row = lasagne.layers.get_output(output_layer, X_batch, dtype="float32")
        output = output_row.T

        output_row_det = lasagne.layers.get_output(output_layer, X_batch,deterministic=True, dtype="float32")

       
        loss_train = lambda_loss(output, y_batch)
        loss_train = loss_train.sum()

        # TODO: (Optionally) You can add regularization if you want - for those interested
        # L1_loss = lasagne.regularization.regularize_network_params(output_layer,lasagne.regularization.l1)
        # L2_loss = lasagne.regularization.regularize_network_params(output_layer,lasagne.regularization.l2)
        # loss_train = loss_train.sum() + L1_loss * L1_reg + L2_loss * L2_reg

        # Parameters you want to update
        all_params = lasagne.layers.get_all_params(output_layer)

        # Update parameters, adam is a particular "flavor" of Gradient Descent
        updates = lasagne.updates.adam(loss_train, all_params)


        # Create two functions:

        # (1) Scoring function, deterministic, does not update parameters, outputs scores
        score_func = theano.function(
            [X_batch],output_row_det,
        )

        # (2) Training function, updates the parameters, outpust loss
        train_func = theano.function(
            [X_batch,y_batch], loss_train,
            updates=updates,
            # givens={
            #     X_batch: dataset['X_train'][batch_slice],
            #     # y_batch: dataset['y_valid'][batch_slice],
            # },
        )

        print "finished create_iter_functions"
        return dict(
            train=train_func,
            out=score_func,
        )

    # TODO: Implement the aggregate (i.e. per document) lambda function

    def lambda_function(self,labels, scores):
        ranking = sorted(range(len(labels)), key=lambda x: -scores[x])
        lambdas = np.zeros(len(ranking)**2).reshape((len(ranking),len(ranking)))
        relevant = filter(lambda x:labels[x]>0, ranking)
        maxDCG = get_max_DCG(labels, len(labels))

        for r1 in relevant:
            for r2 in ranking:
                if labels[r1] == labels[r2]:
                    continue

                lambdas[r1, r2] = -np.fabs(( 1.0 / (1 + np.exp(scores[r1] - scores[r2])))*( 1.0/np.log(r1+2) - 1.0/np.log(r2+2))*(2**labels[r1] - 2**labels[r2]) * (1.0/maxDCG))
                lambdas[r2, r1] = -lambdas[r1, r2]

        aggregated_l = np.sum(lambdas, axis=1)

        return np.array(aggregated_l, dtype='float32')


    def compute_lambdas_theano(self,query, labels):
        scores = self.score(query).flatten()
        result = self.lambda_function(labels, scores[:len(labels)])
        return result

    def train_once(self, X_train, query, labels):
        # TODO: Comment out to obtain the lambdas
        
        lambdas = self.compute_lambdas_theano(query,labels)
        lambdas.resize((BATCH_SIZE, ))
        X_train.resize((BATCH_SIZE, self.feature_count),refcheck=False)
        batch_train_loss = self.iter_funcs['train'](X_train, lambdas)

        return batch_train_loss


    def train(self, train_queries):
        X_trains = train_queries.get_feature_vectors()

        queries = train_queries.values()

        for epoch in itertools.count(1):
            batch_train_losses = []
            random_batch = np.arange(len(queries))
            np.random.shuffle(random_batch)
            l = len(queries)
            for index in xrange(len(queries)):
                random_index = random_batch[index]
                labels = queries[random_index].get_labels()
                sys.stdout.write("\r Query %d / %d" % (index, l))
                sys.stdout.flush()
                batch_train_loss = self.train_once(X_trains[random_index],queries[random_index],labels)
                batch_train_losses.append(batch_train_loss)
            sys.stdout.write("\r")
            sys.stdout.flush()

            avg_train_loss = np.mean(batch_train_losses)

            yield {
                'number': epoch,
                'train_loss': avg_train_loss,
            }

