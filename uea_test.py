import os
import time
import json
import math
import torch
import numpy
import argparse
import weka.core.jvm
import weka.core.converters

import scikit_wrappers


def load_UEA_dataset(path, dataset):
    """
    Loads the UEA dataset given in input in numpy arrays.

    @param path Path where the UCR dataset is located.
    @param dataset Name of the UCR dataset.

    @return Quadruplet containing the training set, the corresponding training
            labels, the testing set and the corresponding testing labels.
    """
    # Initialization needed to load a file with Weka wrappers
    weka.core.jvm.start()
    loader = weka.core.converters.Loader(
        classname="weka.core.converters.ArffLoader"
    )

    train_file = os.path.join(path, dataset, dataset + "_TRAIN.arff")
    test_file = os.path.join(path, dataset, dataset + "_TEST.arff")
    train_weka = loader.load_file(train_file)
    test_weka = loader.load_file(test_file)

    train_size = train_weka.num_instances
    test_size = test_weka.num_instances
    nb_dims = train_weka.get_instance(0).get_relational_value(0).num_instances
    length = train_weka.get_instance(0).get_relational_value(0).num_attributes

    train = numpy.empty((train_size, nb_dims, length))
    test = numpy.empty((test_size, nb_dims, length))
    train_labels = numpy.empty(train_size, dtype=numpy.int)
    test_labels = numpy.empty(test_size, dtype=numpy.int)

    for i in range(train_size):
        train_labels[i] = int(train_weka.get_instance(i).get_value(1))
        time_series = train_weka.get_instance(i).get_relational_value(0)
        for j in range(nb_dims):
            train[i, j] = time_series.get_instance(j).values

    for i in range(test_size):
        test_labels[i] = int(test_weka.get_instance(i).get_value(1))
        time_series = test_weka.get_instance(i).get_relational_value(0)
        for j in range(nb_dims):
            test[i, j] = time_series.get_instance(j).values

    # Normalizing dimensions independently
    for j in range(nb_dims):
        # Post-publication note:
        # Using the testing set to normalize might bias the learned network,
        # but with a limited impact on the reported results on few datasets.
        # See the related discussion here: https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/pull/13.
        mean = numpy.mean(numpy.concatenate([train[:, j], test[:, j]]))
        var = numpy.var(numpy.concatenate([train[:, j], test[:, j]]))
        train[:, j] = (train[:, j] - mean) / math.sqrt(var)
        test[:, j] = (test[:, j] - mean) / math.sqrt(var)

    # Move the labels to {0, ..., L-1}
    labels = numpy.unique(train_labels)
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i
    train_labels = numpy.vectorize(transform.get)(train_labels)
    test_labels = numpy.vectorize(transform.get)(test_labels)

    weka.core.jvm.stop()
    return train, train_labels, test, test_labels


def fit_hyperparameters(file, train, train_labels, cuda, gpu,
                        save_memory=False):
    """
    Creates a classifier from the given set of hyperparameters in the input
    file, fits it and return it.

    @param file Path of a file containing a set of hyperparemeters.
    @param train Training set.
    @param train_labels Labels for the training set.
    @param cuda If True, enables computations on the GPU.
    @param gpu GPU to use if CUDA is enabled.
    @param save_memory If True, save GPU memory by propagating gradients after
           each loss term, instead of doing it after computing the whole loss.
    """
    classifier = scikit_wrappers.CausalCNNEncoderClassifier()

    # Loads a given set of hyperparameters and fits a model with those
    hf = open(os.path.join(file), 'r')
    params = json.load(hf)
    hf.close()
    # Check the number of input channels
    params['in_channels'] = numpy.shape(train)[1]
    params['cuda'] = cuda
    params['gpu'] = gpu
    classifier.set_params(**params)
    return classifier.fit(
        train, train_labels, save_memory=save_memory, verbose=True
    )


def get_config():

    parser = argparse.ArgumentParser(description='Classification tests for UEA repository datasets')

    parser.add_argument('--dataset', type=str, default='data', help='dataset name')
    parser.add_argument('--path', type=str, default='./', help='path where the dataset is located')
    parser.add_argument('--save_path', type=str, default='output', 
                        help='path where the estimator is/should be saved')
    
    parser.add_argument('--cuda', action='store_true', default=False, help='activate to use CUDA')
    parser.add_argument('--gpu', type=int, default=0, 
                        help='index of GPU used for computations (default: 0)')
    
    parser.add_argument('--hyper', type=str, default='default_hyperparameters.json', 
                        help='path of the file of hyperparameters to use; for training; must be a JSON file')
    parser.add_argument('--load', action='store_true', default=False, 
                        help='activate to load the estimator instead of training it')
    parser.add_argument('--fit_classifier', action='store_true', default=False, 
                        help='if not supervised, activate to load the model and retrain the classifier')

    # config = parser.parse_known_args()[0]
    config = parser.parse_args(args=[])
    # print(config)
    
    return config


def main(**kwargs):
    
    # start time
    start_time = time.time()
    print('Start: ', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    # get config
    args = get_config()
    for k in kwargs:
        args.__dict__[k] = kwargs[k]
    print(args)

    # create output dir
    args.save_path = args.save_path + '/' + args.dataset
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if args.cuda and not torch.cuda.is_available():
        print("CUDA is not available, proceeding without it...")
        args.cuda = False

    train, train_labels, test, test_labels = load_UEA_dataset(
        args.path, args.dataset
    )
    if not args.load and not args.fit_classifier:
        classifier = fit_hyperparameters(
            args.hyper, train, train_labels, args.cuda, args.gpu,
            save_memory=True
        )
    else:
        classifier = scikit_wrappers.CausalCNNEncoderClassifier()
        hf = open(
            os.path.join(
                args.save_path, args.dataset + '_hyperparameters.json'
            ), 'r'
        )
        hp_dict = json.load(hf)
        hf.close()
        hp_dict['cuda'] = args.cuda
        hp_dict['gpu'] = args.gpu
        classifier.set_params(**hp_dict)
        classifier.load(os.path.join(args.save_path, args.dataset))

    if not args.load:
        if args.fit_classifier:
            classifier.fit_classifier(classifier.encode(train), train_labels)
        classifier.save(
            os.path.join(args.save_path, args.dataset)
        )
        with open(
            os.path.join(
                args.save_path, args.dataset + '_hyperparameters.json'
            ), 'w'
        ) as fp:
            json.dump(classifier.get_params(), fp)

    print("Test accuracy: " + str(classifier.score(test, test_labels)))

    # end time
    end_time = time.time()
    print('End: ', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    # time consumed
    print('Took %f seconds' % (end_time - start_time))


if __name__ == "__main__":
    
    main(dataset='BasicMotions', path='datasets/UEA/Multivariate2018_arff', save_path='output')
    main(dataset='BasicMotions', path='datasets/UEA/Multivariate2018_arff', save_path='output', 
         cuda=True, gpu=0)




# -----------------------------------------------
# get config
args = get_config()
args.dataset = 'BasicMotions'
args.path = 'datasets/UEA/Multivariate2018_arff'
args.cuda = True
print(args)

# get dataset(numpy.array)
train, train_labels, test, test_labels = load_UEA_dataset(args.path, args.dataset)

''' classifier and accuracy
# classifier
classifier = fit_hyperparameters(args.hyper, train, train_labels, args.cuda, args.gpu, 
                                 save_memory=True)
# accuracy
print("Test accuracy: " + str(classifier.score(test, test_labels)))
'''

# CausalCNNEncoderClassifier
cf = scikit_wrappers.CausalCNNEncoderClassifier()
# Loads a given set of hyperparameters and fits a model with those
hf = open(os.path.join(args.hyper), 'r')
params = json.load(hf)
hf.close()
# Check the number of input channels
params['in_channels'] = numpy.shape(train)[1]
params['cuda'] = args.cuda
params['gpu'] = args.gpu
cf.set_params(**params)

''' fit and accuracy
# fit
cf.fit(train, train_labels, save_memory=False, verbose=True)

# accuracy for test
features_test = cf.encode(X=test, batch_size=10)
print("Test accuracy: " + str(cf.classifier.score(features_test, y=test_labels)))
'''

''' fit
# encoder
cf.encoder = cf.fit_encoder(X=train, y=train_labels, save_memory=False, verbose=True)
# SVM classifier training
features = cf.encode(X)
cf.classifier = cf.fit_classifier(features, y=train_labels)
'''


import math
import numpy
import torch
import sklearn
import sklearn.svm
import sklearn.externals
import sklearn.model_selection

import utils
import losses
import networks

import joblib


''' encoder
'''
# train
train_torch_dataset = utils.Dataset(train)
train_generator = torch.utils.data.DataLoader(train_torch_dataset, batch_size=10, shuffle=True)
# batch
for batch in train_generator:
    break
print(len([_ for _ in train_generator]))
# cf.loss
cf.loss = losses.triplet_loss.TripletLoss(
    compared_length=50, nb_random_samples=10, negative_penalty=1)
# loss parameters
train1 = torch.from_numpy(train)
train2 = train1.cuda(cf.gpu)
batch = batch.cuda(cf.gpu)
encoder = cf.encoder
# loss
loss = cf.loss(batch, encoder, train2, save_memory=False)
loss.backward()


''' loss
'''
batch_size = batch.size(0)
train_size = train2.size(0)
length = min(50, train2.size(2))

# For each batch element, we pick nb_random_samples possible random
# time series in the training set (choice of batches from where the
# negative examples will be sampled)
samples = numpy.random.choice(train_size, size=(10, batch_size))
samples = torch.LongTensor(samples)

# Choice of length of positive and negative samples
length_pos_neg = numpy.random.randint(1, high=length + 1)

# We choose for each batch example a random interval in the time
# series, which is the 'anchor'
# Length of anchors
random_length = numpy.random.randint(length_pos_neg, high=length + 1)
# Start of anchors
beginning_batches = numpy.random.randint(0, high=length - random_length + 1, size=batch_size)

# The positive samples are chosen at random in the chosen anchors
# Start of positive samples in the anchors
beginning_samples_pos = numpy.random.randint(
    0, high=random_length - length_pos_neg + 1, size=batch_size)  
# Start of positive samples in the batch examples
beginning_positive = beginning_batches + beginning_samples_pos
# End of positive samples in the batch examples
end_positive = beginning_positive + length_pos_neg

# We randomly choose nb_random_samples potential negative samples for
# each batch example
beginning_samples_neg = numpy.random.randint(
    0, high=length - length_pos_neg + 1, size=(10, batch_size))

# Anchors representations
representation = encoder(torch.cat(
    [batch[j: j + 1, :, beginning_batches[j]: beginning_batches[j] + random_length] 
     for j in range(batch_size)]
))

# Positive samples representations
positive_representation = encoder(torch.cat(
    [batch[j: j + 1, :, end_positive[j] - length_pos_neg: end_positive[j]] 
     for j in range(batch_size)]
))

size_representation = representation.size(1)
# Positive loss: -logsigmoid of dot product between anchor and positive
# representations
loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
    representation.view(batch_size, 1, size_representation),
    positive_representation.view(batch_size, size_representation, 1)
)))

multiplicative_ratio = 1 / 10
for i in range(10):
    # Negative loss: -logsigmoid of minus the dot product between
    # anchor and negative representations
    negative_representation = encoder(torch.cat(
        [train2[samples[i, j]: samples[i, j] + 1][
            :, :, beginning_samples_neg[i, j]: beginning_samples_neg[i, j] + length_pos_neg] 
        for j in range(batch_size)]
    ))
    # loss
    loss += multiplicative_ratio * -torch.mean(
        torch.nn.functional.logsigmoid(-torch.bmm(
            representation.view(batch_size, 1, size_representation),
            negative_representation.view(batch_size, size_representation, 1)
        ))
    )
print(loss)



''' variables
compared_length: 50
nb_random_samples: 10
negative_penalty: 1

encoder: cf.encoder
train/train2: torch.Size([40, 6, 100])
batch: torch.Size([10, 6, 100])
length: 50
samples: (10, 10)
length_pos_neg: 15 (pos and neg sampling length)
random_length: 20 (anchors sampling length)
beginning_batches: (10,)
beginning_samples_pos: (10,)
beginning_positive: (10,)
end_positive: (10,)
beginning_samples_neg: (10, 10)

representation: (10, 320) (1 times sampling from batch)
positive_representation: (10, 320) (1 times sampling from batch)
size_representation: 320
loss: Tensor(1.0330)
multiplicative_ratio: 0.1
negative_representation: (10, 320) (10 times sampling from train)
'''
