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

train, train_labels, test, test_labels = load_UEA_dataset(args.path, args.dataset)

# classifier = fit_hyperparameters(args.hyper, train, train_labels, args.cuda, args.gpu, 
#                                  save_memory=True)

classifier = scikit_wrappers.CausalCNNEncoderClassifier()
# Loads a given set of hyperparameters and fits a model with those
hf = open(os.path.join(args.hyper), 'r')
params = json.load(hf)
hf.close()
# Check the number of input channels
params['in_channels'] = numpy.shape(train)[1]
params['cuda'] = args.cuda
params['gpu'] = args.gpu
classifier.set_params(**params)
# fit
classifier.fit(train, train_labels, save_memory=False, verbose=True)

# accuracy
print("Test accuracy: " + str(classifier.score(test, test_labels)))






