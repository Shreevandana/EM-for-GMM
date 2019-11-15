#!/usr/bin/env python3
import numpy as np
if not __file__.endswith('_em_gaussian.py'):
    print('ERROR: This file is not named correctly! Please name it as LastName_em_gaussian.py (replacing LastName with your last name)!')
    exit(1)

DATA_PATH = "/u/cs246/data/em/" #TODO: if doing development somewhere other than the cycle server (not recommended), then change this to the directory where your data file is (points.dat)

def parse_data(args):
    num = float
    dtype = np.float32
    data = []
    with open(args.data_file, 'r') as f:
        for line in f:
            data.append([num(t) for t in line.split()])
    dev_cutoff = int(.9*len(data))
    train_xs = np.asarray(data[:dev_cutoff],dtype=dtype)
    dev_xs = np.asarray(data[dev_cutoff:],dtype=dtype) if not args.nodev else None
    return train_xs, dev_xs

def init_model(args,train_xs):
    clusters = []
    dimensions = train_xs.shape[1]
    
    if args.cluster_num:
        lambdas = np.zeros(args.cluster_num)
        mus = np.zeros((args.cluster_num,2))

        for i in range(0,args.cluster_num):
            lambdas[i] = 1/args.cluster_num

        for i in range(0,args.cluster_num):
            mus[i][0] = np.mean(train_xs) + np.random.rand()
            mus[i][1] = np.mean(train_xs) + np.random.rand()
        #TODO: randomly initialize clusters (lambdas, mus, and sigmas)
        if not args.tied:
            sigmas = np.array([np.eye(dimensions)] * args.cluster_num)
        else:
            sigmas = np.eye(dimensions)

        #raise NotImplementedError #remove when random initialization is implemented
    else:
        lambdas = []
        mus = []
        sigmas = []
        with open(args.clusters_file,'r') as f:
            for line in f:
                #each line is a cluster, and looks like this:
                #lambda mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1
                lambda_k, mu_k_1, mu_k_2, sigma_k_0_0, sigma_k_0_1, sigma_k_1_0, sigma_k_1_1 = map(float,line.split())
                lambdas.append(lambda_k)
                mus.append([mu_k_1, mu_k_2])
                sigmas.append([[sigma_k_0_0, sigma_k_0_1], [sigma_k_1_0, sigma_k_1_1]])
        lambdas = np.asarray(lambdas)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(lambdas)

    #TODO: do whatever you want to pack the lambdas, mus, and sigmas into the model variable (just a tuple, or a class, etc.)
    #NOTE: if args.tied was provided, sigmas will have a different shape
    model = [lambdas,mus,sigmas]
    #raise NotImplementedError #remove when model initialization is implemented
    return model

def train_model(model, train_xs, dev_xs, args):
    from scipy.stats import multivariate_normal
    sum = 0
    lambdas = model[0]
    mus = model[1]
    sigmas = model[2]
    n = train_xs.shape[0]
    #n_clusters = len(lambdas) #args.cluster_num
    eta = np.zeros([n,args.cluster_num])
    for i in range(0, args.iterations):
    
    ## E- STEP 
	    for i in range(0,n):
	    	for k in range(0,len(lambdas)):
	    		if args.tied:
	    			px = lambdas[k]*(multivariate_normal(mean=mus[k], cov=sigmas).pdf(train_xs[i]))
	    		else:
	    			px = lambdas[k]*(multivariate_normal(mean=mus[k], cov=sigmas[k]).pdf(train_xs[i]))
	    		eta[i][k] = px
	    
	    eta = eta/np.sum(eta, axis = 1).reshape((n, 1))
	    
	    ##M -STEP

	    for k in range(0,len(lambdas)):
	        lambdas[k] = np.sum(eta[:,k])/n
	        mus[k] = np.dot(eta[:, k].T, train_xs)/np.sum(eta[:, k])
	        if args.tied:
	        	difference = train_xs - mus[k]
	        	sigmas += np.dot(difference.T, eta[:, k].reshape((n, 1)) * difference)                
	        else:
	        	difference = train_xs - mus[k]
	        	sigmas[k] = np.dot(difference.T, eta[:, k].reshape((n, 1)) * difference)
	        	sigmas[k] = sigmas[k]/np.sum(eta[:, k])


    return model

def average_log_likelihood(model, data, args):
    from math import log
    from scipy.stats import multivariate_normal
    lambdas = model[0]
    mus = model[1]
    sigmas = model[2]
    #TODO: implement average LL calculation (log likelihood of the data, divided by the length of the data)
    n = data.shape[0]
    ll_sum = 0.0
    ll_log = 0.0

    for k in range(0,args.cluster_num):
        if args.tied:
            ll_sum += (lambdas[k]*(multivariate_normal(mean=mus[k], cov=sigmas).pdf(data)))
        else:
            ll_sum += (lambdas[k]*(multivariate_normal(mean=mus[k], cov=sigmas[k]).pdf(data)))
    ll_log = np.log(ll_sum)
    ll= np.sum(ll_log)

    #raise NotImplementedError #remove when average log likelihood calculation is implemented
    return (ll/n)

def extract_parameters(model):
    #TODO: extract lambdas, mus, and sigmas from the model and return them (same type and shape as in init_model)
    lambdas = None
    mus = None
    sigmas = None

    lambdas = model[0]
    mus = model[1]
    sigmas = model[2]
    
    #raise NotImplementedError #remove when parameter extraction is implemented
    return lambdas, mus, sigmas

def main():
    import argparse
    import os
    print('Gaussian') #Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points.')
    init_group = parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument('--cluster_num', type=int, help='Randomly initialize this many clusters.')
    init_group.add_argument('--clusters_file', type=str, help='Initialize clusters from this file.')
    parser.add_argument('--nodev', action='store_true', help='If provided, no dev data will be used.')
    parser.add_argument('--data_file', type=str, default=os.path.join(DATA_PATH, 'points.dat'), help='Data file.')
    parser.add_argument('--print_params', action='store_true', help='If provided, learned parameters will also be printed.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of EM iterations to perform')
    parser.add_argument('--tied',action='store_true',help='If provided, use a single covariance matrix for all clusters.')
    args = parser.parse_args()
    if args.tied and args.clusters_file:
        print('You don\'t have to (and should not) implement tied covariances when initializing from a file. Don\'t provide --tied and --clusters_file together.')
        exit(1)

    train_xs, dev_xs = parse_data(args)
    model = init_model(args,train_xs)
    #print("before training",model)
    model = train_model(model, train_xs, dev_xs, args)
    #print("after training",model)
    ll_train = average_log_likelihood(model, train_xs, args)
    print('Train LL: {}'.format(ll_train))
    if not args.nodev:
        ll_dev = average_log_likelihood(model, dev_xs, args)
        print('Dev LL: {}'.format(ll_dev))
    lambdas, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str,a))
        print('Lambdas: {}'.format(intersperse(' | ')(np.nditer(lambdas))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '),mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '),map(lambda s: np.nditer(s),sigmas)))))

if __name__ == '__main__':
    main()