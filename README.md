
# Implement Expectation Maximization (EM) fitting of a mixture of gaussians on a two-dimensional dataset.

# Task


Implement EM fitting of a mixture of gaussians on the two-dimensional data set points.dat. You should try different numbers of mixtures, as well as tied vs. separate covariance matrices for each gaussian. Use the final 1/10 of the data for dev. Plot likelihood on train and dev vs iteration for different numbers of mixtures.

# Files


"em_gmm.py" contains the code for EM for Gaussian Mixture Model "README.md"

# Algorithm


Gaussian Mixture Model using EM algorithm is implemented. It works for both tied and separate covariance matrices. We start by assigning K random gaussians and tied/separate covariance matrices. E and M steps are performed to update the means, covariances and lambda.

# Instructions for running "em_gmm.py"


To run the script "em_gmm.py" type "python3 em_gmm.py" in the commandline The default number of iterations (max_iter) is 15, clusters is 4 and it uses separate covariance matrices.

We can also specify the maximum iterations to run using the optional argument "--max_iter", the number of clusters using "--clusters" and whether to use tied covariance matrix using the optional argument "--tied". The algorithm will stop if the iterations reach the maximum number of iterations or "if the increase in log-likelihood on dev is less than 1e-6".

Please note that the data file "points.dat" should be kept in the same directory as the script.

The code for obtaining the plots is present in the script.

Note: Random seed can be set to a specific value to obtain same results across different runs.

# Results & Interpretation

The log-likelihood was increasing as the iterations progressed as expected. It was observed that the algorithm was performing better with separate covariance matrices. It was also observed that the no. of clusters = 4 was giving better log-likelihood on average.

Sample Output (Separate Covariance matrices): Clusters: 4 Reading data file: points.dat Dimensions of Train: (900, 2) Iteration 0 Iteration 5 Iteration 10 Log-Likelihoods on train: [-3442.9919991286756, -2881.9920115757559, -2847.9069751781194, -2824.3990310675858, -2810.6180114078893, -2802.9940491462426, -2798.3745489152216, -2794.9661425506088, -2791.9966812570228, -2789.1322396009964, -2786.2007389200094, -2783.1093960554995] Log-Likelihoods on dev: [-406.01658497354975, -339.02916996113345, -336.43969494074167, -334.41290169447774, -332.9722830576975, -331.99226238268847, -331.35555533495028, -330.96701295163786, -330.75465486284378, -330.66302445712898, -330.64585197717128, -330.65966119668229]

# References


This was done as a homework problem in the Machine Learning class (CSC 446, Spring 2019) by Prof. Daniel Gildea (https://www.cs.rochester.edu/~gildea/) at the University of Rochester, New York.
