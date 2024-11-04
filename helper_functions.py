import numpy as np
import copy


import time
from numpy.random import rand
from numpy import exp, log, mat
from numpy.linalg import norm

#conda install scipy
from scipy.io import loadmat
from scipy.linalg import svd 
from array import array
#conda install -c anaconda pandas
import pandas as pd
from scipy.stats import pearsonr # pearson correlation
#conda install -c conda-forge python-markdown-math
import math

from itertools import permutations #for finding matched topic index

import statsmodels.api as sm
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from scipy.stats import gmean #calc_c

from sklearn.preprocessing import normalize

import itertools # for generating correlaiton of words


def pos(A):
    A[np.where(A<0)] = 0
    return A

def neg(A):
    A[np.where(A>0)] = 0
    return -A

# initliaze U and V matrices
def NNDSVD(A,k):
    if len(A[np.where(A<0)]) > 0:
        print('the input matrix contains negative elements!')
    m,n = A.shape
    
    W = np.zeros((m,k))
    H = np.zeros((k,n))
    
    tmp=svd(A)
    U = tmp[0][:,0:k+1] #3*3
    S = tmp[1][0:k+1]  # 3*1
    V = tmp[2][0:k+1,:] #3*4
    S=np.diag(S)        #3*3
    
    W[:,0] = np.sqrt(S[0,0]) * abs(U[:,0])
    H[0,:] = np.sqrt(S[0,0]) * abs((V[0,:]))

    i_lst=range(2,k+1,1)
    for i in i_lst:
        uu = copy.deepcopy(U[:,i-1])
        vv = copy.deepcopy(V[i-1,:])
        uu1 = copy.deepcopy(U[:,i-1])
        vv1 = copy.deepcopy(V[i-1,:])
        uup = pos(uu)
        uun = neg(uu1) 
        vvp = pos(vv)
        vvn = neg(vv1)
        n_uup = norm(uup)
        n_vvp = norm(vvp) 
        n_uun = norm(uun) 
        n_vvn = norm(vvn) 
        termp = n_uup*n_vvp
        termn = n_uun*n_vvn
        if (termp >= termn):
            W[:,i-1] = np.sqrt(S[i-1,i-1]*termp)*uup/n_uup 
            H[i-1,:] = np.sqrt(S[i-1,i-1]*termp)*vvp.T/n_vvp
        else:
            W[:,i-1] = np.sqrt(S[i-1,i-1]*termn)*uun/n_uun 
            H[i-1,:] = np.sqrt(S[i-1,i-1]*termn)*vvn.T/n_vvn
    W[np.where(W<0.0000000001)]=0.1;
    H[np.where(H<0.0000000001)]=0.1;
    return (W,H)

# helper function
#If x is 1D, convert it to a column vector. If `return_oneD`, 
# will  also return whether or not the input was 1D.
def vec2col(x,dtype=None,return_oneD=False):
    x = np.asarray(x,dtype=dtype)
    if len(x.shape) > 1:
        oneD = False
    else:
        x = np.atleast_2d(x).T
        oneD = True
    if return_oneD:
        return x,oneD
    return x

def error_plot(ys, yscale='log'):
    plt.figure(figsize=(8, 8))
    plt.xlabel('Step')
    plt.ylabel('Error')
    plt.yscale(yscale)
    plt.plot(range(len(ys)), ys, **kwargs)
    


def nearest_positive_semidefinite(A):
    """Ensure matrix is positive semi-definite using Higham's method, handling complex numbers explicitly."""
    B = (A + A.T) / 2  # Symmetrize the matrix
    _, s, V = np.linalg.svd(B)  # Singular value decomposition
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A_psd = (B + H) / 2  # Initial approximation
    A_psd = (A_psd + A_psd.T) / 2  # Ensure symmetry

    # Enforce real values by eliminating any small imaginary parts
    A_psd = np.real(A_psd)

    # Check for complex eigenvalues and force them to be real
    eigvals = np.linalg.eigvals(A_psd)
    min_eig_real = np.min(np.real(eigvals))  # Take only the real part of the eigenvalues

    if min_eig_real < 0:
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        A_psd += I * (-min_eig_real + spacing)  # Adjust by adding to the diagonal

    return A_psd
    
# def generate_data(eta=1, epsilon=1, n=2000, r=5, m1=10, m2=100, s=2, seed=1, return_val=1):
#     # eta: noise of X2
#     # epsilon: noise of Y
#     # n: sample size
#     # r: number of topics
#     # m1: number of EHR features
#     # m2: number of words
#     # s: number of signal beta's
#     # seed: random seed
#     # return_val: 1 (return data), 2 (return V matrix), 3(return U matrix)
#     np.random.seed(seed)
#     method = "lee"
    
#     # Step 1: Coefficient matrix U (document-topic matrix)
#     index = np.random.rand(n)
#     class_vector = np.zeros(n, dtype=int)
    
#     for i in range(n):
#         index_i = index[i]
#         for l in range(1, r + 1):
#             l_bound = (l - 1) / r
#             u_bound = l / r
#             if l_bound < index_i < u_bound:
#                 class_vector[i] = l
    
#     dist_matrix = np.ones((r, r))
#     np.fill_diagonal(dist_matrix, 10)
    
#     U = np.empty((n, r))
#     for i in range(n):
#         class_i = class_vector[i]
#         d_i = dist_matrix[class_i - 1, :]
#         dirichlet = np.random.dirichlet(d_i, size=1)
#         U[i, :] = dirichlet
    
#     U = pd.DataFrame(U, columns=[f"topic_{i+1}" for i in range(r)])
    
#     # Step 2: Basis matrix V (topic-word matrix)
#     V = np.zeros((r, m2))
#     for l in range(1, r + 1):
#         l_bound = ((l - 1) / r) * m2
#         u_bound = (l / r) * m2
#         V[l - 1, int(l_bound):int(u_bound)] = np.random.uniform(0, 1, size=int(m2 / r))
    
#     # Step 3: Noise
#     mu_e = np.zeros(m2)
#     sigma_e = np.eye(m2)
#     e = np.random.multivariate_normal(mu_e, sigma_e, n)
    
#     # Step 4: Generate X2 from UV adding noise
#     X2 = np.dot(U, V) + eta * e
    
#     # Step 5: Force <= 0 values to be a very small positive number
#     X2[X2 <= 0] = 0.00001
    
#     X2 = pd.DataFrame(X2, columns=[f"word_{i+1}" for i in range(m2)])
    
#     # Step 6: Generate X1 (EHR data)
#     mu = np.repeat([1, 10], [7, m1 - 7]) #repeat values 1 and 10, repeat "1" 7 times and 10 m1-7 times
#     sigma = np.eye(m1)
#     X1 = np.random.multivariate_normal(mu, sigma, n)
#     X1 = pd.DataFrame(X1, columns=[f"EHR_{i+1}" for i in range(m1)])
    
#     # Step 7: Generate W
#     W_random = np.random.binomial(1, 0.5, n)
    
#     # Step 8: Generate Y
#     matrix = np.hstack((X1, U.values))
#     beta_true = np.zeros(m1 + r) 
#     beta_true[m1: (m1+s)] = 3
#     y = np.dot(matrix, beta_true)
    
#     # Noise
#     e = np.random.normal(0, 1, n)
#     y += e * epsilon
    
#     # Step 9: Finalize data
#     data = pd.concat([X1, X2, pd.DataFrame({'W_random': W_random}), U], axis=1)
#     data['y'] = y
    
#     if return_val == 1:
#         return data
#     elif return_val == 2:
#         return pd.DataFrame(V, columns=[f"word_{i+1}" for i in range(m2)])
#     elif return_val == 3:
#         return U

def generate_data_dlearning(seed, n, r, m1, m2, main_topic_signal, topic_weights, eta, epsilon, zeta1, zeta2,gamma1, gamma2, beta0, psi1, psi2, beta1, beta2, beta3, beta4, beta5, rho1,rho2, rho3, v_signal_val, scenario, sparse, correlation_type):
    """Projected gradient descent.
    Inputs:
        seed: random seed
        n: sample size
        r: number of topics
        m1: number of EHR features
        m2: number of words
        eta: noise of X2
        epsilon: noise of Y
        zeta1: effect of X1 on m(x)
        zeta2: effect of X2 on m(x)
        gamma1: effect of U1 on treatment assignment
        gamma2: effect of U2 on treatment assignment
        beta0, psi1, psi2, beta3, beta4: delta_x = beta0 + psi1*X1 + psi2*X2+ beta3*U1 + beta4*U2
        A_noise: noise generating treatment assignment A. p + Unif(-A_noise, A_noise)
        return_val: 1 (return data), 2 (return V matrix), 3(return U matrix)
    Outputs:
        data/V matrix/U matrix
    """
    
    np.random.seed(seed)
    #############
    # Step 1: Coefficient matrix U (document-topic matrix)
    #############
    index = np.random.rand(n) #generate n random numbers, ranges from 0 to 1
    class_vector = np.zeros(n, dtype=int) #generate n zeros, used to store class labels

    # generate an Identity matrix of dimension r x r, with diagnols of 10
    dist_matrix = np.ones((r, r))
    np.fill_diagonal(dist_matrix, main_topic_signal) 
    
    # Generate class labels with weighted probabilities
    rng = np.random.RandomState(0) #make sure class_vector is the same in training, validation and test
    class_vector = rng.choice(np.arange(1, r + 1), size=n, p=topic_weights/np.sum(topic_weights))


    # intercept --> 1 (NO NEED FOR THIS INTERCEPT)
    U0 = np.empty((n, 1))
    U0 = pd.DataFrame(U0, columns=["topic_0"]) 
    U0["topic_0"] = 1

    U = np.empty((n, r)) # initialize U (document-topic) matrix of dimension n x r
    # generate U matrix - dirichlet distributed. each row (document) adds up to 1, with higher probability to be in one class, lower probability to be in the remaining (r-1) classes.

    for i in range(n):
        class_i = class_vector[i]
        d_i = dist_matrix[class_i - 1, :]
        dirichlet = np.random.dirichlet(d_i, size=1)
        U[i, :] = dirichlet 
#     elif sparse == True: # except for the signal topic, all remaining values are zero
#         for i in range(n):
#             class_i = class_vector[i]
#             d_i = dist_matrix[class_i - 1, :]
#             dirichlet = np.random.dirichlet(d_i, size=1).flatten()

#             # Find the index of the dominating column
#             dominant_index = np.argmax(dirichlet)

#             # Set all other values to 0
#             U[i, :] = np.zeros_like(dirichlet)
#             U[i, dominant_index] = dirichlet[dominant_index]

    #without, dimension: n x r
    U = pd.DataFrame(U, columns=[f"topic_{i+1}" for i in range(r)]) #rename topic column
    
    #with intercept, dimension: nx(r+1)
    U_int = pd.concat([U0, U], axis=1) 

    
    #############
    # Step 2: Basis matrix V (topic-word matrix)
    #############
    
    
    if scenario == "A":
        V = np.zeros((r, m2)) # initialize V (topic-word) matrix of dimension r x m2
        if sparse == False:
            V = np.random.uniform(0, 1, (r, m2))
        elif sparse == True:
            V= V

        #Each topic only have m2/r number of words are non-zero. These words aredistributed Uniform(0,5)
        for l in range(1, r + 1):
            l_bound = ((l - 1) / r) * m2
            u_bound = (l / r) * m2
            #V[l - 1, int(l_bound):int(u_bound)] = np.random.gamma(1, 0.1, size=int(m2 / r))
            V[l - 1, int(l_bound):int(u_bound)] = np.random.uniform(1, 5, size=int(m2 / r))
        
        #V[3, 40:60] = np.random.uniform(0, 2, size=int(m2 / r))
        V[3, 60:80] = np.random.uniform(0, 7, size=int(m2 / r))
        V[4, 80:100] = np.random.uniform(0, 7, size=int(m2 / r))
        
#         V[0,0] = v_signal_val/2 #word 1 (say we have word 1 to word 100)
#         V[0,1] = v_signal_val/2 #word 2
#         V[0,2] = v_signal_val/2 #word 3
#         V[0,3] = v_signal_val/2 #word 4
#         V[0,4] = v_signal_val/2 #word 5
#         V[0,5] = v_signal_val/2 #word 6
#         V[0,6] = v_signal_val/2 #word 3
#         V[0,7] = v_signal_val/2 #word 4
#         V[0,8] = v_signal_val/2 #word 3
#         V[0,9] = v_signal_val/2 #word 4
        
        V[0,0] = np.random.uniform(5, 7, size=1)/2
        V[0,1] = np.random.uniform(5, 7, size=1)/2
        V[0,2] = np.random.uniform(5, 7,size=1)/2
        V[0,3] = np.random.uniform(5, 7, size=1)/2
        V[0,4] = np.random.uniform(5, 7, size=1)/2

#         V[1,20] = v_signal_val/2 #word 21 
#         V[1,21] = v_signal_val/2 #word 22
#         V[1,22] = v_signal_val/2 #word 21 
#         V[1,23] = v_signal_val/2 #word 22
#         V[1,24] = v_signal_val/2 #word 21 
#         V[1,25] = v_signal_val/2 #word 22
#         V[1,26] = v_signal_val/2 #word 23
#         V[1,27] = v_signal_val/2 #word 24
#         V[1,28] = v_signal_val/2 #word 22
#         V[1,29] = v_signal_val/2 #word 23
        V[1,20] = np.random.uniform(5, 7,size=1)/2
        V[1,21] = np.random.uniform(5, 7, size=1)/2
        V[1,22] = np.random.uniform(5, 7, size=1)/2
        V[1,23] = np.random.uniform(5, 7, size=1)/2
        V[1,24] = np.random.uniform(5, 7,size=1)/2
        

#         V[2,40] = v_signal_val/2 #word 41 
#         V[2,41] = v_signal_val/2 #word 42
#         V[2,42] = v_signal_val/2 #word 43
#         V[2,43] = v_signal_val/2 #word 44
        V[2,40] = np.random.uniform(5, 7, size=1)/2
        V[2,41] = np.random.uniform(5, 7, size=1)/2
        V[2,42] = np.random.uniform(5, 7,size=1)/2
        V[2,43] = np.random.uniform(5, 7, size=1)/2
        V[2,40] = np.random.uniform(5, 7, size=1)/2


 #         V[3,60] = v_signal_val #word 61 
#         V[3,61] = v_signal_val #word 62
#         V[3,62] = v_signal_val #word 63
#         V[3,63] = v_signal_val #word 64
       

#         V[4,80] = v_signal_val #word 81 
#         V[4,81] = v_signal_val #word 82
#         V[4,82] = v_signal_val #word 83
#         V[4,83] = v_signal_val #word 84
      

        # Define the indices that need to be set to v_signal_val/2
        topic1_indices = list(range(0, 5))  # Words 1 to 10
        topic2_indices = list(range(20, 25))  # Words 1 to 10
            
        # Set the values for both blocks in row 2 of V
#         V[2, topic1_indices] = v_signal_val/2
#         V[2, topic2_indices] = v_signal_val/2
    
        V[2, topic1_indices] = np.random.uniform(5, 7, size= len(topic1_indices))/2
        V[2, topic2_indices] = np.random.uniform(5, 7, size= len(topic1_indices))/2
        
        
    elif scenario == "C": #Scenario: 6 topics, 600 words
        V = np.zeros((r, m2)) # initialize V (topic-word) matrix of dimension r x m2
        V = np.random.uniform(0, 1, (r, m2))

        #Each topic only have m2/r number of words are non-zero. These words aredistributed Uniform(0,1)
        for l in range(1, r + 1):
            l_bound = ((l - 1) / r) * m2
            u_bound = (l / r) * m2
            V[l - 1, int(l_bound):int(u_bound)] = np.random.uniform(0, 5, size=int(m2 / r))
        rows = range(r)
        cols_start = [0, 70, 140, 210, 280, 350]

        # Assign values using nested loops
        for row in rows:
            for col_offset in range(10): #10 = number of signal words in each topic
                V[row, cols_start[row] + col_offset] = v_signal_val
          
    #################################################
    ############Scenario 1 -- Underfitted ############
    #################################################
    elif scenario == "D": #Scenario: 6 topics,120 words
        V = np.zeros((r, m2)) # initialize V (topic-word) matrix of dimension r x m2
        if sparse == False:
            V = np.random.uniform(0, 1, (r, m2))
        elif sparse == True:
            V = V

        #Each topic only have m2/r number of words are non-zero. These words aredistributed Uniform(0,5)
        for l in range(1, r + 1):
            l_bound = ((l - 1) / r) * m2
            u_bound = (l / r) * m2
            V[l - 1, int(l_bound):int(u_bound)] = np.random.uniform(0, 5, size=int(m2 / r))
            #V[l - 1, int(l_bound):int(u_bound)] = np.random.gamma(1, 5, size=int(m2 / r))

#         rows = range(r)
#         cols_start = [0, 20, 40, 60, 80, 100]

#         # Assign values using nested loops
#         for row in rows:
#             for col_offset in range(4): #5 = number of signal words in each topic
#                 V[row, cols_start[row] + col_offset] = v_signal_val

    elif scenario == "E": #Scenario: 6 topics, 120 words
        V = np.zeros((r, m2)) # initialize V (topic-word) matrix of dimension r x m2
        V = np.random.uniform(0, 1, (r, m2))

        #Each topic only have m2/r number of words are non-zero. These words aredistributed Uniform(0,1)
        for l in range(1, r + 1):
            l_bound = ((l - 1) / r) * m2
            u_bound = (l / r) * m2
            V[l - 1, int(l_bound):int(u_bound)] = np.random.uniform(0, 5, size=int(m2 / r))
        rows = range(r)
        cols_start = [0, 20, 40, 60, 80, 100]

        # Assign values using nested loops
        for row in rows:
            for col_offset in range(5): #5 = number of signal words in each topic
                V[row, cols_start[row] + col_offset] = v_signal_val
    elif scenario == "F": #Sparce Scenario: 5 topics, 400 words
        V = np.zeros((r, m2)) # initialize V (topic-word) matrix of dimension r x m2
        V = np.random.uniform(0, 0.01, (r, m2))

        #Each topic only have m2/r number of words are non-zero. These words aredistributed Uniform(0,1)
        for l in range(1, r + 1):
            l_bound = ((l - 1) / r) * m2
            u_bound = (l / r) * m2
            V[l - 1, int(l_bound):int(u_bound)] = np.random.uniform(0, 0.1, size=int(m2 / r))
        rows = range(r)
        cols_start = [0, 80, 160, 240, 320]

        # Assign values using nested loops
        for row in rows:
            for col_offset in range(5): #5 = number of signal words in each topic
                V[row, cols_start[row] + col_offset] = v_signal_val
                
    norm = np.linalg.norm(V) #norm of the whole matrix
    V = V/norm
#     norms = np.linalg.norm(V, axis=1, keepdims=True) #L2 norm of each V vector: ||V_i||_2
#     V = V / norms

    ############# 
    # Step 3: Generate X2 from UV adding noise e
    #############
    mu_e = np.zeros(m2) #initialize 
    sigma_e = np.eye(m2) # creating a diagnal matrix. 
    
    # Add correlation
    ###############################
    #####Correlation Type 1 #######
    ###############################
    if correlation_type == 1:
        if scenario == "A":
            correlated_blocks = [(1, 5), (21, 25), (41, 45), (61, 65), (81, 85)]
        elif scenario == "D":
            correlated_blocks = [(1, 5), (21, 25), (41, 45), (61, 65), (81, 85), (101, 105)]
        for block in correlated_blocks:
            i, j = block
            corr_value = 0.8  # Example correlation value, you can adjust this
            sigma_e[i-1:j, i-1:j] = corr_value  # Set correlations within the block

            # Make diagonal elements in the block 1 (variance = 1)
            np.fill_diagonal(sigma_e[i-1:j, i-1:j], 1)
    ###############################
    #####Correlation Type 2 #######
    ###############################
    elif correlation_type == 2:
        if scenario == "A":
            correlated_blocks = [(1, 15), (21, 35), (41, 55), (61, 75), (81, 95)]
        elif scenario == "D":
            correlated_blocks = [(1, 15), (21, 35), (41, 55), (61, 75), (81, 95), (101, 115)]
        for block in correlated_blocks:
            i, j = block
            corr_value = 0.8  # Example correlation value, you can adjust this
            sigma_e[i-1:j, i-1:j] = corr_value  # Set correlations within the block

            # Make diagonal elements in the block 1 (variance = 1)
            np.fill_diagonal(sigma_e[i-1:j, i-1:j], 1)

    ###############################
    #####Correlation Type 3 #######
    ###############################
    elif correlation_type == 3:
        if scenario == "A":
            correlated_blocks = [(1, 5), (21, 25), (41, 45), (61, 65), (81, 85)]
        elif scenario == "D":
            correlated_blocks = [(1, 5), (21, 25), (41, 45), (61, 65), (81, 85), (101, 105)]
        for block in correlated_blocks:
            i, j = block
            corr_value = 0.8  # Example correlation value, you can adjust this
            sigma_e[i-1:j, i-1:j] = corr_value  # Set correlations within the block

            # Make diagonal elements in the block 1 (variance = 1)
            np.fill_diagonal(sigma_e[i-1:j, i-1:j], 1)

        #####################
        ##################### Correlation accross topics
        source_words = [1, 2, 3]
        target_blocks = [(21, 23)]
        corr_value_across_blocks = 0.8  # Example correlation value for across-block correlations

        for source_word in source_words:
            for block in target_blocks:
                i, j = block
                for target_word in range(i, j+1):  # Loop through target words in each block
                    sigma_e[source_word-1, target_word-1] = corr_value_across_blocks  # Correlate source with target
                    sigma_e[target_word-1, source_word-1] = corr_value_across_blocks  # Ensure symmetry
       #####################################
        source_words = [6, 7, 8, 9, 10]
        target_blocks = [(46, 50), (66, 70)]
        corr_value_across_blocks = 0.2 # Example correlation value for across-block correlations
        for source_word in source_words:
            for block in target_blocks:
                i, j = block
                for target_word in range(i, j+1):  # Loop through target words in each block
                    sigma_e[source_word-1, target_word-1] = corr_value_across_blocks  # Correlate source with target
                    sigma_e[target_word-1, source_word-1] = corr_value_across_blocks  # Ensure symmetry
                    
        #####################################           
        source_words = [46, 47, 48]
        target_blocks = [(66, 70), (86, 90)]
        corr_value_across_blocks = 0.5  # Example correlation value for across-block correlations

        for source_word in source_words:
            for block in target_blocks:
                i, j = block
                for target_word in range(i, j+1):  # Loop through target words in each block
                    sigma_e[source_word-1, target_word-1] = corr_value_across_blocks  # Correlate source with target
                    sigma_e[target_word-1, source_word-1] = corr_value_across_blocks  # Ensure symmetry

    ###############################
    #####Correlation Type 4 #######
    ###############################
    elif correlation_type == 4:
        if scenario == "A":
            correlated_blocks = [(1, 15), (21, 35), (41, 60), (61, 75), (81, 95)]
        elif scenario == "D":
            correlated_blocks = [(1, 15), (21, 35), (41, 60), (61, 75), (81, 95), (101, 115)]
        for block in correlated_blocks:
            i, j = block
            corr_value = 0.8  # Example correlation value, you can adjust this
            sigma_e[i-1:j, i-1:j] = corr_value  # Set correlations within the block

            # Make diagonal elements in the block 1 (variance = 1)
            np.fill_diagonal(sigma_e[i-1:j, i-1:j], 1)

        #####################
        ##################### Correlation accross topics
        source_words = [1, 2, 3, 4, 5]
        target_blocks = [(21, 25)]
        corr_value_across_blocks = 0.9  # Example correlation value for across-block correlations

        for source_word in source_words:
            for block in target_blocks:
                i, j = block
                for target_word in range(i, j+1):  # Loop through target words in each block
                    sigma_e[source_word-1, target_word-1] = corr_value_across_blocks  # Correlate source with target
                    sigma_e[target_word-1, source_word-1] = corr_value_across_blocks  # Ensure symmetry
       #####################################
        source_words = [6, 7, 8, 9, 10]
        target_blocks = [(46, 55), (66, 75)]
        corr_value_across_blocks = 0.5 # Example correlation value for across-block correlations
        for source_word in source_words:
            for block in target_blocks:
                i, j = block
                for target_word in range(i, j+1):  # Loop through target words in each block
                    sigma_e[source_word-1, target_word-1] = corr_value_across_blocks  # Correlate source with target
                    sigma_e[target_word-1, source_word-1] = corr_value_across_blocks  # Ensure symmetry
                    
        #####################################           
        source_words = [46, 47, 48, 49, 50]
        target_blocks = [(66, 80), (86, 100)]
        corr_value_across_blocks = 0.5  # Example correlation value for across-block correlations

        for source_word in source_words:
            for block in target_blocks:
                i, j = block
                for target_word in range(i, j+1):  # Loop through target words in each block
                    sigma_e[source_word-1, target_word-1] = corr_value_across_blocks  # Correlate source with target
                    sigma_e[target_word-1, source_word-1] = corr_value_across_blocks  # Ensure symmetry

    ###############################
    #####Correlation Type 5 #######
    ###############################
    elif correlation_type == 5:
        if scenario == "A":
            correlated_blocks = [(1, 15), (21, 35), (41, 55), (61, 75), (81, 95)]
        elif scenario == "D":
            correlated_blocks = [(1, 15), (21, 35), (41, 55), (61, 75), (81, 95), (101, 115)]
        for block in correlated_blocks:
            i, j = block
            corr_value = 0.6  # Example correlation value, you can adjust this
            sigma_e[i-1:j, i-1:j] = corr_value  # Set correlations within the block

            # Make diagonal elements in the block 1 (variance = 1)
            np.fill_diagonal(sigma_e[i-1:j, i-1:j], 1)

        #####################
        ##################### Correlation accross topics
        source_words = [1, 2, 3, 4, 5]
        target_blocks = [(21, 25)]
        corr_value_across_blocks = 0.9  # Example correlation value for across-block correlations

        for source_word in source_words:
            for block in target_blocks:
                i, j = block
                for target_word in range(i, j+1):  # Loop through target words in each block
                    sigma_e[source_word-1, target_word-1] = corr_value_across_blocks  # Correlate source with target
                    sigma_e[target_word-1, source_word-1] = corr_value_across_blocks  # Ensure symmetry
        #####################################
        source_words = [6, 7, 8, 9, 10]
        target_blocks = [(66, 75), (86,95)]
        corr_value_across_blocks = 0.4 # Example correlation value for across-block correlations
        for source_word in source_words:
            for block in target_blocks:
                i, j = block
                for target_word in range(i, j+1):  # Loop through target words in each block
                    sigma_e[source_word-1, target_word-1] = corr_value_across_blocks  # Correlate source with target
                    sigma_e[target_word-1, source_word-1] = corr_value_across_blocks  # Ensure symmetry
                    
        #####################################           
        source_words = [46, 47, 48, 49, 50, 51,52,53,54,55,56,57,58,59]
        if scenario == "A":
            target_blocks = [(6,20), (26, 40),(66, 80), (86, 100)]
        elif scenario == "D":
            target_blocks = [(6,20), (26, 40),(66, 80), (86, 100), (106, 120)]
        corr_value_across_blocks = 0.9  # Example correlation value for across-block correlations

        for source_word in source_words:
            for block in target_blocks:
                i, j = block
                for target_word in range(i, j+1):  # Loop through target words in each block
                    sigma_e[source_word-1, target_word-1] = corr_value_across_blocks  # Correlate source with target
                    sigma_e[target_word-1, source_word-1] = corr_value_across_blocks  # Ensure symmetry

               

    ##########################################################################################
    #####Correlation Type 6 (For Scenario A, correlation type 6 is the same as type 5) #######
    ##########################################################################################
    elif correlation_type == 6:
        if scenario == "A":
            correlated_blocks = [(1, 15), (21, 35), (41, 55), (61, 75), (81, 95)]

            for block in correlated_blocks:
                i, j = block
                corr_value = 0.6  # Example correlation value, you can adjust this
                sigma_e[i-1:j, i-1:j] = corr_value  # Set correlations within the block

                # Make diagonal elements in the block 1 (variance = 1)
                np.fill_diagonal(sigma_e[i-1:j, i-1:j], 1)

            #####################
            ##################### Correlation accross topics
            source_words = [1, 2, 3, 4, 5]
            target_blocks = [(21, 25)]
            corr_value_across_blocks = 0.9  # Example correlation value for across-block correlations

            for source_word in source_words:
                for block in target_blocks:
                    i, j = block
                    for target_word in range(i, j+1):  # Loop through target words in each block
                        sigma_e[source_word-1, target_word-1] = corr_value_across_blocks  # Correlate source with target
                        sigma_e[target_word-1, source_word-1] = corr_value_across_blocks  # Ensure symmetry
            #####################################
            source_words = [6, 7, 8, 9, 10]
            target_blocks = [(66, 75), (86,95)]
            corr_value_across_blocks = 0.4 # Example correlation value for across-block correlations
            for source_word in source_words:
                for block in target_blocks:
                    i, j = block
                    for target_word in range(i, j+1):  # Loop through target words in each block
                        sigma_e[source_word-1, target_word-1] = corr_value_across_blocks  # Correlate source with target
                        sigma_e[target_word-1, source_word-1] = corr_value_across_blocks  # Ensure symmetry

            #####################################           
            source_words = [46, 47, 48, 49, 50, 51,52,53,54,55,56,57,58,59]
            target_blocks = [(6,20), (26, 40),(66, 80), (86, 100)]
            corr_value_across_blocks = 0.9  # Example correlation value for across-block correlations

            for source_word in source_words:
                for block in target_blocks:
                    i, j = block
                    for target_word in range(i, j+1):  # Loop through target words in each block
                        sigma_e[source_word-1, target_word-1] = corr_value_across_blocks  # Correlate source with target
                        sigma_e[target_word-1, source_word-1] = corr_value_across_blocks  # Ensure symmetry

        elif scenario == "D":
            correlated_blocks = [(1, 15), (21, 35), (41, 55), (61, 75), (81, 95), (101, 115)]

            for block in correlated_blocks:
                i, j = block
                corr_value = 0.4  # Example correlation value, you can adjust this
                sigma_e[i-1:j, i-1:j] = corr_value  # Set correlations within the block

                # Make diagonal elements in the block 1 (variance = 1)
                np.fill_diagonal(sigma_e[i-1:j, i-1:j], 1)
                
            #####################################
            source_words = range(1, 16)  # Words 1 to 15
            target_range = range(66, 80)  # Words 66 to 75
            corr_value_across_blocks = 0.95  # Example correlation value

            # Set correlations for all pairs of source and target words
            for source_word, target_word in itertools.product(source_words, target_range):
                sigma_e[source_word - 1, target_word - 1] = corr_value_across_blocks
                sigma_e[target_word - 1, source_word - 1] = corr_value_across_blocks  # Ensure symmetry
        
            #####################################
            source_words = range(21, 36)  # Words 1 to 15
            target_range = range(86, 100)  # Words 66 to 75
            corr_value_across_blocks = 0.95  # Example correlation value

            # Set correlations for all pairs of source and target words
            for source_word, target_word in itertools.product(source_words, target_range):
                sigma_e[source_word - 1, target_word - 1] = corr_value_across_blocks
                sigma_e[target_word - 1, source_word - 1] = corr_value_across_blocks  # Ensure symmetry

            #####################################
            source_words = range(41, 66)  # Words 1 to 15
            target_range = range(106, 120)  # Words 66 to 75
            corr_value_across_blocks = 0.95 # Example correlation value

            # Set correlations for all pairs of source and target words
            for source_word, target_word in itertools.product(source_words, target_range):
                sigma_e[source_word - 1, target_word - 1] = corr_value_across_blocks
                sigma_e[target_word - 1, source_word - 1] = corr_value_across_blocks  # Ensure symmetry
                    
    # Make sure the covariance matrix is positive semi-definite
    sigma_e = nearest_positive_semidefinite(sigma_e)

    e = np.random.multivariate_normal(mu_e, sigma_e, n) 
    X2 = np.dot(U, V) + eta * e # variance: eta
    X2 = pd.DataFrame(X2, columns=[f"word_{i+1}" for i in range(m2)]) #rename words columns

    #############
    # Step 4: Force <= 0 values to be a very small positive number
    #############
    #X2[X2 <= 0] = 1e-10
    X2[X2 <= 0] = 0
    #############
    # Step 5: Generate X1 (EHR data)
    #############
    mu = np.repeat(0, m1) #repeat values 1 and 10, repeat "1" half of the times and 10 half of the times
    sigma = np.eye(m1)
    X1 = np.random.multivariate_normal(mu, sigma, n)
    X1 = pd.DataFrame(X1, columns=[f"EHR_{i+1}" for i in range(m1)])
    
    X = pd.concat([X1, X2], axis=1) #column bind X1 and X2
    
    #############
    # Step 6: m(X) --mean effect of covariates [X1 for both TMT
    #############
    matrix = np.hstack((X1, U.values))
    zeta_true = np.zeros(m1 + r) 

    zeta_true[1] = zeta1 #these values doesn't affect the treatment effect
    zeta_true[2] = zeta2
    
    zeta_true = vec2col(zeta_true)
    m_x = np.dot(matrix, zeta_true)
    
    #############
    # Step 7. generate delta(x) -- effect of treatment A  (1/2 of ITR d(x))
    #############
    beta_true = np.zeros(m1 + r) 
    beta_true[0] = psi1
    beta_true[1] = psi2
    beta_true[m1] = beta1
    beta_true[m1+1] = beta2
    beta_true[m1+2] = beta3
    beta_true[m1+3] = beta4
    beta_true[m1+4] = beta5
    
    beta_true = vec2col(beta_true)
   
    # words
#     X2_1 = X2["word_1"]
#     X2_2 = X2["word_2"]
#     X2_21 = X2["word_21"]
#     X2_22 = X2["word_22"]
#     X2_41 = X2["word_41"]
#     X2_42 = X2["word_42"]
#     X2_61 = X2["word_61"]
#     X2_62 = X2["word_62"]
    #interaction = rho1*X2_41*X2_61 + rho2*X2_41 + rho3*X2_61 
    #interaction = rho1*X2_41*X2_42*X2_61*X2_62  + rho2*U['topic_1'] * U['topic_2']
    #delta_x =  beta0 + np.dot(matrix, beta_true) + vec2col(interaction) 
    delta_x =  np.dot(matrix, beta_true)
    #############
    # Step 8. generate d(x) -- 
    #############
    d_x = np.where(2*delta_x >0, 1, -1)

    #############
    # Step 9. Generate A -- treatment assignment {1, -1}
    #############
    # Let the treatment assignment be dependent on two EHR features (Ex. age, sex)
    gamma = np.zeros(m1+r)
    gamma[0] = gamma1 #the first EHR 
    gamma[1] = gamma2 #the second EHR

    Xgamma = np.dot(matrix, gamma)
    p_vector = 1/(1+np.exp(-Xgamma)) #or: = np.exp(Xgamma)/(np.exp(Xgamma)+ 1)
    p_vector = vec2col(p_vector)
       
    A = []
    for p_i in p_vector:
        p_i= float(p_i)
        # P(A=1) = p, P(A=-1) = 1-p
        A_i = np.random.choice([1, -1], size=1, p=[p_i, 1 - p_i]) 
        A.append(A_i)
    A = np.array(A)
 
    #############
    # Step 9. Generate Y with error e
    #############
    e = np.random.normal(0, 1, n)
    e= vec2col(e)
    y = m_x + delta_x * A +   e * epsilon #epsilon controls the magnitude of the error
    diff =2 * y * A - delta_x
    rmse_min = np.sqrt(np.linalg.norm(diff**2))

    #############
    # Step 10. Final data
    #############
    data = pd.concat([X1, X2, U_int, 
                 pd.DataFrame(p_vector, columns= ["p"]),
                 pd.DataFrame(A, columns=["A"]),
                 pd.DataFrame(delta_x, columns=["delta_x"]),
                 pd.DataFrame(d_x, columns=["d_x"])], axis=1)
    data['y'] = y
    V_df =  pd.DataFrame(V, columns=[f"word_{i+1}" for i in range(m2)])
    #if return_val == 1:
    #    return data
    #elif return_val == 2:
    #    return pd.DataFrame(V, columns=[f"word_{i+1}" for i in range(m2)])
    #elif return_val == 3:
    #    return U
    return(data, V_df, U, rmse_min)
  
# Example usage:
# Call the function with the appropriate parameters and receive the desired output.
# For example, to get the dataset:
# data = generate_data()

# To get V:
# V = generate_data(return_val=2)

# To get U:
# U = generate_data(return_val=3)


# def generate_data_robinson(eta=1, epsilon=1, n=2000, r=5, m1=10, m2=100, s=2, seed=1, return_val=1):
    
#     # eta: noise of X2
#     # epsilon: noise of Y
#     # n: sample size
#     # r: number of topics
#     # m1: number of EHR features
#     # m2: number of words
#     # s: number of signal beta's
#     # seed: random seed
#     # return_val: 1 (return data), 2 (return V matrix), 3(return U matrix)
#     np.random.seed(seed)
#     method = "lee"
    
#     # Step 1: Coefficient matrix U (document-topic matrix)
#     index = np.random.rand(n)
#     class_vector = np.zeros(n, dtype=int)
    
#     for i in range(n):
#         index_i = index[i]
#         for l in range(1, r + 1):
#             l_bound = (l - 1) / r
#             u_bound = l / r
#             if l_bound < index_i < u_bound:
#                 class_vector[i] = l
    
#     dist_matrix = np.ones((r, r))
#     np.fill_diagonal(dist_matrix, 10)
    
#     U = np.empty((n, r))
#     for i in range(n):
#         class_i = class_vector[i]
#         d_i = dist_matrix[class_i - 1, :]
#         dirichlet = np.random.dirichlet(d_i, size=1)
#         U[i, :] = dirichlet
    
#     U = pd.DataFrame(U, columns=[f"topic_{i+1}" for i in range(r)])
    
#     # Step 2: Basis matrix V (topic-word matrix)
#     V = np.zeros((r, m2))
#     for l in range(1, r + 1):
#         l_bound = ((l - 1) / r) * m2
#         u_bound = (l / r) * m2
#         V[l - 1, int(l_bound):int(u_bound)] = np.random.uniform(0, 1, size=int(m2 / r))
    
#     # Step 3: Noise
#     mu_e = np.zeros(m2)
#     sigma_e = np.eye(m2)
#     e = np.random.multivariate_normal(mu_e, sigma_e, n)
    
#     # Step 4: Generate X2 from UV adding noise
#     X2 = np.dot(U, V) + eta * e
    
#     # Step 5: Force <= 0 values to be a very small positive number
#     X2[X2 <= 0] = 0.00001
    
#     X2 = pd.DataFrame(X2, columns=[f"word_{i+1}" for i in range(m2)])
    
#     # Step 6: Generate X1 (EHR data)
#     mu = np.repeat([1, 10], [7, m1 - 7]) #repeat values 1 and 10, repeat "1" 7 times and 10 m1-7 times
#     sigma = np.eye(m1)
#     X1 = np.random.multivariate_normal(mu, sigma, n)
#     X1 = pd.DataFrame(X1, columns=[f"EHR_{i+1}" for i in range(m1)])
#     X = pd.concat([X1, X2], axis=1) #column bind X1 and X2

#     # Step 7: tao(X)
#     matrix = np.hstack((X1, U.values))
#     beta_true = np.zeros(m1 + r) 
#     beta_true[m1: (m1+s)] = 3
#     tao_x = np.dot(matrix, beta_true)
    
#     # Step 8. generate m(x)
#     m = m1+m2  #total number of predictors = number of EHR features + number of words
#     np.random.seed(0) # Set the random seed for reproducibility
#     theta = np.random.normal(0, 1, m) #randomly generate theta distributed normal with mean0 and sd1
#     theta = vec2col(theta) 
#     m_x = theta.T.dot(X.T)

#     # Step 9. generate e(x)
#     np.random.seed(1) # Set the random seed for reproducibility
#     gamma = np.random.normal(0, 1, m) #randomly generate theta distributed normal with mean0 and sd1
#     gamma = vec2col(gamma)
#     gamma_x = gamma.T.dot(X.T)
#     e_x = 1 / (1 + np.exp(-gamma_x))

    
#     # Step 10. Generate W
#     W = (e_x > 0.5).astype(int)
#     W_tilde = W-e_x
#     W_tilde_matrix = np.diag(W_tilde.flatten())


#     # Step 11. Generate Y
#     e = np.random.normal(0, 1, n)
#     y = m_x + np.dot(W_tilde_matrix, tao_x) +  e * epsilon
#     y = y.T

#     # Final data
#     data = pd.concat([X1, X2, pd.DataFrame(W_tilde.T, columns=["W"]),U], axis=1)
#     data['y'] = y
    
    
#     if return_val == 1:
#         return data
#     elif return_val == 2:
#         return pd.DataFrame(V, columns=[f"word_{i+1}" for i in range(m2)])
#     elif return_val == 3:
#         return U
#     elif return_val == 4:
#         return theta
#     elif return_val == 5:
#         return gamma
    
    
def custom_pearsonr(topic_i, topic_j):
    """
    Calculate Pearson correlation coefficient and p-value, dealing with zero vectors

    Args:
    - topic_i: First topic.
    - topic_j: Second topic.

    Returns:
    - correlation: Pearson correlation coefficient.
    - p_value: p-value.
    """
    if np.all(topic_i == 0) or np.all(topic_j == 0):
        # Return correlation 0 and p-value 1 if either topic_i or topic_j is all zero
        return 0, 1
    elif np.std(topic_i) < 1e-10 or np.std(topic_j) < 1e-10:
        # Return correlation 0 and p-value 1 if the standard deviation of either topic_i or topic_j is 0
        return 0, 1
    else:
        # Calculate correlation and p-value
        correlation, p_value = pearsonr(topic_i, topic_j)
        return correlation, p_value
    
############################################################# Reorder V in the training step, using true V as reference
#used in evaluation
def generate_cor_df_val(Vtrue_df, V_df, R, r):
    # Pairwise Correlation between V_out and Vtrue
    correlation_df = pd.DataFrame()
    #R = Vtrue_df.shape[0]
    #r = V_df.shape[0]
    
    # Estimated V
    V_df = pd.DataFrame(V_df.T)
    num_rows = len(V_df)
    V_df[['word']] = ['word_' + str(i+1) for i in range(num_rows)]
    # Rename all columns follow the pattern: topic_i
    new_column_names = {i: f'topic_{i+1}' for i in range(r+1)}
    V_df.rename(columns=new_column_names, inplace=True)

    # True V
    Vtrue_df = Vtrue_df.T
    Vtrue_df.reset_index(inplace=True)
    Vtrue_df.rename(columns={'index': 'word'}, inplace=True)
    new_column_names = {i: f'topic_{i}' for i in range(R+1)}
    Vtrue_df.rename(columns=new_column_names, inplace=True)

    # Loop through each topic_i in Vtrue_df
    for i in range(1, R+1):  # Assuming you have 5 topics
        topic_i = Vtrue_df[f'topic_{i}']
        correlation_values = {} #Create a dictionary to store correlation values for topic_i
        # Loop through each topic_j in V_df
        for j in range(1, r+1):  # Assuming you have 5 topics
            topic_j = V_df[f'topic_{j}']
            # Calculate the pearson correlation between topic_i and topic_j
            correlation, p_value= custom_pearsonr(topic_i,topic_j)
            # Store the correlation value in the dictionary
            correlation_values[f'topic_{j}'] = correlation
        correlation_df = correlation_df.append(correlation_values, ignore_index=True)
    return correlation_df


# used in gradient descent
def generate_cor_df(Vtrue_df, V_df, R, r):
    # Pairwise Correlation between V_out and Vtrue
    correlation_df = pd.DataFrame()
    #R = Vtrue_df.shape[0]
    #r = V_df.shape[0]
    
    # Estimated V
    V_df = pd.DataFrame(V_df.T)
    num_rows = len(V_df)
    V_df[['word']] = ['word_' + str(i+1) for i in range(num_rows)]
    # Rename all columns follow the pattern: topic_i
    new_column_names = {i: f'topic_{i+1}' for i in range(r+1)}
    V_df.rename(columns=new_column_names, inplace=True)

    # True V
    Vtrue_df = Vtrue_df.T
    Vtrue_df.reset_index(inplace=True)
    Vtrue_df.rename(columns={'index': 'word'}, inplace=True)
    new_column_names = {i: f'topic_{i+1}' for i in range(R+1)}
    Vtrue_df.rename(columns=new_column_names, inplace=True)

    # Loop through each topic_i in Vtrue_df
    for i in range(1, R+1):  # Assuming you have 5 topics
        topic_i = Vtrue_df[f'topic_{i}']
        correlation_values = {} #Create a dictionary to store correlation values for topic_i
        # Loop through each topic_j in V_df
        for j in range(1, r+1):  # Assuming you have 5 topics
            topic_j = V_df[f'topic_{j}']
            # Calculate the pearson correlation between topic_i and topic_j
            correlation, p_value= custom_pearsonr(topic_i,topic_j)
            # Store the correlation value in the dictionary
            correlation_values[f'topic_{j}'] = correlation
        correlation_df = correlation_df.append(correlation_values, ignore_index=True)
    return correlation_df


def find_matched_topic_index(Vtrue, V_out, m2, R, r):
    """
        Vtrue: Simulated V, with dimension R x m2
        V_out: estimated V, with dimension r x m2
    """
    correlation_df = generate_cor_df(Vtrue, V_out, R, r)

    #Match the topics with the highest correlation, then match the topics with the second highest correlation, so on so forth
    matched_pairs = []
    corr_matrix = np.array(correlation_df)
    # Iterate until r pairs are matched
    for _ in range(r):
        # Find the indices of the maximum value in the correlation matrix
        max_index = np.unravel_index(np.nanargmax(corr_matrix), corr_matrix.shape)

        # Record the indices as a matched pair
        matched_pairs.append(max_index)

        # Replace the corresponding row and column with NaN to avoid re-matching
        corr_matrix[max_index[0], :] = np.nan
        corr_matrix[:, max_index[1]] = np.nan

    # Sort matched_pairs by their second element
    matched_pairs_sorted = sorted(matched_pairs, key=lambda x: x[1])

    # Extract the first element of each pair into an array
    matched_index = [pair[0] for pair in matched_pairs_sorted]
    
    return(matched_index, correlation_df)



def find_matched_topic_index_val(Vtrue, V_out, m2, R, r):
    """
        s: the number of top words taking into account
        method: 1-permutation. 2-top words location. 3-correlation
    """
    correlation_df = generate_cor_df_val(Vtrue, V_out, R, r)

    #Match the topics with the highest correlation, then match the topics with the second highest correlation, so on so forth
    matched_pairs = []
    corr_matrix = np.array(correlation_df)
    # Iterate until r pairs are matched
    for _ in range(r):
        # Find the indices of the maximum value in the correlation matrix
        max_index = np.unravel_index(np.nanargmax(corr_matrix), corr_matrix.shape)

        # Record the indices as a matched pair
        matched_pairs.append(max_index)

        # Replace the corresponding row and column with NaN to avoid re-matching
        corr_matrix[max_index[0], :] = np.nan
        corr_matrix[:, max_index[1]] = np.nan

    # Sort matched_pairs by their second element
    matched_pairs_sorted = sorted(matched_pairs, key=lambda x: x[1])

    # Extract the first element of each pair into an array
    matched_index = [pair[0] for pair in matched_pairs_sorted]
    
    return(matched_index, correlation_df)



def reorder_U_val(Vtrue, V, U, m2, exclude_zeros):
    R = Vtrue.shape[0]
    r = V.shape[0]
    n = U.shape[0]
    matched_index, NA = find_matched_topic_index_val(Vtrue=Vtrue, V_out=V,  m2=m2, R=R, r=r)
    matched_index = [x for x in matched_index]
    #print(matched_index)

    # initiate an np array U_r, with all entries 0
    U_r = np.zeros((n, r)) 
    if r == R:
        #U_r = np.column_stack([U[:, index] for index in matched_index])
        for i, index in enumerate(matched_index):
            U_r[:,index] = U[:,i]
    elif r < R:
        U_r = np.zeros((U.shape[0], R))
        U_r[:, matched_index] = U
        
        if exclude_zeros == True:
            U_r = U_r[:, ~np.all(U_r ==0 , axis=0)] #exclude columns with all zeros
        elif exclude_zeros == False:
            U_r = U_r
    return U_r

def reorder_U(Vtrue, V, U, m2, exclude_zeros):
    R = Vtrue.shape[0]
    r = V.shape[0]
    n = U.shape[0]
    matched_index, NA = find_matched_topic_index(Vtrue=Vtrue, V_out=V,  m2=m2, R=R, r=r)
    matched_index = [x for x in matched_index]
    #print(matched_index)

    # initiate an np array U_r, with all entries 0
    U_r = np.zeros((n, r)) 
    if r == R:
        #U_r = np.column_stack([U[:, index] for index in matched_index])
        for i, index in enumerate(matched_index):
            U_r[:,index] = U[:,i]
    elif r < R:
        U_r = np.zeros((U.shape[0], R))
        U_r[:, matched_index] = U
        
        if exclude_zeros == True:
            U_r = U_r[:, ~np.all(U_r ==0 , axis=0)] #exclude columns with all zeros
        elif exclude_zeros == False:
            U_r = U_r
    return U_r

def reorder_V(Vtrue, V, m2, exclude_zeros):
    R = Vtrue.shape[0]
    r = V.shape[0]
    matched_index, NA = find_matched_topic_index(Vtrue, V, m2,R=R, r=r)
    matched_index = [x for x in matched_index]
    #print(matched_index)
    
    # initiate an np array V_r, with all entries 0
    V_r = np.zeros((r, m2)) # r rows and m2 columns
    
    if r == R:
        for i, index in enumerate(matched_index):
            V_r[index] = V[i]
    elif r < R:
        V_r = np.zeros((R, V.shape[1]))
        V_r[matched_index,:] = V
        
        if exclude_zeros == True:
            V_r = V_r[~np.all(V_r ==0 , axis=1), :] #exclude rows with all zeros
        elif exclude_zeros == False:
            V_r = V_r
            
    return V_r

def reorder_V_val(Vtrue, V, m2, exclude_zeros):
    R = Vtrue.shape[0]
    r = V.shape[0]
    matched_index, NA = find_matched_topic_index_val(Vtrue, V, m2,R=R, r=r)
    matched_index = [x for x in matched_index]
    #print(matched_index)
    
    # initiate an np array V_r, with all entries 0
    V_r = np.zeros((r, m2)) # r rows and m2 columns
    
    if r == R:
        for i, index in enumerate(matched_index):
            V_r[index] = V[i]
    elif r < R:
        V_r = np.zeros((R, V.shape[1]))
        V_r[matched_index,:] = V
        
        if exclude_zeros == True:
            V_r = V_r[~np.all(V_r ==0 , axis=1), :] #exclude rows with all zeros
        elif exclude_zeros == False:
            V_r = V_r
            
    return V_r

def reorder_beta(Vtrue, V, beta,m1, m2, exclude_zeros):
    R = Vtrue.shape[0]
    r = V.shape[0]
    # construct matched_index_b2.
    matched_index, NA = find_matched_topic_index(Vtrue=Vtrue, V_out=V, m2=m2, R=R, r=r)
    matched_index = [x for x in matched_index]
    #print(matched_index)
    matched_index_b2 = [x + m1 for x in matched_index]  # add m1 to each element in the list
    matched_index_b2 = [x + 1 for x in matched_index_b2]  # add 1 to each element in the list
    x_index = list(range(m1 + 1))
    matched_index_b2 = x_index + matched_index_b2
    
    #initiate beta_r with all 0
    beta_r = np.zeros(r+m1+1)

    if r == R:
        #beta_r = [beta[i] for i in matched_index_b2]
        for i, index in enumerate(matched_index_b2):
            beta_r[index] = beta[i]
        beta_r = vec2col(beta_r)
    elif r < R:
        beta_r = np.zeros(m1 + R + 1)
        beta_r[matched_index_b2] = np.squeeze(beta)
        beta_r = vec2col(beta_r)
        
        if exclude_zeros == True:
            beta_r = beta_r[beta_r != 0]
            beta_r = vec2col(beta_r)
        elif exclude_zeros == False:
            beta_r = beta_r
    return beta_r


def reorder_beta_val(Vtrue, V, beta,m1, m2, exclude_zeros):
    R = Vtrue.shape[0]
    r = V.shape[0]
    # construct matched_index_b2.
    matched_index, NA = find_matched_topic_index_val(Vtrue=Vtrue, V_out=V, m2=m2, R=R, r=r)
    matched_index = [x for x in matched_index]
    #print(matched_index)
    matched_index_b2 = [x + m1 for x in matched_index]  # add m1 to each element in the list
    matched_index_b2 = [x + 1 for x in matched_index_b2]  # add 1 to each element in the list
    x_index = list(range(m1 + 1))
    matched_index_b2 = x_index + matched_index_b2
    
    #initiate beta_r with all 0
    beta_r = np.zeros(r+m1+1)

    if r == R:
        #beta_r = [beta[i] for i in matched_index_b2]
        for i, index in enumerate(matched_index_b2):
            beta_r[index] = beta[i]
        beta_r = vec2col(beta_r)
    elif r < R:
        beta_r = np.zeros(m1 + R + 1)
        beta_r[matched_index_b2] = np.squeeze(beta)
        beta_r = vec2col(beta_r)
        
        if exclude_zeros == True:
            beta_r = beta_r[beta_r != 0]
            beta_r = vec2col(beta_r)
        elif exclude_zeros == False:
            beta_r = beta_r
    return beta_r

def reorder_beta_ls(Vtrue, V, beta_ls, m1, m2):
    R = Vtrue.shape[0]
    r = V.shape[0]
    # construct matched_index_b2.
    matched_index, NA = find_matched_topic_index(Vtrue=Vtrue, V_out=V,  m2=m2, R=R, r=r)
    matched_index = [x  for x in matched_index]
    matched_index_b2 = [x + m1 -1 for x in matched_index]  # add m1 to each element in the list
    matched_index_b2 = [x for x in matched_index_b2]  # add 1 to each element in the list
    x_index = list(range(m1 + 1))
    matched_index_b2 = x_index + matched_index_b2
    
    beta_ls_r = np.row_stack([beta_ls[index, :] for index in matched_index_b2])
    return beta_ls_r

############################################################# Reorder U in the validation step, using true U as reference

# def generate_cor_df_U(R, r, Utrue, U_val_out):
#     '''
#         In validation step, reorder U using true U as reference.
#     '''
#     Utrue_df = Utrue
    
#     # Pairwise Correlation between V_out and Vtrue
#     correlation_df = pd.DataFrame()
    
#     # Estimated U
#     U_val_df = pd.DataFrame(U_val_out)
#     num_rows = len(U_val_df)
#     U_val_df[['word']] = ['word_' + str(i+1) for i in range(num_rows)]
#     # Rename all columns follow the pattern: topic_i
#     new_column_names = {i: f'topic_{i+1}' for i in range(r)}
#     U_val_df.rename(columns=new_column_names, inplace=True)

#     # True U
    
#     Utrue_df.reset_index(inplace=True)
#     Utrue_df.rename(columns={'index': 'word'}, inplace=True)
#     new_column_names = {i: f'topic_{i+1}' for i in range(R)}
#     Utrue_df.rename(columns=new_column_names, inplace=True)
        
#     # Loop through each topic_i in Vtrue_df
#     for i in range(1, R+1):  # Assuming you have 5 topics
#         topic_i = Utrue_df[f'topic_{i}']
#         correlation_values = {} #Create a dictionary to store correlation values for topic_i
#         # Loop through each topic_j in V_df
#         for j in range(1, r+1):  # Assuming you have 5 topics
#             topic_j = U_val_df[f'topic_{j}']
#             # Calculate the pearson correlation between topic_i and topic_j
#             correlation, p_value= custom_pearsonr(topic_i,topic_j)
#             # Store the correlation value in the dictionary
#             correlation_values[f'topic_{j}'] = correlation
#         correlation_df = correlation_df.append(correlation_values, ignore_index=True)
#     return correlation_df

# def find_matched_topic_index_U(R, r, Utrue, U_val_out):
#     '''
#         In validation step, reorder U using true U as reference.
#     '''
#     correlation_df = generate_cor_df_U(R, r, Utrue, U_val_out)

#     #Match the topics with the highest correlation, then match the topics with the second highest correlation, so on so forth
#     matched_pairs = []
#     corr_matrix = np.array(correlation_df)
#     # Iterate until r pairs are matched
#     for _ in range(r):
#         # Find the indices of the maximum value in the correlation matrix
#         max_index = np.unravel_index(np.nanargmax(corr_matrix), corr_matrix.shape)

#         # Record the indices as a matched pair
#         matched_pairs.append(max_index)

#         # Replace the corresponding row and column with NaN to avoid re-matching
#         corr_matrix[max_index[0], :] = np.nan
#         corr_matrix[:, max_index[1]] = np.nan

#     # Sort matched_pairs by their second element
#     matched_pairs_sorted = sorted(matched_pairs, key=lambda x: x[1])

#     # Extract the first element of each pair into an array
#     matched_index = [pair[0] for pair in matched_pairs_sorted]
    
#     return(matched_index, correlation_df)

# def reorder_U_val(R, r, Utrue, U, exclude_zeros):
#     '''
#         In validation step, reorder U using true U as reference.
#     '''
#     n = U.shape[0]
#     matched_index, NA = find_matched_topic_index_U(R, r, Utrue, U)
#     matched_index = [x for x in matched_index]
#     print(matched_index)

#     # initiate an np array U_r, with all entries 0
#     U_r = np.zeros((n, r)) 
#     if r == R:
#         #U_r = np.column_stack([U[:, index] for index in matched_index])
#         for i, index in enumerate(matched_index):
#             U_r[:,index] = U[:,i]
#     elif r < R:
#         U_r = np.zeros((U.shape[0], R))
#         U_r[:, matched_index] = U
        
#         if exclude_zeros == True:
#             U_r = U_r[:, ~np.all(U_r ==0 , axis=0)] #exclude columns with all zeros
#         elif exclude_zeros == False:
#             U_r = U_r
#     return U_r

# def reorder_beta_val(R, r, Utrue, U, beta, exclude_zeros):

#     # construct matched_index_b2.
#     matched_index, NA = find_matched_topic_index_U(R, r, Utrue, U)
#     matched_index = [x for x in matched_index]
#     print(matched_index)
    
#     matched_index_b2 = [x + m1 for x in matched_index]  # add m1 to each element in the list
#     matched_index_b2 = [x + 1 for x in matched_index_b2]  # add 1 to each element in the list
#     x_index = list(range(m1 + 1))
#     matched_index_b2 = x_index + matched_index_b2
    
#     #initiate beta_r with all 0
#     beta_r = np.zeros(r+m1+1)

#     if r == R:
#         #beta_r = [beta[i] for i in matched_index_b2]
#         for i, index in enumerate(matched_index_b2):
#             beta_r[index] = beta[i]
#         beta_r = vec2col(beta_r)
#     elif r < R:
#         beta_r = np.zeros(m1 + R + 1)
#         beta_r[matched_index_b2] = np.squeeze(beta)
#         beta_r = vec2col(beta_r)
        
#         if exclude_zeros == True:
#             beta_r = beta_r[beta_r != 0]
#             beta_r = vec2col(beta_r)
#         elif exclude_zeros == False:
#             beta_r = beta_r
#     return beta_r

########################################## Finding optimal number of r


def calc_maxdiff_coherence(seed, main_topic_signal, topic_weights,n, m1, m2,gamma1, gamma2,psi1, psi2, beta1, beta2, rho1, rho2, eta, epsilon, R, r_tilde, num_top_words):
    """
        Takes function: calc_c
    """
    # Generate data
    # Scenerio B
    
    data_train, Vtrue_train, NA= generate_data_dlearning(seed=seed, n = n, r=R, m1=m1, m2 = m2, zeta1= 1, zeta2=2, main_topic_signal=main_topic_signal,topic_weights = topic_weights,eta =eta, epsilon=epsilon, gamma1=gamma1, gamma2=gamma2, beta0=0, psi1 = psi1, psi2=psi2,beta1=beta1, beta2=beta2, rho1= rho1, rho2=rho2)

    #print("Train:")
    #print(pd.crosstab(index = data_train['d_x'], columns = 'count')) # Treatment Assignment, controlled by betas
    Y = data_train[['y']].values #array
    A = data_train[['A']].values #array
    X2 =  data_train.iloc[:, m1:m1+m2] #document word 
    X2 = X2.to_numpy()

    # NMF: obtain U_screen and V_screen for screening
    model = NMF(n_components= r_tilde, init= 'nndsvd', random_state=1)
    U_screen = model.fit_transform(X2) #n times r
    V_screen = model.components_


    # calcualte coherence measure for each topic in V_screen
    # only looking at the num_top_words number of top words in each topic in V_screen
    c_values_df, NA =  calc_c(V_screen,Vtrue_train, r_tilde, num_top_words)
    cvalues = c_values_df['Mean']
    #cvalues = np.round(cvalues, 2)

    num_rows, num_cols = U_screen.shape
    pvalues = np.zeros((num_cols,))
    for i in range(num_cols):
        u_i = U_screen[:, i]
        u_i_intercept = sm.add_constant(u_i)  
        model = sm.OLS(2*A*Y, u_i_intercept) ## Fit a linear regression model
        results = model.fit()
        pvalues[i] = results.pvalues[1:]  #p-value
    ############################################################
    pvalues_sorted = np.sort(pvalues)
    pvalues_sorted_transformed = -np.log10(pvalues_sorted)
    pvalues_diff = np.abs(np.diff(pvalues_sorted_transformed))
    max_diff_index = np.argmax(pvalues_diff)
    max_diff = np.round(pvalues_diff.max(),4)
    pvalue_df = pd.DataFrame(pvalues_sorted_transformed, columns=['-log10(p-value)'])

    coherence = np.round(cvalues.mean(),3)
    cvalues = np.round(cvalues, 2)

    pvalue_df['index'] = range(r_tilde)
    pvalue_df['index'] = pvalue_df['index'].astype(str)
    pvalue_df['cvalues'] = cvalues
    coherence_score = pvalue_df['cvalues'].mean()
    return(max_diff, coherence_score)

############################## 
def calc_c(V_screen, Vtrue, r_tilde, num_top_words):
    c_values = []
    top_words_indices_ls = [None] * r_tilde
    # t from 0 to r_tilde from V_screen
    for t in range(r_tilde):
        topic_t = V_screen[t,]
        top_words_indices = np.argsort(topic_t)[-num_top_words:]
        top_words_indices_ls[t] = top_words_indices
        
        V_t = Vtrue.iloc[:, top_words_indices]
       
        m_t = []
        for i in range(num_top_words-1):
            for j in range(i+1,num_top_words):
                w_i = V_t.iloc[:,i]
                w_j = V_t.iloc[:,j]
                w_ij = w_i.dot(w_j)
                #print("Both words in the same document =", w_ij)
                m_t.append(w_ij)
        m_t = pd.Series(m_t)
        #print(m_t)
        c_t = [m_t.mean(), m_t.median(), gmean(m_t)]
        c_values.append(c_t)
    c_values_df = pd.DataFrame(c_values, columns=['Mean', 'Median', 'Geometric Mean'])
    return(c_values_df,top_words_indices_ls)

############################## 
def find_optimal_r(seed, main_topic_signal, topic_weights,n, m1, m2,gamma1, gamma2,psi1, psi2, beta1, beta2, rho1, rho2,eta, epsilon, R, num_top_words, r_min, r_max, plot):
    """
        Takes function: calc_maxdiff_coherence
    """
    max_diff_ls = []
    coherence_score_ls  = []
    for r_tilde in range(r_min, r_max):
        max_diff, coherence_score = calc_maxdiff_coherence(seed, main_topic_signal, topic_weights,n, m1, m2,gamma1, gamma2,psi1, psi2, beta1, beta2, rho1, rho2, eta, epsilon, R, r_tilde, num_top_words)
        max_diff_ls.append(max_diff)
        coherence_score_ls.append(coherence_score)
        
    df = pd.DataFrame({'max_diff_pvalue': max_diff_ls, 'coherence_score': coherence_score_ls})
    df["r_tilde"] = range(r_min, r_max)
    df['max_diff_times_coherence'] = df['max_diff_pvalue'] * df['coherence_score']
    
    # Find the index of the maximum value in the max_diff_pvalue column
    max_y_index = df['max_diff_times_coherence'].idxmax()
    # Get the corresponding r_tilde value
    optimal_r = df.loc[max_y_index, 'r_tilde']
    ############## Plot
    if plot==True:
        # Create figure and axes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot 1: Max Diff P-value vs r_tilde and Coherence Score
        color1 = 'tab:red'
        ax1.set_ylabel('Max Diff P-value', color=color1)
        ax1.plot(df['r_tilde'], df['max_diff_pvalue'], color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)

        # Set grid and x-axis locator
        ax1.grid(True)
        ax1.xaxis.set_major_locator(MultipleLocator(1))

        ax2 = ax1.twinx()
        color2 = 'tab:blue'
        ax2.set_ylabel('Coherence Score', color=color2)
        ax2.plot(df['r_tilde'], df['coherence_score'], color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

        # Plot 2: Max Diff P-value X Coherence Score
        ax3 = plt.subplot(212)  # Second subplot
        ax3.plot(df['r_tilde'], df['max_diff_times_coherence'], color='blue')
        ax3.set_ylabel('Max Diff P-value X Coherence score')
        ax3.set_xlabel('r_tilde')

        # Set grid and x-axis locator
        ax3.grid(True)
        ax3.xaxis.set_major_locator(MultipleLocator(1))

        # Plot vertical line at the maximum value
        ax3.axvline(x=optimal_r, color='red', linestyle='--')

        plt.suptitle('Finding Optimal Number of Topics', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to accommodate suptitle
        plt.show()
 
    return(optimal_r, df)



############################## We don't need method 1 or 2 anymore.
############################## Just as a reference
# def find_matched_topic_index(Vtrue, V_out, s, m2, method, R, r):
#     """
#         s: the number of top words taking into account
#         method: 1-permutation. 2-top words location. 3-correlation
#     """
#     if method == 1:
#         # Using permutation, find concordance for all possible permutations 
        
#         # V true 
#         Vtrue_df = pd.DataFrame(Vtrue.T)
#         Vtrue_df["topic_true"] = Vtrue_df.idxmax(axis=1)
#         Vtrue_df = Vtrue_df[["topic_true"]]
#         Vtrue_df.index = np.arange(0, len(Vtrue_df) ) #change rowname to just the index 

#         # V predicted
#         V_df = pd.DataFrame(V_out.T)
#         V_df["topic_hat"] = V_df.idxmax(axis=1)
#         V_df = V_df[["topic_hat"]]

#         # combine two columns: predicted topic classification and true topic classification
#         df = pd.concat([V_df, Vtrue_df], axis=1)

#         # Create conditions list
#         #conditions  = [ df["topic_true"] == 0, df["topic_true"] == 1, df["topic_true"] == 2, df["topic_true"] == 3, df["topic_true"] == 4]
#         conditions = [df["topic_true"] == i for i in range(r)]

#         # Create choices list
#         choices = list(range(r))


#         # Generate all permutations of choices
#         permutations_list = list(permutations(choices))

#         # Create 5! columns with each using a different permutation
#         for i, perm in enumerate(permutations_list):
#             column_name = f"m{i+1}"
#             conditions = [df["topic_hat"] == int(choice) for choice in perm]
#             df[column_name] = np.select(conditions, choices, default=np.nan)

#         columns_to_check = df.columns[2:]  

#         # Initialize a list to store the counts
#         counts = []

#         # Iterate through the columns and calculate the counts
#         for column_name in columns_to_check:
#             count = len(df[df['topic_true'] == df[column_name]])
#             counts.append(count)

#         #find which permutation gives the highest concordance
#         max_index = counts.index(max(counts))
#         matched_index = permutations_list[max_index]

#         return(matched_index)
    
#     elif method == 2:
#         # Using the index of the true V 
#         # true V
#         r= Vtrue.shape[0]
#         Vtrue_df = pd.DataFrame(Vtrue.T)
#         Vtrue_df["word"] = range(1,m2+1)

#         # V output from the PGD
#         V_df = pd.DataFrame(V_out.T)
#         V_df["word"] = range(1,m2+1) # need to change 101 to m2

#         #########
#         matched_index = list() #correct index

#         for l in range(0, r):
#             #print("----Topic l=",l)
#             diff_ls = list()
#             for i in range(0, r):
#                 #print(i)
#                 #sort the ith column of V_df--topic i, find the top ten words in each topic, since we simulated V with a certain order, we can take the mean of the word, and
#                 #this is the "topic word key"
#                 v_word_mean_out_i = V_df.sort_values(i, ascending =False)["word"][0:s].mean()

#                 # true "topic word key"
#                 v_word_mean_topic_l = Vtrue_df.sort_values(l, ascending =False)["word"][0:s].mean()

#                 #find the difference between the "topic word key" from the V_out and the true V
#                 diff_topic_l = abs(v_word_mean_topic_l - v_word_mean_out_i)
#                 #print(diff_topic_l)
#                 diff_ls.append(diff_topic_l)

#             topic_match_index = pd.Series(diff_ls).idxmin()
#             #print("matched index:", topicl_match_index)
#             matched_index.append(topic_match_index)
            
#     elif method == 3:
#         correlation_df = generate_cor_df(Vtrue, V_out, R, r)

#         R= Vtrue.shape[0]
#         r = V_out.shape[0]

#     #Match the topics with the highest correlation, then match the topics with the second highest correlation, so on so forth
#     matched_pairs = []
#     corr_matrix = np.array(correlation_df)
#     # Iterate until r pairs are matched
#     for _ in range(r):
#         # Find the indices of the maximum value in the correlation matrix
#         max_index = np.unravel_index(np.nanargmax(corr_matrix), corr_matrix.shape)

#         # Record the indices as a matched pair
#         matched_pairs.append(max_index)

#         # Replace the corresponding row and column with NaN to avoid re-matching
#         corr_matrix[max_index[0], :] = np.nan
#         corr_matrix[:, max_index[1]] = np.nan

#     # Sort matched_pairs by their second element
#     matched_pairs_sorted = sorted(matched_pairs, key=lambda x: x[1])

#     # Extract the first element of each pair into an array
#     matched_index = [pair[0] for pair in matched_pairs_sorted]
    
#     return(matched_index, correlation_df)
