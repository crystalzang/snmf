import numpy as np
import copy
from helper_functions import pos, neg, NNDSVD, vec2col, error_plot, reorder_beta
from helper_functions import reorder_U, reorder_V, reorder_beta, reorder_beta_ls, pos, neg, NNDSVD, vec2col, error_plot, find_matched_topic_index

import time
from numpy.random import rand
from numpy import exp, log, mat
from numpy.linalg import norm
from scipy.io import loadmat
from scipy.linalg import svd 
from array import array
import pandas as pd


import math
from itertools import permutations #used in find_matched_topic_index

def objective_U(n,r,m2,X2, W, Y_tilde, U, V, X1, beta, alpha1, lambda_u): 
    """
    Input:
        Y_tilde: transformed Y
        U: original U, no intercept. n x r
        b2: (r+1) x 1
    """
    m1 = X1.shape[1]
    UV=U.dot(V)
    
    if alpha1 == lambda_u == 0:
        F = 0.5/(n*m2)*norm(X2-UV,'fro')**2
    else:
        beta_0 = vec2col(beta[0])  # 1x1
        beta_x = vec2col(beta[1:m1+1]) # m1 x 1
        beta_u = vec2col(beta[m1+1:]) # r x 1

        one_n= vec2col(np.ones(n)) #nx1

        w1b0 = np.sqrt(W).dot(one_n).dot(beta_0) #gradU
        wx1bx = np.sqrt(W).dot(X1).dot(beta_x) #gradU
        wubu = np.sqrt(W).dot(U).dot(beta_u) #gradU
        F = 0.5/(n*m2)*norm(X2-UV,'fro')**2+ alpha1*0.5/float(n)*norm(Y_tilde-w1b0-wx1bx-wubu)**2 +0.5/(n*r)*float(lambda_u)*norm(U,'fro')**2
    return F

def subprob_U_test(n,r,m2,X2, W, Y_tilde, V, U, X1, beta, alpha1,lambda_u, tol, maxiter, a, b, sigma, print_message):
    """Projected gradient descent.
    Input:
        X2: document-word matrix (np array)
        W: n x n
        Y_tilde: transformed
        V: V matrix
        X_tilde: transformed, dimension: n x (1+m1+r)
        beta: dimension: (1+m1+r) x 1
        tol: stopping tolerance
        maxiter: limit of iterations
        step_size: 0: fixed; 1: updating step step size
    Output:
        dimension: n x r
    """ 
    m1 = X1.shape[1]
    VVt=V.dot(V.T)  #gradU
    X2Vt=X2.dot(V.T)  #gradU
    
    if alpha!= 0 or lambda_u != 0:
        beta_0 = vec2col(beta[0])  # 1x1
        beta_x = vec2col(beta[1:m1+1]) # m1 x 1
        beta_u = vec2col(beta[m1+1:])# r x 1

        
        one_n= vec2col(np.ones(n)) #gradU
        w1b0 = np.sqrt(W).dot(one_n).dot(beta_0) #gradU
        wx1bx = np.sqrt(W).dot(X1).dot(beta_x) #gradU
        wubu = np.sqrt(W).dot(U).dot(beta_u) #gradU

    sub_iter = 20
    
    grad_ls = list() #store projected gradient (norm)
    obj_ls = list() #store objective function
    
    for iter in range(maxiter):
        if print_message == True:
            print("--------------------------OUTER LOOP --------iteration:", iter)
        if alpha == lambda_u == 0:
            grad = U.dot(VVt)/(n*m2)-X2Vt/(n*m2)
        else:
            grad = U.dot(VVt)/(n*m2)-X2Vt/(n*m2)- alpha1/n*np.sqrt(W).dot(Y_tilde - w1b0 - wx1bx - wubu).dot(beta_u.T) + float(lambda_u)/(n*r)*U  # gradient of U, n x r
        
        projgrad = np.linalg.norm(grad[np.logical_or(grad < 0, U >0)])
        grad_ls.append(projgrad)
        
        grad_diff = grad_ls[iter-1] - grad_ls[iter]
        if iter > sub_iter and grad_diff < 1e-10:
            if print_message == True:
                print("Difference in current projgrad and previous projgrad=",grad_diff)
                print("(end 1)")
            break
        # search for step size alpha
        objold = objective_U(n,r,m2,X2, W, Y_tilde, U, V,X1, beta, alpha1, lambda_u) #update objective of U   
        obj_ls.append(objold)
        for n_iter in range(sub_iter):
            if print_message == True:
                print("--INNER LOOP --iteration:", n_iter)
            Un = U - a*grad
            Un = np.where(Un > 0, Un, 1e-20)
            d = Un - U
            gradd = np.multiply(grad, d).sum()
            objnew= objective_U(n,r,m2,X2,W, Y_tilde, Un, V, X1, beta, alpha1,lambda_u)  #update objective of U, Un here
            
            suff_decr=objnew-objold-sigma*gradd < tol  #sufficient decrease condition
            
            if print_message == True:
                print('**** objective = ',round(objnew, 4), ", projected gradient norm =",round(projgrad,10), ",suff decr =", round(objnew-objold-sigma*gradd,5))
            #########################################################
            if n_iter == 0:
                decr_alpha = not suff_decr
                Up = U
            # After the first iteration:         
            if decr_alpha:
                if suff_decr:
                    U = Un
                    if print_message == True:
                        print("(end 2)")
                    break
                else:
                    a *= b
            else:
                if not suff_decr or (Up == Un).all():
                    U = Up
                    if print_message == True:
                        print("(end 3)")
                    break
                else:
                    a /= b
                    Up = Un
    return U, grad, iter, grad_ls, obj_ls

def gradient_descent_test(X2, X, W, Y, Y_tilde, U, V, Vtrue, r, beta, alpha1, lambda_u, tol, maxiter, a, b, sigma,print_message):
    """Projected gradient descent.
    
    Inputs:
        X2: document word matrix, n x m2

        Uinit: Init from testing, n x r
        
        V: V_out from training
        Vtrue: simulated V
        tol: tolerance for a relative stopping condition
        
    Returns:
        List of all points computed by projected gradient descent.
        U,V, b2: output solution
        projnorm_ls
    
    """
    n = U.shape[0] #number of sample
    R = Vtrue.shape[0]
    m1 = X.shape[1] - U.shape[1] #number of EHR features
    m2 = V.shape[1] #number of words 
    
    VVt=V.dot(V.T)  #gradU
    X2Vt=X2.dot(V.T)  #gradU
        
    X1 = X[:,0:m1]
    
    if alpha1 == lambda_u == 0: # input W, Y_tilde = None
        gradU = U.dot(VVt) / (n * m2) - X2Vt / (n * m2) 
    
    else:  
        beta_0 = vec2col(beta[0])  # 1x1
        beta_x = vec2col(beta[1:m1+1]) # m1 x 1
        beta_u = vec2col(beta[m1+1:]) # r x 1

        
        one_n= vec2col(np.ones(n)) #gradU
        w1b0 = np.sqrt(W).dot(one_n).dot(beta_0) #gradU
        wx1bx = np.sqrt(W).dot(X1).dot(beta_x) #gradU
        wubu = np.sqrt(W).dot(U).dot(beta_u) #gradU
    
        gradU = U.dot(VVt)/(n*m2)-X2Vt/(n*m2)- alpha1/n*np.sqrt(W).dot(Y_tilde - w1b0 - wx1bx - wubu).dot(beta_u.T) + float(lambda_u)/(n*r)*U  # gradient of U, n x r

    # calculate initial gradient 
    initgrad = np.linalg.norm(gradU,'fro')

    projnorm_ls = list()
    objective_new_ls = list()
    convergence_step = 1000000000000 #meaning non-convergence/exceed max iteration
    
    for iter in range(maxiter): 
       
        objective_new = objective_U(n,r,m2,X2, W, Y_tilde, U, V, X1, beta, alpha1, lambda_u)
        objective_new_ls.append(objective_new)
        
        #Compute projected gradients norm.
        #np.r_: Translates slice objects to concatenation along the first axis.
        projnorm = np.linalg.norm(np.linalg.norm(np.r_[gradU[np.logical_or(gradU<0, U>0)]]))
        projnorm_ls.append(projnorm)

        #stopping condition
        #if projnorm < tol * initgrad:
        #    break
        obj_t1 = objective_new_ls[iter]
        obj_t0 = objective_new_ls[iter-1] 
        obj_diff = obj_t0 - obj_t1 # since objective function is decreasing
        if obj_diff/objective_new_ls[0] < tol: #relative tolerance
            print("!!!At iteration", iter,"Absolute difference in current objective and previous objective / initial objective=",obj_diff/objective_new_ls[0], ", END!!!")
            convergence_step = iter
            break
            
        U, gradU, iterU = subprob_U_test(n,r,m2,X2, W, Y_tilde, V, U, X1, beta, alpha1,lambda_u, tol, maxiter, a, b, sigma, print_message)

    return  U, convergence_step