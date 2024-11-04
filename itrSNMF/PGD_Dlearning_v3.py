import numpy as np
import copy
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

# def eval_ITR(d_x, d_x_hat):
#     tb = pd.crosstab(d_x, d_x_hat)

# #     print(tb)
# #     print(type(tb))
# #     print(tb.shape)
#     # Initialize variables with default values
#     TP = 0
#     TN = 0
#     FP = 0
#     FN = 0

#     if tb.shape[1] == 2:
#         TP = tb.iloc[1, 1]
#         TN = tb.iloc[0, 0]
#         FP = tb.iloc[0, 1]
#         FN = tb.iloc[1, 0]

#     if tb.shape[1] == 1:
#         TP = tb.iloc[1].iloc[0]
#         FP = tb.iloc[0].iloc[0]

#     specificity = TN / (TN + FP) if (TN + FP) != 0 else 0  # Handle division by zero
#     sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0  # Handle division by zero
#     accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0  # Handle division by zero

#     return TP, TN, FP, FN, specificity, sensitivity, accuracy

def eval_ITR(d_x, d_x_hat):
    tb = pd.crosstab(d_x, d_x_hat)

    # Initialize variables with default values
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    print("Run eval_ITR function. Table has shape:", tb.shape, "Table index:", tb.index)
    # Check the dimensions of the crosstab
    if tb.shape == (2, 2):
        # Standard case: both classes are present in predictions and actuals
        TP = tb.iloc[1, 1] if 1 in tb.index and 1 in tb.columns else 0
        TN = tb.iloc[0, 0] if 0 in tb.index and 0 in tb.columns else 0
        FP = tb.iloc[0, 1] if 0 in tb.index and 1 in tb.columns else 0
        FN = tb.iloc[1, 0] if 1 in tb.index and 0 in tb.columns else 0
        print("Standard 2x2 table processed.")

    elif tb.shape == (2, 1):
        # Only one predicted category (could be either 0 or 1)
        if 1 in tb.columns:
            TP = tb.iloc[1, 0] if 1 in tb.index else 0
            FP = tb.iloc[0, 0] if 0 in tb.index else 0
            print("Only predicted class 1 is present.")
        elif -1 in tb.columns:
            TN = tb.iloc[0, 0] if 0 in tb.index else 0
            FN = tb.iloc[1, 0] if 1 in tb.index else 0
            print("Only predicted class -1 is present.")

    elif tb.shape == (1, 2):
        # Only one actual category (could be either 0 or 1)
        if 1 in tb.index:
            TP = tb.iloc[0, 1]
            FN = tb.iloc[0, 0]
            print("Only actual class 1 is present.")
        elif -1 in tb.index:
            TN = tb.iloc[0, 0]
            FP = tb.iloc[0, 1]
            print("Only actual class -1 is present.")

    elif tb.shape == (1, 1):
        # Only one class is present in both actual and predicted
        if 1 in tb.index and 1 in tb.columns:
            TP = tb.iloc[0, 0]
            print("Only class 1 is present in both actual and predicted.")
        elif -1 in tb.index and 0 in tb.columns:
            TN = tb.iloc[0, 0]
            print("Only class -1 is present in both actual and predicted.")

    else:
        print(f"Unexpected table shape: {tb.shape}. Possibly missing categories.")
    
    # Calculate specificity, sensitivity, and accuracy with error handling for division by zero
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0  # True Negative Rate
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0  # True Positive Rate
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0

    return TP, TN, FP, FN, specificity, sensitivity, accuracy


def objective_function(n,r,m2,X2, Y_tilde, U, X_tilde, V, beta, alpha1, lambda_u, lambda_v, lambda_beta): 
    """
    Input:
        Y_tilde: transformed Y
        U: original U, n x r
        V: original V, r x m2
        X_tilde: EHR X1 and U, transformed, with intercept. n x (1+m1+r)
        beta: (1+m1+r) x 1
    """
    UV=U.dot(V)
    Xbeta_tilde= X_tilde.dot(beta)
   
    F = 0.5/(n*m2)*norm(X2-UV,'fro')**2+ alpha1*0.5/n*norm(Y_tilde-Xbeta_tilde)**2 + 0.5/(n*r)*float(lambda_u)*norm(U,'fro')**2  + 0.5/(r*m2)*float(lambda_v)*norm(V,'fro')**2 + 0.5/(r*m2)*float(lambda_beta)*norm(beta)**2
    return F

def objective_b2_dlearning(n,m1,m2,Y_tilde, X_tilde, beta, lambda_beta): 
    """
    Input:
        Y_tilde: transformed Y
        X_tilde: EHR X1, X2, transformed, with intercept. n x (1+m1+m2)
        beta: (1+m1+m2) x 1
    """
    Xbeta_tilde= X_tilde.dot(beta)
    F = 0.5/n*norm(Y_tilde-Xbeta_tilde)**2 + 0.5/(m1+m2)*float(lambda_beta)*norm(beta)**2
    return F


# likelihood function that involves U
def objective_U(n,r,m2,X2, W, Y_tilde, U, V, X1, beta, alpha1, lambda_u): 
    """
    Input:
        Y_tilde: transformed Y
        U: original U, no intercept. n x r
        b2: (r+1) x 1
    """
    m1 = X1.shape[1]
    beta_0 = vec2col(beta[0])  # 1x1
    beta_x = vec2col(beta[1:m1+1]) # m1 x 1
    beta_u = vec2col(beta[m1+1:]) # r x 1
    
    one_n= vec2col(np.ones(n)) #nx1

    UV=U.dot(V)
    w1b0 = np.sqrt(W).dot(one_n).dot(beta_0) #gradU
    wx1bx = np.sqrt(W).dot(X1).dot(beta_x) #gradU
    wubu = np.sqrt(W).dot(U).dot(beta_u) #gradU

    F = 0.5/(n*m2)*norm(X2-UV,'fro')**2+ alpha1*0.5/float(n)*norm(Y_tilde-w1b0-wx1bx-wubu)**2 +0.5/(n*r)*float(lambda_u)*norm(U,'fro')**2
    return F

def objective_V(n,r,m2,X2,U,V, lambda_v):  
    F = 0.5/(float(n)*float(m2))*norm(X2-U.dot(V),'fro')**2 + 0.5/(float(r)*float(m2))*float(lambda_v)*norm(V,'fro')**2
    #F = 0.5*norm(X2-U.dot(V),'fro')**2 + 0.5*float(lambda_v)*norm(V,'fro')**2
    return F

def objective_b2(n,r,m2,Y_tilde, X_tilde,beta, alpha1,lambda_beta): 
    """
    Input:
        Y_tilde: transformed Y
        X_tilde: EHR X1 and U, transformed, with intercept. n x (1+m1+r)
        beta: dimension (1+m1+r) x 1
    """
    Xbeta_tilde= X_tilde.dot(beta)

    F = alpha1/float(n)*0.5*norm(Y_tilde-Xbeta_tilde)**2 + 0.5/(r*m2)*float(lambda_beta)*norm(beta)**2
    #F = alpha1*0.5*norm(Y_tilde-Xbeta_tilde)**2 + 0.5*float(lambda_beta)*norm(beta)**2
    return F

# need to have very small tol since projnorm is small <0.001
def subprob_U(n,r,m2,X2, W, Y_tilde, V, U, X1, beta, alpha1,lambda_u, tol, maxiter, a, b, sigma, print_message, step_size):
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
    
    beta_0 = vec2col(beta[0])  # 1x1
    beta_x = vec2col(beta[1:m1+1]) # m1 x 1
    beta_u = vec2col(beta[m1+1:])# r x 1

    VVt=V.dot(V.T)  #gradU
    X2Vt=X2.dot(V.T)  #gradU
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
            if step_size == 1: #1 meaning updating step size
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
            elif step_size == 0: # fixed step size
                if decr_alpha:
                    U = Un
                    if print_message == True:
                        print("(end 2)")
                    break
                else:
                    if not suff_decr or (Up == Un).all():
                        U = Up
                        if print_message == True:
                            print("(end 3)")
                        break
                    else:
                        Up = Un
    return U, grad, iter, grad_ls, obj_ls

def subprob_V(n,r,m2,X2,U,V, lambda_v,tol,maxiter, a, b, sigma,print_message, step_size):
    """
    Input:
        X2: document-word matrix
        Y: No transformation
        U: No transformation
        
        Vinit: initial solution
        tol: stopping tolerance
        maxiter: limit of iterations
    """    
    UtX2=U.T.dot(X2) 
    UtU=U.T.dot(U) 
    sub_iter = 20
    
    grad_ls=list()
    obj_ls = list()
    
    for iter in range(maxiter):
        if print_message == True:
            print("--------------------------OUTER LOOP-----iteration:", iter)
        grad= UtU.dot(V)/(n*m2)-UtX2/(n*m2)+float(lambda_v) * V/(r*m2)
        #grad= UtU.dot(V)-UtX2+float(lambda_v) * V
        projgrad = np.linalg.norm(grad[np.logical_or(grad < 0, V >0)])
        grad_ls.append(projgrad)
        
        grad_diff = grad_ls[iter-1] - grad_ls[iter]
        if iter > sub_iter and grad_diff < 1e-10:
            if print_message == True:
                print("Difference in current projgrad and previous projgrad=",grad_diff)
                print("(end 1)")
            break
        # search for step size a
        objold = objective_V(n,r,m2,X2,U,V, lambda_v) 
        obj_ls.append(objold)
        
        for n_iter in range(sub_iter):
            if print_message == True:
                print("--inner loop--iteration:", n_iter)
            Vn = V - a*grad #update V
            Vn = np.where(Vn > 0, Vn, 1e-20) #force V to be positive
            d = Vn - V
            gradd = np.multiply(grad, d).sum()
            objnew = objective_V(n,r,m2,X2,U,Vn,lambda_v)
            
            dQd = np.multiply(np.dot(UtU, d), d).sum()
            #suff_decr = 0.99 * gradd + 0.5 * dQd < 0
            
            suff_decr=objnew-objold-sigma*gradd < tol  #sufficient decrease condition
            
            if print_message == True:
                print('**** objective = ',round(objnew, 4), ", projected gradient norm =",round(projgrad,10), ",suff decr =", round(objnew-objold-sigma*gradd,5))
            ##########################################################################
            if n_iter == 0:
                decr_alpha = not suff_decr
                Vp = V
            # After the first iteration:         
            if step_size == 1: #1 meaning updating step size
                if decr_alpha:
                    if suff_decr:
                        V = Vn
                        if print_message == True:
                            print("(end 2)")
                        break
                    else:
                        a *= b
                else:
                    if not suff_decr or (Vp == Vn).all():
                        V = Vp
                        if print_message == True:
                            print("(end 3)")
                        break
                    else:
                        a /= b
                        Vp = Vn
            elif step_size == 0: # fixed step size
                if decr_alpha:
                    V = Vn
                    if print_message == True:
                        print("(end 2)")
                    break
                else:
                    if not suff_decr or (Vp == Vn).all():
                        V = Vp
                        if print_message == True:
                            print("(end 3)")
                        break
                    else:
                        Vp = Vn
    return V, grad, iter, grad_ls, obj_ls

def subprob_b2(n,r,m2,Y_tilde, X_tilde, beta, alpha1, lambda_beta,tol, maxiter, a, b, sigma, print_message, step_size):
    """
    Input:
        Y_tilde: transformed Y
        X_tilde: transformed [1 X1 U], dimension n x (1+m1+r)
        b2init: dimension (r+1) x 1
    """
    m1 = X_tilde.shape[1]-r-1
    XtX_tilde = X_tilde.T.dot(X_tilde)
    
    sub_iter = 20
    
    grad_ls = list()
    obj_ls = list()
    beta_est_ls = []
    for iter in range(maxiter) :
        if print_message == True:
            print("---------------------------OUTER LOOP-----iteration:", iter)
        vector = np.ones((1+r+m1, 1))
        vector[0] = 0 #force the first column to be 0 so that we are not penalizing the intercept

        grad = alpha1/n*XtX_tilde.dot(beta) - alpha1/n*X_tilde.T.dot(Y_tilde) + float(lambda_beta)/(r*m2) * beta * vector #gradient of b2, (1+m1+r) x 1
        #grad = alpha1*XtX_tilde.dot(beta) - alpha1*X_tilde.T.dot(Y_tilde) + float(lambda_beta) * beta #gradient of b2, (r+1) x 1
        projgrad=norm(grad,'fro')
        grad_ls.append(projgrad)
    
        #if projgrad < tol:
        #    break
        grad_diff =  grad_ls[iter-1]- grad_ls[iter]
        if iter > sub_iter and grad_diff < 1e-10:
            if print_message == True:
                print("Difference in current projgrad and previous projgrad=",grad_diff)
                print("(end 1)")
            break
            
        # search for step size alpha
        objold= objective_b2(n,r,m2,Y_tilde, X_tilde, beta, alpha1, lambda_beta) #update objective function
        obj_ls.append(objold)
        for n_iter in range(sub_iter):
            if print_message == True:
                print("--inner loop--iteration:", n_iter)
            betan = beta - a*grad
            d = betan - beta
            gradd = np.multiply(grad, d).sum()

            objnew = objective_b2(n,r,m2,Y_tilde, X_tilde, betan, alpha1,lambda_beta) #update objective function, b2n here
            
            suff_decr=objnew-objold - sigma*gradd < tol  #sufficient decrease condition
            if print_message == True:
                print('**** objective = ',objnew)
            if n_iter < 2: #force 2 iterations
                decr_alpha = not suff_decr
                betap = beta
            if step_size == 1: #updating step size
                if decr_alpha:
                    if suff_decr:
                        beta = betan
                        beta_est_ls.append(beta)
                        #print("end innter loop when inner iteration=", n_iter)
                        break
                    else:
                        a *= b
                else:
                    if not suff_decr or (betap == betan).all():
                        beta = betap
                        beta_est_ls.append(beta)
                        break
                    else:
                        a /= b
                        betap = betan
            elif step_size == 0: # fixed step size
                if decr_alpha:
                    beta = betan
                    if print_message == True:
                        print("(end 2)")
                    break
                else:
                    if not suff_decr or (betap == betan).all():
                        beta = betap
                        if print_message == True:
                            print("(end 3)")
                        break
                    else:
                        betap = betan

    return beta, grad, iter, grad_ls, obj_ls, beta_est_ls

def subprob_b2_dlearning(n,m1, m2,Y_tilde, X_tilde, beta, lambda_beta, tol, maxiter, a, b, sigma, print_message, step_size):
    """
    Input:
        Y_tilde: transformed Y
        X_tilde: transformed [1 X1 X2], dimension n x (1+m1+m2)
        
    """
    #m1 = X_tilde.shape[1]-r-1
    XtX_tilde = X_tilde.T.dot(X_tilde)
    
    sub_iter = 20
    
    grad_ls = list()
    obj_ls = list()
    beta_est_ls = []
    for iter in range(maxiter) :
        if print_message == True:
            print("---------------------------OUTER LOOP-----iteration:", iter)
        vector = np.ones((1+m1+m2, 1))
        vector[0] = 0 #force the first column to be 0 so that we are not penalizing the intercept

        grad = 1/n*XtX_tilde.dot(beta) - 1/n*X_tilde.T.dot(Y_tilde) + float(lambda_beta)/(m1+m2) * beta * vector #gradient of b2, (1+m1+r) x 1
        projgrad=norm(grad,'fro')
        grad_ls.append(projgrad)
    
        #if projgrad < tol:
        #    break
        grad_diff =  grad_ls[iter-1]- grad_ls[iter]
        if iter > sub_iter and grad_diff < 1e-10:
            if print_message == True:
                print("Difference in current projgrad and previous projgrad=",grad_diff)
                print("(end 1)")
            break
            
        # search for step size alpha
        objold= objective_b2_dlearning(n,m1,m2,Y_tilde, X_tilde, beta, lambda_beta)#update objective function
        obj_ls.append(objold)
        for n_iter in range(sub_iter):
            if print_message == True:
                print("--inner loop--iteration:", n_iter)
            betan = beta - a*grad
            d = betan - beta
            gradd = np.multiply(grad, d).sum()

            objnew = objective_b2_dlearning(n,m1,m2,Y_tilde, X_tilde, betan, lambda_beta)#update objective function, b2n here
            
            suff_decr=objnew-objold - sigma*gradd < tol  #sufficient decrease condition
            if print_message == True:
                print('**** objective = ',objnew)
            if n_iter < 2: #force 2 iterations
                decr_alpha = not suff_decr
                betap = beta
            if step_size == 1: #updating step size
                if decr_alpha:
                    if suff_decr:
                        beta = betan
                        beta_est_ls.append(beta)
                        #print("end innter loop when inner iteration=", n_iter)
                        break
                    else:
                        a *= b
                else:
                    if not suff_decr or (betap == betan).all():
                        beta = betap
                        beta_est_ls.append(beta)
                        break
                    else:
                        a /= b
                        betap = betan
            elif step_size == 0: # fixed step size
                if decr_alpha:
                    beta = betan
                    if print_message == True:
                        print("(end 2)")
                    break
                else:
                    if not suff_decr or (betap == betan).all():
                        beta = betap
                        if print_message == True:
                            print("(end 3)")
                        break
                    else:
                        betap = betan
    return beta, grad, iter, grad_ls, obj_ls

def gradient_descent(X2, X, W, A, Y, Y_tilde, Uinit, Vinit, Vtrue, r, beta_init, alpha1, lambda_u, lambda_v,lambda_beta, d_x, tol, maxiter, maxiter_sub, correction, print_message_snmf, print_message, a, b, sigma, step_size, eval):
    """Projected gradient descent.
    
    Inputs:
        X2: document word matrix
        X: [X1, Uinit], dimension: n x (m1+r)
        W: W=1/p_hat if A=1; W=1/(1-p_hat) if A=-1; dimension: nxn
        Y_tilde: transformed Y
        
        U: not transformed U, no intercept, n x r
        U_tilde: transformed U, with intercept, n x (r+1)
        
        Vinit: initialized V matrix, r x m2
        Vtrue: simulated V,r x m2
        
        r= pre-sepcified number of topics to fit
        beta_init: (m1+r+1) x 1
        
        steps: list of scalar step sizes
        tol: tolerance for a relative stopping condition
        timelimit, maxiter: limit of time and iterations
        
        a: default = 1
        b: default = 0.1
        sigma: default = 0.01
        
    Returns:
        List of all points computed by projected gradient descent.
        U,V, b2: output solution
        projnorm_ls
    
    """
    n = Uinit.shape[0] #number of sample
    R = Vtrue.shape[0] #number of topics
    m1 = X.shape[1] - Uinit.shape[1] #number of EHR features
    m2 = Vinit.shape[1] #number of words 
    
    X1 = X[:,0:m1]
    X_tilde = np.insert(X, 0, 1, axis=1)
    X_tilde = np.sqrt(W).dot(X_tilde)

    U = Uinit.copy()
    V = Vinit.copy()
    beta = beta_init.copy() # (1 + m1 + r) x 1


    beta_0 = vec2col(beta[0])  # 1x1
    beta_x = vec2col(beta[1:m1+1]) # m1 x 1
    beta_u = vec2col(beta[m1+1:]) # r x 1

    VVt=V.dot(V.T)  #gradU
    X2Vt=X2.dot(V.T)  #gradU
    one_n= vec2col(np.ones(n)) #gradU
    w1b0 = np.sqrt(W).dot(one_n).dot(beta_0) #gradU
    wx1bx = np.sqrt(W).dot(X1).dot(beta_x) #gradU
    wubu = np.sqrt(W).dot(U).dot(beta_u) #gradU

    UtU= U.T.dot(U)#gradV
    UtX2=U.T.dot(X2) #gradV

    XtX_tilde=X_tilde.T.dot(X_tilde) #gradb2

    vector = np.ones((1+r+m1, 1))
    vector[0] = 0

    gradU = U.dot(VVt)/(n*m2)-X2Vt/(n*m2)- alpha1/n*np.sqrt(W).dot(Y_tilde - w1b0 - wx1bx - wubu).dot(beta_u.T) + float(lambda_u)/(n*r)*U  # gradient of U, n x r
      
    gradV =  UtU.dot(V)/(n*m2)-UtX2/(n*m2)+float(lambda_v) * V/(r*m2) #gradient of V
    
    gradb2 = alpha1/n*XtX_tilde.dot(beta) - alpha1/n*X_tilde.T.dot(Y_tilde) + float(lambda_beta)/(r*m2) * beta * vector #gradient of b2, (r+m1+1) x 1 

    # calculate initial gradient 
    initgrad = norm(gradU,'fro')+ norm(gradV,'fro') + norm(gradb2,'fro') 

    #tolU = max(1e-6, tol) * initgrad
    #tolV = tolU
    #tolb2 = tolV

    projnorm_ls = list()
    objective_new_ls = list()

    #beta_ls = np.empty(1+r+m1, dtype=object)   #update the dimension of beta
    #beta_ls = vec2col(beta_ls)

    gradU_matrix = [] #np.zeros((maxiter_sub, maxiter)
    objU_matrix = []
    
    gradV_matrix = []#update number of maxiter of subprop_V
    objV_matrix = []
    V_ls = []
    U_ls = []
    beta_ls = []
    
    gradb2_matrix = [] #update number of maxiter of subprop_b2
    objb2_matrix = []

    convergence_step = 1000000000000 #meaning non-convergence/exceed max iteration
    
    for iter in range(maxiter): 
        #stopping condition
        
        #Compute projected gradients norm.
        #np.r_: Translates slice objects to concatenation along the first axis.
        projnorm = np.linalg.norm(np.r_[gradU[np.logical_or(gradU<0, U>0)],
                                        gradV[np.logical_or(gradV<0, V>0)],
                                        gradb2.flatten()])


        projnorm_ls.append(projnorm)

        objective_new = objective_function(n, r,m2,X2, Y_tilde, U, X_tilde, V, beta, alpha1, lambda_u, lambda_v, lambda_beta)
        objective_new_ls.append(objective_new)

        if print_message_snmf == True:
                print ("Iteration #:",iter," Proj norm=",projnorm, "Objective function=", objective_new) 

        #if projnorm < tol * initgrad:
        #    break
        obj_t1 = objective_new_ls[iter]
        obj_t0 = objective_new_ls[iter-1] 
        obj_diff = obj_t0 - obj_t1 # since objective function is decreasing
        if iter > 20 and obj_diff/objective_new_ls[0] < tol: #relative tolerance
            print("!!!At iteration", iter,"Absolute difference in current objective and previous objective / initial objective=",obj_diff/objective_new_ls[0], ", END!!!")
            convergence_step = iter
            break

            
        U, gradU, iterU, gradU_ls, objU_ls = subprob_U(n,r,m2,X2, W, Y_tilde, V, U, X1,beta, alpha1, lambda_u, 1e-20, maxiter_sub, a,  b, sigma, print_message, step_size)
        gradU_matrix.append(gradU_ls)
        objU_matrix.append(objU_ls)
        U_ls.append(U)
        
        X_tilde = np.concatenate((X1, U), axis=1)
        X_tilde = np.insert(X_tilde, 0, 1, axis=1)
        X_tilde = np.sqrt(W).dot(X_tilde)

        #if iterU == 0:
        #    tolU = 0.1 * tolU

        V, gradV, iterV, gradV_ls, objV_ls  = subprob_V(n,r,m2,X2, U, V, lambda_v, 1e-20, maxiter_sub, a, b, sigma, print_message, step_size) #update
        gradV_matrix.append(gradV_ls)
        objV_matrix.append(objV_ls)
        V_ls.append(V)
        
        #if iterV == 0:
        #    tolV = 0.1 * tolV
        beta,gradb2,iterb2, gradb2_ls, objb2_ls,  beta_est_ls= subprob_b2(n,r,m2,Y_tilde, X_tilde, 
                                               beta, alpha1, lambda_beta, 1e-10,
                                               maxiter_sub, a,  b, sigma, print_message, step_size) 
        gradb2_matrix.append(gradb2_ls)
        objb2_matrix.append(objb2_ls)
        beta_ls.append(beta) 
        
        #if iterb2==0:
        #    tolb2 = 0.1 * tolb2  

    #######################
    gradU_matrix = np.array(pd.DataFrame(gradU_matrix))
    gradV_matrix = np.array(pd.DataFrame(gradV_matrix))
    gradb2_matrix = np.array(pd.DataFrame(gradb2_matrix))
    
    objU_matrix = np.array(pd.DataFrame(objU_matrix))
    objV_matrix = np.array(pd.DataFrame(objV_matrix))
    objb2_matrix = np.array(pd.DataFrame(objb2_matrix))
    #######################
    # -   
    #No transformation
    delta_x_hat = beta[0] + X1.dot(beta[1:m1+1])+ U.dot(beta[m1+1:])
    #Y_hat = delta_x_hat
    d_x_hat = np.where(delta_x_hat >0, 1,-1)
    d_x_hat = np.squeeze(d_x_hat)
   
    #Transformation
    X1_tilde = np.sqrt(W).dot(X1)
    U_tilde = np.sqrt(W).dot(U)
    Y_tilde_hat = vec2col(np.diag(W) * beta[0])  + X1_tilde.dot(beta[1:m1+1])+ U_tilde.dot(beta[m1+1:])
    
    # Errors to evaluate performance
    # 1. RMSE
    diff = 2 * Y * A - delta_x_hat
    rmse = np.sqrt(np.mean(diff**2))
    
    # 2. Classification Error = 1-accuracy. Only in simulation
    if eval == True:
        NA, NA, NA, NA, specificity, sensitivity, accuracy = eval_ITR(d_x, d_x_hat)
    elif eval == False:
        specificity = sensitivity = accuracy = None
      
    # 3. V Error
    if r < R:
        print("r<R")
        V_R =  reorder_V(Vtrue, V,  m2, False) #dim: R x m2
        
        v_minus_vhat = Vtrue - V_R
        print("V hat has shape: ", V.shape)
        print("V true has shape: ", Vtrue.shape)
        print("V_R has shape: ", V_R.shape)
        print("Since r<R, reorder V hat and then compute v_minus_vhat.")
        v_minus_vhat_norm = np.linalg.norm(v_minus_vhat, 'fro')/(r*m2)

    elif r>R:
        print("r>R")
        v_minus_vhat = 0
        v_minus_vhat_norm = 0
    elif r==R:
        print("r=R")
        v_minus_vhat = Vtrue - V
        v_minus_vhat_norm = np.linalg.norm(v_minus_vhat, 'fro')/(r*m2)

    
    # X2-UV norm
    print("checkpoint 1.")
    x2_minus_uv =  X2 - np.dot(U, V)
    x2_minus_uv_norm = np.linalg.norm(x2_minus_uv, 'fro')/(n*m2)
    
    print("checkpoint 2.")
    ############## Reorder topics
    if correction == False:
        return convergence_step, rmse, x2_minus_uv_norm, v_minus_vhat_norm, specificity, sensitivity, accuracy, Y_tilde_hat, delta_x_hat, U, V, beta, projnorm_ls, objective_new_ls, beta_ls, gradU_matrix, objU_matrix, gradV_matrix, objV_matrix, gradb2_matrix, objb2_matrix, V_ls, U_ls
    
    else:
        beta_r =  reorder_beta(Vtrue, V, beta, m1, m2)
        V_r =  reorder_V(Vtrue, V,  m2)
        U_r =  reorder_U(Vtrue, V, U, m2)
        beta_ls_r =  reorder_beta_ls(Vtrue, V, beta_ls, m1, m2)
    return convergence_step, rmse, x2_minus_uv_norm, v_minus_vhat_norm, specificity, sensitivity, accuracy, Y_tilde_hat, delta_x_hat, U, V, beta, projnorm_ls, objective_new_ls, beta_ls, gradU_matrix, objU_matrix, gradV_matrix, objV_matrix, gradb2_matrix, objb2_matrix, V_ls, U_ls


def gradient_descent_dlearning(X1, X2, W, A, Y, Y_tilde, beta_init,lambda_beta, d_x, tol, maxiter, maxiter_sub, print_message, a, b, sigma, step_size, eval):
    """Projected gradient descent.
    
    Inputs:
        X_tilde: [1, X1, X2], dimension: n x (1+m1+m2)
        Y_tilde: transformed Y
        
        U: not transformed U, no intercept, n x r
        U_tilde: transformed U, with intercept, n x (r+1)
                
        beta_init: (m1+m2+1) x 1
        
        steps: list of scalar step sizes
        tol: tolerance for a relative stopping condition
        timelimit, maxiter: limit of time and iterations
        
        a: default = 1
        b: default = 0.1
        sigma: default = 0.01
        
    Returns:
        List of all points computed by projected gradient descent.
        U,V, b2: output solution
        projnorm_ls
    
    """
    n = X1.shape[0] #number of sample
    m1 = X1.shape[1] #number of EHR features
    m2 = X2.shape[1] #number of words 
    
    X = np.column_stack((X1, X2))

    X_tilde = np.insert(X, 0, 1, axis=1)
    X_tilde = np.sqrt(W).dot(X_tilde)

    beta = beta_init.copy() # (1 + m1 + r) x 1
#     beta_0 = vec2col(beta[0])  # 1x1
#     beta_x1 = vec2col(beta[1:m1+1]) # m1 x 1
#     beta_x2 = vec2col(beta[m1+1:]) # r x 1

    XtX_tilde=X_tilde.T.dot(X_tilde) #gradb2
    vector = np.ones((1+m1+m2, 1))
    vector[0] = 0

    gradb2 = 1/n*XtX_tilde.dot(beta) - 1/n*X_tilde.T.dot(Y_tilde) + float(lambda_beta)/(m1+m2) * beta * vector #gradient of b2, (r+m1+1) x 1 

    # calculate initial gradient 
    initgrad = norm(gradb2,'fro') 

    projnorm_ls = list()
    objective_new_ls = list()
    objective_diff_ls = list()
#     beta_ls = []
#     gradb2_matrix = [] #update number of maxiter of subprop_b2
#     objb2_matrix = []
    convergence_step = 1000000000000 #meaning non-convergence/exceed max iteration
    
    for iter in range(maxiter): 
        
        objective_new =  objective_b2_dlearning(n,m1,m2,Y_tilde, X_tilde, beta, lambda_beta)
        objective_new_ls.append(objective_new)
    
       # print("Objective:",objective_new_ls)
        #print("Current objective:",objective_new)
            
        projnorm = np.linalg.norm(np.r_[gradb2.flatten()])
        projnorm_ls.append(projnorm)

        # Stopping Condition
        obj_t1 = objective_new_ls[iter]
        obj_t0 = objective_new_ls[iter-1] 
        obj_diff = obj_t0 - obj_t1 # since objective function is decreasing
        objective_diff_ls.append(obj_diff)
        #print("objective diff list : ", objective_diff_ls)
        if iter>20 and obj_diff/objective_new_ls[0] < tol: #relative tolerance

            #print("objective_new_ls[0] = ", objective_new_ls[0])
            print("!!!At iteration", iter,"Absolute difference in current objective and previous objective / initial objective=",obj_diff/objective_new_ls[0], ", END!!!")
           
            convergence_step = iter
            break

        beta, gradb2, iterb2, gradb2_ls, objb2_ls = subprob_b2_dlearning(n,m1, m2,Y_tilde, X_tilde, beta, lambda_beta, 1e-20, maxiter_sub, a, b, sigma, print_message, step_size)
        
#         gradb2_matrix.append(gradb2_ls)
#         objb2_matrix.append(objb2_ls)
#         beta_ls.append(beta) 
        
        #if iterb2==0:
        #    tolb2 = 0.1 * tolb2  

    #######################
    #gradb2_matrix = np.array(pd.DataFrame(gradb2_matrix))
    #objb2_matrix = np.array(pd.DataFrame(objb2_matrix))
    #######################
    #No transformation
    delta_x_hat = beta[0] + X1.dot(beta[1:m1+1])+ X2.dot(beta[m1+1:])
    #Y_hat = delta_x_hat
    d_x_hat = np.where(delta_x_hat >0, 1,-1)
    d_x_hat = np.squeeze(d_x_hat)
   
    #Transformation
    X1_tilde = np.sqrt(W).dot(X1)
    X2_tilde = np.sqrt(W).dot(X2)
    Y_tilde_hat = vec2col(np.diag(W) * beta[0]) + X1_tilde.dot(beta[1:m1+1])+ X2_tilde.dot(beta[m1+1:])
    
    # Errors to evaluate performance
    # 1. RMSE
    diff = 2 * Y * A - delta_x_hat
    rmse = np.sqrt(np.mean(diff**2))
    
    # 2. Classification Error = 1-accuracy. Only in simulation
    if eval == True:
        NA, NA, NA, NA, specificity, sensitivity, accuracy = eval_ITR(d_x, d_x_hat)
    elif eval == False:
        specificity = sensitivity = accuracy = None
    
    return convergence_step, rmse, specificity, sensitivity, accuracy, Y_tilde_hat, delta_x_hat, beta, projnorm_ls, objective_new_ls


def pad_with_nans(x, maxiter_sub):
    """
    Pad a numpy array with NaNs to have maxiter_sub columns if the number of columns is less than 100.
    Parameters:
        x (numpy.ndarray): Input array with shape (n, r).

    Returns:
        numpy.ndarray: gradient_descentPadded array with shape (n, 100) if r < 100, else returns x.
    """
    n, r = x.shape
    if r < maxiter_sub:
        new_x = np.full((n, maxiter_sub), np.nan)
        new_x[:, :r] = x
        return new_x
    else:
        return x

def lambda_tuning(tuning_grid, n, m1, m2, r, maxiter, maxiter_sub, X2, X,W, Y_tilde, Uinit, Vinit,Vtrue,beta_init, alpha1,delta_x, tol, correction, print_message_snmf, print_message_tuning, print_message):
    """ lambda tuning for the penalty terms for U, V, and beta
    Input:
        tuning_grid: a set of lambdas for tuning U, V, and beta. Ex. {0, 0.0001,0.001,0.01,0.1}
    """
    R = Vtrue.shape[0]
    tuning_dim = len(tuning_grid)
    eval_ls = np.zeros((tuning_dim, tuning_dim, tuning_dim))

    # beta
    shape = (tuning_dim, tuning_dim, tuning_dim, 1+m1+R)  # Define the shape of the array: (depth, rows, columns)
    beta_out_ls = np.zeros(shape)
    # convergence of beta 
    shape = (tuning_dim, tuning_dim, tuning_dim, 1+m1+R, maxiter+1)
    beta_ls_out_4d = np.zeros(shape)

    # U
    shape = (tuning_dim, tuning_dim, tuning_dim, n, R)
    U_out_ls = np.zeros(shape)
    
    # V
    shape = (tuning_dim, tuning_dim, tuning_dim, R, m2)
    V_out_ls = np.zeros(shape)
    
    # gradient, objective U and V
    shape = (tuning_dim, tuning_dim, tuning_dim, maxiter, maxiter_sub)
    objU_matrix_out = np.zeros(shape)
    gradU_matrix_out = np.zeros(shape)
    objV_matrix_out = np.zeros(shape)
    gradV_matrix_out = np.zeros(shape)
    objb2_matrix_out = np.zeros(shape)
    gradb2_matrix_out = np.zeros(shape)
    
    for i, l_u in enumerate(tuning_grid):
        print("**1.tuning parameter lambda_u (for penality for U) =", l_u)
        for j, l_v in enumerate(tuning_grid):
            print("************2.tuning parameter lambda_v (for penality for V) =", l_v)
            for k, l_beta in enumerate(tuning_grid):
                print("*************************3.tuning parameter lambda_beta (for penality for beta) =", l_beta)
                #this b2_out is already divided by 2, thus comparable to the beta_true
                evaluation_y, U_out, V_out, b2_out, projnorm_ls, objective_new_ls, b2_ls, gradU_matrix, objU_matrix, gradV_matrix, objV_matrix, gradb2_matrix, objb2_matrix = gradient_descent(X2, X, W, Y_tilde, Uinit,Vinit, Vtrue, r, beta_init, alpha1,l_u, l_v, l_beta, delta_x, tol, maxiter, maxiter_sub, correction, print_message_snmf, print_message, a, b)
                eval_ls[i,j,k] = evaluation_y
                beta_out_ls[i, j, k,:] = b2_out.flatten()  # Set the element at index (0, 1, 2) to 10
                U_out_ls[i,j,k,:,:] = U_out
                V_out_ls[i,j,k,:,:] = V_out
                #beta_ls_out_4d[i,j,k,:,:] = b2_ls
                #objU_matrix_out[i,j,k,:,:] = pad_with_nans(objU_matrix, maxiter_sub)
                #gradU_matrix_out[i,j,k,:,:] = pad_with_nans(gradU_matrix, maxiter_sub)
                #objV_matrix_out[i,j,k,:,:] = pad_with_nans(objV_matrix, maxiter_sub)
                #gradV_matrix_out[i,j,k,:,:] = pad_with_nans(gradV_matrix, maxiter_sub)
                #objb2_matrix_out[i,j,k,:,:] = pad_with_nans(objb2_matrix, maxiter_sub)
                #gradb2_matrix_out[i,j,k,:,:] = pad_with_nans(gradb2_matrix, maxiter_sub)
            
    min_index = np.unravel_index(np.argmin(eval_ls), eval_ls.shape)
    print("Optimal lambda_u=", tuning_grid[min_index[0]], ", lambda_v:", tuning_grid[min_index[1]], ", lambda_beta:", tuning_grid[min_index[2]])
    
    beta_out_tuned = beta_out_ls[min_index]
    U_out_tuned = U_out_ls[min_index[0],min_index[1],min_index[2],:,:]
    V_out_tuned =  V_out_ls[min_index[0],min_index[1],min_index[2],:,:]
   # beta_ls_out_tuned = beta_ls_out_4d[min_index[0],min_index[1],min_index[2],:,:]
   
    # these are used to examine the convergence of the program 
    
    #objU_matrix_tuned = objU_matrix_out[min_index[0],min_index[1],min_index[2],:,:]
    #objV_matrix_tuned = objV_matrix_out[min_index[0],min_index[1],min_index[2],:,:]
   # objb2_matrix_tuned = objb2_matrix_out[min_index[0],min_index[1],min_index[2],:,:]
   # gradU_matrix_tuned = gradU_matrix_out[min_index[0],min_index[1],min_index[2],:,:]
   # gradV_matrix_tuned = gradV_matrix_out[min_index[0],min_index[1],min_index[2],:,:]
   # gradb2_matrix_tuned = gradb2_matrix_out[min_index[0],min_index[1],min_index[2],:,:]
    
    beta_ls_out_tuned = objU_matrix_tuned = objV_matrix_tuned =objb2_matrix_tuned = gradU_matrix_tuned = gradV_matrix_tuned = gradb2_matrix_tuned= None
    
    return(min_index,beta_out_tuned,U_out_tuned, V_out_tuned, beta_ls_out_tuned, objU_matrix_tuned, objV_matrix_tuned,objb2_matrix_tuned, gradU_matrix_tuned, gradV_matrix_tuned,gradb2_matrix_tuned)
