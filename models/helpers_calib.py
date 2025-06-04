import numpy as np
import cvxpy as cp

import numpy as np
from scipy.optimize import linprog
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
import torch
import sys
sys.setrecursionlimit(100000)  # Increase the limit


def sample_from_high_uq(logits_V: torch.Tensor):#, mask: torch.Tensor):
    V = logits_V.shape[0]
    num_samples = 5
    probs = logits_V.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=num_samples, replacement=False, generator=None).view(num_samples)


def sample_from_prediction_set(logits_BlV: torch.Tensor, mask, rng=None, num_samples=1) -> torch.Tensor:
    B, l, V = logits_BlV.shape
    logits_BlV.masked_fill_(~mask, -10000000000000000)
    # sample (have to squeeze cuz torch.multinomial can only be used for 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)


def generate_high_uq_samples(uq_classes: torch.Tensor, idx_Bl: torch.Tensor, index_uq): 
    num_classes  = uq_classes.shape[0]
    L = idx_Bl.shape[-1]
    expanded_indcs = idx_Bl.expand(num_classes, L).clone() 
    expanded_indcs[:, index_uq] = uq_classes
    return expanded_indcs

# Minimizing the entropy is somewhat tricky since, by definition the min. values of the entropy 
# are probably gonna be on the vertices of the feasible set. However, these aren't in the solution to 
# the classical optimization algorithms. Thus, the idea is recursivley find the most concentrated distribution under the constraint 
# tau (and thus by the resepctive qhat). 
def min_entropy(x, tau, n_max, n_min): 
    x = np.sort(x)
    x_max = x[-n_max:]
    x_min = x[:n_min]
    rel_x = np.concat([x_min,x_max])
    vec = [1]*n_min + [-1]*n_max
    
    n = len(rel_x)
    pi = cp.Variable(n)
    objective = cp.Maximize(vec @ pi)

    # Constraints
    constraints = [
        cp.sum(pi) == 1,  # Sum constraint
        rel_x @ pi <= tau, # Linear constraint: x1*pi1 + x2*pi2 <= tau
        pi >= 0,           # Non-negativity
        pi <= 1            # Upper bound
    ]

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    try: 
        problem.solve()
    except:
        print('solver failed')
        return(1)
    if problem.status in ["optimal", "optimal_inaccurate"]:
        pi_opt = pi.value 
        eps = 1e-15
        positive_idx = (pi_opt > 0)
        pi_pos = pi_opt[positive_idx]
        entropy = -np.sum(pi_pos * np.log(pi_pos))
        return entropy / np.log(4096)
    else:
        print("No optimal solution found.")
        # recursive strategy for restricting the 
        if (n_max == 2): 
            return(1)
        if(n_max + n_min >= 4096 and n_max == 4096):
            return(1)
        if n_max + n_min == 4096: 
            n_max += n_max
            n_min = 1
        else: 
            n_min += 1
        return(min_entropy(x, tau, n_max=n_max, n_min=n_min))
    
    
# Idea behind that: Use entropy as measure for uncertainty, get the combination of lambdas that maximizes 
# entropy as upper bound of CCR (constraints are defined by the score from the calibration procedure)
def calc_entropy(x, tau, maximize):
    x = x[x != 0 ]
    if maximize: 
        lambda_k = cp.Variable(len(x), nonneg=True)
        # Define entropy function
        entropy = cp.sum(cp.entr(lambda_k))
        # Constraints
        constraints = [
            cp.sum(lambda_k) == 1,  
            x @ lambda_k <= tau,
            lambda_k >= 0, 
            lambda_k <= 1
        ]

        max_entropy_problem = cp.Problem(cp.Maximize(entropy), constraints)
        try: 
            max_entropy_problem.solve(
            solver=cp.MOSEK
        )
        except:
            print('solver failed')
            return(1)
        return max_entropy_problem.value / np.log(4096) #, lambda_k.value
    else: 
        if (x <= tau).any(): 
            # degenerate distribution p present
            return 0
        else: 
            return(min_entropy(x, tau=tau, n_max=1, n_min=1))
        
        
# ------------ Experimental for Rao Entropy
def calc_quadratic_entropy(x, D, tau, maximize=True):
    x = x[x != 0]
    if maximize:
        n = len(x)
        lambda_k = cp.Variable(n, nonneg=True)
        
        # Objective: Maximize lambda^T D lambda (Quadratic/Rao entropy)
        entropy = cp.quad_form(lambda_k, cp.psd_wrap(D))
        constraints = [
            cp.sum(lambda_k) == 1,            
            x @ lambda_k <= tau,              
            lambda_k >= 0,
            lambda_k <= 1
        ]

        # Define and solve the problem
        problem = cp.Problem(cp.Minimize(entropy), constraints)
        problem.solve(solver=cp.MOSEK)

        return problem.value  # No need to normalize since it's not Shannon entropy
    else:
        raise NotImplementedError("Minimizing Rao entropy under constraints not implemented.")
