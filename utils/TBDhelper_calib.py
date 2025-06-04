import numpy as np
import cvxpy as cp

import numpy as np
from scipy.optimize import linprog
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw

def min_entropy(x, tau, n_max, n_min): 
    x = np.sort(x)
    x_max = x[-n_max:]
    x_min = x[:n_min]
    rel_x = np.concat([x_min,x_max])
    vec = [1]*n_min + [-1]*n_max
    
    n = len(rel_x)
    pi = cp.Variable(n)  # pi1, pi2

    # Objective function (maximize pi1 - pi2)
    objective = cp.Maximize(vec @ pi)
    
    # Objective function (maximize pi1 - pi2)
    objective = cp.Maximize(vec @ pi)

    # Constraints
    constraints = [
        cp.sum(pi) == 1,  # Sum constraint
        rel_x @ pi <= tau,     # Linear constraint: x1*pi1 + x2*pi2 <= tau
        pi >= 0,           # Non-negativity
        pi <= 1            # Upper bound
    ]

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()
    # Output results
    if problem.status in ["optimal", "optimal_inaccurate"]:
        print("Optimal Min. found!")
        print("pi values:", pi.value)
        print("Objective function value:", problem.value)
        return problem.value / np.log(4096) #, np.round(pi.value)
    else:
        print("No optimal solution found.")
        if(n_max + n_min >= 4096 and n_max == 4096):
            return(1)
        if n_max + n_min == 4096: 
            n_max += n_max
            n_min = 1
        else: 
            n_min += 1
        return(min_entropy(x, tau, n_max=n_max, n_min=n_min))
    
    

def calc_entropy(x, tau, maximize):
    x = x[x != 0 ]
    #print(x.shape)
    if maximize: 
        lambda_k = cp.Variable(len(x), nonneg=True)

        # Define entropy function
        entropy = cp.sum(cp.entr(lambda_k))  # Negative entropy (minimization)
        # Constraints
        constraints = [
            cp.sum(lambda_k) == 1,  # Probability simplex constraint
            x @ lambda_k <= tau,  # Linear constraint
            lambda_k >= 0, 
            lambda_k <= 1
        ]

        max_entropy_problem = cp.Problem(cp.Maximize(entropy), constraints)

        # Solve the problem
        max_entropy_problem.solve# Solve with Clarabel and custom settings
        max_entropy_problem.solve(
            solver=cp.MOSEK
        )

        # Output results
        print("Lambda values (Max Entropy):", lambda_k.value)
        print("Maximum Entropy:", max_entropy_problem.value /np.log(4096))
        return max_entropy_problem.value /np.log2(4096) #, lambda_k.value
    else: 
        #print(sum(x <= tau))
        if (x <= tau).any(): 
            print("Degenerate Distr. present")
            return 0
        else: 
            return(min_entropy(x, tau=tau, n_max=1, n_min=1))