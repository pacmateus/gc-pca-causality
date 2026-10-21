"""
Geometric Complexity for Causal Inference
==========================================
Implements fGC (functional Geometric Complexity) method for determining causal direction
between categorical variables using compression-based reasoning.

Reference: Page 5 of main.pdf
fGCX→Y = L(TX|Mk) + GC(Mk; TX) + k log(l) + L(ε|Me)

Where:
  L(TX|Mk)      = Shannon entropy encoding of source variable X
  GC(Mk; TX)    = Geometric complexity of multinomial distribution
  k log(l)      = Complexity of encoding function f (k categories → l categories)
  L(ε|Me)       = Shannon entropy of residuals/errors
"""

import pandas as pd
import numpy as np
from scipy.stats import entropy
import math
from statistics import mean 
import time
from synthetic_data_generation import *


def computefreq(data, k):
    """
    Compute frequency distribution of data values.
    
    Args:
        data: pandas Series of categorical data
        k: list of categories to compute frequencies for
    
    Returns:
        list: normalized frequencies (probabilities) for each category
    """
    r = []
    value_counts = data.value_counts(sort=False)

    for i in k:
        r.append(value_counts[i]/data.size)
    return r

def logfact(n):
    """
    Compute log factorial (log n!).
    Used in geometric complexity calculations.
    
    Args:
        n: integer value
    
    Returns:
        float: sum of log(2) + log(3) + ... + log(n-1)
    """
    r = 0
    for j in range(2,n):
        r = r + math.log(j)
    return r


def LShannon(data,k):
    """
        Computes the Shannon Entropy of the data with k categories

        Args:
        data: pandas Series of categorical data
        k: list of unique categories
    """
    n=data.size
    return n*entropy(computefreq(data, k))

def GCMult(data, k):
    """
    Compute Geometric Complexity (GC) of a multinomial distribution.

    Implements the GC formula from the paper's Theorem 3.6 (Sec. 3):
      GC(M_k; T_X) = -log(n!) + sum log(x_j!) + n*H(p_hat)
        + (k-1)/2 * log(1/(2*pi))               [dimension term, N=1, n=k-1]
        + log( pi^(k/2)/Gamma(k/2) * n^((k-1)/2) )
        + (k-2)*(k-1)/(24n)                     [Ricci scalar curvature]

    The volume term is the corrected Riemannian volume of the simplex with the
    Fisher metric g_ij = n*(delta_ij/p_i + 1/p_k): the positive orthant of
    S^(k-1) of radius 2*sqrt(n) has volume pi^(k/2)/Gamma(k/2) * n^((k-1)/2).
    It replaces the earlier pair
      (k-1)/2*log(1/(2*pi)) + log(n*pi^(k/2)/(2^(k-3)*Gamma(k/2))),
    which grew like n rather than n^((k-1)/2) and so biased the k-dependence of
    the criterion (fixed Sep 2026; see the paper's Theorem 3.6).

    Args:
        data: pandas Series of categorical data
        k: list of unique categories

    Returns:
        float: Geometric complexity value (in nats)
    """
    n = data.size
    # Value counts for multinomial coefficient
    vc = data.value_counts(sort=False)

    # Shannon entropy term: n*H(p̂)
    r = LShannon(data, k)

    # Multinomial coefficient: -log(n!) + sum(log(x_j!))
    r += -math.lgamma(n + 1) + sum(math.lgamma(c + 1) for c in vc.values)

    # Dimension term, from Eq. (14) with N = 1 and n = dim Delta^{k-1} = k-1:
    #   (k-1)/2 * log(1/(2*pi))
    r += (-(len(k)-1) / 2) * math.log(2*math.pi)

    # Volume term: log( pi^(k/2)/Gamma(k/2) * n^((k-1)/2) )
    r += len(k)/2*math.log(math.pi) - math.log(math.gamma(len(k)/2)) \
        + (len(k)-1)/2*math.log(n)

    # Ricci scalar curvature correction: (k-2)(k-1)/(24n)
    r += (len(k)-2)*(len(k)-1)/(24*n)

    return r


def compute_f(dtorig, rorig, ddest, rdest):
    """
    Infer the function f by finding the best mapping from source to destination.
    
    For each source category x, f(x) is determined as the most frequently observed
    paired destination category y. This implements: f(x) = argmax_y count(x,y)
    
    Args:
        dtorig: list of source variable values
        rorig: list of unique source categories (support of X)
        ddest: list of destination variable values (must be same length as dtorig)
        rdest: list of unique destination categories (support of Y)
    
    Returns:
        tuple: (mapping dict, frequency dict)
          - mapping[x] = y: best destination for each source category
          - frequency[x] = p: relative frequency of best pairing for category x
    """

    aux = {}
    
    for i in rorig:
        for j in rdest:
            aux[(i,j)] = 0
    
    if(len(dtorig) != len(ddest)):
        return #levantar erro?
    
    for i in range(0, len(dtorig)):
        aux[(dtorig[i],ddest[i])] += 1 

    r = {}
    for i in rorig:
        maxi = -1
        pmax = -1
        total = 0
        for j in rdest:
            total = total + aux[(i,j)]
            if(maxi < aux[(i,j)]):
                maxi = aux[(i,j)]
                pmax = j
        r[i] = pmax

    freq=0
    for i in range(0, len(dtorig)):
        if(ddest[i]==r[dtorig[i]]):
            freq += 1 
    
    freq = freq/len(dtorig)
    return r,freq


def errorMultprob(r, fs, x, y, k):
    """
    Compute error probability under multinomial error model.
    
    Error model Me: probability (1-p) for zero error, uniform (p/(k-1)) over k-1 other categories.
    If prediction f(x) matches actual y: return high probability freq[x].
    If mismatch: return low probability (1-freq[x])/(rdest-1).
    
    Args:
        r: function mapping (f(x) = r[x])
        fs: frequency of y=f(x) in the data
        x: source category
        y: observed destination category
        k: number of destination categories
    
    Returns:
        float: probability P(Y=y | fs, error model)
    """
    if y == r[x]:
        return fs
    else:
        return (1-fs)/(k-1)


def logerror(dtorig, rorig, ddest, rdest):
    """
    Compute L(ε|Me): Shannon entropy of residuals under error model.
    
    This corresponds to the last term in:
    fGCX→Y = L(TX|Mk) + GC(Mk; TX) + k log(l) + L(ε|Me)
    
    Accumulates -log probabilities of observed residuals given:
    - Inferred function f from source to destination
    - Multinomial error model Me
    
    Args:
        dtorig: list of source variable values
        rorig: list of unique source categories
        ddest: list of destination variable values
        rdest: list of unique destination categories
    
    Returns:
        float: Total log error (sum of -log P(error) for each sample)
    """
    if(len(dtorig) != len(ddest)):
        return

    r = 0
    bestmatch,fa = compute_f(dtorig, rorig, ddest, rdest)

    for i in range(len(dtorig)):
        #print(errorMultprob(bestmatch, freq, dtorig[i], ddest[i], len(rdest)))
        r = r - math.log(errorMultprob(bestmatch, fa, dtorig[i], ddest[i], len(rdest)))
        
    return r


def causalcompl(dtorig, ddest):
    """
    Compute complete fGC score for one causal direction.
    
    Computes: fGCX→Y = L(TX|Mk) + GC(Mk; TX) + k log(l) + L(ε|Me)
    
    Where:
    - L(TX|Mk)       = Shannon entropy encoding of source variable (in GCMult)
    - GC(Mk; TX)     = Geometric complexity of multinomial (from GCMult)
    - k log(l)       = len(rorig) * log(len(rdest)): function mapping complexity
    - L(ε|Me)        = logerror(): residual encoding under error model
    
    Args:
        dtorig: list of source variable values
        ddest: list of destination variable values
    
    Returns:
        float: Total compression cost for this causal direction (lower = better)
    """
    dtorig = pd.Series(dtorig)
    ddest = pd.Series(ddest)
    rorig = dtorig.value_counts().index.to_list()
    rdest = ddest.value_counts().index.to_list()
    
    return GCMult(dtorig, rorig) + len(rorig)*math.log(len(rdest)) + logerror(dtorig, rorig, ddest, rdest)


def causality(x, y):
    """
    Evaluate causal direction using geometric complexity method.
    
    Compares compression cost of X→Y vs Y→X:
    - If fGCX→Y < fGCY→X: X causes Y (returned value negative)
    - If fGCX→Y > fGCY→X: Y causes X (returned value positive)
    
    The direction with LOWER complexity is the true causal direction,
    because true causal relationships compress better.
    
    Parameters:
        x (list): List of x data points (source variable)
        y (list): List of y data points (destination variable)
    
    Returns:
        tuple: (dirXY, dirYX)
          - dirXY: fGC score for X→Y direction
          - dirYX: fGC score for Y→X direction
          Decision: if dirXY < dirYX, then X causes Y
    """
    dirXY = causalcompl(x, y)
    dirYX = causalcompl(y, x)

    return dirXY, dirYX





def main():
    """
    Test harness for geometric complexity causal inference.
    
    Generates synthetic data with known causal relationships (X → Y)
    and evaluates whether the fGC method correctly identifies causality.
    
    Tests across:
    - Multiple cardinalities (k_len): [2,3,5,8,10]
    - Multiple error distributions (y_len): [2,3,5,8,10]
    - Multiple data sizes (data_len): [50]
    - 100 runs per configuration
    
    Tracks:
    - tracker_acc: accuracy (how often fGCY→X > fGCX→Y correctly identifies X→Y)
    - tracker_diff: average difference (fGCY→X - fGCX→Y) per configuration
    - runtimes_avg: execution time per configuration
    """
    data_len = [50]
    #data_len = [100, 1000, 5000, 10000, 50000, 100000]
    #k_len = [10, 50, 100, 200]
    k_len = [2,3,5,8,10]
    y_len = [2,3,5,8,10]
    runtimes = []
    tracker_acc = {}
    tracker_diff = {}
    runtimes_avg = {}
    runs = 100

    print("Ola")
    for y in y_len:
        for k in k_len:
            for N in data_len:
                cnter = 0
                diff_acc = []
                runtimes = []
                for i in range(runs):
                    # Generate random source variable X with k categories
                    dt = np.random.randint(0, k, size=N)
                    
                    # Generate dependent variable Y = f(X) + error
                    par = createSynthetic(k, y, dt, 10, 'dependent')
                    
                    # Time the causality inference
                    start = time.time()
                    a, b = causality(dt, par)  # a=fGCX→Y, b=fGCY→X
                    finish = time.time()
                    runtime = finish - start
                    
                    # Count correct direction identification (should be b > a)
                    if b > a:
                        cnter += 1
                    
                    # Track difference for this run
                    diff_acc.append(b - a)
                    runtimes.append(runtime)
                
                # Store aggregate statistics for this (k,y) configuration
                if (k, y) not in tracker_acc:
                    tracker_acc[(k, y)] = []
                if (k, y) not in tracker_diff:
                    tracker_diff[(k, y)] = []
                
                # Accuracy: proportion of runs with correct direction
                tracker_acc[(k, y)].append((N, cnter/runs))
                # Average difference: mean(fGCY→X - fGCX→Y) should be positive
                tracker_diff[(k, y)].append((N, mean(diff_acc)))
                # Average runtime per configuration
                runtimes_avg[(N, (k, y))] = mean(runtimes)


    print(tracker_acc)
    print(tracker_diff)
    print(runtimes_avg)

if __name__ == '__main__':
    main()