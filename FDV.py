#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/9 16:21
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : FDV.py
# @Statement : Fuzzy decision variable (FDV) framework
# @Reference : Yang X, Zou J, Yang S, et al. A Fuzzy Decision Variables Framework for Large-Scale Multiobjective Optimization[J]. IEEE Transactions on Evolutionary Computation, 2023, 27(3): 445-459.
import numpy as np
import matplotlib.pyplot as plt


def cal_obj(pop, nobj=3):
    # DTLZ5
    g = np.sum((pop[:, nobj - 1:] - 0.5) ** 2, axis=1)
    temp = np.tile(g.reshape((g.shape[0], 1)), (1, nobj - 2))
    pop[:, 1: nobj - 1] = (1 + 2 * temp * pop[:, 1: nobj - 1]) / (2 + 2 * temp)
    temp1 = np.concatenate((np.ones((g.shape[0], 1)), np.cos(pop[:, : nobj - 1] * np.pi / 2)), axis=1)
    temp2 = np.concatenate((np.ones((g.shape[0], 1)), np.sin(pop[:, np.arange(nobj - 2, -1, -1)] * np.pi / 2)), axis=1)
    return np.tile((1 + g).reshape(g.shape[0], 1), (1, nobj)) * np.fliplr(np.cumprod(temp1, axis=1)) * temp2


def nd_sort(objs):
    # fast non-domination sort
    (npop, nobj) = objs.shape
    n = np.zeros(npop, dtype=int)  # the number of individuals that dominate this individual
    s = []  # the index of individuals that dominated by this individual
    rank = np.zeros(npop, dtype=int)
    ind = 1
    pfs = {ind: []}  # Pareto fronts
    for i in range(npop):
        s.append([])
        for j in range(npop):
            if i != j:
                less = equal = more = 0
                for k in range(nobj):
                    if objs[i, k] < objs[j, k]:
                        less += 1
                    elif objs[i, k] == objs[j, k]:
                        equal += 1
                    else:
                        more += 1
                if less == 0 and equal != nobj:
                    n[i] += 1
                elif more == 0 and equal != nobj:
                    s[i].append(j)
        if n[i] == 0:
            pfs[ind].append(i)
            rank[i] = ind
    while pfs[ind]:
        pfs[ind + 1] = []
        for i in pfs[ind]:
            for j in s[i]:
                n[j] -= 1
                if n[j] == 0:
                    pfs[ind + 1].append(j)
                    rank[j] = ind + 1
        ind += 1
    pfs.pop(ind)
    return pfs, rank


def crowding_distance(objs, pfs):
    # crowding distance
    (npop, nobj) = objs.shape
    cd = np.zeros(npop)
    for key in pfs.keys():
        pf = pfs[key]
        temp_obj = objs[pf]
        fmin = np.min(temp_obj, axis=0)
        fmax = np.max(temp_obj, axis=0)
        df = fmax - fmin
        for i in range(nobj):
            if df[i] != 0:
                rank = np.argsort(temp_obj[:, i])
                cd[pf[rank[0]]] = np.inf
                cd[pf[rank[-1]]] = np.inf
                for j in range(1, len(pf) - 1):
                    cd[pf[rank[j]]] += (objs[pf[rank[j + 1]], i] - objs[pf[rank[j]], i]) / df[i]
    return cd


def selection(pop, rank, cd):
    # binary tournament selection
    (npop, nvar) = pop.shape
    mating_pool = np.zeros((npop, nvar))
    for i in range(npop):
        [ind1, ind2] = np.random.choice(npop, 2, replace=False)
        if rank[ind1] < rank[ind2]:
            mating_pool[i] = pop[ind1]
        elif rank[ind1] == rank[ind2]:
            mating_pool[i] = pop[ind1] if cd[ind1] > cd[ind2] else pop[ind2]
        else:
            mating_pool[i] = pop[ind2]
    return mating_pool


def crossover(mating_pool, lb, ub, eta_c):
    # simulated binary crossover (SBX)
    (noff, nvar) = mating_pool.shape
    nm = int(noff / 2)
    parent1 = mating_pool[:nm]
    parent2 = mating_pool[nm:]
    beta = np.zeros((nm, nvar))
    mu = np.random.random((nm, nvar))
    flag1 = mu <= 0.5
    flag2 = ~flag1
    beta[flag1] = (2 * mu[flag1]) ** (1 / (eta_c + 1))
    beta[flag2] = (2 - 2 * mu[flag2]) ** (-1 / (eta_c + 1))
    beta = beta * (-1) ** np.random.randint(0, 2, (nm, nvar))
    beta[np.random.random((nm, nvar)) < 0.5] = 1
    beta[np.tile(np.random.random((nm, 1)) > 1, (1, nvar))] = 1
    offspring1 = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
    offspring2 = (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
    offspring = np.concatenate((offspring1, offspring2), axis=0)
    offspring = np.min((offspring, np.tile(ub, (noff, 1))), axis=0)
    offspring = np.max((offspring, np.tile(lb, (noff, 1))), axis=0)
    return offspring


def mutation(pop, lb, ub, eta_m):
    # polynomial mutation
    (npop, nvar) = pop.shape
    lb = np.tile(lb, (npop, 1))
    ub = np.tile(ub, (npop, 1))
    site = np.random.random((npop, nvar)) < 1 / nvar
    mu = np.random.random((npop, nvar))
    delta1 = (pop - lb) / (ub - lb)
    delta2 = (ub - pop) / (ub - lb)
    temp = np.logical_and(site, mu <= 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - delta1[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1)
    temp = np.logical_and(site, mu > 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - delta2[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)))
    pop = np.min((pop, ub), axis=0)
    pop = np.max((pop, lb), axis=0)
    return pop


def GA_operator(pop, rank, cd, lb, ub, eta_c, eta_m):
    # genetic algorithm
    mating_pool = selection(pop, rank, cd)
    off = crossover(mating_pool, lb, ub, eta_c)
    off = mutation(off, lb, ub, eta_m)
    return off


def FDV_operator(pop, lb, ub, t, iter, Rate, Acc):
    # fuzzy decision variable
    total = 1
    S = int(np.floor(np.sqrt(2 * Rate * total / Acc)))
    Step = np.zeros(S + 2)
    for i in range(S):
        Step[i + 1] = (S * (i + 1) - (i + 1) ** 2 / 2) * Acc
    Step[-1] = Rate * total  # compensation step
    R = ub - lb
    ratio = (t + 1) / iter
    for i in range(S + 1):
        if Step[i] < ratio <= Step[i + 1]:
            gamma_a = R * 10 ** (-i - 1) * np.floor(10 ** (i + 1) * 1 / R * (pop - lb)) + lb
            gamma_b = R * 10 ** (-i - 1) * np.ceil(10 ** (i + 1) * 1 / R * (pop - lb)) + lb
            miu1 = 1 / (pop - gamma_a)
            miu2 = 1 / (gamma_b - pop)
            flag = miu1 > miu2
            pop = gamma_b
            pop[flag] = gamma_a[flag]
    return pop, cal_obj(pop)


def environmental_selection(pop, objs, npop):
    # the environmental selection of NSGA-II
    [pfs, rank] = nd_sort(objs)
    cd = crowding_distance(objs, pfs)
    temp_list = []
    for i in range(len(pop)):
        temp_list.append([pop[i], objs[i], rank[i], cd[i]])
    temp_list.sort(key=lambda x: (x[2], -x[3]))
    next_pop = np.zeros((npop, pop.shape[1]))
    next_objs = np.zeros((npop, objs.shape[1]))
    next_rank = np.zeros(npop)
    next_cd = np.zeros(npop)
    for i in range(npop):
        next_pop[i] = temp_list[i][0]
        next_objs[i] = temp_list[i][1]
        next_rank[i] = temp_list[i][2]
        next_cd[i] = temp_list[i][3]
    return next_pop, next_objs, next_rank, next_cd


def main(npop, iter, lb, ub, Rate=0.8, Acc=0.4, eta_c=20, eta_m=20):
    """
    The main function
    :param npop: population size
    :param iter: iteration number
    :param lb: lower bound
    :param ub: upper bound
    :param Rate: fuzzy evolution rate (default = 0.8)
    :param Acc: step acceleration (default = 0.4)
    :param eta_c: spread factor distribution index (default = 20)
    :param eta_m: perturbance factor distribution index (default = 20)
    :return:
    """
    # Step 1. Initialization
    nvar = len(lb)  # the dimension of decision space
    pop = np.random.uniform(lb, ub, (npop, nvar))  # population
    objs = cal_obj(pop)  # objectives
    [pfs, rank] = nd_sort(objs)  # Pareto rank
    cd = crowding_distance(objs, pfs)  # crowding distance

    # Step 2. Optimization
    for t in range(iter):

        if (t + 1) % 500 == 0:
            print('Iteration: ' + str(t + 1) + ' completed.')

        # Step 2.1. GA operation
        off = GA_operator(pop, rank, cd, lb, ub, eta_c, eta_m)

        # Step 2.2. FDV operation
        off, off_objs = FDV_operator(off, lb, ub, t, iter, Rate, Acc)

        # Step 2.3. Environmental selection
        pop, objs, rank, cd = environmental_selection(np.concatenate((pop, off), axis=0), np.concatenate((objs, off_objs), axis=0), npop)

    # Step 3. Sort the results
    pf = objs[rank == 1]
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.view_init(45, 45)
    x = [o[0] for o in pf]
    y = [o[1] for o in pf]
    z = [o[2] for o in pf]
    ax.scatter(x, y, z, color='red')
    ax.set_xlabel('objective 1')
    ax.set_ylabel('objective 2')
    ax.set_zlabel('objective 3')
    plt.title('The Pareto front of DTLZ5')
    plt.savefig('Pareto front NSGA-II')
    plt.show()


if __name__ == '__main__':
    main(100, 10000, np.array([0] * 100), np.array([1] * 100))
