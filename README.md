### FDV: Fuzzy decision variable for large-scale multiobjective optimization

##### Reference: Yang X, Zou J, Yang S, et al. A fuzzy decision variables framework for large-scale multiobjective optimization[J]. IEEE Transactions on Evolutionary Computation, 2023, 27(3): 445-459.

##### FDV is a framework for LSMOP. FDV divides the entire optimization process into two stages: (1) fuzzy stage: decision variables are blurred to reduce the decision space; (2) precision stage optimizes original decision variables. This code cooperates FDV and NSGA-II.

| Variables | Meaning                                              |
| --------- | ---------------------------------------------------- |
| npop      | Population size                                      |
| iter      | Iteration number                                     |
| lb        | Lower bound                                          |
| ub        | Upper bound                                          |
| Rate      | Fuzzy evolution rate (default = 0.8)                 |
| Acc       | Step acceleration (default = 0.4)                    |
| eta_c     | Spread factor distribution index (default = 30)      |
| eta_m     | Perturbance factor distribution index (default = 20) |
| nvar      | The dimension of decision space                      |
| pop       | Population                                           |
| objs      | Objectives                                           |
| rank      | Pareto rank                                          |
| cd        | Crowding distance                                    |
| pf        | Pareto front                                         |



#### Test problem: DTLZ5

$$
\begin{aligned}
	& \theta_i = \frac{\pi}{4(1 + g(x_M))}(1 + 2g(x_M)x_i), \quad i = 1, \cdots, n \\
	& g(x_M) = \sum_{x_i \in x_M} (x_i - 0.5) ^ 2 \\
	& \min \\
	& f_1(x) = (1 + g(x_M)) \cos(\theta_i \pi / 2) \cdots \cos(\theta_{M-2} \pi /2) \cos(\theta_{M - 1} \pi /2) \\
	& f_2(x) = (1 + g(x_M)) \cos(\theta_i \pi / 2) \cdots \cos(\theta_{M-2} \pi /2) \sin(\theta_{M - 1} \pi /2) \\
	& f_3(x) = (1 + g(x_M)) \cos(\theta_i \pi / 2) \cdots \sin(\theta_{M-2} \pi /2) \\
	& \vdots \\
	& f_M(x) = (1 + g(x_M)) \sin(\theta_1 \pi /2) \\
	& \text{subject to} \\
	& x_i \in [0, 1], \quad \forall i = 1, \cdots, n
\end{aligned}
$$



#### Example

```python
if __name__ == '__main__':
    main(100, 10000, np.array([0] * 100), np.array([1] * 100))
```

##### Output:

![Pareto front](/Users/xavier/Desktop/Xavier Ma/个人算法主页/Fuzzy Decision Variable/Pareto front.png)



