import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt


x = cp.Variable(5, nonneg=True)
c = np.array([0.05, 0.27, 0.19, 0.185, 0.185])
Aeq = np.array([1, 1.01, 1.02, 1.045, 1.065])
M = 10000
q = np.array([0.025, 0.015, 0.055, 0.026])

a_list = []
Q_list = []
X_list = []

a = 0.0
while a < 0.05:
    objective = cp.Maximize(c @ x)
    constraints = [
        Aeq @ x == M,
        q @ x[1:] <= a * M,
        x >= 0  
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"求解失败, a = {a:.3f}, 状态: {prob.status}")
        a += 0.001
        continue
    
    total_risk = q @ x.value[1:]
    a_list.append(a)
    Q_list.append(total_risk)
    X_list.append(x.value.copy())
    a += 0.001

plt.figure(figsize=(8, 5))
plt.plot(a_list, Q_list, 'k*-', markersize=4)
plt.xlabel('a', fontsize=14)
plt.ylabel('Q', fontsize=14)
plt.title('a - Q')
plt.grid(True, alpha=0.3)
plt.show()
