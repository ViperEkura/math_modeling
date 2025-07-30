import cvxpy as cp
import numpy as np

# 定义变量（4x4 矩阵，非负）
x = cp.Variable((4, 4), nonneg=True)

# 定义目标函数
cost = (
    2800 * cp.sum(x[:, 0]) +
    4500 * cp.sum(x[0:3, 1]) + 
    6000 * cp.sum(x[0:2, 2]) +
    7300 * x[0, 3]
)
objective = cp.Minimize(cost)

# 定义约束条件
constraints = [
    cp.sum(x[0, :]) >= 15,                           
    cp.sum(x[0, 1:4]) + cp.sum(x[1, 0:3]) >= 10,
    x[0, 2] + x[0, 3] + x[1, 1] + x[1, 2] + x[2, 0] + x[2, 1] >= 20,
    x[0, 3] + x[1, 2] + x[2, 1] + x[3, 0] >= 12
]

# 求解问题
problem = cp.Problem(objective, constraints)
problem.solve()

print(f"status: {problem.status}")
print("matrix x:")
print(np.round(x.value, 2))
print(f"z = {problem.value:.2f}")
