# example 2 - 5

import cvxpy as cp

# 定义决策变量（6个非负整数变量）
x = cp.Variable(6, integer=True, nonneg=True)

# 定义目标函数（最小化 x1 + x2 + ... + x6）
objective = cp.Minimize(cp.sum(x))

# 定义约束条件
constraints = [
    x[0] + x[5] >= 35,  # x1 + x6 ≥ 35
    x[0] + x[1] >= 40,   # x1 + x2 ≥ 40
    x[1] + x[2] >= 50,   # x2 + x3 ≥ 50
    x[2] + x[3] >= 45,   # x3 + x4 ≥ 45
    x[3] + x[4] >= 55,   # x4 + x5 ≥ 55
    x[4] + x[5] >= 30    # x5 + x6 ≥ 30
]

# 创建问题并求解
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.CBC, verbose=True)

# 输出结果
print("z:", problem.value)

for i in range(1, 7):
    print("x[", i, "] =", x[i - 1].value)