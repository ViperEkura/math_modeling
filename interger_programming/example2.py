# example 2 - 6

import cvxpy as cp
import numpy as np

# 费用矩阵 (4 家公司 × 5 个门店)
c = np.array([
    [15.0, 13.8, 12.5, 11.0, 14.3],  # A
    [14.5, 14.0, 13.2, 10.5, 15.0],  # B
    [13.8, 13.0, 12.8, 11.3, 14.6],  # C
    [14.7, 13.6, 13.0, 11.6, 14.0]   # D
])

# 定义决策变量
x = cp.Variable((4, 5), boolean=True)

# 目标函数：总费用最小化
objective = cp.Minimize(cp.sum(cp.multiply(c, x)))

# 约束条件
constraints = [
    # 每个门店只能被一家公司负责（列和为 1）
    cp.sum(x[:, j]) == 1 for j in range(5)
] + [
    # 每家公司最多负责 2 个门店（行和 ≤ 2）
    cp.sum(x[i, :]) <= 2 for i in range(4)
]

# 建立并求解问题
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.GLPK_MI)

# 输出结果
print("最优总费用：", problem.value)
print("\n分配方案: ")
for i in range(4):
    for j in range(5):
        if x[i, j].value > 0:
            print(f"公司 {['A','B','C','D'][i]} 负责门店 {j+1}，费用 {c[i,j]} 万元")