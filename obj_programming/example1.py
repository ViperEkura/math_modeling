import cvxpy as cp
import numpy as np

# 定义决策变量
x1 = cp.Variable()  # 第一个决策变量
x2 = cp.Variable()  # 第二个决策变量
d1_minus = cp.Variable(nonneg=True)  # 目标1的负偏差
d1_plus = cp.Variable(nonneg=True)   # 目标1的正偏差
d2_minus = cp.Variable(nonneg=True)  # 目标2的负偏差
d2_plus = cp.Variable(nonneg=True)   # 目标2的正偏差
d3_minus = cp.Variable(nonneg=True)  # 目标3的负偏差
d3_plus = cp.Variable(nonneg=True)   # 目标3的正偏差

# 假设优先级权重 P1, P2, P3 (这里需要根据实际情况设定具体数值)
# 为了演示，我们假设 P1=1, P2=1, P3=1
P1 = 1
P2 = 1
P3 = 1

# 目标函数: 最小化加权偏差
objective = cp.Minimize(P1 * d1_plus + P2 * (d2_minus + d2_plus) + P3 * d3_minus)

# 约束条件
constraints = [
    2 * x1 + x2 <= 11,                   # 资源约束
    x1 - x2 + d1_minus - d1_plus == 0,   # 目标1约束
    x1 + 2 * x2 + d2_minus - d2_plus == 10,  # 目标2约束
    8 * x1 + 10 * x2 + d3_minus - d3_plus == 56,  # 目标3约束
    x1 >= 0,
    x2 >= 0
    # d_i 变量已定义为非负，无需额外约束
]

# 创建问题并求解
prob = cp.Problem(objective, constraints)
prob.solve()

# 输出结果
print(f"最优值: {prob.value}")
print(f"x1 = {x1.value}")
print(f"x2 = {x2.value}")
print(f"d1_minus = {d1_minus.value}")
print(f"d1_plus = {d1_plus.value}")
print(f"d2_minus = {d2_minus.value}")
print(f"d2_plus = {d2_plus.value}")
print(f"d3_minus = {d3_minus.value}")
print(f"d3_plus = {d3_plus.value}")