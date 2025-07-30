import cvxpy as cp

# 定义变量（非负约束）
x1 = cp.Variable(nonneg=True)
x2 = cp.Variable(nonneg=True)
x3 = cp.Variable(nonneg=True)

# 定义目标函数（最大化）
objective = cp.Maximize(2 * x1 + 3 * x2 - 5 * x3)

# 定义约束条件
constraints = [
    x1 + x2 + x3 == 7,                  # 等式约束
    2 * x1 - 5 * x2 + x3 >= 10,         # 不等式约束（≥）
    x1 + 3 * x2 + x3 <= 12,             # 不等式约束（≤）
]

# 创建问题并求解
problem = cp.Problem(objective, constraints)
problem.solve()


print("Status:", problem.status)
print(f"x1 = {x1.value:.4f}, x2 = {x2.value:.4f}, x3 = {x3.value:.4f}")
print(f"z = {problem.value:.4f}")