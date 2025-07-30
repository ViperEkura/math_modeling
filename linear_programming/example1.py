import cvxpy as cp

# 定义变量（非负约束）
x1 = cp.Variable(nonneg=True)
x2 = cp.Variable(nonneg=True)

# 定义目标函数（最大化）
objective = cp.Maximize(4 * x1 + 3 * x2)
# 定义约束条件
constraints = [
    2 * x1 + x2 <= 10,
    x1 + x2 <= 8,
    x2 <= 7
]

prob = cp.Problem(objective, constraints)
prob.solve()

print("Status:", prob.status)
print(f"x1 = {x1.value:.2f}, x2 = {x2.value:.2f}")
print(f"z = {prob.value:.2f}")