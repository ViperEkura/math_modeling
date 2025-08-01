import cvxpy as cp
import numpy as np

# 定义常量
c1 = np.array([-2, -3])
c2 = np.array([1, 2])
a = np.array([[0.5, 0.25], [0.2, 0.2], [1, 5], [-1, -1]])
b = np.array([8, 4, 72, -10])

# ===== 步骤1: 定义变量和基础约束 =====
x = cp.Variable(2, nonneg=True)
constraints = [a @ x <= b]

# ===== 步骤2: 求解加权目标问题 (0.5*c1 + 0.5*c2) =====
obj1 = (0.5 * c1 + 0.5 * c2) @ x
prob1 = cp.Problem(cp.Minimize(obj1), constraints)
prob1.solve(cp.CBC)
sx = x.value
f1 = -c1 @ sx
f2 = c2 @ sx

print(f"sol1.x = {sx}, fval1 = {prob1.value:.4f}")
print(f"f1 = {f1:.4f}, f2 = {f2:.4f}\n")

# ===== 步骤3: 分别求解单目标优化 =====
# 子问题1: 最小化 c1*x
prob21 = cp.Problem(cp.Minimize(c1 @ x), constraints)
prob21.solve()
sx21 = x.value.copy()  # 保存解
fval21 = prob21.value

# 子问题2: 最小化 c2*x
prob22 = cp.Problem(cp.Minimize(c2 @ x), constraints)
prob22.solve()
sx22 = x.value.copy()  # 保存解
fval22 = prob22.value

print(f"sol21.x = {np.around(sx21, 2)}, fval21 = {fval21:.4f}")
print(f"sol22.x = {np.around(sx22, 2)}, fval22 = {fval22:.4f}\n")

# ===== 步骤4: 求解最小二乘问题 =====
prob23 = cp.Problem(cp.Minimize((c1 @ x - fval21)**2 + (c2 @ x - fval22)**2), constraints)
prob23.solve()
sx23 = x.value
fval23 = prob23.value

print(f"sol23.x = {sx23}, fval23 = {fval23:.4f}\n")

# ===== 步骤5: 约束c1*x为最小值时最小化c2*x =====
constraints2 = constraints + [c1 @ x == fval21]
prob3 = cp.Problem(cp.Minimize(c2 @ x), constraints2)
prob3.solve()
sx3 = x.value
fval3 = prob3.value

print(f"sol3.x = {sx3}, fval3 = {fval3:.4f}")