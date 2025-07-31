import cvxpy as cp
import numpy as np

# 初始化问题
x = cp.Variable(2, nonneg=True)  # 2维非负变量

# 定义系数
c1 = np.array([-2, -3])
c2 = np.array([1, 2])
a = np.array([[0.5, 0.25], [0.2, 0.2], [1, 5], [-1, -1]])
b = np.array([8, 4, 72, -10])

# 基础约束
constraints = [a @ x <= b]

# --- 第一部分：线性组合目标 ---
obj1 = 0.5 * c1 @ x + 0.5 * c2 @ x
prob1 = cp.Problem(cp.Minimize(obj1), constraints)
prob1.solve()

print("第一部分：线性组合目标")
print("最优值 fval1 =", np.around(prob1.value, 2))  
print("最优解 sx =", np.around(x.value, 2))  
f1 = -c1 @ x.value  # 计算实际目标值 f1 = 2x₁ + 3x₂
f2 = c2 @ x.value    # 计算实际目标值 f2 = x₁ + 2x₂
print(f"f1 = {np.around(f1, 2)}, f2 = {np.around(f2, 2)}\n")  

# --- 第二部分：单目标优化 ---
# 最小化 c1*x
prob21 = cp.Problem(cp.Minimize(c1 @ x), constraints)
prob21.solve()
sx21 = x.value.copy()
fval21 = prob21.value
print("第二部分：最小化 c1*x")
print("最优值 fval21 =", np.around(fval21, 2))  
print("最优解 sx21 =", np.around(sx21, 2))  

# 最小化 c2*x
prob22 = cp.Problem(cp.Minimize(c2 @ x), constraints)
prob22.solve()
sx22 = x.value.copy()
fval22 = prob22.value
print("\n最小化 c2*x")
print("最优值 fval22 =", np.around(fval22, 2))  
print("最优解 sx22 =", np.around(sx22, 2))  

# 最小化到理想点的距离平方
prob23 = cp.Problem(
    cp.Minimize((c1 @ x - fval21)**2 + (c2 @ x - fval22)**2),
    constraints
)
prob23.solve()
sx23 = x.value.copy()
fval23 = prob23.value
print("\n最小化到理想点的距离平方")
print("最优值 fval23 =", np.around(fval23, 2))  
print("最优解 sx23 =", np.around(sx23, 2))  
print(f"实际目标值: f1 = {np.around(-c1@sx23, 2)}, f2 = {np.around(c2@sx23, 2)}\n")  

# --- 第三部分：分层优化 ---
# 在 c1*x = fval21 的条件下最小化 c2*x
constraints3 = constraints + [c1 @ x == fval21]
prob3 = cp.Problem(cp.Minimize(c2 @ x), constraints3)
prob3.solve()
sx3 = x.value.copy()
fval3 = prob3.value
print("第三部分：分层优化")
print("最优值 fval3 =", np.around(fval3, 2))  
print("最优解 sx3 =", np.around(sx3, 2))  
print(f"实际目标值: f1 = {np.around(-c1@sx3, 2)}, f2 = {np.around(c2@sx3, 2)}")  