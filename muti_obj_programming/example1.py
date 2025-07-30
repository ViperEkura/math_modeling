import numpy as np
import cvxpy as cp


x = cp.Variable(2, nonneg=True)  # 非负变量

# 定义目标函数系数
c1 = np.array([-2, -3])
c2 = np.array([1, 2])

# 定义约束矩阵和向量
a = np.array([[0.5, 0.25], [0.2, 0.2], [1, 5], [-1, -1]])
b = np.array([8, 4, 72, -10])

# 添加约束条件
constraints = [a @ x <= b]

# 定义第一个目标函数
obj1 = 0.5 * c1 @ x + 0.5 * c2 @ x

# 创建一个新的问题实例，并设置目标函数
prob1 = cp.Problem(cp.Minimize(obj1), constraints)

# 求解问题
prob1.solve()
sol1 = x.value
fval1 = prob1.value

# 计算两个目标函数的值
f1 = -c1 @ sol1
f2 = c2 @ sol1

# 显示结果
print("Solution: ")
print(sol1)
print("Objective function value: ")
print(fval1)
print("Values of the two objective functions:")
print(f1)
print(f2)

# 创建一个新的问题实例，并设置第一个目标函数
prob21 = cp.Problem(cp.Minimize(c1 @ x), constraints)

# 求解第一个优化问题
prob21.solve()
sol21 = x.value
fval21 = prob21.value

# 创建一个新的问题实例，并设置第二个目标函数
prob22 = cp.Problem(cp.Minimize(c2 @ x), constraints)

# 求解第二个优化问题
prob22.solve()
sol22 = x.value
fval22 = prob22.value

# 创建一个新的问题实例，并设置第三个目标函数（最小化两个目标函数值与之前求得的最优值的偏差平方和）
obj23 = cp.sum_squares(c1 @ x - fval21) + cp.sum_squares(c2 @ x - fval22)
prob23 = cp.Problem(cp.Minimize(obj23), constraints)

# 求解第三个优化问题
prob23.solve()
sol23 = x.value
fval23 = prob23.value

# 创建一个新的问题实例，并设置目标函数和约束条件
prob3 = cp.Problem(cp.Minimize(c2 @ x), constraints + [c1 @ x == fval21])

# 求解优化问题
prob3.solve()
sol3 = x.value
fval3 = prob3.value