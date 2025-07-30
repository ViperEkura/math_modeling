import cvxpy as cp
import numpy as np

# 网点坐标数据
coordinates = np.array([
    [9.488, 5.681],
    [8.792, 10.38],
    [11.59, 3.929],
    [11.56, 4.432],
    [5.675, 9.965],
    [9.849, 17.66],
    [9.175, 6.151],
    [13.13, 11.85],
    [15.46, 8.872],
    [15.54, 15.58]
])
n = coordinates.shape[0]

# 计算欧几里得距离矩阵
def calculate_distance_matrix(coords):
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
    return distance_matrix

d = calculate_distance_matrix(coordinates)

x = cp.Variable(n, boolean=True)     # x[i] = 1 表示在网点i建站
y = cp.Variable((n, n), boolean=True) # y[i,j] = 1 表示网点j被网点i覆盖

# 4. 目标函数
objective = cp.Minimize(cp.sum(x))

# 5. 约束条件
constraints = []

# 约束1：每个网点至少被一个供应站覆盖
for j in range(n):
    constraints.append(cp.sum(y[:, j]) >= 1)

# 约束2：每个供应站最多覆盖5个网点（包括自己）
for i in range(n):
    constraints.append(cp.sum(y[i, :]) <= 5)

# 约束3：自覆盖约束 x[i] == y[i,i]
for i in range(n):
    constraints.append(y[i, i] == x[i])

# 约束4：距离约束 d[i,j] * y[i,j] <= 10 * x[i]
#        且 y[i,j] <= x[i]
for i in range(n):
    for j in range(n):
        if i != j:
            constraints.append(d[i, j] * y[i, j] <= 10 * x[i])
        constraints.append(y[i, j] <= x[i])

# 6. 建立并求解问题
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.GLPK_MI, verbose=True)

# 7. 输出结果
print("状态:", problem.status)
print("最优目标值:", problem.value)
print("\n供应站选址方案：")
supply_stations = []
for i in range(n):
    if x[i].value > 0:
        supply_stations.append(i + 1)
        print(f"在网点 {i + 1} 建立供应站")

print("\n覆盖关系：")
for i in range(n):
    for j in range(n):
        if y[i, j].value > 0:
            print(f"网点 {j + 1} 被网点 {i + 1} 的供应站覆盖")