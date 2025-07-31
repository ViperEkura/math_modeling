import cvxpy as cp
import numpy as np

# 定义优化变量
x = cp.Variable(2, nonneg=True)  # 2维非负变量
dp = cp.Variable(4, nonneg=True) # 4维非负正偏差
dm = cp.Variable(4, nonneg=True) # 4维非负负偏差

# 定义基础约束
constraints = [
    2 * cp.sum(x) <= 12,  # 约束1: 2*(x1+x2) <= 12
    
    # 目标约束（带偏差变量）
    200*x[0] + 300*x[1] + dm[0] - dp[0] == 1500,
    2*x[0] - x[1] + dm[1] - dp[1] == 0,
    4*x[0] + dm[2] - dp[2] == 16,
    5*x[1] + dm[3] - dp[3] == 15
]

# 定义多目标分量
mobj = [
    dm[0],                    # 第一目标: dm1
    dp[1] + dm[1],            # 第二目标: dp2 + dm2
    3*dp[2] + 3*dm[2] + dp[3] # 第三目标: 3*dp3 + 3*dm3 + dp4
]

# 初始化目标上限
goal = np.array([100000.0, 100000.0, 100000.0])

# 多级目标优化
for i in range(3):
    # 添加当前目标上限约束
    current_constraints = constraints + [
        mobj[0] <= goal[0],
        mobj[1] <= goal[1],
        mobj[2] <= goal[2]
    ]
    
    # 创建优化问题（最小化当前目标）
    prob = cp.Problem(cp.Minimize(mobj[i]), current_constraints)
    prob.solve()
    
    print(f'第{i+1}级目标计算结果如下：')
    print(f'目标值:{prob.value:.2f}')
    print(f'x = {np.around(x.value, 2)}')
    print(f'dm = {np.around(dm.value, 2)}')
    print(f'dp = {np.around(dp.value, 2)}')
    
    # 更新当前目标上限
    goal[i] = prob.value