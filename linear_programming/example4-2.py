import cvxpy as cp
import numpy as np

# =================== 数据输入 ===================
# 资产数量 n=4
n = 4
M = 10000  # 总资金 M = 10000 元
r0 = 0.05  # 银行利率

# 四种资产的数据（按 s1, s2, s3, s4）
r = np.array([0.28, 0.21, 0.23, 0.25])       # 收益率
q = np.array([0.025, 0.015, 0.055, 0.026])   # 风险损失率
p = np.array([0.01, 0.02, 0.045, 0.065])     # 交易费率
u = np.array([103, 198, 52, 40])              # 最低购买额（本题暂不处理分段费用）

# 目标：总收益至少为 k * M
k = 0.26  # 例如：要求总收益 >= 20% × 10000 = 2000 元

# =================== 变量定义 ===================
# x[0] = 银行存款；x[1:] = s1, s2, s3, s4 投资额
x = cp.Variable(n + 1, nonneg=True)
R = cp.Variable()  # 辅助变量：最大风险

# =================== 目标函数 ===================
# 最小化最大单项风险
objective = cp.Minimize(R)

# =================== 约束条件 ===================
constraints = []

# 1. 每个资产的风险不超过 R
for i in range(1, n + 1):
    constraints.append(q[i - 1] * x[i] <= R)

# 2. 总收益 >= k * M
# 收益 = 银行收益 + 其他资产净收益
# 银行收益: r0 * x[0]
# 其他资产净收益: (r_i - p_i) * x[i] （因为交易费已支出）
total_return = r0 * x[0]
for i in range(1, n + 1):
    total_return += (r[i - 1] - p[i - 1]) * x[i]

constraints.append(total_return >= k * M)

# 3. 总支出（含交易费）等于 M
total_cost = x[0]  # 银行无交易费
for i in range(1, n + 1):
    total_cost += (1 + p[i - 1]) * x[i]  # 花费 = 本金 + 交易费
constraints.append(total_cost == M)

# =================== 求解问题 ===================
prob = cp.Problem(objective, constraints)
prob.solve()

# =================== 输出结果 ===================
if prob.status in ["optimal", "optimal_inaccurate"]:
    print(f"总体最小风险 R = {R.value:.2f} 元")
    print(f"投资组合（单位：元）:")
    print(f"  银行存款: {x[0].value:.2f}")
    for i in range(1, n + 1):
        print(f"  s{i}: {x[i].value:.2f}")
    
    # 计算实际总收益
    actual_return = r0 * x[0].value
    for i in range(1, n + 1):
        actual_return += (r[i - 1] - p[i - 1]) * x[i].value
    print(f"实际总收益: {actual_return:.2f} 元 (目标: {k*M:.2f} 元)")
    
    # 验证总支出
    total_spent = x[0].value
    for i in range(1, n + 1):
        total_spent += (1 + p[i - 1]) * x[i].value
    print(f"总支出（含交易费）: {total_spent:.2f} 元")
    
else:
    print(f"❌ 求解失败，状态: {prob.status}")