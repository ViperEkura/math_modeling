import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# =================== 数据输入 ===================
n = 4
M = 10000  # 总资金
r0 = 0.05  # 银行利率

# 四种资产数据：s1, s2, s3, s4
r = np.array([0.28, 0.21, 0.23, 0.25])       # 收益率
q = np.array([0.025, 0.015, 0.055, 0.026])   # 风险损失率
p = np.array([0.01, 0.02, 0.045, 0.065])     # 交易费率

# =================== 设置 k 的范围 ===================
k_min = 0.05
k_max = 0.30
k_step = 0.01
k_values = np.arange(k_min, k_max + k_step, k_step)

# 存储结果
R_values = []
s1_values = []
s2_values = []
s3_values = []
s4_values = []
bank_values = []
feasible_k = []

# =================== 遍历每个 k 求解 ===================
for k in k_values:
    x = cp.Variable(n + 1, nonneg=True)
    R = cp.Variable()

    objective = cp.Minimize(R)
    constraints = []

    # 风险约束
    for i in range(1, n + 1):
        constraints.append(q[i - 1] * x[i] <= R)

    # 收益约束
    total_return = r0 * x[0] + sum((r[i - 1] - p[i - 1]) * x[i] for i in range(1, n + 1))
    constraints.append(total_return >= k * M)

    # 资金约束
    total_cost = x[0] + sum((1 + p[i - 1]) * x[i] for i in range(1, n + 1))
    constraints.append(total_cost == M)

    prob = cp.Problem(objective, constraints)
    prob.solve()

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"⚠️  k = {k:.2f} 不可行")
        continue

    feasible_k.append(k)
    R_values.append(R.value)
    bank_values.append(x[0].value)
    s1_values.append(x[1].value)
    s2_values.append(x[2].value)
    s3_values.append(x[3].value)
    s4_values.append(x[4].value)

# =================== 绘图：合并为双子图 ===================
if len(feasible_k) == 0:
    print("❌ 所有 k 值均无可行解，请检查参数。")
else:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    # --- 上图：k-R 曲线 ---
    ax1.plot(feasible_k, R_values, 'b-o', linewidth=2, markersize=4, label='最小风险 $R$')
    ax1.set_ylabel('最小可实现风险 $R$（元）', fontsize=11)
    ax1.set_title('收益目标 $k$ 与风险及投资组合的关系', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.4)
    ax1.legend()
    ax1.set_ylim(bottom=0)  # 从 0 开始

    # --- 下图：投资组合 ---
    ax2.plot(feasible_k, s1_values, 'o-', label='s1', linewidth=2, markersize=4)
    ax2.plot(feasible_k, s2_values, 's-', label='s2', linewidth=2, markersize=4)
    ax2.plot(feasible_k, s3_values, '^-', label='s3', linewidth=2, markersize=4)
    ax2.plot(feasible_k, s4_values, 'D-', label='s4', linewidth=2, markersize=4)
    ax2.plot(feasible_k, bank_values, 'v-', label='银行存款', linewidth=2, markersize=4)

    ax2.set_xlabel('最低收益要求 $k$', fontsize=12)
    ax2.set_ylabel('投资金额（元）', fontsize=11)
    ax2.grid(True, alpha=0.4)
    ax2.legend()
    ax2.set_ylim(bottom=0)

    # 调整子图间距
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)  # 增加上下图间距
    plt.savefig('investment_portfolio.png', dpi=300)
    plt.show()

    # =================== 输出信息 ===================
    print(f"✅ 可行收益范围: k ∈ [{feasible_k[0]:.2f}, {feasible_k[-1]:.2f}]")
    print(f"对应风险范围: R ∈ [{R_values[0]:.1f}, {R_values[-1]:.1f}] 元")

    k_final = feasible_k[-1]
    print(f"\n📈 当追求最高可行收益 k = {k_final:.2f} 时的投资组合（单位：元）:")
    print(f"  银行存款: {bank_values[-1]:.2f}")
    print(f"  s1:       {s1_values[-1]:.2f}")
    print(f"  s2:       {s2_values[-1]:.2f}")
    print(f"  s3:       {s3_values[-1]:.2f}")
    print(f"  s4:       {s4_values[-1]:.2f}")
    print(f"实际风险 R = {R_values[-1]:.2f} 元")