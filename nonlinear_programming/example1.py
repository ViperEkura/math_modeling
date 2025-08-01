import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cvxpy as cp


def main_problem():
    # 定义决策变量
    x1 = cp.Variable()
    x2 = cp.Variable()
    
    # 定义目标函数（二次规划形式）
    Q = np.array([[-0.01, -0.0035], 
                  [-0.0035, -0.01]])  # 二次项系数矩阵
    c = np.array([144, 174])  # 线性项系数
    
    # 目标函数：x^T Q x + c^T x - 400000
    objective = cp.Maximize(cp.quad_form(cp.hstack([x1, x2]), Q) + c @ cp.hstack([x1, x2]) - 400000)
    
    # 约束条件
    constraints = [x1 >= 0, x2 >= 0]
    
    # 求解问题
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    # 获取结果
    x10 = np.around(x1.value, 2)
    x20 = np.around(x2.value, 2)
    f0 = prob.value
    
    # 计算其他指标
    p1 = 339 - 0.01*x10 - 0.003*x20
    p2 = 399 - 0.004*x10 - 0.01*x20
    total_cost = 400000 + 195*x10 + 225*x20
    rate = f0 / total_cost
    
    print(f"最优解: x1 = {x10}, x2 = {x20}")
    print(f"最大利润: f0 = {f0}")
    print(f"19英寸平均售价: p1 = {p1:.2f}")
    print(f"21英寸平均售价: p2 = {p2:.2f}")
    print(f"总成本: c = {total_cost}")
    print(f"利润率: rate = {rate:.4f}")
    
    # 绘制三维曲面
    x1_vals = np.linspace(0, 10000, 100)
    x2_vals = np.linspace(0, 10000, 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    
    # 计算目标函数值
    F = (339 - 0.01*X1 - 0.003*X2)*X1 + (399 - 0.004*X1 - 0.01*X2)*X2 - (400000 + 195*X1 + 225*X2)
    
    fig = plt.figure(figsize=(14, 6))
    
    # 3D曲面图
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X1, X2, F, cmap='viridis', alpha=0.6)
    ax1.scatter(x10, x20, f0, color='red', s=100, label='Optimal Point')
    ax1.set_xlabel('$x_1$', fontsize=12)
    ax1.set_ylabel('$x_2$', fontsize=12)
    ax1.set_zlabel('Profit', fontsize=12)
    ax1.set_title('Profit Function Surface', fontsize=14)
    ax1.legend()
    
    # 等高线图
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X1, X2, F, levels=20, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.scatter(x10, x20, color='red', s=100, label='Optimal Point')
    ax2.set_xlabel('$x_1$', fontsize=12)
    ax2.set_ylabel('$x_2$', fontsize=12)
    ax2.set_title('Profit Contour Plot', fontsize=14)
    ax2.legend()
    plt.tight_layout()
    plt.savefig('profit_optimization.png', dpi=300)
    plt.show()
    
    return x10, x20, f0

# 第二部分：灵敏度分析
def sensitivity_analysis():
    import sympy as sp
    
    # 定义符号变量
    x1, x2, a = sp.symbols('x1 x2 a')
    
    # 定义目标函数
    f = (339 - a*x1 - 0.003*x2)*x1 + (399 - 0.004*x1 - 0.01*x2)*x2 - (400000 + 195*x1 + 225*x2)
    f = sp.simplify(f)
    
    # 计算偏导数
    f1 = sp.diff(f, x1)
    f2 = sp.diff(f, x2)
    
    # 求解驻点
    sol = sp.solve((f1, f2), (x1, x2))
    x10_expr = sol[x1]
    x20_expr = sol[x2]
    
    print("\n灵敏度分析:")
    print("x1 关于 a 的表达式:")
    sp.pprint(x10_expr)
    print("\nx2 关于 a 的表达式:")
    sp.pprint(x20_expr)
    
    # 绘制 x1 和 x2 关于 a 的变化
    a_vals = np.linspace(0.002, 0.02, 100)
    
    # 将符号表达式转换为数值函数
    x10_func = sp.lambdify(a, x10_expr, 'numpy')
    x20_func = sp.lambdify(a, x20_expr, 'numpy')
    
    x10_vals = x10_func(a_vals)
    x20_vals = x20_func(a_vals)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(a_vals, x10_vals, 'b-', linewidth=2)
    ax1.set_xlabel('$a$', fontsize=12)
    ax1.set_ylabel('$x_1$', fontsize=12)
    ax1.set_title('Optimal $x_1$ as function of $a$', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2.plot(a_vals, x20_vals, 'r-', linewidth=2)
    ax2.set_xlabel('$a$', fontsize=12)
    ax2.set_ylabel('$x_2$', fontsize=12)
    ax2.set_title('Optimal $x_2$ as function of $a$', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('sensitivity_variables.png', dpi=300)
    plt.show()
    
    # 计算灵敏度
    dx1 = sp.diff(x10_expr, a)
    dx10 = dx1.subs(a, 0.01)
    sx1a = dx10 * 0.01 / 4735
    
    dx2 = sp.diff(x20_expr, a)
    dx20 = dx2.subs(a, 0.01)
    sx2a = dx20 * 0.01 / 7043
    
    print(f"\n灵敏度系数 (a=0.01):")
    print(f"dx1/da = {dx10.evalf():.2f}")
    print(f"sx1a = (dx1/da) * (a/x1) = {sx1a.evalf():.4f}")
    print(f"dx2/da = {dx20.evalf():.2f}")
    print(f"sx2a = (dx2/da) * (a/x2) = {sx2a.evalf():.4f}")
    
    # 绘制利润关于 a 的变化
    F_expr = f.subs({x1: x10_expr, x2: x20_expr})
    F_expr = sp.simplify(F_expr)
    F_func = sp.lambdify(a, F_expr, 'numpy')
    F_vals = F_func(a_vals)
    
    plt.figure(figsize=(8, 5))
    plt.plot(a_vals, F_vals, 'g-', linewidth=2)
    plt.xlabel('$a$', fontsize=12)
    plt.ylabel('Profit', fontsize=12)
    plt.title('Optimal Profit as function of $a$', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('sensitivity_profit.png', dpi=300)
    plt.show()
    
    # 计算利润相对误差
    Sya = -4735**2 * 0.01 / 553641
    f3 = f.subs({x1: 4735, x2: 7043, a: 0.011})
    f4 = F_expr.subs(a, 0.011)
    delta = (f4 - f3) / f4
    
    print(f"\n当 a 从 0.01 增加到 0.011 时:")
    print(f"近似利润: f3 = {f3.evalf():.2f}")
    print(f"精确利润: f4 = {f4.evalf():.2f}")
    print(f"利润相对误差: delta = {delta.evalf():.6f}")

# 执行主程序
if __name__ == "__main__":
    main_problem()
    sensitivity_analysis()