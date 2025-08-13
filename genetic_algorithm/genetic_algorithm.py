import os
import numpy as np
import matplotlib.pyplot as plt
from math import acos, cos, sin, pi
import random


def load_data(filename):
    """加载数据文件并处理坐标"""
    sj0 = np.loadtxt(filename)
    x = sj0[:, 0:8:2].flatten()  # 提取所有x坐标（经度）
    y = sj0[:, 1:8:2].flatten()  # 提取所有y坐标（纬度）
    sj = np.column_stack((x, y))
    d1 = [70, 40]  # 基地坐标（起点/终点）
    xy = np.vstack((d1, sj, d1))  # 组合完整坐标
    return xy * pi / 180  # 将角度转换为弧度

def calculate_distance_matrix(xy):
    """计算所有点之间的球面距离矩阵"""
    n = len(xy)
    d = np.zeros((n, n))
    for i in range(n-1):
        for j in range(i+1, n):
            # 使用球面距离公式计算两点间距离
            d[i, j] = 6370 * acos(cos(xy[i, 0]-xy[j, 0]) * cos(xy[i, 1]) * 
                        cos(xy[j, 1]) + sin(xy[i, 1]) * sin(xy[j, 1]))
    return d + d.T  # 使距离矩阵对称

def initialize_population(w, d):
    """使用改良圈算法初始化种群"""
    J = np.zeros((w, 102))
    for k in range(w):
        c = random.sample(range(100), 100)  # 生成1-100的随机排列
        c1 = [0] + [x+1 for x in c] + [101]  # 构建初始解（添加起点终点）
        
        # 改良圈算法优化路径
        for t in range(102):  # 最大优化次数
            flag = 0  # 优化标志
            for m in range(100):
                for n in range(m+2, 101):
                    # 如果交换路径段能缩短总距离
                    if (d[c1[m], c1[n]] + d[c1[m+1], c1[n+1]] < 
                        d[c1[m], c1[m+1]] + d[c1[n], c1[n+1]]):
                        c1[m+1:n+1] = c1[n:m:-1]  # 执行路径段交换
                        flag = 1  # 标记已优化
            if flag == 0:  # 如果没有优化则退出
                J[k, c1] = range(102)  # 记录当前解
                break
    
    J[:, 0] = 0  # 确保起点固定
    return J / 102  # 将解编码到[0,1]区间

def genetic_algorithm(d, J, w=50, g=100):
    """遗传算法主函数"""
    for k in range(g):  # 进化代数循环
        A = J.copy()  # 子代种群
        
        # 交叉操作
        c = random.sample(range(w), w)  # 随机选择交叉个体对
        for i in range(0, w, 2):  
            F = 2 + int(100 * random.random())  # 随机交叉点
            # 执行交叉交换
            temp = A[c[i], F:102].copy()
            A[c[i], F:102] = A[c[i+1], F:102]
            A[c[i+1], F:102] = temp  
        
        # 变异操作
        by = []  # 变异个体索引
        while not by:
            by = [i for i in range(w) if random.random() < 0.1]  # 按概率选择变异个体
        
        B = A[by, :]  # 需要变异的个体
        for j in range(len(by)):
            bw = sorted([2 + int(100 * random.random()) for _ in range(3)])  # 3个变异点
            # 执行变异交换
            B[j, :] = np.concatenate((
                B[j, :bw[0]],
                B[j, bw[1]:bw[2]],
                B[j, bw[0]:bw[1]],
                B[j, bw[2]:]
            )) 
        
        # 合并父代和子代种群
        G = np.vstack((J, A, B))  
        
        # 计算所有个体适应度（路径长度）
        SG = np.sort(G, axis=1)
        ind1 = np.argsort(G, axis=1)  # 解码路径序列
        num = G.shape[0]
        long = np.zeros(num)
        for j in range(num):
            for i in range(101):
                long[j] += d[ind1[j, i], ind1[j, i+1]]  # 累计路径长度
        
        # 选择最优个体
        ind2 = np.argsort(long)  # 按路径长度排序
        slong = np.sort(long)
        J = G[ind2[:w], :]  # 选择前w个最优个体
    
    # 返回最优解
    path = ind1[ind2[0], :]  # 最优路径
    flong = slong[0]  # 最短距离
    return path, flong

def plot_path(xy, path):
    """绘制路径图"""
    xx = xy[path, 0]  # 路径经度序列
    yy = xy[path, 1]  # 路径纬度序列
    plt.plot(xx, yy, '-o')  # 绘制带标记点的路径线
    plt.show()

def main():
    script_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_path, 'data12_1.txt')
    xy = load_data(data_path)
    
    d = calculate_distance_matrix(xy)
    
    w, g = 50, 100
    J = initialize_population(w, d)
    
    path, flong = genetic_algorithm(d, J, w, g)
    
    print("Path:", path)
    print("Path length:", flong)
    plot_path(xy, path)

if __name__ == "__main__":
    main()