"""
约束代数符号计算 - 使用SymPy计算四面体上的泊松括号

根据v2文档要求：
- 必须用SymPy显式计算 {G_i, G_j}, {G_i, H_j}, {H_i, H_j}
- 判断代数是否闭合
- 分析异常项结构
"""

import sympy as sp
from sympy import symbols, cos, sin, diff, simplify, expand, collect
from sympy import Matrix, eye, zeros
import numpy as np

print("=" * 60)
print("  约束代数符号计算（SymPy）")
print("  四面体图：4节点，6边，4面")
print("=" * 60)

print("\n## 第一步：定义符号变量")

A01, A02, A03, A12, A13, A23 = symbols('A01 A02 A03 A12 A13 A23', real=True)
E01, E02, E03, E12, E13, E23 = symbols('E01 E02 E03 E12 E13 E23', real=True)
beta = symbols('beta', positive=True, real=True)

A = {
    (0,1): A01, (1,0): -A01,
    (0,2): A02, (2,0): -A02,
    (0,3): A03, (3,0): -A03,
    (1,2): A12, (2,1): -A12,
    (1,3): A13, (3,1): -A13,
    (2,3): A23, (3,2): -A23,
}

E = {
    (0,1): E01, (1,0): -E01,
    (0,2): E02, (2,0): -E02,
    (0,3): E03, (3,0): -E03,
    (1,2): E12, (2,1): -E12,
    (1,3): E13, (3,1): -E13,
    (2,3): E23, (3,2): -E23,
}

print("  A_ij: 相位变量（反对称）")
print("  E^ij: 电场变量（反对称）")
print("  beta: 耦合常数")

print("\n## 第二步：定义约束")

def G_constraint(i):
    """高斯约束 G_i = Σ_j E^ij"""
    result = 0
    for j in range(4):
        if i != j:
            result += E[(i,j)]
    return result

G = [G_constraint(i) for i in range(4)]

print("\n高斯约束 G_i:")
for i in range(4):
    print(f"  G_{i} = {G[i]}")

def F_triangle(i, j, k):
    """面上回路相位 F_△ = A_ij + A_jk + A_ki"""
    return A[(i,j)] + A[(j,k)] + A[(k,i)]

triangles = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]

print("\n回路相位 F_△:")
for tri in triangles:
    i, j, k = tri
    F = F_triangle(i, j, k)
    print(f"  F_({i}{j}{k}) = {F}")

def H_constraint(i):
    """局域哈密顿约束 H_i"""
    edges_from_i = []
    for j in range(4):
        if i != j:
            edges_from_i.append((i,j))
    
    degree = len(edges_from_i)
    
    kinetic = 0
    for edge in edges_from_i:
        kinetic += E[edge]**2 / 2
    
    curvature = 0
    for tri in triangles:
        if i in tri:
            idx = tri.index(i)
            j = tri[(idx+1)%3]
            k = tri[(idx+2)%3]
            F = F_triangle(tri[0], tri[1], tri[2])
            curvature += cos(F)
    
    return kinetic/degree - beta * curvature

H = [H_constraint(i) for i in range(4)]

print("\n局域哈密顿约束 H_i:")
for i in range(4):
    print(f"  H_{i} = {H[i]}")

print("\n## 第三步：计算泊松括号")

def poisson_bracket(f, g, var_type='A'):
    """
    计算泊松括号 {f, g}
    
    泊松括号定义：
    {f, g} = Σ_e (∂f/∂E_e * ∂g/∂A_e - ∂f/∂A_e * ∂g/∂E_e)
    """
    edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    
    result = 0
    for edge in edges:
        E_var = E[edge]
        A_var = A[edge]
        
        df_dE = diff(f, E_var)
        dg_dA = diff(g, A_var)
        
        df_dA = diff(f, A_var)
        dg_dE = diff(g, E_var)
        
        result += df_dE * dg_dA - df_dA * dg_dE
    
    return simplify(result)

print("\n### 3.1 高斯约束代数 {G_i, G_j}")

GG_matrix = Matrix.zeros(4, 4)
for i in range(4):
    for j in range(4):
        bracket = poisson_bracket(G[i], G[j])
        GG_matrix[i,j] = simplify(bracket)

print("\n{G_i, G_j} 矩阵:")
sp.pprint(GG_matrix)

GG_is_zero = all(GG_matrix[i,j] == 0 for i in range(4) for j in range(4))
if GG_is_zero:
    print("\n✅ 高斯约束代数闭合：{G_i, G_j} = 0")
else:
    print("\n❌ 高斯约束代数不闭合！")

print("\n### 3.2 高斯-哈密顿约束代数 {G_i, H_j}")

GH_matrix = Matrix.zeros(4, 4)
for i in range(4):
    for j in range(4):
        bracket = poisson_bracket(G[i], H[j])
        GH_matrix[i,j] = simplify(bracket)

print("\n{G_i, H_j} 矩阵:")
sp.pprint(GH_matrix)

print("\n### 3.3 哈密顿约束代数 {H_i, H_j}")

HH_matrix = Matrix.zeros(4, 4)
for i in range(4):
    for j in range(i+1, 4):
        bracket = poisson_bracket(H[i], H[j])
        HH_matrix[i,j] = simplify(bracket)
        HH_matrix[j,i] = -HH_matrix[i,j]

print("\n{H_i, H_j} 矩阵:")
sp.pprint(HH_matrix)

print("\n## 第四步：分析异常项")

print("\n### 4.1 检查 {H_i, H_j} 是否为零")

HH_is_zero = all(HH_matrix[i,j] == 0 for i in range(4) for j in range(4) if i != j)

if HH_is_zero:
    print("\n✅ 哈密顿约束代数闭合：{H_i, H_j} = 0")
    print("   基础层是第一类约束系统")
else:
    print("\n❌ 哈密顿约束代数不闭合！")
    print("   存在异常项")
    
    print("\n### 4.2 提取异常项")
    
    for i in range(4):
        for j in range(i+1, 4):
            if HH_matrix[i,j] != 0:
                print(f"\n异常项 {{H_{i}, H_{j}}} = {HH_matrix[i,j]}")

print("\n### 4.3 检查异常项结构")

print("\n异常项的一般结构应该是：")
print("  {H_i, H_j} ~ sin(F_△) * E_e")
print("\n其中 F_△ = A_ij + A_jk + A_ki 是回路相位")

print("\n## 第五步：数值验证")

print("\n使用随机数值验证符号计算结果...")

np.random.seed(12345)
A_vals = {
    (0,1): 0.1 * (np.random.rand() - 0.5),
    (0,2): 0.1 * (np.random.rand() - 0.5),
    (0,3): 0.1 * (np.random.rand() - 0.5),
    (1,2): 0.1 * (np.random.rand() - 0.5),
    (1,3): 0.1 * (np.random.rand() - 0.5),
    (2,3): 0.1 * (np.random.rand() - 0.5),
}

for key in list(A_vals.keys()):
    i, j = key
    A_vals[(j,i)] = -A_vals[key]

E_vals = {
    (0,1): 0.1 * (np.random.rand() - 0.5),
    (0,2): 0.1 * (np.random.rand() - 0.5),
    (0,3): 0.1 * (np.random.rand() - 0.5),
    (1,2): 0.1 * (np.random.rand() - 0.5),
    (1,3): 0.1 * (np.random.rand() - 0.5),
    (2,3): 0.1 * (np.random.rand() - 0.5),
}

for key in list(E_vals.keys()):
    i, j = key
    E_vals[(j,i)] = -E_vals[key]

beta_val = 1.0

subs_dict = {
    A01: A_vals[(0,1)],
    A02: A_vals[(0,2)],
    A03: A_vals[(0,3)],
    A12: A_vals[(1,2)],
    A13: A_vals[(1,3)],
    A23: A_vals[(2,3)],
    E01: E_vals[(0,1)],
    E02: E_vals[(0,2)],
    E03: E_vals[(0,3)],
    E12: E_vals[(1,2)],
    E13: E_vals[(1,3)],
    E23: E_vals[(2,3)],
    beta: beta_val,
}

print("\n数值代入后的泊松括号：")
print("\n{H_i, H_j} 数值矩阵:")

HH_numeric = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        val = HH_matrix[i,j].subs(subs_dict).evalf()
        HH_numeric[i,j] = float(val)

print(HH_numeric)

max_HH = np.max(np.abs(HH_numeric))
print(f"\n最大异常值：{max_HH:.6e}")

if max_HH < 1e-10:
    print("\n✅ 数值验证：代数闭合")
elif max_HH < 1e-6:
    print(f"\n⚠️ 数值验证：小异常（{max_HH:.2e}）")
else:
    print(f"\n❌ 数值验证：显著异常（{max_HH:.2e}）")

print("\n## 第六步：结论")

print("\n根据v2文档的生死判据：")
print("\n判据0：约束代数闭合性检验")

if GG_is_zero:
    print("  ✅ {G_i, G_j} = 0：高斯约束代数闭合")
else:
    print("  ❌ {G_i, G_j} ≠ 0：高斯约束代数不闭合")

if HH_is_zero:
    print("  ✅ {H_i, H_j} = 0：哈密顿约束代数闭合")
    print("\n结论：基础层是第一类约束系统，理论存活进入下一步")
else:
    print("  ❌ {H_i, H_j} ≠ 0：哈密顿约束代数不闭合")
    print("\n结论：基础层不是第一类约束系统")
    print("      根据路线A，此具体格点实现失败")
    print("      必须修改哈密顿约束定义或图结构")

print("\n" + "=" * 60)
print("  符号计算完成")
print("=" * 60)
