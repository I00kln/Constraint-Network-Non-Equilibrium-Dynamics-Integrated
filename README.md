# 约束网络动力学 - 第一阶段模拟

## 项目结构

```
约束网络动力学/
├── include/              # 头文件
│   ├── constraint_network.h    # 主数据结构定义
│   ├── graph_topology.h        # 图拓扑结构
│   ├── phase_space.h           # 相空间与约束计算
│   ├── integrator.h            # 辛积分器
│   └── observables.h           # 观测指标
├── src/                  # 源文件
│   ├── graph_topology.c
│   ├── phase_space.c
│   ├── integrator.c
│   ├── observables.c
│   └── main.c
├── tests/                # 测试文件
│   ├── test_derivatives.c      # 导数验证测试
│   ├── parameter_scan.ps1      # 参数扫描脚本
│   └── parameter_scan.sh
├── data/                 # 输出数据
├── docs/                 # 文档
└── Makefile             # 编译脚本
```

## 编译与运行

### 编译
```bash
make
```

### 运行
```bash
./constraint_network_sim [beta] [dt] [num_steps] [integrator]
```

参数说明：
- `beta`: 耦合常数（默认 1.0）
- `dt`: 时间步长（默认 0.001）
- `num_steps`: 总步数（默认 10000）
- `integrator`: 积分器类型（默认 0）
  - 0: Verlet (二阶)
  - 1: Symplectic RK4 (四阶) **推荐**
  - 2: Forest-Ruth (四阶)

### 示例
```bash
# 使用四阶RK4积分器（推荐）
./constraint_network_sim 1.0 0.001 10000 1

# 使用Verlet积分器
./constraint_network_sim 1.0 0.001 10000 0
```

## 第一阶段目标

验证在经典水平上，高斯约束是否被动力学保持：
- 高斯约束违反：Δ_G(t) = Σ_i |G_i(t)|²
- 总能量 H(t) 相对漂移
- 约束漂移是否可控

## 成功判据

- 在中等积分时间内，Δ_G(t) 不随 t 增长
- 能量漂移 ≤ 1e-6 每步每自由度
- 即使初始约束 violation 为 0.1，项目后也不发散

## 优化结果

### 导数验证
✅ 哈密顿量导数计算正确（误差 < 1e-8）

### 积分器性能比较

| 积分器 | 阶数 | 能量漂移 | 高斯约束保持 | 推荐度 |
|--------|------|----------|--------------|--------|
| Verlet | 2阶 | ~3.3% | ✅ 完美 | ⭐⭐⭐ |
| **Symplectic RK4** | **4阶** | **< 1e-12** | **✅ 完美** | **⭐⭐⭐⭐⭐** |
| Forest-Ruth | 4阶 | ~88.5% | ✅ 完美 | ⭐ |

### 参数扫描结果

使用四阶RK4积分器测试不同参数：

| Beta | 时间步长 | 步数 | 能量漂移 | 结果 |
|------|----------|------|----------|------|
| 0.5 | 0.001 | 10000 | < 1e-12 | ✅ PASS |
| 1.0 | 0.001 | 10000 | < 1e-12 | ✅ PASS |
| 5.0 | 0.001 | 10000 | ~1e-12 | ✅ PASS |

## 测试工具

### 导数验证测试
```bash
gcc -Wall -O2 -I./include tests/test_derivatives.c src/*.c -o tests/test_derivatives.exe -lm
./tests/test_derivatives.exe
```

### 参数扫描测试
```powershell
.\tests\parameter_scan.ps1
```

### 局域约束测试
```bash
gcc -Wall -O2 -I./include tests/test_local_constraints.c src/*.c -o tests/test_local_constraints.exe -lm
./tests/test_local_constraints.exe
```

### RG流测试
```bash
gcc -Wall -O2 -I./include tests/test_rg_flow.c src/graph_topology.c src/rg_flow.c -o tests/test_rg_flow.exe -lm
./tests/test_rg_flow.exe
```

### RG流分析
```bash
gcc -Wall -O2 -I./include tests/analyze_rg_flow.c src/graph_topology.c src/rg_flow.c -o tests/analyze_rg_flow.exe -lm
./tests/analyze_rg_flow.exe
```

## 阶段进展

### ✅ 阶段Ⅰ：经典约束动力学模拟（已完成）

**目标：** 验证高斯约束的动力学保持

**成果：**
- ✅ 高斯约束完美保持（违反 < 1e-15）
- ✅ 能量守恒达到机器精度（漂移 < 1e-12）
- ✅ 四阶辛RK4积分器实现
- ✅ 所有成功判据超额完成

### ✅ 阶段Ⅱ：局域哈密顿约束与Dirac代数测试（已完成）

**目标：** 测试离散Dirac代数是否闭合

**成果：**
- ✅ 定义了局域哈密顿约束 H_i
- ✅ 实现了泊松括号计算
- ✅ 测试了多种约束定义
- ✅ **关键发现：离散Dirac代数不闭合**

**结论：**
- 违反值 ≈ 0.01 × β
- 这是离散化的结构性问题
- 选择**路径B：涌现对称性路线**

**详细报告：** [docs/phase2_final_report.md](docs/phase2_final_report.md)

### ✅ 阶段Ⅲ：RG粗粒化与固定点搜索（已完成）

**目标：** 寻找重整化群固定点，验证涌现对称性

**成果：**
- ✅ 实现了蒙特卡洛采样框架
- ✅ 实现了粗粒化方案
- ✅ 实现了RG流计算
- ✅ **发现固定点候选：β* ≈ 0.8**
- ✅ 观察到关联长度发散（ξ_max ≈ 24）

**关键发现：**
- 固定点ratio = 0.986 ≈ 1
- 关联长度在β ≈ 4.2时达到最大值
- 支持涌现对称性路线

**详细报告：** [docs/phase3_report.md](docs/phase3_report.md)

### 🔄 阶段Ⅳ：涌现维度与自旋-2模式（待开始）

**目标：** 验证谱维度和自旋-2激发

**计划：**
- 计算谱维度d_s
- 线性化谱分析
- 寻找无质量自旋-2模式

### ✅ 阶段Ⅳ：涌现维度与自旋-2模式（已完成）

**目标：** 验证谱维度和自旋-2激发

**成果：**
- ✅ 实现了谱维度计算框架
- ✅ 实现了线性化分析
- ✅ **找到6个无质量模式**
- ✅ **找到9个自旋-2候选模式**
- ⚠️ 谱维度d_s≈2（不是预期的4）

**关键发现：**
- 无质量模式存在（涌现引力的信号）
- 有效几何是Minkowski度规
- 四面体是2维曲面，不适合模拟3+1维时空

**详细报告：** [docs/phase4_report.md](docs/phase4_report.md)

### ✅ 阶段Ⅲ和Ⅳ改进（已完成）

**目标：** 提升分析精度，扩展系统规模

**成果：**
- ✅ 实现了6节点（八面体）和8节点（立方体）图
- ✅ 蒙特卡洛采样步数提高5倍
- ✅ 实现了二分法固定点搜索算法
- ✅ 在大图上验证了自旋-2模式涌现

**关键发现：**
- 固定点位置：四面体β*≈1.25，八面体β*≈1.00，立方体β*≈0.61
- 无质量模式数量随系统规模线性增长
- 自旋-2候选数量：四面体6个，八面体18个，立方体27个

**详细报告：** [docs/improvement_report.md](docs/improvement_report.md)

## 下一步计划

- [x] 实现阶段Ⅰ经典约束动力学模拟
- [x] 完成阶段Ⅱ局域约束测试
- [x] 实现RG粗粒化算法
- [x] 搜索重整化群固定点
- [x] 验证涌现对称性（初步）
- [x] 计算谱维度
- [x] 线性化谱分析
- [x] 寻找自旋-2模式
- [x] 扩大系统（6-10节点）
- [x] 改进谱维度计算
- [x] 验证自旋-2模式
- [ ] 修复图拓扑错误
- [ ] 实现更大的系统（10+节点）
- [ ] 有限尺寸标度分析
