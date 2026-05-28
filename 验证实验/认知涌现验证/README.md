# 认知与规律涌现框架 - 数值验证系统

## 概述

本验证系统用于数值验证《认知与规律涌现框架》中的六条核心定理，为理论框架提供数值层面的确认。

## 核心验证内容

### 定理1：离散到连续的桥接定理（Γ-收敛）
- 离散变分闭包算子的构造与迭代
- 路径作用量的计算
- Γ-收敛的下界条件和恢复条件验证
- 离散与连续极限的一致性检验

### 定理2：认知最优控制定理（动态HJB方程）
- Hamilton-Jacobi-Bellman方程的数值求解
- Lax-Oleinik半群性质验证
- 粘性解的验证

### 定理3：认知静态传播定理（静态Eikonal方程）
- Eikonal方程的数值求解（迭代法和Fast Marching Method）
- 认知势的计算
- 费马原理验证

### 定理4：认知几何涌现定理（Finsler度量）
- 稳定范数（Mather β-函数）的计算
- Finsler度量的涌现验证
- 各向异性与不可逆性验证
- 基本张量的正定性验证

### 定理5：认知相变与分类定理（弱KAM结构）
- Mañé临界值的计算
- Aubry集的识别
- Peierls势垒的计算
- 校准曲线的构造

### 定理6：认知跃迁定理（大偏差理论）
- Freidlin-Wentzell大偏差理论验证
- 跃迁概率的指数衰减验证
- 逃逸时间的Kramers公式验证
- Instanton路径的计算

## 文件结构

```
认知涌现验证/
├── discrete_closure_dynamics.py    # 定理1验证模块
├── hjb_eikonal_solver.py           # 定理2和定理3验证模块
├── finsler_emergence.py            # 定理4验证模块
├── weak_kam_structure.py           # 定理5验证模块
├── cognitive_transition.py         # 定理6验证模块
├── visualization.py                # 可视化工具
├── main_verification.py            # 主验证脚本
└── README.md                       # 本文档
```

## 使用方法

### 1. 完整验证

运行所有定理的完整验证：

```bash
python main_verification.py --mode full
```

这将：
- 依次验证所有六条定理
- 生成可视化结果
- 保存验证报告和结果数据

### 2. 快速验证

运行快速验证模式（验证关键定理）：

```bash
python main_verification.py --mode quick
```

### 3. 单定理验证

验证特定定理：

```bash
python main_verification.py --mode single --theorem 1
```

其中 `--theorem` 参数为1-6的整数。

### 4. 其他选项

```bash
# 禁用可视化
python main_verification.py --mode full --no-viz

# 指定输出目录
python main_verification.py --mode full --output ./my_results
```

## 输出结果

验证完成后，将在输出目录生成以下文件：

1. **验证结果JSON文件**：包含所有验证结果的详细数据
2. **验证报告TXT文件**：人类可读的汇总报告
3. **可视化图像**：
   - `potential_landscape.png`：势函数景观
   - `finsler_unit_ball.png`：Finsler度量单位球
   - `transition_dynamics.png`：跃迁动力学
   - `aubry_set.png`：Aubry集可视化
   - `verification_summary.png`：验证结果汇总

## 验证结果解读

### 验证状态
- ✓ 通过：定理在数值层面得到确认
- ✗ 未通过：需要进一步调整参数或改进算法

### 关键指标

1. **Γ-收敛验证**：
   - 下界条件误差
   - 恢复条件误差

2. **HJB方程验证**：
   - 粘性解残差
   - 半群性质误差

3. **Finsler度量验证**：
   - 齐次性误差
   - 凸性违反次数
   - 基本张量最小特征值

4. **弱KAM验证**：
   - Aubry集大小
   - Peierls势垒高度

5. **跃迁验证**：
   - 势垒高度
   - 跃迁概率衰减率

## 理论背景

本验证系统基于以下理论框架：

### 核心公理
- **公理A**：认知架构流形
- **公理B**：认知折射率
- **公理C**：认知Finsler度量
- **公理D**：变分闭包动力学

### 数学工具
- Γ-收敛理论
- Hamilton-Jacobi方程理论
- 弱KAM理论
- 大偏差理论
- Finsler几何

## 注意事项

1. **计算时间**：完整验证可能需要几分钟时间
2. **内存需求**：某些验证需要较大的内存，建议至少8GB
3. **随机性**：部分验证使用Monte Carlo方法，结果可能有轻微波动
4. **参数调整**：如需更高精度，可调整各模块中的配置参数

## 依赖库

```
numpy
scipy
matplotlib
```

## 扩展开发

如需添加新的验证内容：

1. 在相应模块中添加新的验证函数
2. 在 `main_verification.py` 中调用新函数
3. 在 `visualization.py` 中添加相应的可视化方法

## 联系方式

如有问题或建议，请参考主项目文档或联系项目维护者。
