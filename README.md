# DM-JCR 论文复现

| 文件                         | 内容                    | 公式           |
| -------------------------- | --------------------- | ------------ |
| `mobility.py`              | 车辆、UAV 位置更新           | （4）（5）       |
| `channel.py`               | 距离、信道增益、SINR、速率       | （6）—（8）      |
| `random_channel.py`        | 遮挡、Rayleigh/Rician 衰落 | 随机化（7）（8）    |
| `task_model.py`            | 直接卸载时延和能耗             | （12）（13）直接分支 |
| `relay_model.py`           | UAV 中继任务模型            | （12）（13）中继分支 |
| `resource_allocation.py`   | 三类任务的联合资源投影与评价       | （17）（27）     |
| `environment.py`           | 环境快照及固定尺寸张量编码         | （16）          |
| `strategy_codec.py`        | 三类原始策略与统一策略张量双向转换    | 扩散模型接口       |
| `check_dynamic_channel.py` | 动态信道集成实验              | （4）—（8）      |

## 统一参数配置

所有可执行实验脚本统一读取 [`configs/reproduction.toml`](configs/reproduction.toml)。配置文件将参数按来源分为三组：

- `paper.*`：论文明确给出的数值或范围；
- `assumptions.*`：论文没有给出、由本复现补充的模型假设；
- `experiments.*`：各检查脚本使用的演示场景、任务、信道和原始策略数值。

默认运行：

```powershell
python -m scripts.check_dynamic_channel
python -m scripts.check_resource_allocation
python -m scripts.check_relay_resource_allocation
python -m scripts.check_equation17_all_tasks
python -m scripts.generate_diffusion_dataset
python -m scripts.train_dm_jcr
python -m scripts.sample_dm_jcr
```

也可以为任一脚本指定另一份兼容配置：

```powershell
python -m scripts.check_resource_allocation --config configs/reproduction.toml
```

新增实验参数时应先写入 TOML，再由脚本读取；不要在脚本中新增场景硬编码。数值计算函数仍通过显式参数接收配置值，以便单元测试和复用。

## 扩散模型数据接口

`EnvironmentSnapshot` 汇总公式（16）中的信道、节点运动、遮挡、任务、剩余资源、拓扑和任务—节点映射。`encode_environment` 按“车辆 → UAV → RSU、同类节点 ID 升序”编码为固定尺寸张量，并返回节点与任务掩码。

`encode_raw_strategy` 把直接计算、中继计算和 UAV 辅助 V2V 三类原始资源分数编码为每任务 8 个语义槽位；`decode_raw_strategy` 可无损恢复现有资源分配对象。补零上限与归一化尺度统一配置在 `paper.system` 和 `assumptions.tensor` 中。下一阶段可直接在这两个接口之上实现论文的去噪扩散网络和训练流程。

## 扩散模型数据集

`data_generation.py` 自动生成车辆、UAV、RSU、随机信道和三类任务，使用公式（9）—（11）确定固定卸载映射，并通过公式（17）（27）评价多个候选资源策略。保存的 `x0` 是候选搜索找到的高质量标签，不宣称是全局最优解。

快速生成小型数据集：

```powershell
python -m scripts.generate_diffusion_dataset --train-samples 8 --validation-samples 2 --test-samples 2 --candidates 16
```

每个划分保存为压缩 NPZ，既包含扁平条件 `E` 和标签 `x0`，也保留 `node_features`、`channel_gains`、`blockage`、`topology`、`task_features`、`task_node_indices`、两类掩码、目标值和可行性。`manifest.json` 记录尺寸、随机种子和各划分统计。三个划分使用同一主种子派生出的不同随机子流，保证可重复且互不复用样本。

## DM-JCR 扩散模型

扩散实现严格采用论文给出的主流程：K-Means 环境分类、两隐藏层 MLP 分类器、对称全连接升降维网络、`28×28` 单通道 U-Net、两次下采样和上采样、`64→128→256` 特征通道、1000 个扩散步、公式（19）的确定性反向采样，以及公式（24）的 `δJ²` 自定义损失。扩散训练不使用数据集中的 `x0` 专家标签；`x0` 仅用于质量对照，因为论文明确说明其方法不依赖专家数据集。

论文未公开的 beta 调度、K-Means 类别数、隐藏层尺寸、Dropout、学习率、训练轮数、类别 prompt 注入和非负输出变换均集中在 `assumptions.diffusion`，没有混入 `paper.diffusion`。

完整训练默认执行论文的 1000 步反向过程，计算量较大：

```powershell
python -m scripts.train_dm_jcr --device auto
python -m scripts.check_dm_jcr_inference --device auto
```

只验证训练闭环时可以显式减少样本、轮数和扩散步数；这种模式不用于报告论文结果：

```powershell
python -m scripts.train_dm_jcr --device cpu --limit-samples 8 `
  --classifier-epochs 1 --autoencoder-epochs 1 --diffusion-epochs 1 `
  --batch-size 8 --denoising-steps 1 `
  --checkpoint outputs/checkpoints/dm_jcr_smoke.pt
```
