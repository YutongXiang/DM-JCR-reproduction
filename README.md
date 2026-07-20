# Reproduction of DM-JCR

| 文件                         | 内容                    | 公式           |
| -------------------------- | --------------------- | ------------ |
| `mobility.py`              | 车辆、UAV 位置更新           | （4）（5）       |
| `channel.py`               | 距离、信道增益、SINR、速率       | （6）—（8）      |
| `random_channel.py`        | 遮挡、Rayleigh/Rician 衰落 | 随机化（7）（8）    |
| `task_model.py`            | 直接卸载时延和能耗             | （12）（13）直接分支 |
| `relay_model.py`           | UAV 中继任务模型            | （12）（13）中继分支 |
| `check_dynamic_channel.py` | 动态信道集成实验              | （4）—（8）      |
