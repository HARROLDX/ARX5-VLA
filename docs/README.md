# 文档目录

[返回项目首页](../README.md)

## ARX5 实验笔记

根目录原有的三份笔记统一放在 `docs/notes/`，保留原始实验内容与命令。笔记中的绝对路径、设备编号和实验结论对应原现场环境，使用前需核对。

1. [机械臂驱动说明](notes/机械臂驱动说明.md)：CAN、ROS2、单臂与双臂遥操作。
2. [数据采集流程笔记](notes/数据采集流程笔记.md)：采集、回放、RGB/深度与数据导出。
3. [运行与流程笔记](notes/运行与流程笔记.md)：数据转换、Diffusion Policy / VLA 训练、AWBC 和部署记录。

## 数据与部署

- [数据集准备](dataset.md)
- [快速归一化统计](norm_stats_fast.md)
- [远程推理](tda_remote_inference.md)
- [ARX 推理环境与执行方式](../train_deploy_alignment/inference/arx/README.md)
- [ARX DAgger 采集](../train_deploy_alignment/dagger/arx/README.md)
- [Docker 环境](../scripts/docker/docker_README.md)

## 方法与模块

- [kai0 上游参考与引用](kai0-reference.md)
- [Model Arithmetic](../model_arithmetic/README.md)
- [Stage Advantage](../stage_advantage/README.md)
- [Train-Deploy Alignment](../train_deploy_alignment/README.md)

模块 README 保留在对应代码目录，便于就近维护；本页作为统一入口。首页的两幅动画统一为 400 × 226 像素，原始 demo 视频保留在 [arx_demo.mp4](arx_demo.mp4)。
