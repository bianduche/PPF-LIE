# 实验结果目录

此目录用于存储训练和测试实验的结果。

## 目录结构

```
experiments/
├── lol/           # LOL数据集实验结果
│   ├── logs/      # TensorBoard日志
│   ├── checkpoints/# 模型检查点
│   └── configs/   # 实验配置
├── ml/            # ML数据集实验结果
│   ├── logs/
│   ├── checkpoints/
│   └── configs/
└── sice/          # SICE数据集实验结果
    ├── logs/
    ├── checkpoints/
    └── configs/
```

## 训练日志

训练过程中会自动记录：

- 损失曲线 (TensorBoard)
- 验证集指标
- 中间可视化结果
- 模型检查点

## 最佳模型

最佳模型检查点将保存为：
- `best_model.pth` - 最佳验证PSNR模型
- `latest_model.pth` - 最新检查点

## 恢复训练

```bash
# 从检查点恢复训练
python train.py --config configs/train_lol.yaml --resume experiments/lol/checkpoints/latest_model.pth
```

## 实验对比

| 实验 | 数据集 | 最佳PSNR | 最佳SSIM | 备注 |
|------|--------|----------|----------|------|
| exp001 | LOL | 25.71 | 0.92 | 基准 |
| exp002 | ML | 25.96 | 0.89 | 工业场景 |
| exp003 | SICE | 24.66 | 0.94 | 多曝光 |
