# Waveform Inversion — UNet Baseline

Yale/UNC‑CH Kaggle 竞赛 baseline，纯 PyTorch，无外部依赖（除 numpy）。

## 文件
- `working_space/config.py` 统一超参数配置（训练、推理、数据尺寸、归一化和提交格式）
- `train.py`        训练脚本（含 UNet 定义、数据加载、训练/验证循环）
- `infer.py`        推理脚本（读测试 .npy → 预测 → 写 submission.csv）
- `compute_stats.py` 统计速度图均值/标准差，用于归一化
- `README.md`       本文件

## 在 Kaggle Notebook 使用

训练、推理和测试使用的超参数及默认地址集中在 `working_space/config.py`。默认地址均基于
项目根目录解析：训练数据为 `input/waveform-inversion/train_samples`，测试数据为
`input/waveform-inversion/test`，输出保存到 `output`。命令行传入的参数会覆盖对应默认值。

### 1. 挂载数据
把打印出的 `mean` / `std` 填进 `working_space/config.py` 的 `Cfg.vel_mean` / `Cfg.vel_std`。
把竞赛数据集加到 Notebook input，确认路径结构：
```
/kaggle/input/waveform-inversion/
    train_samples/
        FlatVel_A/data/*.npy
        FlatVel_A/model/*.npy
        ...
    test/*.npy
    sample_submission.csv
```

### 2. 统计归一化参数（首次必跑）
```python
!python compute_stats.py \
    --data_dir /kaggle/input/waveform-inversion/train_samples
```
把打印出的 `mean` / `std` 填进 `train.py` 顶部 `Cfg.vel_mean` / `Cfg.vel_std`。

### 3. 训练
```python
!python train.py \
    --data_dir /kaggle/input/waveform-inversion/train_samples \
    --out_dir /kaggle/working \
    --epochs 30 --batch_size 8
```
产出 `best_unet.pth` 和 `history.json`。

### 4. 推理提交
```python
!python infer.py \
    --ckpt /kaggle/working/best_unet.pth \
    --test_dir /kaggle/input/waveform-inversion/test \
    --out /kaggle/working/submission.csv
```
直接下载 `submission.csv` 上传即可。

## 本地调试
```bash
# 假设数据在 ./data/train_samples 和 ./data/test
python compute_stats.py --data_dir ./data/train_samples
python train.py --data_dir ./data/train_samples --out_dir ./out
python infer.py  --ckpt ./out/best_unet.pth --test_dir ./data/test --out ./out/submission.csv
```

## 预期分数
- 纯 UNet + 全部 10 个家族数据：公开榜 MAE ≈ 20–28
- 加入完整 OpenFWI 数据 + 数据增强：可压到 15 左右
- 接可微 FWI 后处理（参考 2nd/5th 方案）：冲 7–10

## 改进方向（按性价比排序）
1. **数据**：下载完整 OpenFWI（107 GB），按家族分别训练或共享 backbone + family embedding
2. **输入表征**：`log1p(|seis|)` 之外尝试 SVD 截断、包络提取、多通道 FFT 幅值
3. **模型**：UNet → ConvNeXt / ViT+RoPE / CaFormer；用 family embedding 做条件化
4. **损失**：MAE + 少量 TV loss 保边缘；SSIM 辅助
5. **后处理**：DL 出初值 → 可微波动方程 FWI 精修（L-BFGS / Gauss-Newton）
6. **集成**：多 checkpoint / 多模型加权融合 + hill-climb 权重搜索

## 注意事项
- 波形数值动态范围大，务必做 `log1p(|x|)` 或 z-score
- 速度图归一化均值/标准差请从实际数据统计得到，不要直接用默认值
- 测试集只交奇数列（x_1, x_3, ..., x_69），共 35 列；`infer.py` 已处理
- 每个 oid 有 70 行（y_0 … y_69），不要漏行
- 内存不足时改小 `batch_size`，或用 `mmap_mode="r"`（已默认开启）
