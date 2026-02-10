# TactileManipulation

基于 [ManipTrans](https://github.com/ManipTrans/ManipTrans)（CVPR 2025）迁移到 **IsaacLab** 的灵巧手操作训练框架。通过残差强化学习（Residual RL）将人手抓取操作迁移到机器人灵巧手上。

与原版 ManipTrans（IsaacGym + Hydra 配置）不同，本项目使用 **IsaacLab**（基于 Isaac Sim）作为仿真后端，采用 argparse 命令行接口，支持两种训练模式：
- **Imitator**：手部轨迹模仿学习（无物体），训练 base policy
- **Residual**：冻结 imitator + 残差 PPO 策略，训练手-物操作

## 项目结构

```
TactileManipulation/
├── main/
│   ├── dataset/
│   │   ├── mano2dexhand.py      # MANO → 灵巧手 retargeting（IK 优化）
│   │   └── factory.py           # 数据集工厂（ManipDataFactory）
│   └── rl/
│       └── train.py             # 统一训练/测试入口
├── maniptrans_envs/
│   ├── tasks/
│   │   ├── dexhand_manip_env.py     # 残差操作环境（手 + 物体）
│   │   ├── dexhand_imitator_env.py  # 模仿学习环境（仅手）
│   │   └── agents/                  # RL Games PPO 配置
│   └── lib/envs/dexhands/           # 灵巧手定义（Inspire, Shadow, Allegro 等）
├── data/                            # 数据集目录
└── assets/                          # URDF、预训练 imitator 权重
```

## 数据预处理

将人手 MANO 关节轨迹 retarget 到灵巧手 DOF 空间，生成训练用 demo 数据。

```bash
# 单手（右手 Inspire，GRAB 数据集 g0）
python main/dataset/mano2dexhand.py --data_idx g0 --dexhand inspire --side right --headless --iter 2000

# 单手（左手）
python main/dataset/mano2dexhand.py --data_idx g0 --dexhand inspire --side left --headless --iter 2000

# 双手（OakInk V2 数据）
python main/dataset/mano2dexhand.py --data_idx 20aed@0 --side right --dexhand inspire --headless --iter 7000
python main/dataset/mano2dexhand.py --data_idx 20aed@0 --side left --dexhand inspire --headless --iter 7000
```

其他灵巧手将 `--dexhand inspire` 替换为 `shadow`、`allegro`、`artimano`、`xhand`、`inspireftp`，迭代次数可能需要调整（Shadow 3000, Allegro 4000 等）。

## 训练

### Imitator 训练（手部轨迹模仿，无物体）

```bash
# 右手 Inspire，GRAB g0
python main/rl/train.py --mode imitator --dexhand inspire --side right \
    --num_envs 4096 --learning_rate 5e-4 --max_epochs 5000 \
    --random_state_init --actions_moving_average 0.4 \
    --data_indices g0 --experiment imitator_g0_inspire --headless
```

### Residual 训练（手+物体操作）

需要提供预训练的 imitator checkpoint 作为 base policy。

```bash
# 右手 Inspire，GRAB g0
python main/rl/train.py --mode residual --dexhand inspire --side right \
    --num_envs 4096 --learning_rate 2e-4 --max_epochs 5000 \
    --random_state_init --actions_moving_average 0.4 \
    --rh_base_checkpoint assets/imitator_rh_inspire.pth \
    --data_indices g0 --early_stop_epochs 100 \
    --experiment cross_g0_inspire --headless
```

### 常用参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `residual` | 训练模式：`imitator` 或 `residual` |
| `--dexhand` | `inspire` | 灵巧手类型 |
| `--side` | `right` | 手侧：`right` / `left` |
| `--num_envs` | `4096` | 并行环境数 |
| `--learning_rate` | `2e-4` | 学习率 |
| `--actions_moving_average` | `0.4` | 动作平滑系数 |
| `--random_state_init` | `False` | 随机初始状态（RSI） |
| `--use_pid_control` | `False` | PID 手腕控制（Shadow/Allegro 等需要） |
| `--data_indices` | `g0` | 数据序列索引，支持多个（空格分隔） |
| `--headless` | — | 无头模式（无渲染窗口） |

## 测试

```bash
# Imitator 测试
python main/rl/train.py --mode imitator --dexhand inspire --side right \
    --num_envs 4 --test --data_indices g0 \
    --actions_moving_average 0.4 \
    --checkpoint runs/imitator_g0_inspire__MM-DD-HH-MM-SS/nn/imitator_g0_inspire.pth

# Residual 测试
python main/rl/train.py --mode residual --dexhand inspire --side right \
    --num_envs 4 --test --data_indices g0 \
    --actions_moving_average 0.4 \
    --rh_base_checkpoint assets/imitator_rh_inspire.pth \
    --checkpoint runs/cross_g0_inspire__MM-DD-HH-MM-SS/nn/cross_g0_inspire.pth
```

去掉 `--headless` 可以打开可视化窗口。

## 与原版 ManipTrans 的主要区别

| | 原版 ManipTrans | TactileManipulation |
|---|---|---|
| 仿真后端 | IsaacGym | IsaacLab (Isaac Sim) |
| 四元数格式 | xyzw | wxyz |
| 配置方式 | Hydra | argparse |
| 训练入口 | `main/rl/train.py task=ResDexHand ...` | `main/rl/train.py --mode residual ...` |
| 环境基类 | `VecTask` | `DirectRLEnv` |
| Python | 3.8 | 3.10+ |

## 致谢

- [ManipTrans](https://github.com/ManipTrans/ManipTrans) (CVPR 2025)
- [IsaacLab](https://github.com/isaac-sim/IsaacLab)
- [RL Games](https://github.com/Denys88/rl_games)

## 引用

```bibtex
@inproceedings{li2025maniptrans,
    title={Maniptrans: Efficient dexterous bimanual manipulation transfer via residual learning},
    author={Li, Kailin and Li, Puhao and Liu, Tengyu and Li, Yuyang and Huang, Siyuan},
    booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
    year={2025}
}
```

## License

[GPL v3](LICENSE)
