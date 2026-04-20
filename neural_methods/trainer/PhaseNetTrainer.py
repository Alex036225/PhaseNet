# 文件名: PhaseNetTrainer.py

"""
PhaseNet Trainer (DDP-ready, FP16 AMP + per-source immediate backward)
- 兼容单卡 / 单机多卡 / 多机多卡（DistributedDataParallel）
- 训练：对每个 train 源 forward 后立刻 backward（/len(sources) 缩放），避免多份计算图同时驻留导致 OOM
- 启用 AMP（强制 FP16，避免 GRU 的 BF16 不支持问题），并允许 TF32
- 测试阶段跨进程聚合预测与标签，仅主进程计算指标与保存
- 测试结束后自动：
    (1) 可视化预测时域曲线（分段）
    (2) 分段 PSD 频谱可视化（致密、平滑）
    (3) 显著性图（柔和配色，低透明度叠加）
"""

import os
from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import cv2

# --- 可选：可视化依赖（Agg 后端，避免无显卡/无显示环境报错） ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm  # 柔和配色

# fig_show.py（如果在 PYTHONPATH 或项目根目录可直接 import）
try:
    from fig_show import draw as draw_ppg  # 你文件里的主绘图函数名是 draw(rppg, gt)
except Exception:
    draw_ppg = None

from evaluation.metrics import calculate_metrics
from neural_methods.loss.SpectralLoss import SpectralLoss
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhaseNet.PhaseNet import PhaseNet
from neural_methods.trainer.BaseTrainer import BaseTrainer

# ========= 性能开关：TF32（Ampere+ 有效：如 4090） =========
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =========================
# 内嵌：分布式小工具（自包含）
# =========================
def ddp_is_available():
    return dist.is_available()

def ddp_is_initialized():
    return ddp_is_available() and dist.is_initialized()

def setup_distributed(backend="nccl"):
    """根据 torchrun 环境变量初始化分布式（若已初始化则跳过）"""
    if ddp_is_initialized():
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

def cleanup_distributed():
    if ddp_is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def get_rank():
    if ddp_is_initialized():
        return dist.get_rank()
    return 0

def get_world_size():
    if ddp_is_initialized():
        return dist.get_world_size()
    return 1

def is_main_process():
    return get_rank() == 0

def barrier():
    if ddp_is_initialized():
        dist.barrier()

def all_gather_pyobj(obj):
    """在所有进程上 all_gather 任意 Python 对象，返回 list（每个rank一个对象）"""
    if not ddp_is_initialized():
        return [obj]
    world_size = get_world_size()
    out_list = [None for _ in range(world_size)]
    dist.all_gather_object(out_list, obj)
    return out_list


# =========================
# Trainer
# =========================
class PhaseNetTrainer(BaseTrainer):
    """Trainer for PhaseNet, DDP + FP16 AMP + per-source backward."""

    def __init__(self, config, data_loader):
        """Initializes trainer parameters, model, loss, and optimizer."""
        super().__init__()

        # 1) 分布式初始化
        setup_distributed(backend="nccl")

        # 2) 设备
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")

        self.config = config
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.min_valid_loss = None
        self.best_epoch = 0

        # Loss 权重
        self.beta = 0.2   # 重构损失权重
        self.gama = 0.0   # 频谱损失权重

        # 损失函数
        self.loss_fn_spec = SpectralLoss()
        self.loss_fn_sup = Neg_Pearson()

        # 3) 模型
        self.model = PhaseNet(
            feature_dim=config.MODEL.PHASENET.PARAMS.FEATURE_DIM,
            latent_dim=config.MODEL.PHASENET.PARAMS.LATENT_DIM,
            hidden_dim=config.MODEL.PHASENET.PARAMS.HIDDEN_DIM,
        ).to(self.device)

        # 4) DDP 包裹
        if ddp_is_initialized():
            self.model = DDP(
                self.model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                output_device=self.device.index if self.device.type == "cuda" else None,
                find_unused_parameters=False
            )

        # 5) AMP（强制 FP16；避免 GRU 的 BF16 不支持）
        from torch.cuda.amp import GradScaler
        self.scaler = GradScaler(enabled=True)  # FP16 需要 scaler

        # 6) 训练模式准备
        if "train_and_test" in config.TOOLBOX_MODE:
            # 多源训练：取所有 train* loader 的最小长度
            self.num_train_batches = min(
                len(loader) for key, loader in data_loader.items() if key.startswith('train')
            )
            # 优化器
            self.optimizer = optim.Adam(self._model_parameters(), lr=config.TRAIN.LR)

            # 梯度累计（可选配置项）
            self.accum_steps = int(getattr(self.config.TRAIN, "ACCUM_STEPS", 1))
            if self.accum_steps < 1:
                self.accum_steps = 1

        elif config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError(f"PhaseNetTrainer initialized in incorrect toolbox mode: {config.TOOLBOX_MODE}!")

    # 便捷函数：拿到底层模型参数（兼容 DDP/非DDP）
    def _raw_model(self):
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _model_parameters(self):
        return self._raw_model().parameters()

    @staticmethod
    def _safe_zscore(x, eps=1e-4):
        x = x.float()
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True).clamp_min(eps)
        z = (x - mean) / std
        return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_sup_spec_loss(self, pred, label):
        pred_norm = self._safe_zscore(pred)
        label_norm = self._safe_zscore(label)
        sup_loss = self.loss_fn_sup(pred_norm, label_norm).float()
        spec_loss = self.loss_fn_spec(pred_norm, label_norm).float()
        sup_loss = torch.nan_to_num(sup_loss, nan=0.0, posinf=1.0, neginf=1.0)
        spec_loss = torch.nan_to_num(spec_loss, nan=0.0, posinf=1.0, neginf=1.0)
        return sup_loss, spec_loss

    def _compute_sup_loss(self, pred, label):
        pred_norm = self._safe_zscore(pred)
        label_norm = self._safe_zscore(label)
        sup_loss = self.loss_fn_sup(pred_norm, label_norm).float()
        return torch.nan_to_num(sup_loss, nan=0.0, posinf=1.0, neginf=1.0)

    def train(self, data_loader):
        """Training routine with per-source immediate backward to lower peak memory."""

        from torch.cuda.amp import autocast

        train_keys = [key for key in data_loader.keys() if key.startswith('train')]
        iterators = {key: iter(data_loader[key]) for key in train_keys}
        num_batches = self.num_train_batches

        for epoch in range(self.max_epoch_num):
            # DDP: 每个 epoch 设置 sampler 的 epoch
            for key in train_keys:
                sampler = getattr(data_loader[key], "sampler", None)
                if isinstance(sampler, DistributedSampler):
                    sampler.set_epoch(epoch)

            if is_main_process():
                print(f"\n==== Training Epoch: {epoch + 1}/{self.max_epoch_num} ====")

            self.model.train()

            epoch_total_loss = 0.0
            epoch_sup_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_spec_loss = 0.0

            # 梯度清零（支持梯度累计）
            self.optimizer.zero_grad(set_to_none=True)

            with tqdm(total=num_batches, ncols=120, desc=f"Epoch {epoch + 1}", disable=not is_main_process()) as tbar:
                for idx in range(num_batches):

                    # 用于日志显示的数值累计（不保计算图）
                    batch_sup_loss_val = 0.0
                    batch_recon_loss_val = 0.0
                    batch_spec_loss_val = 0.0

                    # ============ 核心：每个源 forward 后立刻 backward ============
                    for key in train_keys:
                        try:
                            batch = next(iterators[key])
                        except StopIteration:
                            iterators[key] = iter(data_loader[key])
                            batch = next(iterators[key])

                        video_data = batch[0].to(torch.float32).to(self.device, non_blocking=True)
                        bvp_label = batch[1].to(torch.float32).to(self.device, non_blocking=True)

                        # AMP autocast（强制 FP16）
                        with autocast(dtype=torch.float16, enabled=torch.cuda.is_available()):
                            rPPG_pred, recon_loss = self.model(video_data)
                            sup_loss, spec_loss = self._compute_sup_spec_loss(rPPG_pred, bvp_label)
                            recon_loss = torch.nan_to_num(recon_loss.float(), nan=0.0, posinf=1.0, neginf=1.0)

                            # 本源总损失
                            loss = sup_loss + self.beta * recon_loss + self.gama * spec_loss
                            loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=1.0)

                            # 平均到所有源（与原逻辑“先平均再一次 backward”等效）
                            loss = loss / len(train_keys)

                            # 梯度累计：进一步除以 accum_steps
                            loss = loss / self.accum_steps

                        # 立刻反传，释放本源计算图
                        self.scaler.scale(loss).backward()

                        # 日志数值（CPU 标量）
                        batch_sup_loss_val   += float(sup_loss.detach().item())
                        batch_recon_loss_val += float(recon_loss.detach().item())
                        batch_spec_loss_val  += float(spec_loss.detach().item())
                    # ============================================================

                    # 累计到一定步数再 step（或每步 step）
                    if ((idx + 1) % self.accum_steps) == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)

                    # 用于显示的平均值
                    avg_sup_loss   = batch_sup_loss_val   / len(train_keys)
                    avg_recon_loss = batch_recon_loss_val / len(train_keys)
                    avg_spec_loss  = batch_spec_loss_val  / len(train_keys)
                    total_loss_val = avg_sup_loss + self.beta * avg_recon_loss + self.gama * avg_spec_loss

                    # 累计 epoch 级日志
                    epoch_total_loss += total_loss_val
                    epoch_sup_loss   += avg_sup_loss
                    epoch_recon_loss += avg_recon_loss
                    epoch_spec_loss  += avg_spec_loss

                    if is_main_process():
                        tbar.set_postfix(OrderedDict(
                            total_loss=f"{epoch_total_loss / (idx + 1):.4f}",
                            sup_loss=f"{epoch_sup_loss / (idx + 1):.4f}",
                            recon_loss=f"{epoch_recon_loss / (idx + 1):.4f}",
                            spec_loss=f"{epoch_spec_loss / (idx + 1):.4f}",
                        ))
                        tbar.update(1)

            # 仅主进程保存
            if is_main_process():
                self.save_model(epoch)
                print("\n==== Testing after Epoch {} ====".format(epoch + 1))

            # 测试（各进程各自推理，主进程聚合）
            self.test(data_loader, use_current_model=True)

        if is_main_process():
            print("Training finished!")
            if not self.config.TEST.USE_LAST_EPOCH and self.min_valid_loss is not None:
                print(f"Best model was saved at epoch {self.best_epoch} with validation loss {self.min_valid_loss:.4f}")

    def valid(self, data_loader, epoch):
        """Evaluates the model on the validation set and updates best checkpoint."""
        if "valid" not in data_loader or data_loader["valid"] is None:
            if is_main_process():
                print("No validation set found, skipping validation.")
            return

        if is_main_process():
            print("\n==== Validating ====")

        sampler = getattr(data_loader["valid"], "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        self.model.eval()
        running_valid_loss = []

        from torch.cuda.amp import autocast
        with torch.no_grad():
            for batch in tqdm(data_loader["valid"], ncols=100, desc="Validation", disable=not is_main_process()):
                video_data = batch[0].to(torch.float32).to(self.device, non_blocking=True)
                bvp_label = batch[1].to(torch.float32).to(self.device, non_blocking=True)

                # 推理可选用 autocast 省显存（强制 FP16）
                with autocast(dtype=torch.float16, enabled=torch.cuda.is_available()):
                    outputs = self.model(video_data)
                    rPPG_pred = outputs[0]
                    sup_loss = self._compute_sup_loss(rPPG_pred, bvp_label)

                running_valid_loss.append(float(sup_loss.detach().item()))

        # 跨进程聚合
        all_loss_lists = all_gather_pyobj(running_valid_loss)
        merged_losses = []
        for ls in all_loss_lists:
            merged_losses.extend(ls)
        avg_valid_loss = float(np.mean(merged_losses)) if len(merged_losses) > 0 else float('inf')

        if is_main_process():
            print(f"Average Validation Loss: {avg_valid_loss:.4f}")
            if not self.config.TEST.USE_LAST_EPOCH:
                if self.min_valid_loss is None or avg_valid_loss < self.min_valid_loss:
                    self.min_valid_loss = avg_valid_loss
                    self.best_epoch = epoch
                    print(f"New best model found! Saving checkpoint for epoch {self.best_epoch}.")
                    self.save_model(self.best_epoch, is_best=True)

        return avg_valid_loss

    def test(self, data_loader, use_current_model=False):
        """Runs the model on test sets. 支持DDP聚合，并在结束后进行可视化与显著性图生成。"""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        # 若不是使用当前模型，需要加载权重（所有进程都要 load）
        if not use_current_model:
            if self.config.TOOLBOX_MODE == "only_test":
                if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                    raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
                path = self.config.INFERENCE.MODEL_PATH
                if is_main_process():
                    print("Testing uses pretrained model!")
                    print(path)
            else:
                if self.config.TEST.USE_LAST_EPOCH:
                    path = os.path.join(self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                    if is_main_process():
                        print("Testing uses last epoch as non-pretrained model!")
                        print(path)
                else:
                    path = os.path.join(self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                    if is_main_process():
                        print("Testing uses best epoch selected using model selection as non-pretrained model!")
                        print(path)
            state = torch.load(path, map_location=self.device)
            self._raw_model().load_state_dict(state)

        self.model = self.model.to(self.device)
        self.model.eval()
        if is_main_process():
            print("Running model evaluation on the testing dataset!")

        predictions_local = dict()
        labels_local = dict()

        sampler = getattr(data_loader["test"], "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(0)

        from torch.cuda.amp import autocast
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80, disable=not is_main_process())):
                batch_size = test_batch[0].shape[0]
                data = test_batch[0].to(self.device, non_blocking=True)
                label = test_batch[1].to(self.device, non_blocking=True)

                # 推理 autocast（强制 FP16）
                with autocast(dtype=torch.float16, enabled=torch.cuda.is_available()):
                    pred_ppg_test, _ = self.model(data)

                # 统一搬到 CPU，避免在测试阶段长期持有 GPU Tensor 导致额外内存压力
                label_cpu = label.detach().float().cpu()
                pred_cpu = pred_ppg_test.detach().float().cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions_local:
                        predictions_local[subj_index] = dict()
                        labels_local[subj_index] = dict()
                    predictions_local[subj_index][sort_index] = pred_cpu[idx].clone()
                    labels_local[subj_index][sort_index] = label_cpu[idx].clone()

                del data, label, pred_ppg_test, label_cpu, pred_cpu

        # DDP: 聚合到主进程
        all_pred_parts = all_gather_pyobj(predictions_local)
        all_label_parts = all_gather_pyobj(labels_local)

        if is_main_process():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            predictions = {}
            labels = {}
            for part in all_pred_parts:
                for sid, seqs in part.items():
                    predictions.setdefault(sid, {}).update(seqs)
            for part in all_label_parts:
                for sid, seqs in part.items():
                    labels.setdefault(sid, {}).update(seqs)

            print('')
            calculate_metrics(predictions, labels, self.config)
            if self.config.TEST.OUTPUT_SAVE_DIR:
                self.save_test_outputs(predictions, labels, self.config)

            # === 可视化（按需打开/默认打开）===
            # try:
            #     self._visualize_predictions(predictions, labels)
            # except Exception as e:
            #     print(f"[viz] 可视化跳过：{e}")
            # try:
            #     self._visualize_psd_segmented(predictions, labels)
            # except Exception as e:
            #     print(f"[psd-seg] 分段频谱可视化跳过：{e}")
            # # === 显著性图 ===
            # try:
            #     self._run_saliency_on_test(data_loader)
            # except Exception as e:
                # print(f"[saliency] 显著性可视化跳过：{e}")

        barrier()  # 确保主进程完成日志/保存后再退出

    def save_model(self, index, is_best=False):
        """Saves the model checkpoint. 仅主进程执行。"""
        if not is_main_process():
            return

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        state_dict = self._raw_model().state_dict()

        model_path = os.path.join(self.model_dir, f"{self.model_file_name}_Epoch{index}.pth")
        torch.save(state_dict, model_path)
        print(f"Saved model to {model_path}")

        if is_best:
            best_model_path = os.path.join(self.model_dir, f"{self.model_file_name}_Best.pth")
            torch.save(state_dict, best_model_path)
            print(f"Saved best model to {best_model_path}")
    
    def _visualize_predictions(self, predictions: dict, labels: dict):
        """
        将聚合后的 predictions/labels 画图到 OUTPUT_SAVE_DIR/vis/ 下。
        - 优先调用 fig_show.draw(rppg, gt)
        - 若 fig_show 不可用，使用简易 fallback
        - 在每个 subject 上：先按 sort_index 拼接，再用 GT 的均值/方差做 z-score 归一化
        """
        out_root = getattr(self.config.TEST, "OUTPUT_SAVE_DIR", None)
        if not out_root:
            print("[viz] TEST.OUTPUT_SAVE_DIR 未设置，跳过可视化。")
            return

        import numpy as np
        vis_dir = os.path.join(out_root, "vis")
        os.makedirs(vis_dir, exist_ok=True)

        max_vis = int(getattr(self.config.TEST, "MAX_VISUALIZE_SUBJECTS", 8))
        saved = 0

        def _to_numpy(x):
            try:
                import torch
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().numpy()
            except Exception:
                pass
            return np.array(x)

        for sid in sorted(predictions.keys()):
            if sid not in labels:
                continue

            idxs = sorted(predictions[sid].keys())
            # 收集并拼接
            pred_seq = [_to_numpy(predictions[sid][k]).squeeze() for k in idxs]
            gt_seq   = [_to_numpy(labels[sid][k]).squeeze()      for k in idxs]
            if len(pred_seq) == 0 or len(gt_seq) == 0:
                continue

            pred = np.concatenate(pred_seq, axis=-1)
            gt   = np.concatenate(gt_seq,   axis=-1)

            # --- 分别标准化（各自 z-score），避免被对方方差“稀释” ---
            def _zscore(x):
                m = float(np.mean(x))
                s = float(np.std(x))
                if not np.isfinite(s) or s < 1e-8:
                    s = 1e-8
                return (x - m) / s

            pred = _zscore(pred)
            gt   = _zscore(gt)

            # ===== 每 128 帧一张图（分段可视化） =====
            L = int(getattr(self.config.TEST, "VIS_WINDOW", 300))        # 每张的帧数
            S = int(getattr(self.config.TEST, "VIS_STRIDE", L))          # 步长；默认为不重叠
            MAX_SEGS = int(getattr(self.config.TEST, "MAX_VISUALIZE_SEGS", 999))  # 每个 subject 最多张数

            sid_dir = os.path.join(vis_dir, str(sid))
            os.makedirs(sid_dir, exist_ok=True)

            seg_count = 0
            for start in range(0, gt.size, S):
                if seg_count >= MAX_SEGS:
                    break
                end = min(start + L, gt.size)

                pred_vis = pred[start:end]
                gt_vis   = gt[start:end]
                if gt_vis.size < 8:      # 太短就跳过
                    continue

                try:
                    # 每段单独出一张
                    plt.figure(figsize=(10, 2.5))
                    if draw_ppg is not None:
                        # 你的 fig_show.draw(pred, gt)
                        draw_ppg(pred_vis, gt_vis)
                    else:
                        plt.plot(pred_vis, label="rPPG", linewidth=1)
                        plt.plot(gt_vis,   label="GT-PPG", linewidth=1)
                        plt.legend(frameon=False)
                        plt.xlabel("Frame")
                        plt.ylabel("Amplitude (z-score)")
                    plt.tight_layout()
                    plt.savefig(os.path.join(sid_dir, f"{sid}_seg{seg_count:04d}.png"), dpi=200, bbox_inches="tight")
                    plt.close()
                except Exception as e:
                    print(f"[viz] 绘制 {sid} 段 {seg_count} 失败：{e}")

                seg_count += 1

            print(f"[viz] {sid}: 已保存 {seg_count} 张分段图到 {sid_dir}")

    def _visualize_psd_segmented(self, predictions: dict, labels: dict):
        """
        分段频谱可视化（40–180 BPM，平滑致密曲线）：
        - 每段做 Welch(高 nfft) -> 映射到 BPM -> 插值到致密网格 -> 轻度平滑 -> 归一化
        - 保存到 OUTPUT_SAVE_DIR/vis_psd_segments/<sid>/<sid>_segXXXX_psd.png
        """
        out_root = getattr(self.config.TEST, "OUTPUT_SAVE_DIR", None)
        if not out_root:
            print("[psd-seg] TEST.OUTPUT_SAVE_DIR 未设置，跳过分段 PSD 可视化。")
            return

        try:
            from scipy.signal import welch, savgol_filter
        except Exception:
            print("[psd-seg] 需要 scipy.signal.welch 和 savgol_filter，未安装则跳过。")
            return

        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.ndimage import gaussian_filter1d

        vis_root = os.path.join(out_root, "vis_psd_segments")
        os.makedirs(vis_root, exist_ok=True)

        # 采样率：TEST.VIS_FS > DATA.FS > 30
        fs = float(getattr(self.config.TEST, "VIS_FS",
                getattr(getattr(self.config, "DATA", object()), "FS", 30.0)))

        # Welch 参数（分辨率更高，更接近“平滑曲线”的观感）
        base_nperseg = int(getattr(self.config.TEST, "PSD_NPERSEG", 128))
        # 更高 nfft 提升频率分辨率（对短段会自动截断到段长的下一次幂）
        def _nice_nfft(seg_len):
            # 至少 256，倾向 1024/2048；太短的段也给个 >=256 的 nfft 做零填充
            if seg_len <= 0:
                return 256
            p = 1
            while (1 << p) < max(256, seg_len * 4):  # *4 的零填充
                p += 1
            return 1 << p

        # 窗口/步长：默认 128 帧一段
        L = int(getattr(self.config.TEST, "VIS_WINDOW", 128))
        S = int(getattr(self.config.TEST, "VIS_STRIDE", L))
        MAX_SEGS = int(getattr(self.config.TEST, "MAX_VISUALIZE_SEGS", 999))
        MIN_LEN = int(getattr(self.config.TEST, "MIN_SEG_LEN", 32))
        MAX_SUBJ = int(getattr(self.config.TEST, "MAX_VISUALIZE_SUBJECTS", 64))

        # 目标 BPM 网格（致密）
        bpm_grid = np.linspace(40.0, 180.0, 701)  # 步长 ~0.2 BPM
        # 平滑强度（可以在 config 覆盖）
        use_savgol = bool(getattr(self.config.TEST, "PSD_USE_SAVGOL", True))
        sg_win = int(getattr(self.config.TEST, "PSD_SG_WIN", 21))  # 必须为奇数
        sg_poly = int(getattr(self.config.TEST, "PSD_SG_POLY", 3))
        gauss_sigma = float(getattr(self.config.TEST, "PSD_GAUSS_SIGMA", 1.2))

        def _to_numpy(x):
            try:
                import torch
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().numpy()
            except Exception:
                pass
            return np.array(x)

        def _psd_dense(x):
            """对一段信号计算致密、平滑的 BPM-PSD 曲线，返回 (bpm_grid, psd_norm)。"""
            x = np.asarray(x, dtype=np.float32)
            if x.size < MIN_LEN:
                return None, None
            # z-score
            m, s = float(np.mean(x)), float(np.std(x))
            s = 1e-8 if (not np.isfinite(s) or s < 1e-8) else s
            x = (x - m) / s

            nperseg = max(8, min(base_nperseg, x.size))
            nfft = _nice_nfft(x.size)

            f, pxx = welch(
                x, fs=fs, nperseg=nperseg, nfft=nfft,
                detrend="constant", scaling="density", return_onesided=True
            )
            bpm = f * 60.0
            mask = (bpm >= 40.0) & (bpm <= 180.0)
            if not np.any(mask):
                return None, None
            bpm, pxx = bpm[mask], pxx[mask]

            # 插值到致密网格
            grid = bpm_grid
            pxx_dense = np.interp(grid, bpm, pxx, left=pxx[0], right=pxx[-1])

            # 轻度平滑（Savitzky-Golay 或 Gaussian）
            if use_savgol:
                # 窗宽需为奇数且 <= 序列长度
                win = min(len(pxx_dense) - (1 - len(pxx_dense) % 2), sg_win)
                if win < 5:  # 太短时退化为 Gaussian
                    pxx_dense = gaussian_filter1d(pxx_dense, sigma=gauss_sigma)
                else:
                    if win % 2 == 0:
                        win -= 1
                    win = max(win, sg_poly + 2 + (sg_poly % 2))  # 确保>poly且为奇数
                    pxx_dense = savgol_filter(pxx_dense, window_length=win, polyorder=sg_poly, mode="interp")
            else:
                pxx_dense = gaussian_filter1d(pxx_dense, sigma=gauss_sigma)

            # 归一化到 [0,1]
            pmax = float(np.max(pxx_dense))
            if not np.isfinite(pmax) or pmax <= 0:
                return None, None
            pxx_dense = pxx_dense / pmax
            return grid, pxx_dense

        shown_subjects = 0
        for sid in sorted(predictions.keys()):
            if sid not in labels:
                continue
            if shown_subjects >= MAX_SUBJ:
                break

            idxs = sorted(predictions[sid].keys())
            pred_seq = [_to_numpy(predictions[sid][k]).squeeze() for k in idxs]
            gt_seq   = [_to_numpy(labels[sid][k]).squeeze()      for k in idxs]
            if len(pred_seq) == 0 or len(gt_seq) == 0:
                continue
            pred_all = np.concatenate(pred_seq, axis=-1)
            gt_all   = np.concatenate(gt_seq,   axis=-1)

            sid_dir = os.path.join(vis_root, str(sid))
            os.makedirs(sid_dir, exist_ok=True)

            seg_count = 0
            for start in range(0, gt_all.size, S):
                if seg_count >= MAX_SEGS:
                    break
                end = min(start + L, gt_all.size)
                if end - start < MIN_LEN:
                    continue

                bpm_gt,   psd_gt   = _psd_dense(gt_all[start:end])
                bpm_pred, psd_pred = _psd_dense(pred_all[start:end])
                if bpm_gt is None or bpm_pred is None:
                    continue

                # 画图（极简风格）
                try:
                    plt.figure(figsize=(6.2, 2.4))
                    ax = plt.gca()
                    plt.plot(bpm_gt,   psd_gt,   linewidth=2.0, label="GT-PPG")
                    plt.plot(bpm_pred, psd_pred, linewidth=2.0, label="rPPG")
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    plt.yticks([])
                    plt.xlim(40, 180)
                    plt.xticks(list(range(40, 181, 20)))
                    ax.tick_params(axis='x', colors='gray', labelsize=10)
                    plt.xlabel("BPM", fontsize=11)
                    leg = plt.legend(loc='upper right', frameon=True, ncol=1,
                                    bbox_to_anchor=(0.98, 0.9), handlelength=1.6,
                                    borderpad=0.6, borderaxespad=0.1)
                    leg.get_frame().set_alpha(0.9)
                    leg.get_frame().set_edgecolor("#d0d0d0")
                    leg.get_frame().set_linewidth(1.0)
                    plt.tight_layout()
                    out_path = os.path.join(sid_dir, f"{sid}_seg{seg_count:04d}_psd.png")
                    plt.savefig(out_path, dpi=220, bbox_inches="tight")
                    plt.close()
                except Exception as e:
                    print(f"[psd-seg] 绘制 {sid} 段 {seg_count} 失败：{e}")
                    continue

                seg_count += 1

            print(f"[psd-seg] {sid}: 已保存 {seg_count} 张分段 PSD 图到 {sid_dir}")
            shown_subjects += 1

    def _run_saliency_on_test(self, data_loader):
        """
        Generates and saves saliency maps for a few batches of the test set.
        This must be run on the main process to avoid file writing conflicts.
        """
        if not is_main_process():
            return
            
        out_root = getattr(self.config.TEST, "OUTPUT_SAVE_DIR", None)
        if not out_root:
            print("[saliency] TEST.OUTPUT_SAVE_DIR is not set, skipping saliency map generation.")
            return

        saliency_dir = os.path.join(out_root, "saliency_maps")
        orig_dir = os.path.join(saliency_dir, 'original_frames')
        map_dir = os.path.join(saliency_dir, 'heatmaps')
        os.makedirs(orig_dir, exist_ok=True)
        os.makedirs(map_dir, exist_ok=True)
        
        print("\n[saliency] Starting saliency map generation...")

        # We will only visualize a few batches to save time and disk space
        max_batches_to_visualize = 3  # You can adjust this number
        batches_done = 0

        # 1. 初始状态设为 eval
        self.model.eval() 

        with torch.enable_grad():
            for i, batch in enumerate(data_loader['test']):
                if batches_done >= max_batches_to_visualize:
                    break

                video_frames = batch[0].to(self.device, non_blocking=True)
                label = batch[1].to(self.device, non_blocking=True)
                video_frames.requires_grad = True

                # --- 核心修改部分 ---
                # 2. 为了 cuDNN 的 RNN backward，在计算梯度前临时切换到 train 模式
                self.model.train()

                # 前向传播
                with autocast(dtype=torch.float16, enabled=torch.cuda.is_available()):
                    model_output = self.model(video_frames)
                    if isinstance(model_output, (list, tuple)):
                        pred_ppg = model_output[0]
                    else:
                        pred_ppg = model_output

                    loss = self._compute_sup_loss(pred_ppg, label)

                # 反向传播 (在 train 模式下进行)
                loss.backward()

                # 3. 梯度计算完毕，立刻切回 eval 模式，这是至关重要的一步
                self.model.eval()

                # The gradient is now stored in video_frames.grad
                saliency = video_frames.grad.data.abs() # Shape: [B, C, T, H, W]

                # Take the maximum value across the color channels to get a single value per pixel
                saliency, _ = torch.max(saliency, dim=1) # Shape: [B, T, H, W]

                # Move to CPU and convert to NumPy for saving
                video_np = video_frames.detach().cpu().numpy()
                saliency_np = saliency.cpu().numpy()

                # --- Save the frames and maps ---
                batch_size = video_np.shape[0]
                for b in range(batch_size):
                    # Process each frame in the video clip
                    for t in range(video_np.shape[2]):
                        # 1. Save the original frame
                        frame = video_np[b, :, t, :, :]
                        frame = np.transpose(frame, (1, 2, 0)) # H, W, C
                        
                        # Denormalize if necessary (assuming it was normalized to [-1, 1] or [0, 1])
                        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
                        frame = (frame * 255).astype(np.uint8)
                        
                        frame_filename = os.path.join(orig_dir, f'batch{i}_vid{b}_frame{t:03d}.png')
                        plt.imsave(frame_filename, frame)

                        # 2. Save the saliency map
                        saliency_map = saliency_np[b, t, :, :] # H, W
                        
                        # Normalize the map for visualization
                        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
                        
                        map_filename = os.path.join(map_dir, f'batch{i}_vid{b}_frame{t:03d}_map.png')
                        # Use a colormap like 'jet' to get the blue-to-red effect
                        plt.imsave(map_filename, saliency_map, cmap='jet')

                print(f"[saliency] Saved visualizations for batch {i+1}/{max_batches_to_visualize}")
                batches_done += 1

        print(f"[saliency] Saliency map generation complete. Files saved in {saliency_dir}")
