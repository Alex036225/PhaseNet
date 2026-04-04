import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format
from torch.nn.utils.parametrizations import weight_norm


# ---- 1) 极简因果卷积块（左填充 + 扩张 + 残差）----
class CausalBlock(nn.Module):
    def __init__(self, ch, kernel_size=3, dilation=1, dropout=0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(ch, ch, kernel_size, padding=0, dilation=dilation, bias=False)
        self.act  = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):             # x: (B, C, T)
        pad = (self.kernel_size - 1) * self.dilation
        y = F.pad(x, (pad, 0))        # 只在左侧补零，保证严格因果
        y = self.conv(y)
        y = self.act(y)
        y = self.drop(y)
        return y + x                  # 残差

# ---- 2) 极简因果卷积堆叠（用于替换 temporal_model）----
class TemporalCausalConvMinimal(nn.Module):
    """
    输入 (B, T, input_size) -> 输出 (B, T, output_size)
    仅使用 N 层因果卷积做消融；不做门控/归一化。
    """
    def __init__(self, input_size, output_size, hidden=128,
                 num_layers=3, kernel_size=3, dropout=0.1, dilation_base=2):
        super().__init__()
        self.inp = nn.Conv1d(input_size, hidden, 1)
        self.blocks = nn.ModuleList([
            CausalBlock(hidden, kernel_size=kernel_size, dilation=(dilation_base ** i), dropout=dropout)
            for i in range(num_layers)
        ])
        self.out = nn.Conv1d(hidden, output_size, 1)

    def forward(self, x):             # x: (B, T, F)
        x = x.permute(0, 2, 1)        # -> (B, F, T)
        x = self.inp(x)
        for b in self.blocks:
            x = b(x)
        x = self.out(x)               # (B, output_size, T)
        return x.permute(0, 2, 1)     # -> (B, T, output_size)



class TIM(nn.Module):
    """
    非循环时序交错模块 (Non-Circular Temporal Interlacing Module)

    核心思想:
    创建一个全新的、混合了时序信息的通道表示。与循环版本不同，此版本在处理
    时间序列的边界时使用零填充，而不是循环移位。这可以防止信息从序列的
    一端泄露到另一端，更适合在线或流式处理任务。

    操作 (Vectorized):
    1. 初始化一个全零的输出张量，这天然地处理了边界填充问题。
    2. 将需要来自“现在”信息的通道内容直接从输入复制到输出。
    3. 通过对时间和通道维度进行切片，将来自“未来”和“过去”的信息复制到
       输出张量的相应位置。
       
    这是一个零计算开销 (Zero-FLOP) 模块。
    """
    def __init__(self, channels: int, future_ratio: float = 1/8, past_ratio: float = 1/8):
        """
        Args:
            channels (int): 输入特征的通道数。
            future_ratio (float): 从未来帧获取信息的通道比例。
            past_ratio (float): 从过去帧获取信息的通道比例。
        """
        super().__init__()
        # 计算具体通道数
        f = int(channels * future_ratio)
        b = int(channels * past_ratio)
        
        # 确保通道数不为负
        self.future_channels = max(f, 0)
        self.past_channels = max(b, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [B, C, T, H, W]。

        Returns:
            torch.Tensor: 输出张量，形状与输入相同。
        """
        # 获取维度信息
        B, C, T, H, W = x.shape
        
        f = self.future_channels
        b = self.past_channels
        
        # 如果不需要交错或者只有一个时间帧，则直接返回原始输入
        if f + b == 0 or T == 1:
            return x

        # 1. 初始化输出张量为全零
        #    这优雅地处理了所有边界情况的零填充
        output = torch.zeros_like(x)

        # 2. 填充来自“现在”的信息 (present -> present)
        #    将输入中不需要移动的通道直接复制到输出
        if C > f + b:
             output[:, f+b:, ...] = x[:, f+b:, ...]

        # 3. 填充来自“未来”的信息 (future -> present)
        #    输出的 t=0..T-2 帧，其内容来自输入的 t=1..T-1 帧
        #    输出在 t=T-1 处的值保持为0 (因为没有 t=T 的未来帧)
        if f > 0:
            output[:, :f, :-1, ...] = x[:, :f, 1:, ...]

        # 4. 填充来自“过去”的信息 (past -> present)
        #    输出的 t=1..T-1 帧，其内容来自输入的 t=0..T-2 帧
        #    输出在 t=0 处的值保持为0 (因为没有 t=-1 的过去帧)
        if b > 0:
            output[:, f+b:f+b, 1:, ...] = x[:, f+b:f+b, :-1, ...]
            
        return output

    

# --- 视觉编码器部分 (无变动) ---
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y.expand_as(x)

# =========================
# EfficientSpatioTemporalBlock（插入 TIM 于 1x1 之后、(1,3,3) 之前）
# =========================
class EfficientSpatioTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_kernel=3, expand_ratio=4,
                 tsm_forward_ratio=1/8, tsm_backward_ratio=1/8):
        super(EfficientSpatioTemporalBlock, self).__init__()
        self.use_residual = in_channels == out_channels
        hidden_dim = in_channels * expand_ratio

        # Stage 1: 逐点扩张
        self.stage1 = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.InstanceNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        # --- 在此处插入 TIM 以做浅层时序混合 ---
        # self.tsm = WTSM(hidden_dim, forward_ratio=tsm_forward_ratio, backward_ratio=tsm_backward_ratio)

        self.tsm = TIM(hidden_dim, future_ratio=tsm_forward_ratio, past_ratio=tsm_backward_ratio)

        # Stage 2: 空间深度可分离 (1,3,3)
        self.stage2 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(1, 3, 3),
                      padding=(0, 1, 1), groups=hidden_dim, bias=False),
            nn.InstanceNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Stage 3: 显式时间深度卷积 (k,1,1) —— 保留原有设计
        self.stage3 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(temporal_kernel, 1, 1),
                      padding=(temporal_kernel // 2, 0, 0), groups=hidden_dim, bias=False),
            nn.InstanceNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.se = SELayer(hidden_dim)

        # Stage 4: 投影回通道数
        self.proj = nn.Sequential(
            nn.Conv3d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels)
        )

        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x):
        shortcut = x
        x = self.stage1(x)
        x = self.tsm(x)        # <--- 浅层时序混合
        x = self.stage2(x)
        x = self.stage3(x)     # <--- 保留原时序深度卷积
        x = self.se(x)
        main_out = self.proj(x)
        if self.use_residual:
            return self.pool(main_out + shortcut)
        else:
            return self.pool(main_out)

# --- 新增：空间注意力头模块 (保持不变) ---
class SpatialAttentionHead(nn.Module):
    """
    空间注意力头模块 (Spatial Attention Head)
    功能: 为输入的特征图生成一个空间注意力图，让模型自主学习关注哪些区域。
    """
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        raw_attention = self.attention_conv(x_reshaped)
        raw_attention_flat = raw_attention.view(B * T, 1, H * W)
        attention_weights = F.softmax(raw_attention_flat, dim=2)
        attention_map = attention_weights.view(B * T, 1, H, W)
        final_attention_map = attention_map.view(B, T, 1, H, W).permute(0, 2, 1, 3, 4)
        return final_attention_map

# --- 辅助模块 (替换 BN1d → IN1d，保持不变) ---
class Decoder1D(nn.Module):
    def __init__(self, latent_dim=32, feature_dim=128, start_len=8, num_blocks=3):
        super().__init__()
        self.start_channels = feature_dim // (2 ** (num_blocks - 1))
        if self.start_channels == 0:
            self.start_channels = 1
        self.start_len = start_len
        self.initial_dense = nn.Sequential(
            nn.Linear(latent_dim, self.start_channels * start_len),
            nn.ReLU(inplace=True)
        )
        blocks = []
        in_c = self.start_channels
        for i in range(num_blocks):
            out_c = in_c * 2
            blocks.append(nn.ConvTranspose1d(in_c, out_c, 4, 2, 1))
            blocks.append(nn.InstanceNorm1d(out_c))
            blocks.append(nn.ReLU())
            in_c = out_c
        blocks.append(nn.Conv1d(in_c, in_c, kernel_size=3, padding=1))
        self.deconv = nn.Sequential(*blocks)
        final_len = start_len * (2 ** num_blocks)
        self.final_fc = nn.Linear(final_len * in_c, feature_dim)

    def forward(self, c):
        x = self.initial_dense(c).view(-1, self.start_channels, self.start_len)
        x = self.deconv(x).flatten(1)
        return self.final_fc(x)

class SequenceRegressor(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden_dim, batch_first=True, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, z_seq):
        out, _ = self.gru(z_seq)
        return self.head(out).squeeze(-1)

# --- 门控 TCN (无修改) ---
class GatedTemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(GatedTemporalBlock, self).__init__()
        self.conv_linear = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.conv_gate = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        linear = self.conv_linear(x)
        gate = self.conv_gate(x)
        out = torch.tanh(linear) * torch.sigmoid(gate)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        if res.size(1) != out.size(1):
            res = res[:, :out.size(1), :]
        return self.relu(out + res)

class GatedTCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.2):
        super(GatedTCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size // 2
            layers.append(GatedTemporalBlock(in_channels, out_channels, kernel_size, 1, dilation_size, padding, dropout))
        self.network = nn.Sequential(*layers)
        self.final_conv = nn.Conv1d(num_channels[-1], output_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.network(x)
        out = self.final_conv(out)
        return out.permute(0, 2, 1)

# --- 主模型：MDNet + 空间注意力 (保持你原本逻辑) ---
class MDNet(nn.Module):
    def __init__(self, feature_dim=128, latent_dim=32, hidden_dim=128, tcn_layers=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        # 1. 视觉编码器
        self.base_encoder = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
            nn.InstanceNorm3d(16),
            nn.ReLU(inplace=True),
            EfficientSpatioTemporalBlock(16, 32),
            EfficientSpatioTemporalBlock(32, 64),
            EfficientSpatioTemporalBlock(64, 128),
        )

        # 1.5 空间注意力
        self.attention_head = SpatialAttentionHead(in_channels=128)

        self.encoder_head = nn.Linear(128, feature_dim)

        # 2. 门控 TCN 时序建模
        tcn_channels = [hidden_dim] * tcn_layers
        self.temporal_model = GatedTCN(
            input_size=feature_dim * 2,
            output_size=feature_dim,
            num_channels=tcn_channels,
            kernel_size=3
        )


        # self.temporal_model = TemporalCausalConvMinimal(
        #     input_size=feature_dim * 2,   # 你前面拼了 z_raw 与 v_raw
        #     output_size=feature_dim,
        #     hidden=hidden_dim,
        #     num_layers=3,                 # 消融主变量：2 或 3 层即可
        #     kernel_size=3,
        #     dropout=0.1,
        #     dilation_base=2
        # )

        # 3. 投影与重建模块
        self.projection_encoder = nn.Sequential(
            nn.Linear(feature_dim, (latent_dim + feature_dim) // 2),
            nn.ReLU(),
            nn.Linear((latent_dim + feature_dim) // 2, latent_dim)
        )
        self.decoder = Decoder1D(latent_dim=latent_dim, feature_dim=feature_dim)
        self.sim_alpha = 0.1

        # 4. 回归头
        self.regressor_head = SequenceRegressor(feature_dim=feature_dim)

    def forward(self, video_clip):
        if video_clip.dim() != 5:
            raise ValueError(f"Expected 5D input, got {video_clip.dim()}D")
        if video_clip.shape[1] == 3:
            pass
        elif video_clip.shape[2] == 3:
            video_clip = video_clip.permute(0, 2, 1, 3, 4)
        else:
            raise ValueError(f"Invalid input shape: {video_clip.shape}. Channel dim must be 3.")

        B, C, T, H, W = video_clip.shape

        # 1. 视觉编码
        x_feat = self.base_encoder(video_clip)

        # 1.5. 空间注意力
        attention_map = self.attention_head(x_feat)
        x_attended = x_feat * attention_map
        x = torch.sum(x_attended, dim=[-1, -2])  # B x 128 x T

        # B, C, T, H, W = x_feat.shape
        # # 先把空间维度展平
        # x_flat = x_feat.view(B, C, T, H * W)
        # # 在空间维度做 softmax 归一化
        # weights = F.softmax(x_flat, dim=-1)
        # # 按权重加权求和，相当于 B × C × T
        # x = torch.sum(x_flat * weights, dim=-1)


        x = x.permute(0, 2, 1)
        z_raw = self.encoder_head(x)

        # 2. 动态特征
        v_raw = torch.zeros_like(z_raw)
        v_raw[:, 1:] = z_raw[:, 1:] - z_raw[:, :-1]
        dynamic_features = torch.cat([z_raw, v_raw], dim=-1)

        # 3. 门控 TCN
        z_clean_seq = self.temporal_model(dynamic_features)

        # 4. 回归预测
        pred = self.regressor_head(z_clean_seq)

        # 5. 正则化损失
        recon_loss = torch.tensor(0.0, device=video_clip.device)
        if self.training:
            z_clean_flat = z_clean_seq.reshape(B * T, -1)
            z_raw_flat = z_raw.reshape(B * T, -1)
            c_star = self.projection_encoder(z_clean_flat)
            z_proj = self.decoder(c_star)
            mse_loss = F.mse_loss(z_proj, z_raw_flat)
            cos_sim = F.cosine_similarity(z_proj, z_raw_flat, dim=-1).mean()
            sim_loss = 1 - cos_sim
            recon_loss = mse_loss + self.sim_alpha * sim_loss

        return pred, recon_loss

# --- 测试与计算量分析 ---
if __name__ == "__main__":
    model = MDNet(feature_dim=128, latent_dim=32, hidden_dim=128, tcn_layers=4)
    model.eval()

    batch_size = 1
    seq_len = 128
    height = 128
    width = 128
    dummy_input = torch.randn(batch_size, 3, seq_len, height, width)

    # 注意：由于注意力模块中存在动态变形，thop可能无法精确计算MACs，但仍可用于评估参数量
    try:
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        macs, params = clever_format([macs, params], "%.3f")
        print(f"模型总参数量 (Parameters): {params}")
        print(f"模型总计算量 (MACs): {macs} (输入: T={seq_len}, H={height}, W={width})")
    except Exception as e:
        print(f"无法使用thop进行计算量分析，可能由于模型中的动态操作。错误: {e}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型总参数量 (手动计算): {total_params/1e6:.3f}M")
