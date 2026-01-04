import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
from layers.LiftingScheme import LiftingScheme, InverseLiftingScheme
from layers.RevIN import RevIN


def normalization(channels: int):
    return nn.InstanceNorm1d(num_features=channels)


class AdpWaveletBlock(nn.Module):
    # def __init__(self, in_channels, kernel_size, share_weights, simple_lifting, regu_details, regu_approx):
    def __init__(self, configs, input_size):
        super(AdpWaveletBlock, self).__init__()
        self.regu_details = configs.regu_details
        self.regu_approx = configs.regu_approx
        if self.regu_approx + self.regu_details > 0.0:
            self.loss_details = nn.SmoothL1Loss()

        self.wavelet = LiftingScheme(configs.enc_in, k_size=configs.lifting_kernel_size, input_size=input_size)
        self.norm_x = normalization(configs.enc_in)
        self.norm_d = normalization(configs.enc_in)

    def forward(self, x):
        (c, d) = self.wavelet(x)
        x = c

        r = None
        if(self.regu_approx + self.regu_details != 0.0):
            if self.regu_details:
                rd = self.regu_details * d.abs().mean()
            if self.regu_approx:
                rc = self.regu_approx * torch.dist(c.mean(), x.mean(), p=2)
            if self.regu_approx == 0.0:
                r = rd
            elif self.regu_details == 0.0:
                r = rc
            else:
                r = rd + rc

        x = self.norm_x(x)
        d = self.norm_d(d)

        return x, r, d


class InverseAdpWaveletBlock(nn.Module):
    # def __init__(self, in_channels, kernel_size, share_weights, simple_lifting):
    def __init__(self, configs, input_size):
        super(InverseAdpWaveletBlock, self).__init__()
        self.inverse_wavelet = InverseLiftingScheme(configs.enc_in, input_size=input_size, kernel_size=configs.lifting_kernel_size)

    def forward(self, c, d):
        reconstructed = self.inverse_wavelet(c, d)
        return reconstructed




class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        # print(x.shape)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)



class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


# class ExpMovingAverage(nn.Module):
#     """
#     [New] Exponential Moving Average (EMA) block
#     用于捕捉更灵敏的趋势，减少滞后
#     """
#
#     def __init__(self, alpha=0.5):
#         super(ExpMovingAverage, self).__init__()
#         # alpha 是平滑因子，范围 (0, 1)。
#         # 可以设为固定值，也可以设为可学习参数。这里为了简单设为超参数。
#         self.alpha = alpha
#
#     def forward(self, x):
#         # x shape: [Batch, Length, Channel]
#         # EMA 公式: S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
#
#         # 初始化 ema 序列，第一个值通常等于输入序列的第一个值
#         ema = torch.zeros_like(x)
#         ema[:, 0, :] = x[:, 0, :]
#
#         # 简单的循环实现 (对于时间序列预测，序列长度通常不会特别大，循环开销可接受)
#         # 如果追求极致性能，可以使用 CUDA kernel 或 cumsum 近似优化
#         for t in range(1, x.size(1)):
#             ema[:, t, :] = self.alpha * x[:, t, :] + (1 - self.alpha) * ema[:, t - 1, :]
#
#         return ema

class series_decomp1(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp1, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

# class series_decomp1(nn.Module):
#     """
#     [Updated] Series decomposition block with Hybrid Smoothing
#     结合 SMA 和 EMA 进行趋势分离
#     """
#
#     def __init__(self, kernel_size, alpha=0.5):
#         super(series_decomp1, self).__init__()
#
#         # 1. 传统的移动平均 (SMA) - 负责全局平滑
#         self.moving_avg = moving_avg(kernel_size, stride=1)
#
#         # 2. 新增的指数移动平均 (EMA) - 负责快速响应
#         self.exp_moving_avg = ExpMovingAverage(alpha=alpha)
#
#         # 3. 融合权重 (Learnable Fusion Weight)
#         # 初始化为 0.5，表示两者各占一半。使用 sigmoid 确保权重在 0-1 之间。
#         self.fusion_weight = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
#
#     def forward(self, x):
#         # 计算 SMA 趋势
#         trend_sma = self.moving_avg(x)
#
#         # 计算 EMA 趋势
#         trend_ema = self.exp_moving_avg(x)
#
#         # 融合趋势 (Hybrid Trend)
#         # weight 经过 sigmoid 限制在 (0, 1) 范围
#         lambda_w = torch.sigmoid(self.fusion_weight)
#
#         # 公式: X_trend = lambda * SMA + (1 - lambda) * EMA
#         trend_hybrid = lambda_w * trend_sma + (1 - lambda_w) * trend_ema
#
#         # 计算残差/季节项
#         res = x - trend_hybrid
#
#         return res, trend_hybrid

# class series_decomp1(nn.Module):
#     def __init__(self, kernel_size, alpha=0.5):
#         super(series_decomp1, self).__init__()
#         self.moving_avg = moving_avg(kernel_size, stride=1)  # SMA
#         self.exp_moving_avg = ExpMovingAverage(alpha=alpha)  # EMA
#         self.gamma = nn.Parameter(torch.tensor(1.0))  # 修正力度
#
#     def forward(self, x):
#         # 1. 先用 SMA 提取一个粗略的、稳定的趋势
#         trend_base = self.moving_avg(x)
#
#         # 2. 计算当前输入与粗略趋势的“偏差” (Deviation)
#         # 这个偏差里包含了：由于 SMA 滞后导致的丢失趋势 + 高频噪声
#         deviation = x - trend_base
#
#         # 3. 用 EMA 平滑这个偏差
#         # 目的是提取出“滞后的趋势部分”，滤掉“纯噪声”
#         trend_correction = self.exp_moving_avg(deviation)
#
#         # 4. 将修正量加回基础趋势
#         trend_final = trend_base + self.gamma * trend_correction
#
#         res = x - trend_final
#         return res, trend_final

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        # self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.patch_num = int((self.seq_len // (2 ** configs.lifting_levels)) // configs.patch_len)
        self.encoder_levels = nn.ModuleList()
        self.linear_levels = nn.ModuleList()
        self.coef_linear_levels = nn.ModuleList()
        self.coef_dec_levels = nn.ModuleList()

        # self.revin_layer = RevIN(configs.enc_in, affine=False, subtract_last=False)
        kernel_size = 23
        self.decompsition = series_decomp1(kernel_size)

        in_planes = configs.enc_in
        input_size = self.seq_len
        expand_ratio = 1

        for i in range(configs.lifting_levels):
            self.encoder_levels.add_module(
                'encoder_level_' + str(i),
                AdpWaveletBlock(configs, input_size)
            )
            in_planes *= 1
            input_size = input_size // 2
            self.linear_levels.add_module(
                'linear_level_' + str(i),
                nn.Sequential(
                    nn.Linear(input_size, input_size * expand_ratio),
                    # nn.Tanh()
                )
            )
            self.coef_linear_levels.add_module(
                'linear_level_' + str(i),
                nn.Sequential(
                    nn.Linear(input_size, input_size * expand_ratio),
                    # nn.Tanh()
                )
            )
            self.coef_dec_levels.add_module(
                'linear_level_' + str(i),
                nn.Sequential(
                    nn.Linear(input_size, input_size * expand_ratio),
                    # nn.Tanh()
                )
            )

        self.input_size = input_size
        self.decoder_levels = nn.ModuleList()

        for i in range(configs.lifting_levels - 1, -1, -1):
            self.decoder_levels.add_module(
                'decoder_level_' + str(i),
                InverseAdpWaveletBlock(configs, input_size=input_size)
            )
            in_planes //= 1
            input_size *= 2

        self.en_embedding = EnEmbedding(configs.enc_in, configs.d_model, self.patch_len, configs.dropout)
        self.ex_embedding = DataEmbedding_inverted(self.seq_len // (2 ** configs.lifting_levels), configs.d_model, configs.embed, configs.freq,
                                                   configs.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(configs.enc_in, self.head_nf, self.seq_len // (2 ** configs.lifting_levels),
                                head_dropout=configs.dropout)

        self.linear_seasonal = nn.Sequential(
            nn.Linear(self.seq_len // (2 ** configs.lifting_levels), configs.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(configs.hidden_size, self.seq_len // (2 ** configs.lifting_levels))
        )

        self.linear_trend = nn.Sequential(
            nn.Linear(self.seq_len // (2 ** configs.lifting_levels), configs.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(configs.hidden_size, self.seq_len // (2 ** configs.lifting_levels))
        )



        self.endlinear = nn.Sequential(
            nn.Linear(12, 64),
            # nn.ReLU(),
            nn.Linear(64, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 3)
        )


        self.projection = nn.Linear(self.seq_len, configs.pred_len, bias=True)


    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, clusters=None):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        _, _, N = x_enc.shape



        x_enc = x_enc.permute(0, 2, 1)  # [256, 12, 168]
        encoded_coefficients = []
        x_embedding_levels = []
        coef_embedding_levels = []
        # Encoding
        for l, l_linear, c_linear in zip(self.encoder_levels, self.linear_levels, self.coef_linear_levels):
            # print("Level", level, "x_shape:", x.shape)
            x_enc, r, details = l(x_enc)
            # print("The size of x is", x.size(), "and the size of details is", details.size(), l_linear)
            encoded_coefficients.append(details)
            coef_embedding_levels.append(c_linear(details))
            x_embedding_levels.append(l_linear(x_enc))
        # Embedding
        x_enc = x_enc.permute(0, 2, 1)
        # Xer
        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))  # [batch*12, patch_num+1, d_model]
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)  # [batch, 12, d_model]

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)
        # Amp
        seasonal, trend = self.decompsition(x_enc)
        seasonal = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
        trend = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
        amp_dec_out = seasonal + trend
        # 并
        # dec_out = torch.cat([amp_dec_out, dec_out], dim=-1)
        # x_dec = self.endlinear(dec_out).permute(0, 2, 1)
        x_dec = dec_out + amp_dec_out
        x_dec = x_dec.permute(0, 2, 1)
        # Decoding
        for dec, x_emb_level, coef_emb_level, c_linear in zip(self.decoder_levels, x_embedding_levels[::-1],
                                                              coef_embedding_levels[::-1], self.coef_dec_levels[::-1]):
            details = encoded_coefficients.pop()
            details = coef_emb_level + c_linear(details)
            x_dec = x_dec + x_emb_level
            x_dec = dec(x_dec, details)
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :]
        # dec_out = self.endlinear(dec_out)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        dec_out = self.endlinear(dec_out)

        return dec_out


    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        x_enc = x_enc.squeeze(1).permute(0, 2, 1)
        B, L, C = x_enc.shape
        # if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        # else:
        #     return None