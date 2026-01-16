import argparse


class BaseConfig:
    """基类，包含所有模型共享的超参数"""

    def __init__(self):
        self.task_name = 'long_term_forecast'
        self.is_training = 1
        self.model_id = 'test'

        self.seq_len = 168
        self.pred_len = 24
        self.label_len = self.pred_len

        self.embed = 'timeF'
        self.activation = 'gelu'
        self.use_norm = 1
        self.channel_independence = 1
        self.decomp_method = 'moving_avg'
        self.moving_avg = 24
        self.factor = 1



class DeWAttNet(BaseConfig):
    def __init__(self):
        super().__init__()
        # self.n_clusters = 3
        self.lifting_levels = 1
        self.d_model = 64
        self.freq = 'h'
        self.embed = 'timeF'
        self.output_attention = False
        self.enc_in = 12
        self.regu_details = 0.01
        self.regu_approx = 0.01
        self.lifting_kernel_size = 7
        self.n_heads = 2
        self.e_layers = 1
        self.d_ff = 64

        self.lr = 0.002
        self.batchsize = 256
        self.epochs = 50
        self.weightdecay = 0.3
        self.decaypatience = 4

        self.patch_len = 24
        self.dropout = 0.2
        self.hidden_size = 256






def get_config(model_name: str):
    """根据模型名称获取特定的超参数配置"""

    if model_name == 'DeWAttNet':
        return DeWAttNet()


    else:
        raise ValueError(f"Unsupported model name: {model_name}")
