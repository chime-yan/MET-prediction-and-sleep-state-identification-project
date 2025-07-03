import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, IterableDataset, get_worker_info


class LearnableDiagonalProjection(nn.Module):
    """
    模型内可学习对角仿射映射（这也是 Embedding 的一种简化形式（线性 embedding））

    对输入特征做每通道独立缩放(scale) + 偏移(shift)，
    且仅使用向量参数而非矩阵，保证映射矩阵始终为对角结构，避免维度混合并保持全秩，以保证信息完整。
    用广播机制高效实现了对角矩阵变换的效果，从而在不显式构造对角矩阵的情况下，达到了:
        仿射变换（每维独立线性变换）；
        保持对角结构（不混合维度）；
        保持全秩（只要每个 scale[i] ≠ 0）；
        参数量最小化（只需两个 input_dim 维度的向量）；
        高效计算（避免矩阵乘法，节省内存和算力）
    """

    def __init__(self, input_dim):
        super().__init__()
        # 仅使用对角向量参数，scale 和 shift 形状为 (input_dim,)
        # 等价于对角矩阵 W = diag(scale), b = shift
        self.scale = nn.Parameter(torch.ones(input_dim))
        self.shift = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        """
        x: Tensor of shape (batch, seq_len, input_dim)
        返回: Tensor of same shape
        """

        dim = x.shape[2]
        # 对角仿射映射，不存在任何跨通道权重：
        # x_proj[..., i] = scale[i] * x[..., i] + shift[i]
        x_proj = x * self.scale.view(1, 1, dim) + self.shift.view(1, 1, dim)
        return x_proj


class MultiFileCSVIterableDataset(IterableDataset):
    def __init__(self, file_list: list, batch_size: int, usecols: list, seq_len: int,
                 question_type='regression', classification_column_count=1,
                 reg_classifier=False):
        """
        参数说明：
          file_list: CSV 文件的路径列表
          batch_size: batch_size = time_steps - seq_len
          usecols: 要读取的列列表，例如 ['hour_sin', 'hour_cos', 'x', 'y', 'z', 'age', 'sex', 'MET']
          seq_len: 滑动窗口的长度
          question_type: 选择 regression or classification
          classification_column_count: 分类问题的标签列的数量

          time_steps: 每次读取时间步的数量
          reg_classifier: 是否开启回归任务辅助分类头时的读取数据模式（默认False）
        """

        super(MultiFileCSVIterableDataset, self).__init__()
        self.file_list = file_list
        self.batch_size = batch_size
        self.usecols = usecols
        self.seq_len = seq_len

        self.time_steps = self.batch_size + self.seq_len

        self.question_type = question_type

        if classification_column_count >= 1:
            self.classification_column_count = classification_column_count
        else:
            raise ValueError('classification_column_count must be >= 1')

        self.reg_classifier = reg_classifier

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:
            # 只有一个 worker（num_workers=0）
            file_subset = self.file_list

        else:
            # 多 worker，分配不同文件
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            file_subset = self.file_list[worker_id::num_workers]  # 让每个 worker 读取不同的文件
            print(f"Worker {worker_id + 1}/{num_workers}: 处理 {len(file_subset)} 个文件")

        # 流式读取分配到的文件
        for file_path in file_subset:
            init_hidden = True
            for chunk in pd.read_csv(file_path, chunksize=self.time_steps, usecols=self.usecols):
                chunk = chunk[self.usecols] # 调整列顺序
                data = torch.tensor(chunk.values, dtype=torch.float)

                if data.size(0) < self.time_steps:
                    continue

                # 1. 用 unfold 在时间维度上一次性取出 (seq_len + 1) 行，步长 1
                #    得到 shape = (time_steps - seq_len, seq_len + 1, feature_dim + 1)
                data = data.unfold(0, self.seq_len + 1, 1)
                data = data.transpose(1, 2)

                # 2. 分离 windows 和 targets
                if self.question_type == 'regression' and self.reg_classifier:
                    windows = data[:, :self.seq_len, :-2]
                    targets = data[:, -1, -2:]

                elif self.question_type == 'regression':
                    #    windows: 每个窗口的前 seq_len 行、所有特征列（不含标签）
                    windows = data[:, :self.seq_len, :-1]  # shape = (batch_size, seq_len, feature_dim)

                    #    targets: 每个窗口最后一行的标签列
                    targets = data[:, -1, -1]  # shape = (batch_size,)

                elif self.question_type == 'classification':
                    windows = data[:, :self.seq_len, :-self.classification_column_count] # shape = (batch_size, seq_len, feature_dim + classification_column_count)
                    targets = data[:, -1, -self.classification_column_count:]  # shape = (batch_size, classification_column_count)

                else:
                    raise NotImplementedError

                yield windows, targets, init_hidden
                init_hidden = False


class EarlyStopping:
    def __init__(self, patience=10, delta=0.01, path='best_model.pth'):
        """
        参数：
        patience: 容忍多少个epoch验证集性能没有提升
        delta: 用于定义“性能提升”的阈值
        """

        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model, hidden=None):
        if self.best_loss is None:
            self.best_loss = val_loss

            if hidden is not None:
                hidden = tuple(h.detach() for h in hidden)
                # 保存最佳模型
                torch.save({'model': model.state_dict(),
                            'hidden': hidden},
                           self.path)
                print('Saved first epoch model')

            else:
                # 保存最佳模型
                torch.save({'model': model.state_dict()}, self.path)
                print('Saved first epoch model')

        elif val_loss < self.best_loss - self.delta:
            print(f"Best Val Loss Updated: {self.best_loss:.4f} → {val_loss:.4f}")
            self.best_loss = val_loss
            self.counter = 0

            if hidden is not None:
                hidden = tuple(h.detach() for h in hidden)
                # 保存最佳模型
                torch.save({'model': model.state_dict(),
                            'hidden': hidden},
                           self.path)
                print(f'Saved Best Val Loss model, Best val loss: {self.best_loss:.4f}')

            else:
                # 保存最佳模型
                torch.save({'model': model.state_dict()}, self.path)
                print(f'Saved Best Val Loss model, Best val loss: {self.best_loss:.4f}')

        else:
            self.counter += 1
            if self.counter > self.patience:
                self.early_stop = True


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layers:int , hidden_neurons: list,
                 activation_fn=None, last_activation=None,
                 dropout_way=False, dropout=0.3, flatten=False, batch_norm=False):
        super(MLP, self).__init__()
        """
            基本参数：
            - input_size: 输入层的特征数量
            - output_size: 输出维度
    
            结构超参数：
            - hidden_layers: 隐藏层数量
            - hidden_neurons: 每个隐藏层的神经元数量，列表格式
    
            结构基本参数：
            - activation_fn: 默认的隐藏层激活函数(默认为 ReLU)    
            - last_activation: 最后隐藏层到输出层的激活函数（可选）
    
            - dropout_way: 是否启用暂退法(默认为False)
            - dropout: 隐藏层丢弃概率(默认0.3)   # dropout 是结构超参数
            - flatten: 是否在第一层添加展平层（默认False）
            - batch_norm: 是否在每个隐藏层后添加 batch_norm 层（默认False）
        """

        self.activation_fn = activation_fn if activation_fn else nn.ReLU()
        self.batch_norm = batch_norm


        layers = []
        if flatten:
            layers.append(nn.Flatten())

        if hidden_layers == 0:
            layers.append(nn.Linear(input_size, output_size))
            if last_activation is not None:
                layers.append(last_activation)

            else:
                pass

        else:
            assert hidden_layers == len(hidden_neurons)

            layers.append(nn.Linear(input_size, hidden_neurons[0]))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_neurons[0]))

            layers.append(self.activation_fn)

            if dropout_way:
                layers.append(nn.Dropout(dropout))

            for i in range(1, hidden_layers):
                layers.append(nn.Linear(hidden_neurons[i - 1], hidden_neurons[i]))
                if self.batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_neurons[i]))

                layers.append(self.activation_fn)

                if dropout_way:
                    layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(hidden_neurons[-1], output_size))
            if last_activation is not None:
                layers.append(last_activation)

        self.network = nn.Sequential(*layers)
        self._init_params()

    def forward(self, x):
        return self.network(x)

    # 初始化所有层的权重和偏置，包括输出层
    def _init_params(self):
        num_layers = len(self.network)
        linear_indices = [i for i, l in enumerate(self.network) if isinstance(l, nn.Linear)]
        last_linear_index = linear_indices[-1]

        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear):
                next_activation = None
                is_output_layer = (i == last_linear_index)

                # 如果不是输出层，找下一激活函数（考虑是否跳过 BatchNorm）
                if not is_output_layer:
                    step = 2 if self.batch_norm else 1
                    if i + step < num_layers:
                        next_layer = self.network[i + step]
                        if isinstance(next_layer, (nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.GELU,
                                                   nn.Tanh, nn.Sigmoid, nn.ELU)):
                            next_activation = next_layer

                else:
                    if i + 1 < num_layers:
                        next_layer = self.network[i + 1]
                        if isinstance(next_layer, (nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.GELU,
                                                   nn.Tanh, nn.Sigmoid, nn.ELU)):
                            next_activation = next_layer

                # 初始化策略选择
                if next_activation is not None:
                    if isinstance(next_activation, (nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.GELU, nn.ELU)):
                        nn.init.kaiming_uniform_(layer.weight)

                    elif isinstance(next_activation, (nn.Tanh, nn.Sigmoid)):
                        nn.init.xavier_uniform_(layer.weight)

                else:
                    # 输出层没有激活时，默认用 Xavier
                    nn.init.xavier_uniform_(layer.weight)

                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k  # 特征维度

    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (batch_size, heads, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)   # (B, H, S, S)

        B, _, S, _ = Q.shape

        # 如果 mask 是 True，自动生成因果下三角 mask
        if mask is True:
            # 先造 (S, S) 下三角，再扩展到 (B, 1, S, S)
            base = torch.tril(torch.ones(S, S, dtype=torch.bool, device=Q.device))
            mask_tensor = base.unsqueeze(0).unsqueeze(1).expand(B, 1, S, S)
        else:
            mask_tensor = mask  # 可能是 None or 具体 Tensor

        if mask_tensor is not None:
            # mask_tensor == False 的位置填 -1e9
            # .masked_fill(cond, value) 会把 cond == True 的元素替换成 value，而保留其他位置不变
            scores = scores.masked_fill(~mask_tensor, float('-1e9'))

        attn_weights = F.softmax(scores, dim=-1)    # (B, H, S, S)
        output = torch.matmul(attn_weights, V)      # (B, H, S, d_k)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=None):
        super().__init__()
        self.d_model = d_model

        # 自动计算头数
        if num_heads is None:
            # 默认策略：每头至少32维
            self.num_heads = max(1, d_model // 32)
            self.num_heads = min(8, self.num_heads)  # 不超过8头
        else:
            self.num_heads = num_heads

        assert d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) 必须要被 num_heads ({self.num_heads}) 整除！"

        self.d_k = self.d_model // self.num_heads
        self.attn = ScaledDotProductAttention(self.d_k)

        # 投影矩阵
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self._init_weights()

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        # 先按token维度(时间维度)拆分特征，再按头部维度重新组织
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 线性变换并分头
        Q = self.split_heads(self.W_Q(Q))
        K = self.split_heads(self.W_K(K))
        V = self.split_heads(self.W_V(V))

        # 计算注意力
        x, attn_weights = self.attn(Q, K, V, mask)

        # 合并头部并输出
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_O(x), attn_weights

    def _init_weights(self):
        # Q/K/V使用截断正态
        std = (1.0 / self.d_model) ** 0.5
        std = max(0.02, min(std, 0.5))

        nn.init.trunc_normal_(self.W_Q.weight, std=std)
        nn.init.trunc_normal_(self.W_K.weight, std=std)
        nn.init.trunc_normal_(self.W_V.weight, std=std)
        nn.init.zeros_(self.W_Q.bias)
        nn.init.zeros_(self.W_K.bias)
        nn.init.zeros_(self.W_V.bias)

        # 输出投影使用Xavier
        nn.init.xavier_uniform_(self.W_O.weight)
        nn.init.zeros_(self.W_O.bias)


class LightweightAttention(nn.Module):
    """
    小 hidden_size(hidden < 32) 下使用的轻量化注意力机制。
    模拟自注意力加权，但避免多头计算复杂度。
    """

    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

        self.query = nn.Linear(self.d_k, self.d_k)
        self.key = nn.Linear(self.d_k, self.d_k)
        self.value = nn.Linear(self.d_k, self.d_k)

        self._init_weights()

    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (batch, seq_len, d_k)
        q = self.query(Q)  # (B, S, D)
        k = self.key(K)    # (B, S, D)
        v = self.value(V)  # (B, S, D)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B, S_q, S_k), S_q = S_k

        B, S, _ = Q.shape

        # 如果 mask 是 True，自动生成因果下三角 mask
        if mask is True:
            # 先造 (S, S) 下三角，再扩展到 (B, S, S)
            base = torch.tril(torch.ones(S, S, dtype=torch.bool, device=Q.device))
            mask_tensor = base.unsqueeze(0).expand(B, S, S)
        else:
            mask_tensor = mask  # 可能是 None or 具体 Sensor

        if mask_tensor is not None:
            # mask_tensor == False 的位置填 -1e9
            # .masked_fill(cond, value) 会把 cond == True 的元素替换成 value，而保留其他位置不变
            scores = scores.masked_fill(~mask_tensor, float('-1e9'))

        weights = F.softmax(scores, dim=-1)  # (B, S, S)
        output = torch.matmul(weights, v)  # (B, S, D)
        return output, weights

    def _init_weights(self):
        # Q/K/V使用截断正态
        std = (1.0 / self.d_k) ** 0.5
        std = max(0.02, min(std, 0.5))

        nn.init.trunc_normal_(self.query.weight, std=std)
        nn.init.trunc_normal_(self.key.weight, std=std)
        nn.init.trunc_normal_(self.value.weight, std=std)
        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.key.bias)
        nn.init.zeros_(self.value.bias)


class AttentionLSTMWithMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int,
                 mlp_hidden_layers: int, mlp_hidden_neurons: list,
                 mlp_activation_fn=None, mlp_last_activation=None,
                 mlp_dropout_way=False, mlp_dropout=0.3,
                 mlp_flatten=False, mlp_batch_norm=False,
                 bidirectional=False, lstm_dropout=0.,
                 diagonal_projection=False, attn_heads=None,
                 lstm_layers=1, attn_dropout=0.3, attn_mask = None, reg_classifier=False,
                 classifier_hidden_size=32, classifier_output_size=2, used_reg_classifier=True):
        super(AttentionLSTMWithMLP, self).__init__()
        """
            参数：
            - input_size: 输入层的特征数量
            - output_size: 输出维度

            - hidden_size: LSTM 隐藏层特征维度

            - mlp_hidden_layers: MLP 隐藏层数量
            - mlp_hidden_neurons: 每个 MLP 隐藏层的神经元数量，列表格式

            - mlp_activation_fn: 默认的 MLP 隐藏层激活函数(默认为 ReLU)    
            - mlp_last_activation: MLP 最后隐藏层到输出层的激活函数（可选）

            - mlp_dropout_way: MLP 是否启用暂退法(默认为False)
            - mlp_dropout: MLP 隐藏层丢弃概率(默认0.3)
            - mlp_flatten: 是否在 MLP 的第一层添加展平层（默认False）
            - mlp_batch_norm: 是否在每个隐藏层后添加 batch_norm 层（默认False）
            
            - bidirectional: 是否启用双向LSTM（默认False）
            - lstm_dropout: 是否启用2层以上LSTM层之间的丢弃(默认0.)
            
            - diagonal_projection: 是否启用可学习对角仿射映射，用于调整数据范围到有效梯度范围（默认False）
            - attn_heads: 多头自注意力机制的头数量，默认为None,即自动计算
            - lstm_layers: LSTM 结构单元数(默认1)
            - attn_dropout: 自注意力机制丢弃概率(默认0.3)
            - attn_mask: 若传入mask（torch.Tensor），则使用传入的mask；若为True，则使用内置的下三角mask
            - reg_classifier: 是否开启回归任务辅助分类头结构（默认False）
            - classifier_hidden_size: 辅助分类头隐藏层（一层）神经元数量(默认32)
            - classifier_output_size: 辅助分类头的输出维度(默认2，即二分类)
            - used_reg_classifier: 前向传播时是否使用辅助分类头（默认True）
        """

        self.diagonal_projection = diagonal_projection
        self.mask = attn_mask
        self.reg_classifier = reg_classifier
        self.used_reg_classifier = used_reg_classifier

        if self.diagonal_projection:
            self.diagonal_projection_layer = LearnableDiagonalProjection(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

        if hidden_size >= 32:
            self.attention = MultiHeadAttention(
                d_model=hidden_size,
                num_heads=attn_heads
            )
        else:
            self.attention = LightweightAttention(hidden_size)

        self.attn_dropout_layer = nn.Dropout(attn_dropout)

        self.mlp = MLP(input_size=hidden_size, output_size=output_size, hidden_layers=mlp_hidden_layers,
                       hidden_neurons=mlp_hidden_neurons, activation_fn=mlp_activation_fn,
                       last_activation=mlp_last_activation, dropout_way=mlp_dropout_way, dropout=mlp_dropout,
                       flatten=mlp_flatten, batch_norm=mlp_batch_norm)

        if self.reg_classifier:
            # 辅助分类头
            self.classifier = MLP(input_size=hidden_size, output_size=classifier_output_size, hidden_layers=1,
                                  hidden_neurons=[classifier_hidden_size], activation_fn=mlp_activation_fn,
                                  dropout_way=mlp_dropout_way, dropout=mlp_dropout)

        self._init_lstm_params()

    def forward(self, x, hidden, return_sequences=False):
        if self.diagonal_projection:
            x = self.diagonal_projection_layer(x)

        lstm_out, hidden = self.lstm(x, hidden) # 输出形状：(batch_size, seq_len, hidden_size),当batch_first=True时
        norm_lstm_out = self.layer_norm(lstm_out)  # (B, S, D)

        attn_out, _ = self.attention(norm_lstm_out, norm_lstm_out, norm_lstm_out, mask=self.mask)
        attn_out = self.attn_dropout_layer(attn_out)
        lstm_out = attn_out + lstm_out  # 残差连接

        if return_sequences:
            out = lstm_out # 输出形状：(batch_size, seq_len, hidden_size)
            out = out.reshape(-1, lstm_out.size(2)) # (batch_size * seq_len, output_size)

            norm_out = self.layer_norm(out)
            model_out = self.mlp(norm_out)  # (batch_size * seq_len, output_size)

        else:
            out = lstm_out[:, -1, :]  # 取最后一个时间步的输出, 输出形状：(batch_size, hidden_size)，即(B, D)

            # # attn_weights: (B, H, S_q, S_k)，把多头注意力权重拿出来做池化
            # weights_mean = attn_weights.mean(dim=1)  # (B, S_q, S_k)
            # weights_last_query = weights_mean[:, -1, :]  # (B, S_k)
            # weights_norm = torch.softmax(weights_last_query, dim=-1)  # (B, S_k)
            #
            # # 输出形状：(batch_size, hidden_size)，即(B, D)
            # pooled = torch.bmm(weights_norm.unsqueeze(1), out).squeeze(1)

            norm_out = self.layer_norm(out)
            model_out = self.mlp(norm_out) # 输出形状：(batch_size, output_size)

        if self.reg_classifier and self.used_reg_classifier:
            class_logits = self.classifier(norm_out) # 输出形状：(batch_size, classifier_output_size) 或 (batch_size * seq_len, classifier_output_size)
            return model_out, hidden, class_logits
        else:
            return model_out, hidden

    def _init_lstm_params(self):
        # 对于 LSTM 层，每一层包含多组参数：
        # weight_ih_l[k]：输入到隐藏状态的权重
        # weight_hh_l[k]：隐藏状态到隐藏状态的权重
        # bias_ih_l[k] 和 bias_hh_l[k]：偏置
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                # 使用 Xavier 均匀分布初始化输入到隐藏层的权重，保证输入信号在传播过程中不过度缩放。
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # 使用正交初始化隐藏到隐藏层的权重，保持长期状态传递的稳定性，避免梯度爆炸或消失。
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # 将偏置初始化为 0，将遗忘门的偏置设为 1
                nn.init.zeros_(param.data)

                # LSTM 的偏置通常被分为 4 个部分（顺序通常为 input, forget, cell, output）
                # 将 forget gate 的偏置初始化为 1，让模型在初始状态下更容易保留长期记忆。
                n = param.size(0)  # n = 4 * hidden_size
                start, end = n // 4, n // 2  # 偏置的排列顺序为：i, f, g, o
                param.data[start:end].fill_(1)  # 将遗忘门对应部分设置为 1


def get_csv_dataloader(file_list: list, csv_usecols_list: list, batch_size: int, seq_len: int,
                       num_workers=0, prefetch_factor: int = None, question_type='regression',
                       classification_column_count=1, reg_classifier=False):
    usecols = csv_usecols_list

    # 创建 DataLoader
    dataset = MultiFileCSVIterableDataset(file_list, batch_size, usecols, seq_len, question_type,
                                          classification_column_count, reg_classifier)
    # 注意：对于 IterableDataset，DataLoader 的 batch_size 参数应设置为 None
    dataloader = DataLoader(dataset, batch_size=None, num_workers=num_workers, prefetch_factor=prefetch_factor,
                            shuffle=False, pin_memory=True)
    # persistent_workers=True

    return dataloader

def train_model(model, train_loader, loss_fn, optimizer, device, question_type='regression',
                is_soft_labels=False, class_prob_path=None, hidden_func=False,
                batch_warmup_scheduler=None, batch_lr_scheduler=None,
                reg_classifier=False, classifier_loss_alpha=0.2):
    model.train()

    total_loss = 0
    total_samples = 0
    hidden = None

    for batch_idx, (inputs, targets, init_hidden) in enumerate(train_loader):
        if inputs.dim() != 3:
            inputs = inputs.unsqueeze(1)

        if hidden is not None:
            # 将 hidden 的梯度清空
            hidden = tuple(h.detach() for h in hidden)

        if hidden is not None and init_hidden:
            hidden = None

        optimizer.zero_grad()

        if hidden_func:
            if reg_classifier:
                outputs, hidden, class_logits = model.forward(inputs.to(device), hidden)
            else:
                outputs, hidden = model.forward(inputs.to(device), hidden)

        else:
            if reg_classifier:
                outputs, _, class_logits = model.forward(inputs.to(device), hidden)
            else:
                outputs, _ = model.forward(inputs.to(device), hidden)

        if question_type == 'regression':
            if reg_classifier:
                loss = (1 - classifier_loss_alpha) * loss_fn(outputs, targets[:, 1].unsqueeze(1).to(device)) + \
                    classifier_loss_alpha * nn.CrossEntropyLoss()(class_logits, targets[:, 0].long().to(device))
            else:
                loss = loss_fn(outputs, targets.unsqueeze(1).to(device))

        elif question_type == 'classification':
            if not is_soft_labels:
                loss = loss_fn(outputs, targets.to(torch.long).to(device))
            else:
                loss = loss_fn(outputs, targets.to(torch.float).to(device))

            prob_out = F.softmax(outputs.detach(), dim=1)

        else:
            raise NotImplementedError

        loss.backward()
        optimizer.step()

        if batch_warmup_scheduler is not None:
            if not batch_warmup_scheduler.is_finished():
                batch_warmup_scheduler.step_batch()

            elif (batch_lr_scheduler is not None) and batch_warmup_scheduler.is_finished():
                batch_lr_scheduler.step()

        elif (batch_warmup_scheduler is None) and batch_lr_scheduler is not None:
            batch_lr_scheduler.step()

        # 累积损失加权
        batch_count = inputs.size(0)
        total_loss += loss.item() * batch_count
        total_samples += batch_count

        if question_type == 'classification' and class_prob_path is not None:
            pred_dict = {f"prob_{label}": prob_out[:, label].cpu().tolist()
                         for label in range(prob_out.size(1))}

            if not is_soft_labels:
                targets_dict = {'targets': targets.cpu().tolist()}
                df_dict = pred_dict | targets_dict

                # 组织数据
                batch_df = pd.DataFrame(df_dict)
            else:
                targets_dict = {f"targets_{label}": targets[:, label].cpu().tolist()
                                for label in range(targets.size(1))}
                df_dict = pred_dict | targets_dict

                batch_df = pd.DataFrame(df_dict)

            # 流式写入csv
            if batch_idx == 0:
                batch_df.to_csv(class_prob_path, index=False, mode="w")  # 第一个 batch 写入，带表头

            else:
                batch_df.to_csv(class_prob_path, index=False, mode="a", header=False)  # 追加写入，不重复写表头

            # print(f"Batch {batch_idx + 1} 的 prob_out 和 targets 写入 {class_prob_path}")

    train_avg_loss = total_loss / total_samples
    return model, train_avg_loss

# 验证过程
def validate_model(model, val_loader, loss_fn, device, question_type='regression',
                   is_soft_labels=False, class_val_prob_path: str = None, hidden_func=False,
                   reg_mu_target_lower: float = None, reg_mu_target_upper: float = None,
                   reg_classifier=False, classifier_loss_alpha=0.2):
    model.eval()

    with torch.no_grad():
        total_loss = 0
        total_samples = 0
        val_hidden = None

        total_mu_target_loss = 0
        total_mu_target_sum = 0

        for batch_idx, (val_inputs, val_targets, init_hidden) in enumerate(val_loader):
            if val_inputs.dim() != 3:
                val_inputs = val_inputs.unsqueeze(1)

            if hidden_func:
                if val_hidden is not None and init_hidden:
                    val_hidden = None

                if reg_classifier:
                    val_outputs, val_hidden, val_class_logits = \
                    model.forward(val_inputs.to(device), val_hidden)
                else:
                    val_outputs, val_hidden = model.forward(val_inputs.to(device), val_hidden)

            else:
                if reg_classifier:
                    val_outputs, _, val_class_logits = model.forward(val_inputs.to(device), val_hidden)
                else:
                    val_outputs, _ = model.forward(val_inputs.to(device), val_hidden)

            if question_type == 'regression':
                if reg_classifier:
                    val_loss = (1 - classifier_loss_alpha) * loss_fn(val_outputs, val_targets[:, 1].unsqueeze(1).to(device)) + \
                        classifier_loss_alpha * nn.CrossEntropyLoss()(val_class_logits, val_targets[:, 0].long().to(device))
                else:
                    val_loss = loss_fn(val_outputs, val_targets.unsqueeze(1).to(device))

                if reg_mu_target_lower is not None and reg_mu_target_upper is not None:
                    mu_target_loss, mu_target_sum = (
                    _boundary_mae(val_outputs.detach(), val_targets[:, 1].unsqueeze(1).to(device),
                                  lower=reg_mu_target_lower, upper=reg_mu_target_upper))

            elif question_type == 'classification':
                if not is_soft_labels:
                    val_loss = loss_fn(val_outputs, val_targets.to(torch.long).to(device))
                else:
                    val_loss = loss_fn(val_outputs, val_targets.to(torch.float).to(device))

                val_prob_out = F.softmax(val_outputs.detach(), dim=1)

            else:
                raise NotImplementedError


            # 累积损失加权
            batch_count = val_inputs.size(0)
            total_loss += val_loss.item() * batch_count
            total_samples += batch_count

            if question_type == 'regression' and reg_mu_target_lower is not None and reg_mu_target_upper is not None:
                total_mu_target_loss += mu_target_loss * mu_target_sum
                total_mu_target_sum += mu_target_sum


            if question_type == 'classification' and class_val_prob_path is not None:
                val_pred_dict = {f"prob_{label}": val_prob_out[:, label].cpu().tolist()
                                 for label in range(val_prob_out.size(1))}

                if not is_soft_labels:
                    val_targets_dict = {'targets': val_targets.cpu().tolist()}
                    df_dict = val_pred_dict | val_targets_dict

                    # 组织数据
                    batch_df = pd.DataFrame(df_dict)
                else:
                    val_targets_dict = {f"targets_{label}": val_targets[:, label].cpu().tolist()
                                        for label in range(val_targets.size(1))}
                    df_dict = val_pred_dict | val_targets_dict

                    batch_df = pd.DataFrame(df_dict)

                # 流式写入csv
                if batch_idx == 0:
                    batch_df.to_csv(class_val_prob_path, index=False, mode="w")  # 第一个 batch 写入，带表头

                else:
                    batch_df.to_csv(class_val_prob_path, index=False, mode="a", header=False)  # 追加写入，不重复写表头

                # print(f"Batch {batch_idx + 1} 的 val_prob_out 和 val_targets 写入 {class_val_prob_path}")

        val_avg_loss = total_loss / total_samples
        if question_type == 'regression' and reg_mu_target_lower is not None and reg_mu_target_upper is not None:
            val_avg_mu_target_loss = total_mu_target_loss / total_mu_target_sum
            return val_avg_loss, val_avg_mu_target_loss
        else:
            return val_avg_loss

def predict(model, device, pred_x, pred_y=None, loss_fn=None, hidden_func=False, pred_hidden=None,
            question_type='regression', is_soft_labels=False, return_sequences=False, anti_transform_func=None,
            **anti_transform_kwargs):
    model.eval()

    pred_x = torch.tensor(pred_x, dtype=torch.float).unsqueeze(0).to(device)

    if pred_y is not None:
        if question_type == 'regression':
            pred_y = torch.tensor(pred_y, dtype=torch.float).unsqueeze(1).to(device)

        elif question_type == 'classification':
            if not is_soft_labels:
                pred_y = torch.tensor(pred_y, dtype=torch.long).to(device)
            else:
                pred_y = torch.tensor(pred_y, dtype=torch.float).to(device)

        else:
            raise NotImplementedError

    with torch.no_grad():
        if hidden_func:
            pred, pred_hidden = model.forward(pred_x, pred_hidden, return_sequences)  # 保持 pred_hidden 跨步传递

            if pred_y is not None and loss_fn is not None:
                if question_type == 'regression':
                    # 映射回原始空间进行评估
                    if anti_transform_func is not None:
                        pred = anti_transform_func(pred, **anti_transform_kwargs)
                        pred_loss = loss_fn(pred, pred_y)

                    else:
                        pred_loss = loss_fn(pred, pred_y)

                    pred = pred.squeeze(1).detach().cpu().numpy()
                    return pred, pred_loss.item(), pred_hidden

                elif question_type == 'classification':
                    pred_loss = loss_fn(pred, pred_y)

                    prob_out = F.softmax(pred.detach(), dim=1)
                    _, prob_out_index = torch.max(prob_out, 1)
                    return prob_out_index.cpu().numpy(), prob_out.cpu().numpy(), pred_loss.item(), pred_hidden

            else:
                if question_type == 'regression':
                    pred = pred.squeeze(1).detach().cpu().numpy()
                    return pred, pred_hidden

                elif question_type == 'classification':
                    prob_out = F.softmax(pred.detach(), dim=1)
                    _, prob_out_index = torch.max(prob_out, 1)
                    return prob_out_index.cpu().numpy(), pred_hidden

        else:
            pred, _ = model.forward(pred_x, pred_hidden, return_sequences)

            if pred_y is not None:
                if question_type == 'regression':
                    # 映射回原始空间进行评估
                    if anti_transform_func is not None:
                        pred = anti_transform_func(pred, **anti_transform_kwargs)
                        pred_loss = loss_fn(pred, pred_y)

                    else:
                        pred_loss = loss_fn(pred, pred_y)

                    pred = pred.squeeze(1).detach().cpu().numpy()
                    return pred, pred_loss.item()

                elif question_type == 'classification':
                    pred_loss = loss_fn(pred, pred_y)

                    prob_out = F.softmax(pred.detach(), dim=1)
                    _, prob_out_index = torch.max(prob_out, 1)
                    return prob_out_index.cpu().numpy(), prob_out.cpu().numpy(), pred_loss.item()

            else:
                if question_type == 'regression':
                    pred = pred.squeeze(1).detach().cpu().numpy()
                    return pred

                elif question_type == 'classification':
                    prob_out = F.softmax(pred.detach(), dim=1)
                    _, prob_out_index = torch.max(prob_out, 1)
                    return prob_out_index.cpu().numpy()

def _boundary_mae(preds: torch.Tensor, targets: torch.Tensor, lower=0.8, upper=1.2):
    mask = ((targets >= lower) & (targets <= upper)).to(preds.device)
    if mask.sum() == 0:
        return 0, 0  # 区间内无样本
    return torch.abs(preds[mask] - targets[mask]).mean(), mask.sum()
