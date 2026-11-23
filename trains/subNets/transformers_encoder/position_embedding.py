import math
import torch
import torch.nn as nn

def make_positions(tensor, padding_idx, left_pad):
    """
    将非填充（padding）标记的位置替换为它们的“位置编号”。
    位置编号从 padding_idx+1 开始；填充符号保持不变。
    left_pad=True 表示填充在序列左侧，否则填充在右侧。
    """
    # 1. 准备一个足够长的 “位置索引” 缓冲区
    max_pos = padding_idx + 1 + tensor.size(1)
    device = tensor.get_device()
    buf_name = f'range_buf_{device}'
    # 如果缓存不存在，则初始化一个同类型的空张量
    if not hasattr(make_positions, buf_name):
        setattr(make_positions, buf_name, tensor.new())
    # 确保缓存与输入同类型
    setattr(make_positions, buf_name, getattr(make_positions, buf_name).type_as(tensor))
    # 如果当前缓存长度不够，就填充 1,2,3… 的位置索引
    if getattr(make_positions, buf_name).numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=getattr(make_positions, buf_name))
    # 2. 构造 mask，标记哪些位置不是 padding
    mask = tensor.ne(padding_idx)
    # 3. 从缓存中取出前 seq_len 个位置索引，扩展到与 tensor 同形
    positions = getattr(make_positions, buf_name)[:tensor.size(1)].expand_as(tensor)
    # 4. 如果填充在左侧，需要将位置编号往后平移
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    # 5. 最终用 masked_scatter_ 将非 padding 的原始值替换为位置编号
    new_tensor = tensor.clone()
    return new_tensor.masked_scatter_(mask, positions[mask]).long()


class SinusoidalPositionalEmbedding(nn.Module):
    """
    产生任意长度的正余弦位置嵌入（sinusoidal positional embedding）。
    支持左侧或右侧 padding，padding 对应的索引位置全为 0 向量。
    """

    def __init__(self, embedding_dim, padding_idx=0, left_pad=0, init_size=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = dict()   # device --> actual weight; due to nn.DataParallel :-(
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """
        构造正余弦位置编码矩阵，shape = (num_embeddings, embedding_dim)。
        位置嵌入
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        """
        输入 input: shape = (batch_size, seq_len)，其中元素为 token IDs（其中 padding_idx 被视为填充）。
        输出位置嵌入: shape = (batch_size, seq_len, embedding_dim)。embedding_dim表示每个 token 的位置嵌入向量，而填充位置的嵌入全为 0。
        """
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        device = input.get_device()
        if device not in self.weights or max_pos > self.weights[device].size(0):
            # recompute/expand embeddings if needed
            self.weights[device] = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights[device] = self.weights[device].type_as(self._float_tensor).to(input.device)
        positions = make_positions(input, self.padding_idx, self.left_pad)
        return self.weights[device].index_select(0, positions.contiguous().view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number