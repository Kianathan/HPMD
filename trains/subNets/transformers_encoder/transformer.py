import math
import torch
import torch.nn.functional as F
from torch import nn
from .multihead_attention import MultiheadAttention
from .position_embedding import SinusoidalPositionalEmbedding
#每一个Encoder的输入是下一层Encoder输出，最底层Encoder的输入是原始的输入(法语句子)；Decoder也是类似，但是最后一层Encoder的输出会输入给每一个Decoder层，这是Attention机制的要求。
#每一层的Encoder都是相同的结构，它由一个Self-Attention层和一个前馈网络(全连接网络)组成
#每一层的Decoder也是相同的结构，它除了Self-Attention层和全连接层之外还多了一个Attention层，这个Attention层使得Decoder在解码时会考虑最后一层Encoder所有时刻的输出。
class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """
    # Transformer 编码器由 *args.encoder_layers* 层组成。每一层都是一个 :class:`TransformerEncoderLayer`。
    # 参数：
    # embed_tokens (torch.nn.Embedding)：输入嵌入
    # num_heads (int)：heads 数量
    # layers (int)：层数
    # attn_dropout (float)：应用于注意权重的 dropout
    # relu_dropout (float)：应用于残差块第一层的 dropout
    # res_dropout (float)：应用于残差块的 dropout
    # attn_mask (bool)：是否对注意权重应用 mask

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False):
        super().__init__()
        self.dropout = embed_dropout      # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim) #embed_dim=256，embed_scale=16
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.attn_mask = attn_mask

        # （2）堆叠多个 EncoderLayer
        self.layers = nn.ModuleList([])
        for layer in range(layers):     #layers=4
            new_layer = TransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)

        # （3）最后的 LayerNorm（若 normalize=True）
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    #x_in_k = None, x_in_v = None
    def forward(self, x_in, x_in_k = None, x_in_v = None):
        """
        Args:
            x_in (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_v (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # 参数：
        # x_in (FloatTensor)：形状为 `(src_len, batch, embed_dim)` 的嵌入输入
        # x_in_k (FloatTensor)：形状为 `(src_len, batch, embed_dim)` 的嵌入输入
        # x_in_v (FloatTensor)：形状为 `(src_len, batch, embed_dim)` 的嵌入输入
        # 返回：
        # dict：
        # - **encoder_out** (Tensor)：最后一个编码器层的输出，形状为 `(src_len, batch, embed_dim)`
        # - **encoder_padding_mask** (ByteTensor)：填充元素的位置，形状为 `(batch, src_len)`
        
        # embed tokens and positions
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            # 词嵌入的第 0 维已经能区分不同位置，足够用来生成位置编码
            #相加后，x 同时包含词嵌入和位置信息
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            # embed tokens and positions    
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.embed_positions is not None:
                x_k += self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
                x_v += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)
        
        # encoder layers，intermediates保存每一个层的输出结果
        intermediates = [x]
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)
            intermediates.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        return x

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.
    Args:
        embed_dim: Embedding dimension
    """

    # 编码器层模块。
    # 在原始论文中，每个操作（多头注意力机制或 FFN）都使用以下方式进行后处理：`dropout -> add residual -> layernorm`。在 tensor2tensor 代码中，他们建议，当使用 layernorm 预处理每个层，并使用以下方式进行后处理时，学习效果会更稳健：
    # `dropout -> add residual`。我们默认使用论文中的方法，但可以通过将
    # *args.encoder_normalize_before* 设置为 ``True`` 来启用 tensor2tensor 方法。
    # 参数：
    # embed_dim：嵌入维度

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)   # The "Add & Norm" part in the paper
        self.fc2 = Linear(4*self.embed_dim, self.embed_dim)
        # 构造两个归一化层，每个归一化会对每个样本的所有 embed_dim 个通道独立做均值-方差归一化，并且学习可训练的缩放（γ）和平移（β）参数。这两个分别用于自注意力前后归一化
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            x_k (Tensor): same as x
            x_v (Tensor): same as x
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        
        # 参数：
        # x (Tensor)：输入层，形状为 `(seq_len, batch, embed_dim)`
        # encoder_padding_mask (ByteTensor)：二进制 ByteTensor，形状为 `(batch, src_len)`，其中填充元素用 ``1`` 表示。
        # x_k (Tensor)：与 x 相同
        # x_v (Tensor)：与 x 相同
        # 返回：
        # 编码后的输出，形状为 `(batch, src_len, embed_dim)

        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None #此处为None
        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True) 
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        #Dropout + 残差连接 + LayerNorm（后归一化）
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        #前馈网络 FFN + 残差连接 + LayerNorm
        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    #控制 LayerNorm 位置（before 或 after）
    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after   
        if after ^ self.normalize_before:   #互斥表达式，仅仅有一个为真才执行
            return self.layer_norms[i](x)
        else:
            return x

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)

#这个函数 buffered_future_mask 的作用是在自回归（causal）注意力机制中为 query—key 序列生成一个“未来时刻”屏蔽矩阵（future mask），以确保每个位置只能看到它自己及之前的位置，屏蔽掉之后的时刻。
def buffered_future_mask(tensor, tensor2=None):
    #获取序列长度
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    #将构造的全 1 张量中所有位置填成负无穷大（-inf），以便后续与注意力分数相加时完全屏蔽（softmax 后权重为 0）。
    #再取上三角部分，其中 k = 1 + abs(dim2-dim1) 控制从哪一条对角线开始：
    #   对于 dim1 == dim2（最常见的自注意力场景），abs(dim2-dim1)=0，所以 k=1，表示从主对角线的正上方第一条开始全部保留（即把对角线以上全部置 -inf），从而屏蔽“未来”位置。
    #   如果 dim1 != dim2，k 也能正确偏移，保证 query 中的每个位置都只能看到最靠近它的 key 及之前的位置。
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if tensor.is_cuda:
        future_mask = future_mask.to(tensor.device)
    #返回一个形状恰好为 (dim1, dim2) 的 mask 矩阵。
    return future_mask[:dim1, :dim2]


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:    #偏置初始化为0
        nn.init.constant_(m.bias, 0.)
    return m

#归一化函数
def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


if __name__ == '__main__':
    encoder = TransformerEncoder(300, 4, 2)
    x = torch.tensor(torch.rand(20, 2, 300))
    print(encoder(x).shape)
