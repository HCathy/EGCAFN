import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

# 增加数据扰动
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(GCNLayer, self).__init__()
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))

    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(2)
        batch, l = D.shape
        D1 = torch.reshape(D, (batch * l, 1))
        D1 = D1.squeeze(1)
        D2 = torch.pow(D1, -0.5)
        D2 = torch.reshape(D2, (batch, l))
        D_hat = torch.zeros([batch, l, l], dtype=torch.float)
        for i in range(batch):
            D_hat[i] = torch.diag(D2[i])
        return D_hat.cuda()

    def forward(self, H, A):
        nodes_count = A.shape[1]
        I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)

        (batch, l, c) = H.shape
        H1 = torch.reshape(H, (batch * l, c))
        H2 = self.BN(H1)
        H = torch.reshape(H2, (batch, l, c))

        D_hat = self.A_to_D_inv(A)
        A_hat = torch.matmul(D_hat, torch.matmul(A, D_hat))
        A_hat = I + A_hat
        output = torch.matmul(A_hat, self.GCN_liner_out_1(H))
        output = self.Activition(output)
        return output

class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, embed_dim: int = 128, num_heads: int = 8, dropout: float = 0.6):
        super(GraphAttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.linear = nn.Linear(128, embed_dim)  # 确保输入维度正确
        self.output_linear = nn.Linear(embed_dim, output_dim)  # 确保输出维度正确
        self.dropout_layer = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, H, A):
        batch_size, nodes_count, feature_dim = H.size()
        H = self.linear(H)
        attention_weights = self.compute_attention_weights(H, A)
        H = H.permute(1, 0, 2)
        attention_weights = attention_weights.unsqueeze(1)
        attention_weights = attention_weights.expand(-1, self.num_heads, -1,
                                                     -1)
        attention_weights = attention_weights.reshape(batch_size * self.num_heads, nodes_count,
                                                      nodes_count)

        H, _ = self.attention(H, H, H, attn_mask=attention_weights)
        H = H.permute(1, 0, 2)
        H = self.output_linear(H)
        H = self.dropout_layer(H)

        return H

    def compute_attention_weights(self, H, A):
        attention_scores = torch.bmm(H, H.transpose(1, 2))
        attention_scores = self.leaky_relu(attention_scores)
        attention_weights = attention_scores * A
        attention_weights = F.softmax(attention_weights, dim=-1)
        return attention_weights

# class GCN_Layer(nn.Module):
#     def __init__(self, input_dim: int, gcn_out_dim: int, gat_out_dim: int, dropout: float = 0.6):
#         super(GCN_Layer, self).__init__()
#         # 定义 GCN 部分
#         self.gcn = GCNLayer(input_dim, gcn_out_dim)
#         # 新增
#         self.residual = nn.Linear(input_dim, gcn_out_dim)
#
#     def forward(self, H, A):
#         H_gcn = self.gcn(H, A)
#         return H_gcn
#
# class GAT_Layer(nn.Module):
#     def __init__(self, input_dim: int, gcn_out_dim: int,  dropout: float = 0.6):
#         super(GAT_Layer, self).__init__()
#         # 定义 GCN 部分
#         self.gat = GraphAttentionLayer(input_dim, gcn_out_dim)
#
#         self.residual = nn.Linear(128, gcn_out_dim)
#     def forward(self, H, A):
#         # residual = self.residual(H)
#         H_gat = self.gat(H, A)
#         return H_gat

# 新eca
class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        # k_size = 5
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class Spatial_attention(nn.Module):
    def __init__(self, in_channels, kernel_size=3, out_channels=1, stride=1, padding=1):
        super(Spatial_attention, self).__init__()
        self.conv1 = nn.Conv2d(2, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0)  # 多尺度信息
        self.bn = nn.BatchNorm2d(out_channels)  # 添加归一化层
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat((avg_out, max_out), 1)
        y = self.conv1(y)
        y = self.conv2(y)  # 多尺度信息
        y = self.act(y)
        return x * y.expand_as(x)


class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.conv7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3)
        self.bn = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        x = torch.cat([x1, x3, x5, x7], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True,
                 num_layers=4):
        super(DepthwiseSeparableBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(
                    DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias))
            else:
                layers.append(
                    DepthwiseSeparableConv2d(out_channels, out_channels, kernel_size, 1, padding, dilation, bias))

        self.block = nn.Sequential(*layers)
        self.use_residual = (in_channels == out_channels and stride == 1)
        if not self.use_residual:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=bias),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.block(x)
        if self.use_residual:
            x += identity
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()  # 确保不使用 inplace=True
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class GSSA(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.,
            group_spatial_size=3  # 调整为 3
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.group_spatial_size = group_spatial_size
        inner_dim = dim_head * heads

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias=False)

        self.group_tokens = nn.Parameter(torch.randn(dim))

        self.group_tokens_to_qk = nn.Sequential(
            nn.LayerNorm(dim_head),
            nn.GELU(),
            Rearrange('b h n c -> b (h c) n'),
            nn.Conv1d(inner_dim, inner_dim * 2, 1),
            Rearrange('b (h c) n -> b h n c', h=heads),
        )

        self.group_attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        batch, height, width, heads, gss = x.shape[0], *x.shape[-2:], self.heads, self.group_spatial_size
        # print(height)
        # print(width)
        # print(heads)
        assert (height % gss) == 0 and (
                width % gss) == 0, f'height {height} and width {width} must be divisible by group spatial size {gss}'
        num_groups = (height // gss) * (width // gss)
        x = rearrange(x, 'b c (h g1) (w g2) -> (b h w) c (g1 g2)', g1=gss, g2=gss)
        w = repeat(self.group_tokens, 'c -> b c 1', b=x.shape[0])
        x = torch.cat((w, x), dim=-1)
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h=heads), (q, k, v))
        q = q * self.scale
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        group_tokens, grouped_fmaps = out[:, :, 0], out[:, :, 1:]

        if num_groups == 1:
            fmap = rearrange(grouped_fmaps, '(b x y) h (g1 g2) d -> b (h d) (x g1) (y g2)', x=height // gss,
                             y=width // gss, g=gss, g2=gss)
            return self.to_out(fmap)
        group_tokens = rearrange(group_tokens, '(b x y) h d -> b h (x y) d', x=height // gss, y=width // gss)
        grouped_fmaps = rearrange(grouped_fmaps, '(b x y) h n d -> b h (x y) n d', x=height // gss, y=width // gss)
        w_q, w_k = self.group_tokens_to_qk(group_tokens).chunk(2, dim=-1)
        w_q = w_q * self.scale
        w_dots = torch.einsum('b h i d, b h j d -> b h i j', w_q, w_k)
        w_attn = self.group_attend(w_dots)
        aggregated_grouped_fmap = torch.einsum('b h i j, b h j w d -> b h i w d', w_attn, grouped_fmaps)
        fmap = rearrange(aggregated_grouped_fmap, 'b h (x y) (g1 g2) d -> b (h d) (x g1) (y g2)', x=height // gss,
                         y=width // gss, g1=gss, g2=gss)
        return self.to_out(fmap)


class EGCAFN(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, dim, hidden_dim=64, num_heads=8):
        super(EGCAFN, self).__init__()
        self.class_count = class_count
        self.channel = changel
        self.height = height
        self.width = width
        self.dim = dim
        layers_count = 6
        layers_count_1 = 8
        self.GCN_Branch = nn.Sequential()
        self.GAT_Branch = nn.Sequential()
        # Define GCN and GAT branches
        self.GCN_Branch = nn.Sequential(
            *[GCNLayer(self.channel if i == 0 else 128, 128) for i in range(6)]
        )

        self.GAT_Branch = nn.Sequential(
            *[GraphAttentionLayer(self.channel if i == 0 else 128, 128, num_heads=num_heads, dropout=0.6) for i in
              range(6)]
        )
        self.ca = eca_layer(64)
        self.sa = Spatial_attention(64, 3, 64, 1, 1)
        self.fc = nn.Linear(192, 64)
        self.multi_scale_fusion = MultiScaleFusion(41, 64)  # 确保输入和输出通道数一致

        # 添加深度可分离卷积
        self.depthwise_separable_conv = DepthwiseSeparableBlock(128, 128, 3, stride=2, padding=1, num_layers=3)
        self.residual_block = ResidualBlock(128, 64)
        self.gssa = GSSA(dim=64, heads=num_heads, dim_head=16, dropout=0.1, group_spatial_size=3)
        self.fc = nn.Linear(192, 64)

        self.to_latent = nn.Identity()
        self.BN = nn.BatchNorm1d(192)
        self.mlp_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.Linear(64, self.class_count)
        )


    def forward(self, x: torch.Tensor, A: torch.Tensor, indexs_train):

        (batch, h, w, c) = x.shape
        H = torch.reshape(x, (batch, h * w, c))
        residual = H.clone()

        for layer in self.GCN_Branch:
            H = layer(H, A)

        if residual.shape[-1] != H.shape[-1]:
            linear_layer = nn.Linear(residual.shape[-1], H.shape[-1]).to(H.device)
            residual = linear_layer(residual)
        H = H + residual

        for layer in self.GAT_Branch:
            H = layer(H, A)

        H = H.permute(0, 2, 1).unsqueeze(-1)
        H = self.depthwise_separable_conv(H)
        H = self.residual_block(H)
        H = H.squeeze(-1).permute(0, 2, 1)

        H = self.multi_scale_fusion(H.unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)
        H = H.unsqueeze(-1)

        pad_height = (self.gssa.group_spatial_size - (
                    H.shape[1] % self.gssa.group_spatial_size)) % self.gssa.group_spatial_size
        H = F.pad(H, (0, 0, 0, pad_height))
        pad_width = (self.gssa.group_spatial_size - (
                    H.shape[1] % self.gssa.group_spatial_size)) % self.gssa.group_spatial_size
        if pad_width > 0:
            H = F.pad(H, (0, pad_width, 0, 0))

        H = self.gssa(H)
        H = H[:, :, :, 0].permute(0, 2, 1)

        H = self.ca(H) * self.sa(H.unsqueeze(2)).squeeze(2)
        H = H + H

        tr_in = H
        x = self.to_latent(tr_in[:, 0])
        x = self.mlp_head(x)

        return x
