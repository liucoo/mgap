from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch.nn import Parameter
import torch

import torch.nn as nn
import numpy as np

from typing import Union, Optional, Callable
from torch_scatter import scatter_add, scatter_max
from torch_geometric.utils import softmax

import math


import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, ChebConv, GraphConv


def uniform(size, tensor):
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))


def topk(x, ratio, batch, min_score=None, tol=1e-7):
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter_max(x, batch)[0][batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero(as_tuple=False).view(-1)
    else:
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes, ),
                             torch.finfo(x.dtype).min)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        if isinstance(ratio, int):
            k = num_nodes.new_full((num_nodes.size(0), ), ratio)
            k = torch.min(k, num_nodes)
        else:
            k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)

        mask = [
            torch.arange(k[i], dtype=torch.long, device=x.device) +
            i * max_num_nodes for i in range(batch_size)
        ]
        mask = torch.cat(mask, dim=0)

        perm = perm[mask]

    return perm


def filter_adj(edge_index, edge_attr, perm, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr



class TopKPooling(torch.nn.Module):
    r""":math:`\mathrm{top}_k` pooling operator from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_, `"Towards Sparse
    Hierarchical Graph Classifiers" <https://arxiv.org/abs/1811.01287>`_
    and `"Understanding Attention and Generalization in Graph Neural
    Networks" <https://arxiv.org/abs/1905.02850>`_ papers
    if min_score :math:`\tilde{\alpha}` is None:
        .. math::
            \mathbf{y} &= \frac{\mathbf{X}\mathbf{p}}{\| \mathbf{p} \|}
            \mathbf{i} &= \mathrm{top}_k(\mathbf{y})
            \mathbf{X}^{\prime} &= (\mathbf{X} \odot
            \mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}
            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}
    if min_score :math:`\tilde{\alpha}` is a value in [0, 1]:
        .. math::
            \mathbf{y} &= \mathrm{softmax}(\mathbf{X}\mathbf{p})
            \mathbf{i} &= \mathbf{y}_i > \tilde{\alpha}
            \mathbf{X}^{\prime} &= (\mathbf{X} \odot \mathbf{y})_{\mathbf{i}}
            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}},
    where nodes are dropped based on a learnable projection score
    :math:`\mathbf{p}`.
    Args:
        in_channels (int): Size of each input sample.
        ratio (float or int): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`, or the value
            of :math:`k` itself, depending on whether the type of :obj:`ratio`
            is :obj:`float` or :obj:`int`.
            This value is ignored if :obj:`min_score` is not :obj:`None`.
            (default: :obj:`0.5`)
        min_score (float, optional): Minimal node score :math:`\tilde{\alpha}`
            which is used to compute indices of pooled nodes
            :math:`\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}`.
            When this value is not :obj:`None`, the :obj:`ratio` argument is
            ignored. (default: :obj:`None`)
        multiplier (float, optional): Coefficient by which features gets
            multiplied after pooling. This can be useful for large graphs and
            when :obj:`min_score` is used. (default: :obj:`1`)
        nonlinearity (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.tanh`)
    """
    def __init__(self, in_channels: int, ratio: Union[int, float] = 0.5,
                 min_score: Optional[float] = None, multiplier: float = 1.,
                 nonlinearity: Callable = torch.tanh,
                 cus_drop_ratio = 0 ):
        super(TopKPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.weight = Parameter(torch.Tensor(1, in_channels))

        self.dropout = torch.nn.Dropout(cus_drop_ratio)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.weight)

    def forward(self, x, edge_index, edge_attr=None, batch=None, attn=None):
        """"""

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = (attn * self.weight).sum(dim=-1)

        if self.min_score is None:
            score = self.nonlinearity(score / self.weight.norm(p=2, dim=-1))
        else:
            score = softmax(score, batch)

        sc = self.dropout(score)

        perm = topk(sc, self.ratio, batch, self.min_score)
        x_ae = x[perm]
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, x_ae

    def __repr__(self):
        return '{}({}, {}={}, multiplier={})'.format(
            self.__class__.__name__, self.in_channels,
            'ratio' if self.min_score is None else 'min_score',
            self.ratio if self.min_score is None else self.min_score,
            self.multiplier)


class SAGPool(torch.nn.Module):
    def __init__(self,in_channels,
                 ratio=0.8,
                 Conv=GCNConv,
                 non_linearity=torch.tanh,
                 cus_drop_ratio= 0):

        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
        self.dropout = torch.nn.Dropout(cus_drop_ratio)


    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.score_layer(x,edge_index).squeeze(-1)

        sc = self.dropout(score)

        perm = topk(sc, self.ratio, batch)
        x_ae = x[perm]
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, x_ae



class GSAPool(torch.nn.Module):

    def __init__(self, in_channels, pooling_ratio=0.5, alpha=0.6, pooling_conv="GCNConv", fusion_conv="True",
                        min_score=None, multiplier=1,
                        non_linearity=torch.tanh,
                        cus_drop_ratio =0):
        super(GSAPool,self).__init__()
        self.in_channels = in_channels

        self.ratio = pooling_ratio
        self.alpha = alpha

        self.sbtl_layer = self.conv_selection(pooling_conv, in_channels)
        self.fbtl_layer = nn.Linear(in_channels, 1)
        self.fusion = self.conv_selection(fusion_conv, in_channels, conv_type=1)

        self.min_score = min_score
        self.multiplier = multiplier
        self.fusion_flag = 0
        if(fusion_conv!="false"):
            self.fusion_flag = 1
        self.non_linearity = non_linearity

        self.dropout = torch.nn.Dropout(cus_drop_ratio)

    def conv_selection(self, conv, in_channels, conv_type=0):
        if(conv_type == 0):
            out_channels = 1
        elif(conv_type == 1):
            out_channels = in_channels
        if(conv == "GCNConv"):
            return GCNConv(in_channels,out_channels)
        elif(conv == "ChebConv"):
            return ChebConv(in_channels,out_channels,1)
        elif(conv == "SAGEConv"):
            return SAGEConv(in_channels,out_channels)
        elif(conv == "GATConv"):
            return GATConv(in_channels,out_channels, heads=1, concat=True)
        elif(conv == "GraphConv"):
            return GraphConv(in_channels,out_channels)
        else:
            raise ValueError

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        #SBTL
        score_s = self.sbtl_layer(x,edge_index).squeeze()
        #FBTL
        score_f = self.fbtl_layer(x).squeeze()
        #hyperparametr alpha
        score = score_s*self.alpha + score_f*(1-self.alpha)

        score = score.unsqueeze(-1) if score.dim()==0 else score

        if self.min_score is None:
            score = self.non_linearity(score)
        else:
            score = softmax(score, batch)

        sc = self.dropout(score)
        perm = topk(sc, self.ratio, batch)

        #fusion
        if(self.fusion_flag == 1):
            x = self.fusion(x, edge_index)
        x_ae = x[perm]
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, x_ae



