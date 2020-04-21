from bonsai_graph.ops import *
from bonsai_graph.pruner import *
  
import torch
import torch.nn as nn
import torch.nn.functional as F


def gat_message(edges):
    if 'norm' in edges.src:
        msg = edges.src['ft'] * edges.src['norm']
        return {'ft': edges.src['ft'], 'a2': edges.src['a2'], 'a1': edges.src['a1'], 'norm': msg}
    return {'ft': edges.src['ft'], 'a2': edges.src['a2'], 'a1': edges.src['a1']}


class NASLayer(nn.Module):
    def __init__(self, attention_type, aggregator_type, act, head_num, in_channels, out_channels=8, concat=True,
                 dropout=0.6, pooling_dim=128, residual=False, batch_normal=True):
        '''
        build one layer of GNN
        :param attention_type:
        :param aggregator_type:
        :param act: Activation function
        :param head_num: head num, in another word repeat time of current ops
        :param in_channels: input dimension
        :param out_channels: output dimension
        :param concat: concat output. get average when concat is False
        :param dropout: dropput for current layer
        :param pooling_dim: hidden layer dimension; set for pooling aggregator
        :param residual: whether current layer has  skip-connection
        :param batch_normal: whether current layer need batch_normal
        '''
        super().__init__()
        self.attention_type = attention_type
        self.aggregator_type = aggregator_type
        self.act = act
        self.name = '{}_{}_{}'.format(aggregator_type,attention_type,act)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = int(head_num)
        self.concat = concat
        self.dropout = dropout
        self.attention_dim = 64 if attention_type in ['cos', 'generalized_linear'] else 1
        self.pooling_dim = pooling_dim
        self.batch_normal = batch_normal

        self.bn = nn.BatchNorm1d(self.in_channels, momentum=0.5)
        self.prp = nn.ModuleList()
        self.red = nn.ModuleList()
        self.fnl = nn.ModuleList()
        self.agg = nn.ModuleList()
        for hid in range(self.num_heads):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.prp.append(AttentionPrepare(in_channels, out_channels, self.attention_dim, dropout))
            agg = aggs[aggregator_type](out_channels, pooling_dim)
            self.agg.append(agg)
            self.red.append(attents[attention_type](dropout, self.attention_dim, agg))
            self.fnl.append(GATFinalize(hid, in_channels, out_channels, acts[act], residual))


    def forward(self, features, g):
        last = self.bn(features) if self.batch_normal else features

        for hid in range(self.num_heads):
            # prepare
            g.ndata.update(self.prp[hid](last))
            # message passing
            g.update_all(gat_message, self.red[hid], self.fnl[hid])
            
        # merge all the heads
        if not self.concat:
            output = g.pop_n_repr('head0')
            for hid in range(1, self.num_heads):
                output = torch.add(output, g.pop_n_repr('head%d' % hid))
            output = output / self.num_heads
        else:
            output = torch.cat([g.pop_n_repr('head%d' % hid) for hid in range(self.num_heads)], dim=1)
        del last
        return output
    
    
class MultiLayer(nn.Module):
    def __init__(self, head_num, in_channels, out_channels=8, layer_ops=None, concat=True, dropout=0.6, pooling_dim=128, residual=False, batch_normal=True, prune=True):
        super().__init__()
        layers = []
        self.out_size = out_channels*head_num if concat else out_channels
        
        if layer_ops is None:
            for act in acts.keys():
                for att in attents.keys():
                    for agg in aggs.keys():
                        op = NASLayer(att, agg, act, head_num, in_channels, out_channels, concat, dropout, pooling_dim, residual, batch_normal)
                        p = Pruner(init=.0025) if prune else None
                        layers.append(nn.ModuleDict({'op':op,'pruner':p}))
        else:
            for op in layer_ops:
                op = NASLayer(op['att'], op['agg'], op['act'], head_num, in_channels, out_channels, concat, dropout, pooling_dim, residual, batch_normal)
                p = Pruner(init=.0025) if prune else None
                layers.append(nn.ModuleDict({'op':op,'pruner':p}))
        self.layers = nn.ModuleList(layers)
        self.prune = prune
        if concat:
            self.bn = nn.BatchNorm1d(head_num*out_channels, momentum=0.5)
        else:
            self.bn = nn.BatchNorm1d(out_channels, momentum=0.5)
        
    def forward(self, features, g):
        out = torch.zeros((features.shape[0], self.out_size), dtype=features.dtype, device=features.device)
        for i, layer in enumerate(self.layers):
            if self.prune:
                if not layer['pruner'].deadheaded:
                    out += layer['pruner'](layer['op'](features, g))
            else:
                 out += layer['op'](features, g)
        return out

