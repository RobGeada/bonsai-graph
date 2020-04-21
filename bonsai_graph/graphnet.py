import torch
import torch.nn as nn
import torch.nn.functional as F

from bonsai_graph.ops import *
from bonsai_graph.layer import *
from bonsai_graph.utils.tensor_utils import cache_stats

class GraphNet(torch.nn.Module):

    def __init__(self, num_feat, num_label, layer_nums, head_num, prune, layers_ops, drop_out=0.6, multi_label=False, batch_normal=True, residual=True, state_num=5):
        super().__init__()
        self.multi_label = multi_label
        self.num_feat = num_feat
        self.num_label = num_label
        self.dropout = drop_out
        self.residual = residual
        
        self.layer_nums = layer_nums
        self.head_num = head_num
        self.prune = prune
        self.layers_ops = layers_ops

        # layer module
        self.build_model(batch_normal, drop_out, num_feat, num_label, state_num)

    def build_model(self, batch_normal, drop_out, num_feat, num_label, state_num):
        self.layers = nn.ModuleList()
        self.build_hidden_layers(batch_normal, drop_out, self.layer_nums, num_feat, num_label, self.prune)

    
    def print_ops(self):
        for i, multilayer in enumerate(self.layers):
            print("{}: {}".format(i, [layer['op'].name for layer in multilayer.layers if not layer['pruner'].deadheaded]))
        

    def build_hidden_layers(self, batch_normal, drop_out, layer_nums, num_feat, num_label, prune):
        # build hidden layer
        head_num = self.head_num
        for i in range(layer_nums):
            # compute input
            in_channels = num_feat if i == 0 else out_channels * head_num   
            concat = i != layer_nums - 1
            out_channels = 3 if not concat else 2
                       
            residual = False and self.residual if i == 0 else True and self.residual 
            self.layers.append(MultiLayer(head_num, 
                                          in_channels,
                                          out_channels, 
                                          self.layers_ops[i], 
                                          dropout=drop_out,
                                          concat=concat, 
                                          residual=residual,
                                          batch_normal=batch_normal, 
                                          prune=prune))

    def track_pruners(self):
        [layer['pruner'].track_gates() for multilayer in self.layers for layer in multilayer.layers]

        
    def deadhead(self):
        deadheads = [layer['pruner'].deadhead() for multilayer in self.layers for layer in multilayer.layers]
        remaining = sum([not layer['pruner'].deadheaded for multilayer in self.layers for layer in multilayer.layers])
        print("Deadheaded {} ops. Remaining: {}".format(sum(deadheads),remaining)) 

            
    def compression(self):
        pruners = [layer['pruner'].sg() for multilayer in self.layers for layer in multilayer.layers]
        return sum(pruners).detach().item()/len(pruners)
            
    def forward(self, feat, g, verbose=False):
        output = feat
        for i, layer in enumerate(self.layers):
            output = layer(output, g)
            if verbose:
                print("{}: {}".format(i,cache_stats()))
        return output
