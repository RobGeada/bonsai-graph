import os
import time
from collections import namedtuple


import numpy as np
import pickle as pkl
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import load_data

from bonsai_graph.graphnet import GraphNet


def load(kwargs, save_file=".pkl"):
    kwarg_nt = namedtuple('kwarg', kwargs.keys())(*kwargs.values())

    save_file = 'data/' + kwarg_nt.dataset + save_file 
    if os.path.exists(save_file):
        with open(save_file, "rb") as f:
            return pkl.load(f)
    else:
        datas = load_data(kwarg_nt)
        with open(save_file, "wb") as f:
            pkl.dump(datas, f)
        return datas


def evaluate(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()


def compression_loss(model, comp_lambda):
    nums, dens = [], []
    w = model.layers[0].layers[0]['pruner'].weight
    zero = torch.tensor([0.], device=w.device)
    for multilayer in model.layers:
        pruners = [layer['pruner'].sg() for layer in multilayer.layers]
        dens += [len(pruners)]
        nums += [sum(pruners)]
    nums = torch.cat(nums)
    dens = torch.tensor(dens, device=w.device, dtype=w.dtype)
    comp_ratio = torch.div(nums, dens)
    comp = torch.norm(1/dens - comp_ratio)
    loss = comp_lambda*comp
    return loss


# manager the train process of GNN on citation dataset
class CitationGNNManager(object):
    def __init__(self, kwargs):

        self.kwargs = kwargs

        if  kwargs['dataset'] in ["cora", "citeseer", "pubmed"]:
            self.data = load(kwargs)
            self.kwargs['in_feats'] = self.in_feats = self.data.features.shape[1]
            self.kwargs['num_class'] = self.n_classes = self.data.num_labels
            print(self.n_classes)

        self.kwargs = kwargs
        self.drop_out = kwargs.get('in_drop',.6)
        self.multi_label = kwargs.get('multi_label', False)
        self.lr = kwargs.get('lr', .005)
        self.weight_decay = kwargs.get('weight_decay', 5e-4)
        self.loss_fn = torch.nn.BCELoss()
        
        self.epochs = kwargs.get('epochs', 300)
        self.train_graph_index = 0
        self.train_set_length = 10
        self.loss_fn = torch.nn.functional.nll_loss
        self.num_layers = self.kwargs['num_layers']
        self.head_num = self.kwargs['head_num']
        self.prune = self.kwargs['prune']
        self.layers_ops = self.kwargs.get('layers_ops',[None for _ in range(self.num_layers)])
        self.cuda = True
        
        self.build_model()

    def build_model(self):
        self.model = GraphNet(self.in_feats, self.n_classes, self.num_layers, self.head_num, self.prune, self.layers_ops, drop_out=self.drop_out, multi_label=False, batch_normal=False)
        if self.cuda:
            self.model.cuda()
        
    # train from scratch
    def train(self):
        try:
            # use optimizer
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            model, val_acc, perf = self.run_model(self.model, optimizer, self.loss_fn, self.data, self.epochs, cuda=self.cuda, show_info=True, return_best=True)
            return perf
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
                raise e
                val_acc = 0
            else:
                raise e


    def run_model(self, model, optimizer, loss_fn, data, epochs, early_stop=5,
                 return_best=False, cuda=True, show_info=False):

        dur = []
        begin_time = time.time()
        best_performance = 0
        min_val_loss = float("inf")
        min_train_loss = float("inf")
        model_val_acc = 0
        features, g, labels, mask, val_mask, test_mask, n_edges = CitationGNNManager.prepare_data(data, cuda)
        mask = mask.to(torch.bool)
        val_mask = val_mask.to(torch.bool)
        test_mask = test_mask.to(torch.bool)
        annealer = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=self.lr/3)

        for epoch in range(1, epochs):
            if epoch%8==0 and self.prune:
                model.deadhead()
                model.print_ops()
            model.train()
            t0 = time.time()
            # forward
            logits = model(features, g, verbose=epoch==1)
            logits = F.log_softmax(logits, 1)
            loss = loss_fn(logits[mask], labels[mask])
            if self.prune:
                model.track_pruners()
                comp_loss = compression_loss(model, .1)
                loss += comp_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #annealer.step()
            train_loss = loss.item()
            del logits

            # evaluate
            model.eval()
            logits = model(features, g)
            logits = F.log_softmax(logits, 1)
            train_acc = evaluate(logits, labels, mask)
            dur.append(time.time() - t0)

            val_loss = float(loss_fn(logits[val_mask], labels[val_mask]))
            val_acc = evaluate(logits, labels, val_mask)
            test_acc = evaluate(logits, labels, test_mask)
            del logits

            if val_loss < min_val_loss:  # and train_loss < min_train_loss
                min_val_loss = val_loss
                min_train_loss = train_loss
                model_val_acc = val_acc
                if test_acc > best_performance:
                    best_performance = test_acc
            if show_info:
                if self.prune:
                    print(
                        "Epoch {:05d} | Loss (L: {:.4f}, C: {:.4f}) | Time(s) {:.4f} | comp {:.2f} | acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}".format(
                            epoch, loss.item(), comp_loss.item(), np.mean(dur), model.compression(), train_acc, val_acc, test_acc))
                else:
                        print(
                        "Epoch {:05d} | Loss (L: {:.4f}) | Time(s) {:.4f} | comp n/a | acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}".format(
                            epoch, loss.item(), np.mean(dur), train_acc, val_acc, test_acc))

        print(f"val_score:{model_val_acc},test_score:{best_performance}")
        if return_best:
            return model, model_val_acc, best_performance
        else:
            return model, model_val_acc

    def update_ops(self):
        layer_preserve = [[layer['op'].name for layer in multilayer.layers if not layer['pruner'].deadheaded] for multilayer in self.model.layers]
        layers_ops = []
        for layer in layer_preserve:
            layer_ops = []
            new_aggs, new_atts, new_acts = [],[],[]
            for op in layer:
                if 'generalized_linear' in op:
                    op=op.replace("generalized_linear",'generalizedlinear')
                if 'gat_sym' in op:
                    op=op.replace("gat_sym",'gatsym')
                agg, att, act = op.split("_")
                layer_ops.append({'att':att,'agg':agg,'act':act})
            layers_ops.append(layer_ops)
        self.layers_ops = layers_ops
        
        
    @staticmethod
    def prepare_data(data, cuda=True):
        features = torch.FloatTensor(data.features)
        labels = torch.LongTensor(data.labels)
        mask = torch.ByteTensor(data.train_mask)
        test_mask = torch.ByteTensor(data.test_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        n_edges = data.graph.number_of_edges()
        # create DGL graph
        g = DGLGraph(data.graph)
        # add self loop
        g.add_edges(g.nodes(), g.nodes())
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0

        if cuda:
            features = features.cuda()
            labels = labels.cuda()
            norm = norm.cuda()
        g.ndata['norm'] = norm.unsqueeze(1)
        return features, g, labels, mask, val_mask, test_mask, n_edges