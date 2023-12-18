import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import softmax_kernel_transformation, create_projection_matrix, kernelized_gumbel_softmax, kernelized_softmax, add_conv_relational_bias

BIG_CONSTANT = 1e8



class GKAMP(nn.Module):
    '''
    Global Kernelized Attention Message Passing
    '''
    def __init__(self, in_channels, out_channels, num_heads, kernel_transformation=softmax_kernel_transformation, projection_matrix_type='a',
                 nb_random_features=10, use_gumbel=True, nb_gumbel_sample=10, rb_order=0, rb_trans='sigmoid', use_edge_loss=False):
        super(GKAMP, self).__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        self.Wv = nn.Linear(in_channels, out_channels * num_heads)
        self.Wo = nn.Linear(out_channels * num_heads, out_channels)
        if rb_order >= 1:
            self.b = torch.nn.Parameter(torch.FloatTensor(rb_order, num_heads), requires_grad=True)
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.kernel_transformation = kernel_transformation
        self.projection_matrix_type = projection_matrix_type
        self.nb_random_features = nb_random_features
        self.use_gumbel = use_gumbel
        self.nb_gumbel_sample = nb_gumbel_sample
        self.rb_order = rb_order
        self.rb_trans = rb_trans
        self.use_edge_loss = use_edge_loss

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        self.Wv.reset_parameters()
        self.Wo.reset_parameters()
        if self.rb_order >= 1:
            if self.rb_trans == 'sigmoid':
                torch.nn.init.constant_(self.b, 0.1)
            elif self.rb_trans == 'identity':
                torch.nn.init.constant_(self.b, 1.0)

    def forward(self, z, feat, adjs, tau):
        B, N = z.size(0), z.size(1)
        query = self.Wq(feat).reshape(-1, N, self.num_heads, self.out_channels)
        key = self.Wk(feat).reshape(-1, N, self.num_heads, self.out_channels)
        value = self.Wv(z).reshape(-1, N, self.num_heads, self.out_channels)
        if self.projection_matrix_type is None:
            projection_matrix = None
        else:
            dim = query.shape[-1]
            seed = torch.ceil(torch.abs(torch.sum(query) * BIG_CONSTANT)).to(torch.int32)
            projection_matrix = create_projection_matrix(
                self.nb_random_features, dim, seed=seed).to(query.device)
        # compute all-pair message passing update and attn weight on input edges, requires O(N) or O(N + E)
        if self.use_gumbel and self.training:  # only using Gumbel noise for training
            z_next = kernelized_gumbel_softmax(query,key,value,self.kernel_transformation,projection_matrix,adjs[0],
                                                  self.nb_gumbel_sample, tau, self.use_edge_loss)
        else:
            z_next = kernelized_softmax(query, key, value, self.kernel_transformation, projection_matrix, adjs[0],
                                                tau, self.use_edge_loss)
        for i in range(self.rb_order):
            z_next += add_conv_relational_bias(value, adjs[i], self.b[i], self.rb_trans)

        z_next = self.Wo(z_next.flatten(-2, -1))

        return z_next

class MultiInfSurrogate(nn.Module):
    def __init__(self, n, item_num, hidden_channels, out_channels, num_layers=2, num_heads=4, dropout=0.0,
                 kernel_transformation=softmax_kernel_transformation, nb_random_features=30, use_bn=True, use_gumbel=True,
                 use_residual=True, use_act=False, use_bias=True, nb_gumbel_sample=10, rb_order=0, rb_trans='sigmoid'):
        super(MultiInfSurrogate, self).__init__()
        self.item_num = item_num
        self.hidden_channels = hidden_channels
        self.mp_layers = nn.ModuleList()
        self.infemb = nn.Embedding(item_num, hidden_channels)
        self.infemb.weight.data[0].fill_(0)
        self.fc2 = nn.Linear(hidden_channels, out_channels) 
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        self.dropout = dropout
        for i in range(num_layers):
            self.mp_layers.append(
                GKAMP(hidden_channels, hidden_channels, num_heads=num_heads, kernel_transformation=kernel_transformation,
                              nb_random_features=nb_random_features, use_gumbel=use_gumbel, nb_gumbel_sample=nb_gumbel_sample,
                               rb_order=rb_order, rb_trans=rb_trans))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.node_feats = nn.Embedding(n, hidden_channels)
        self.prefer_bias = nn.Parameter(torch.zeros(n, hidden_channels))
        self.act_fn = nn.ReLU()
        self.use_bn = use_bn
        self.use_bias = use_bias
        self.use_residual = use_residual
        self.use_act = use_act
        self.loss = nn.L1Loss()
    
    def reset_parameters(self):
        for mp in self.mps:
            mp.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.infemb.reset_parameters()
        self.infemb.weight.data[0].fill_(0)
        self.fc2.reset_parameters()

    def forward(self, inf_state, adj, tau=1.0):
        # inf_state [B, N]
        B = inf_state.shape[0]
        z = self.infemb(inf_state) # [B, N, D]
        layer_ = []
        if self.use_bn:
            z = self.bns[0](z)
        z = self.act_fn(z)
        z = F.dropout(z, p=self.dropout, training=self.training)
        layer_.append(z)
        for i, mp in enumerate(self.mp_layers):
            z = mp(z, self.node_feats.weight, adj, tau)
            if self.use_bias:
                z = z + self.prefer_bias 
            if self.use_residual:
                z += layer_[i]
            if self.use_bn:
                z = self.bns[i+1](z)
            if self.use_act:
                z = self.act_fn(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
            layer_.append(z)
        z = self.fc2(z) 
        z = self.act_fn(z)
        z = z.view(B,-1) #[B, N*D]

        return torch.sum(z,1)

    def lastlayer(self, inf_state, adj, tau=1.0):
        # inf_state [B, N]
        B = inf_state.shape[0]
        z = self.infemb(inf_state) # [B, N, D]
        layer_ = []
        if self.use_bn:
            z = self.bns[0](z)
        z = self.act_fn(z)
        z = F.dropout(z, p=self.dropout, training=self.training)
        layer_.append(z)
        for i, mp in enumerate(self.mp_layers):
            z = mp(z, self.node_feats.weight, adj, tau)
            if self.use_bias:
                z = z + self.prefer_bias 
            if self.use_residual:
                z += layer_[i]
            if self.use_bn:
                z = self.bns[i+1](z)
            if self.use_act:
                z = self.act_fn(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
            layer_.append(z)
        z = self.fc2(z) 
        z = self.act_fn(z) #[B, N, M]
        z = torch.sum(z,-1) #[B, N]
        return z
