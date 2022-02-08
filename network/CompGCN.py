from network.Modules import *

__all__ = ['CompGCN']


class CompGCN(nn.Module):

    def __init__(self, adjs, d_model, d_inner, phi_mode='mul', dropout=0.1, device=None):

        super(CompGCN, self).__init__()

        self.num_relns = len(adjs)
        self.phi_mode = phi_mode
        self.device = device
        if self.device is None:
            self.device = adjs[0].device

        self.w_in = XavierLinear(d_model, d_model)
        self.w_out = XavierLinear(d_model, d_model)
        self.w_loop = XavierLinear(d_model, d_model)
        self.w_rel = XavierLinear(d_model, d_model)
        # self.bn = torch.nn.BatchNorm1d(d_model)
        self.pos_ffn_l = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_r = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

        adjs_in = []
        adjs_out = []
        adjs_loop = []
        for adj in adjs:
            adj = self.sym_transform(adj).unsqueeze(0)
            A_in = torch.triu(adj)
            A_out = torch.tril(adj)
            A_loop = torch.eye(A_in.shape[-1], A_in.shape[-1]).unsqueeze(0).type(torch.float).to(self.device)
            A_loop = A_loop/A_loop.sum()
            adjs_in.append(A_in)
            adjs_out.append(A_out)
            adjs_loop.append(A_loop)

        self.adjs_in = adjs_in
        self.adjs_out = adjs_out
        self.adjs_loop = adjs_loop

    def sym_transform(self, A):
        D_inv = torch.diag(A.sum(dim=1).pow(-0.5))
        D_inv[D_inv == float('inf')] = 0
        A = torch.matmul(torch.matmul(D_inv, A), D_inv)
        A = A/A.sum()
        return A

    def phi(self, label_emb, reln_emb, mode):
        if mode == 'mul':
            return (label_emb * reln_emb)
        elif mode == 'sub':
            return (label_emb - reln_emb).type(torch.float)

    def forward(self, labels, relns):

        out_labels = 0
        for r in range(self.num_relns):

            reln_in = torch.index_select(relns, 0, torch.tensor([r]).to(self.device))
            input_in = self.phi(labels, reln_in, self.phi_mode)
            output_in = torch.matmul(self.adjs_in[r], self.w_in(input_in))

            reln_out = torch.index_select(relns, 0, torch.tensor([self.num_relns + r]).to(self.device))
            input_out = self.phi(labels, reln_out, self.phi_mode)
            output_out = torch.matmul(self.adjs_out[r], self.w_out(input_out))

            reln_loop = torch.index_select(relns, 0, torch.tensor([2 * self.num_relns + r]).to(self.device))
            input_loop = self.phi(labels, reln_loop, self.phi_mode)
            output_loop = torch.matmul(self.adjs_loop[r], self.w_loop(input_loop))

            out_labels += output_in * (1 / 3) + output_out * (1 / 3) + output_loop * (1 / 3)

#        out_labels = self.bn(out_labels.squeeze()).unsqueeze(0)
        out_labels = out_labels/self.num_relns
        out_labels = self.pos_ffn_l(out_labels)
        out_relns = self.pos_ffn_r(relns)
        # out_relns = torch.nn.functional.relu(self.w_rel(relns))

        return out_labels, out_relns
