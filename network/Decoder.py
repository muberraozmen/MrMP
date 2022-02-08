from network.Modules import *

__all__ = ['Decoder']


class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn1 = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, dec_enc_attn_mask=None):
        dec_output = self.enc_attn(dec_input, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn1(dec_output)

        return dec_output


class Decoder(nn.Module):

    def __init__(self, opt):

        super().__init__()

        self.layer_stack = nn.ModuleList([
            DecoderLayer(opt.d_model, opt.d_inner, opt.n_head, opt.d_k, opt.d_v, dropout=opt.dropout)
            for _ in range(opt.n_layers_dec)])

    def forward(self, dec_input, enc_outputs, mask):
        dec_output = dec_input
        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output, enc_outputs[-1], dec_enc_attn_mask=mask)
        return dec_output



