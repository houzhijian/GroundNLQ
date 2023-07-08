import torch
from torch import nn
from torch.nn import functional as F

from .models import register_backbone
from .blocks import (get_sinusoid_encoding, TransformerBlock, MaskedConv1D, LayerNorm)


@register_backbone("convTransformer")
class ConvTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """

    def __init__(
            self,
            n_vid_in,  # input video feature dimension
            n_txt_in,  # input text feature dimension
            n_embd,  # embedding dimension (after convolution)
            n_head,  # number of head for self-attention in transformers
            n_embd_ks,  # conv kernel size of the embedding network
            max_len,  # max sequence length
            arch=(2, 2, 2, 0, 5),  # (#convs, #stem transformers, #branch transformers)
            mha_win_size=[-1] * 6,  # size of local window for mha
            scale_factor=2,  # dowsampling rate for the branch,
            with_ln=False,  # if to attach layernorm after conv
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0,  # dropout rate for the projection / MLP
            path_pdrop=0.0,  # droput rate for drop path
            use_abs_pe=False,  # use absolute position embedding
            use_rel_pe=False,  # use relative position embedding
    ):
        super().__init__()
        assert len(arch) == 5
        assert len(mha_win_size) == (1 + arch[3] + arch[4])
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd ** 0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # vid_embedding network using convs
        self.vid_embd = nn.ModuleList()
        self.vid_embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = n_vid_in
            else:
                in_channels = n_embd
            self.vid_embd.append(MaskedConv1D(
                in_channels, n_embd, n_embd_ks,
                stride=1, padding=n_embd_ks // 2, bias=(not with_ln)
            )
            )
            if with_ln:
                self.vid_embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.vid_embd_norm.append(nn.Identity())

        # txt_embedding network using linear projection
        self.txt_embd = nn.ModuleList()
        self.txt_embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = n_txt_in
            else:
                in_channels = n_embd
            self.txt_embd.append(MaskedConv1D(
                in_channels, n_embd, 1,
                stride=1, padding=0, bias=(not with_ln)
            )
            )
            if with_ln:
                self.txt_embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.txt_embd_norm.append(nn.Identity())

        # stem network using (vanilla) transformer
        self.vid_stem = nn.ModuleList()
        for idx in range(arch[2]):
            self.vid_stem.append(TransformerBlock(
                n_embd, n_head,
                n_ds_strides=(1, 1),
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=self.mha_win_size[0],
                use_rel_pe=self.use_rel_pe,
                use_cross_modal=True,
            )
            )

        self.txt_stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.txt_stem.append(TransformerBlock(
                n_embd, n_head,
                n_ds_strides=(1, 1),
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=-1,
                use_rel_pe=self.use_rel_pe,
                use_cross_modal=False,
            )
            )

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[3]):
            self.branch.append(TransformerBlock(
                n_embd, n_head,
                n_ds_strides=(self.scale_factor, self.scale_factor),
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=self.mha_win_size[1 + idx],
                use_rel_pe=self.use_rel_pe,
                use_cross_modal=True,
            )
        )

        for idx in range(arch[4]):
            self.branch.append(TransformerBlock(
                n_embd, n_head,
                n_ds_strides=(self.scale_factor, self.scale_factor),
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=self.mha_win_size[1 + idx],
                use_rel_pe=self.use_rel_pe,
                use_cross_modal=False,
            )
        )

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, src_vid, src_vid_mask, src_txt, src_txt_mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = src_vid.size()

        # vid_embedding network
        for idx in range(len(self.vid_embd)):
            src_vid, src_vid_mask = self.vid_embd[idx](src_vid, src_vid_mask)
            src_vid = self.relu(self.vid_embd_norm[idx](src_vid))

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        assert src_txt is not None

        # txt_embedding network
        for idx in range(len(self.txt_embd)):
            src_txt, src_txt_mask = self.txt_embd[idx](src_txt, src_txt_mask)
            src_txt = self.relu(self.txt_embd_norm[idx](src_txt))

        src_query = src_txt
        src_query_mask = src_txt_mask

        # txt_stem transformer
        for idx in range(len(self.txt_stem)):
            src_query, src_query_mask = self.txt_stem[idx](src_query, src_query_mask)

        # vid_stem transformer
        for idx in range(len(self.vid_stem)):
            src_vid, src_vid_mask = self.vid_stem[idx](src_vid, src_vid_mask, src_query, src_query_mask)

        # prep for outputs
        out_feats = tuple()
        out_masks = tuple()
        # 1x resolution
        out_feats += (src_vid,)
        out_masks += (src_vid_mask,)

        # main branch with downsampling
        for idx in range(len(self.branch)):
            src_vid, src_vid_mask = self.branch[idx](src_vid, src_vid_mask, src_query, src_query_mask)
            out_feats += (src_vid,)
            out_masks += (src_vid_mask,)

        return out_feats, out_masks

