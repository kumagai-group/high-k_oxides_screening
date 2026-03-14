import torch
import torch.nn as nn
from torch_scatter import scatter
from e3nn.o3 import Irreps
from e3nn import o3
from e3nn import nn as e3_nn
from e3nn.math import soft_one_hot_linspace
from typing import Union


class Convolution(nn.Module):
    def __init__(
        self,
        irreps_in: Union[int, str, Irreps],
        irreps_sh: Union[int, str, Irreps],
        irreps_out: Union[int, str, Irreps],
        dim_edge_attr: int = 64,
        act=nn.ReLU(),
        debug: bool = False
    ):
        super().__init__()
        self.debug = debug

        self.irreps_in = Irreps(irreps_in) if isinstance(irreps_in, (str, int)) else irreps_in
        self.irreps_sh = Irreps(irreps_sh) if isinstance(irreps_sh, (str, int)) else irreps_sh
        self.irreps_out = Irreps(irreps_out) if isinstance(irreps_out, (str, int)) else irreps_out

        self.mul_l0in = int(self.irreps_in[0][0])
        self.mul_l0out = int(self.irreps_out[0][0])
        self.dim_edge_attr = dim_edge_attr
        self.act = act

        self.gate = e3_nn.Gate(
            Irreps(f"{self.mul_l0out}x0e"), [self.act],
            Irreps(f"{sum([_irrep[0] for _irrep in self.irreps_out[1:]])}x0e"), [self.act],
            Irreps("+".join([f"{_irrep[0]}x{_irrep[1]}" for _irrep in self.irreps_out[1:]]))
        )
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_sh, self.gate.irreps_in,
            internal_weights=False, shared_weights=False
        )
        self.embedding_edge = e3_nn.FullyConnectedNet(
            [2 * self.mul_l0in + self.dim_edge_attr, self.tp.weight_numel], act=self.act
        )

    def forward(self, x, edge_attr, Yij, edge_index):
        if self.debug:
            print(f"[Convolution] input x: {x.shape}, edge_attr: {edge_attr.shape}")

        emb_input = torch.cat(
            [x[edge_index[0]][:, :self.mul_l0in], x[edge_index[1]][:, :self.mul_l0in], edge_attr],
            dim=-1
        )
        w = self.embedding_edge(emb_input)

        msg = self.tp(x[edge_index[1]], Yij, w)
        out = scatter(msg, edge_index[0], dim=0, reduce="mean")

        out = self.gate(out)
        return x + out


class E3NNModel(nn.Module):
    def __init__(
        self,
        irreps_in: Irreps,
        irreps_sh: Irreps,
        irreps_hid: Irreps,
        irreps_out: Irreps,
        dim_edge: int,
        n_conv: int,
        activation: nn.Module,
        is_site_pred: bool,
        output_allsite: bool,
    ):
        super().__init__()
        self.irreps_in = Irreps(irreps_in) if isinstance(irreps_in, (str, int)) else irreps_in
        self.irreps_sh = Irreps(irreps_sh) if isinstance(irreps_sh, (str, int)) else irreps_sh
        self.irreps_hid = Irreps(irreps_hid) if isinstance(irreps_hid, (str, int)) else irreps_hid
        self.irreps_out = Irreps(irreps_out) if isinstance(irreps_out, (str, int)) else irreps_out

        self.act = activation
        self.dim_edge_attr = dim_edge
        self.is_site_pred = is_site_pred
        self.output_allsite = output_allsite

        self.embedding = o3.Linear(self.irreps_in, self.irreps_hid, biases=True)
        self.conv_layers = nn.ModuleList([
            Convolution(
                irreps_in=self.irreps_hid, 
                irreps_sh=self.irreps_sh, 
                irreps_out=self.irreps_hid, 
                dim_edge_attr=self.dim_edge_attr, 
                act=self.act, 
            )
            for _ in range(n_conv)
        ])

        self.gate_out = e3_nn.Gate(
            Irreps(f"{self.irreps_hid[0]}"), [self.act],
            Irreps(f"{sum([_irrep[0] for _irrep in self.irreps_hid[1:]])}x0e"), [self.act],
            Irreps("+".join([f"{_irrep[0]}x{_irrep[1]}" for _irrep in self.irreps_hid[1:]]))
        )
        self.fc_bgate = o3.Linear(self.irreps_hid, self.gate_out.irreps_in, biases=True)
        self.fc_pgate = o3.Linear(self.gate_out.irreps_out, self.irreps_out, biases=True)

        self.site_pooling = SitePooling()

    def forward(self, batch_data):
        x = self._embed_node_features(batch_data.attrs_node)
        Yij = self._compute_sph_harm(batch_data.rijs_relative)
        e_attr = self._embed_edge_attr(batch_data.attrs_edge)

        for conv in self.conv_layers:
            x = conv(x, e_attr, Yij, batch_data.indexes_edge)
        x = self._post_conv(x)
        x = self._pooling(batch_data, x)
        return x

    def _embed_node_features(self, x):
        return self.embedding(x)

    def _compute_sph_harm(self, xij):
        Yij = o3.spherical_harmonics(
            l=self.irreps_sh, 
            x=xij,
            normalize=True,
            normalization="component",
        )
        return Yij

    def _embed_edge_attr(self, dist):
        dist = dist.squeeze(-1) if dist.ndim > 1 else dist
        edge_attr = soft_one_hot_linspace(
            x=dist, start=0.0, end=5.0,
            number=self.dim_edge_attr, basis="gaussian", cutoff=True
        )
        return edge_attr

    def _post_conv(self, x):
        x_bgate = self.fc_bgate(x)
        x_gate = self.gate_out(x_bgate)
        return self.fc_pgate(x_gate)

    def _pooling(self, batch_data, x):
        if self.is_site_pred:
            # if not self.output_allsite:
            #     x = self.site_pooling(x, pooling_mask=batch_data.pooling_mask)
            return x
        else:
            return scatter(x, batch_data.batch, dim=0, reduce="mean")


class SitePooling(nn.Module):
    def forward(self, x, pooling_mask):
        indices = [idx for group in pooling_mask for idx in group]
        return x[indices]
