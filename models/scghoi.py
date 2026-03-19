import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .driverhoi import (
    HandGCN, ROIFeatureExtractor, DriverEncoder, DeviceEncoder,
    DVEdgeEncoder, DDEdgeEncoder,
    mlp, safe_norm, pairwise_diff, DriverHOIModel
)

class SpatialConditionNet(nn.Module):
    def __init__(self, edge_dim=256, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, edge_feat):
        return self.net(edge_feat).squeeze(-1)


class SCGConvLayer(nn.Module):
    def __init__(self, node_dim=256, edge_dim=256):
        super().__init__()
        self.msg_mlp = mlp([node_dim + edge_dim, node_dim])
        self.update_mlp = mlp([node_dim * 2, node_dim])

    def forward(self, H, E, A_norm):
        B, M, D = H.shape

        H_expand = H.unsqueeze(1).expand(-1, M, -1, -1)       # [B, M, M, D]
        msg_input = torch.cat([H_expand, E], dim=-1)           # [B, M, M, 2D]
        msg_raw = self.msg_mlp(msg_input)                      # [B, M, M, D]

        m = torch.einsum('bij,bijd->bid', A_norm, msg_raw)    # [B, M, D]

        H_new = H + self.update_mlp(torch.cat([H, m], dim=-1))

        return H_new


class SCGCore(nn.Module):
    def __init__(self, node_dim=256, edge_dim=256, num_layers=3):
        super().__init__()
        self.cond_net = SpatialConditionNet(edge_dim, hidden=128)
        self.layers = nn.ModuleList([
            SCGConvLayer(node_dim, edge_dim) for _ in range(num_layers)
        ])

    def forward(self, H0, E):
        B, M, _ = H0.shape

        A = self.cond_net(E)                                    # [B, M, M]

        diag_mask = 1.0 - torch.eye(M, device=H0.device).unsqueeze(0)
        A = A * diag_mask

        A_norm = A / (A.sum(dim=-1, keepdim=True) + 1e-6)

        H = H0
        for layer in self.layers:
            H = layer(H, E, A_norm)

        return H, A


class SCGHOIModel(DriverHOIModel):
    def __init__(self, num_act=4, num_cat=4, node_dim=256, num_devices=31, ablation='baseline'):
        nn.Module.__init__(self)
        self.ablation = ablation
        self.num_devices = num_devices

        self.hand_gcn = HandGCN()
        self.roi_extractor = ROIFeatureExtractor(out_dim=256)
        self.driver_enc = DriverEncoder(ablation=ablation)
        self.device_enc = DeviceEncoder(cat_num=num_cat, ablation=ablation)

        self.dv_edge = DVEdgeEncoder(node_dim=node_dim, ablation=ablation)
        self.dd_edge = DDEdgeEncoder(ablation=ablation)

        self.scg_core = SCGCore(node_dim=node_dim, edge_dim=node_dim, num_layers=3)

        self.act_head = nn.Linear(node_dim, num_act)
        self.dev_head = mlp([node_dim * 2, 256, num_devices + 1])

    def forward(self, hand_kps, dev_geom, dev_cat, f_hand_roi=None, f_dev_roi=None):
        B, N, _ = dev_geom.shape

        f_hand_gcn = self.hand_gcn(hand_kps)
        if self.ablation == 'no_pose':
            f_hand_gcn = torch.zeros_like(f_hand_gcn)

        h_d = self.driver_enc(f_hand_gcn, f_hand_roi)
        h_v = self.device_enc(dev_geom, dev_cat, f_dev_roi)

        p_h = hand_kps.mean(dim=1)
        p_v = dev_geom[..., :3]
        e_dv = self.dv_edge(h_d, h_v, p_h, p_v, hand_kps, dev_cat)
        e_dd = self.dd_edge(p_v, dev_geom[..., 3:6])

        M = N + 1
        E_all = torch.zeros(B, M, M, 256, device=hand_kps.device)
        E_all[:, 0, 1:, :] = e_dv
        E_all[:, 1:, 0, :] = e_dv
        E_all[:, 1:, 1:, :] = e_dd

        H0 = torch.zeros(B, M, 256, device=hand_kps.device)
        H0[:, 0, :] = h_d
        H0[:, 1:, :] = h_v

        H_T, A = self.scg_core(H0, E_all)

        h_d_T = H_T[:, 0, :]
        h_v_T = H_T[:, 1:, :]

        act_logits = self.act_head(h_d_T)

        h_d_expand = h_d_T.unsqueeze(1).expand(-1, N, -1)
        s_all_raw = self.dev_head(torch.cat([h_v_T, h_d_expand], dim=-1))
        dev_logits_raw = s_all_raw.mean(dim=1)
        dev_prob = torch.softmax(dev_logits_raw, dim=-1)

        return {
            "act_logits": act_logits,
            "dev_logits_raw": dev_logits_raw,
            "dev_prob": dev_prob,
            "adjacency": A
        }