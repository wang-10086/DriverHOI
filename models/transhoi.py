import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .driverhoi import (
    HandGCN, ROIFeatureExtractor, DriverEncoder, DeviceEncoder,
    mlp, safe_norm, DriverHOIModel
)

class GeometricPositionalEncoding(nn.Module):
    def __init__(self, geo_dim=7, d_model=256):
        super().__init__()
        self.proj = mlp([geo_dim, 128, d_model])

    def forward(self, geo_features):
        return self.proj(geo_features)


class TransHOILayer(nn.Module):
    def __init__(self, d_model=256, nhead=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.cross_attn_d = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn_v = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.self_attn_v = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ffn_d = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model)
        )
        self.ffn_v = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model)
        )
        self.norm_d1 = nn.LayerNorm(d_model)
        self.norm_d2 = nn.LayerNorm(d_model)
        self.norm_v1 = nn.LayerNorm(d_model)
        self.norm_v2 = nn.LayerNorm(d_model)
        self.norm_v3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_d, h_v):
        h_d_attn, attn_weights = self.cross_attn_d(h_d, h_v, h_v)
        h_d = self.norm_d1(h_d + self.dropout(h_d_attn))
        h_d = self.norm_d2(h_d + self.dropout(self.ffn_d(h_d)))

        h_v_cross, _ = self.cross_attn_v(h_v, h_d, h_d)
        h_v = self.norm_v1(h_v + self.dropout(h_v_cross))

        h_v_self, _ = self.self_attn_v(h_v, h_v, h_v)
        h_v = self.norm_v2(h_v + self.dropout(h_v_self))

        h_v = self.norm_v3(h_v + self.dropout(self.ffn_v(h_v)))

        return h_d, h_v, attn_weights


class TransHOICore(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransHOILayer(d_model, nhead, dim_feedforward=d_model * 2, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, h_d, h_v):
        h_d = h_d.unsqueeze(1)  # [B, 1, 256]
        attn = None
        for layer in self.layers:
            h_d, h_v, attn = layer(h_d, h_v)
        return h_d.squeeze(1), h_v, attn

class TransHOIModel(DriverHOIModel):
    def __init__(self, num_act=4, num_cat=4, node_dim=256, num_devices=31, ablation='baseline'):
        nn.Module.__init__(self)
        self.ablation = ablation
        self.num_devices = num_devices

        self.hand_gcn = HandGCN()
        self.roi_extractor = ROIFeatureExtractor(out_dim=256)
        self.driver_enc = DriverEncoder(ablation=ablation)
        self.device_enc = DeviceEncoder(cat_num=num_cat, ablation=ablation)

        self.geo_pe = GeometricPositionalEncoding(geo_dim=7, d_model=node_dim)
        self.interaction_core = TransHOICore(
            d_model=node_dim, nhead=4, num_layers=3, dropout=0.1
        )

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

        dv = p_v - p_h.unsqueeze(1)
        r = safe_norm(dv, dim=-1, keepdim=True)
        dv_dir = dv / (r + 1e-6)

        palm = hand_kps[:, [0, 5, 9, 13, 17]].mean(dim=1)
        wrist = hand_kps[:, 0]
        u_h = palm - wrist
        u_h = u_h / (safe_norm(u_h, dim=-1, keepdim=True) + 1e-6)

        rel = hand_kps - wrist.unsqueeze(1)
        proj = (rel * u_h.unsqueeze(1)).sum(-1)
        k_far = hand_kps[torch.arange(B, device=hand_kps.device), proj.argmax(dim=1)]
        dist_hand_dev = safe_norm(p_v - k_far.unsqueeze(1), dim=-1, keepdim=True)
        cos_align = (dv_dir * u_h.unsqueeze(1)).sum(-1, keepdim=True)

        if self.ablation == 'no_pose':
            cos_align = torch.zeros_like(cos_align)
        if self.ablation == 'no_geom':
            dv_dir = torch.zeros_like(dv_dir)
            r = torch.zeros_like(r)
            dist_hand_dev = torch.zeros_like(dist_hand_dev)

        geo_raw = torch.cat([dv_dir, r, r ** 2, dist_hand_dev, cos_align], dim=-1)
        geo_encoding = self.geo_pe(geo_raw)
        h_v = h_v + geo_encoding

        h_d_T, h_v_T, attn_weights = self.interaction_core(h_d, h_v)

        act_logits = self.act_head(h_d_T)

        h_d_expand = h_d_T.unsqueeze(1).expand(-1, N, -1)
        s_all_raw = self.dev_head(torch.cat([h_v_T, h_d_expand], dim=-1))
        dev_logits_raw = s_all_raw.mean(dim=1)
        dev_prob = torch.softmax(dev_logits_raw, dim=-1)

        adjacency = attn_weights.squeeze(1) if attn_weights is not None else None

        return {
            "act_logits": act_logits,
            "dev_logits_raw": dev_logits_raw,
            "dev_prob": dev_prob,
            "adjacency": adjacency
        }