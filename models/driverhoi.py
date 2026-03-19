import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.models as models

from utils import project_3d_to_2d, get_tip_point


def mlp(channels, act_last=False, dropout=0.0):
    layers = []
    for i in range(len(channels) - 1):
        layers.append(nn.Linear(channels[i], channels[i + 1]))
        if i < len(channels) - 2 or act_last:
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def pairwise_diff(x):
    xi = x.unsqueeze(2)
    xj = x.unsqueeze(1)
    return xj - xi


def safe_norm(x, dim=-1, eps=1e-6, keepdim=False):
    return torch.sqrt(torch.clamp((x ** 2).sum(dim=dim, keepdim=keepdim), min=eps))


class ROIFeatureExtractor(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        backbone = models.resnet18(weights=None)
        state_dict = torch.load("resnet18.pth")
        backbone.load_state_dict(state_dict)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(backbone.fc.in_features, out_dim)

    def _fix_bbox(self, H, W, bbox, min_size=16, margin=4):
        x1, y1, x2, y2 = bbox
        if any([not np.isfinite(v) for v in [x1, y1, x2, y2]]): return 0, 0, W, H
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        x1, x2 = int(max(0, min(W - 1, x1))), int(max(0, min(W, x2)))
        y1, y2 = int(max(0, min(H - 1, y1))), int(max(0, min(H, y2)))
        w, h = x2 - x1, y2 - y1
        if w < min_size or h < min_size:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            half_w, half_h = max(min_size // 2, w // 2), max(min_size // 2, h // 2)
            x1, x2 = max(0, cx - half_w - margin), min(W, cx + half_w + margin)
            y1, y2 = max(0, cy - half_h - margin), min(H, cy + half_h + margin)
        if x2 <= x1 or y2 <= y1: return 0, 0, W, H
        return x1, y1, x2, y2

    def extract_roi(self, image, bbox):
        _, H, W = image.shape
        x1, y1, x2, y2 = self._fix_bbox(H, W, bbox)
        roi = image[:, y1:y2, x1:x2]
        if roi.numel() == 0 or roi.shape[-1] == 0 or roi.shape[-2] == 0:
            roi = image
        return TF.resize(roi, [224, 224], antialias=True).unsqueeze(0)

    def forward(self, images_tensor, all_bboxes):
        rois = []
        lengths = []
        for b, bboxes in enumerate(all_bboxes):
            img = images_tensor[b]
            lengths.append(len(bboxes))
            for bbox in bboxes:
                rois.append(self.extract_roi(img, bbox))

        if len(rois) == 0:
            return [torch.zeros((l, self.proj.out_features), device=images_tensor.device) for l in lengths]

        batch_rois = torch.cat(rois, dim=0)
        chunk_size = 256
        feats = []
        with torch.no_grad():
            for i in range(0, batch_rois.size(0), chunk_size):
                chunk = batch_rois[i:i + chunk_size]
                f = self.backbone(chunk)
                feats.append(f.view(f.size(0), -1))

        f_all = torch.cat(feats, dim=0)
        f_proj = self.proj(f_all)
        return torch.split(f_proj, lengths)


class HandGCN(nn.Module):
    def __init__(self, in_dim=3, hidden=64, out_dim=64, dropout=0.3):
        super().__init__()
        self.register_buffer("A", self._build_adjacency(21))
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.lin_out = nn.Linear(hidden, out_dim)

    def _build_adjacency(self, J):
        A = torch.eye(J)
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
                 (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
        for i, j in edges:
            A[i, j] = A[j, i] = 1
        A = A / A.sum(1, keepdim=True)
        return A

    def _normalize(self, K):
        center = K.mean(dim=1, keepdim=True)            # [B, 1, 3]
        K_centered = K - center

        ref_bone = safe_norm(K[:, 9:10, :] - K[:, 0:1, :], dim=-1, keepdim=True)  # [B, 1, 1]
        ref_bone = ref_bone.clamp(min=1e-4)
        return K_centered / ref_bone

    def forward(self, K):
        K = self._normalize(K)
        H = F.relu(self.lin1(torch.matmul(self.A, K)))
        H = F.relu(self.lin2(torch.matmul(self.A, H)))
        H = self.drop(H)
        return self.lin_out(H.mean(dim=1))


class DriverEncoder(nn.Module):
    def __init__(self, hand_gcn_out=64, hand_roi_dim=256, out_dim=256, ablation='baseline'):
        super().__init__()
        self.ablation = ablation

        self.pose_gate = nn.Linear(hand_gcn_out, hand_gcn_out)
        nn.init.constant_(self.pose_gate.bias, -2.0)
        self.pose_drop = nn.Dropout(0.3)
        self.node_mlp = mlp([hand_gcn_out + hand_roi_dim, 256, out_dim])

    def forward(self, f_hand_gcn, f_hand_roi=None):
        if self.ablation != 'with_visual' or f_hand_roi is None:
            f_hand_roi = torch.zeros(f_hand_gcn.size(0), 256, device=f_hand_gcn.device)


        gate = torch.sigmoid(self.pose_gate(f_hand_gcn))
        f_hand_gcn = self.pose_drop(gate * f_hand_gcn)

        x = torch.cat([f_hand_gcn, f_hand_roi], dim=-1)
        return self.node_mlp(x)


class DeviceEncoder(nn.Module):
    def __init__(self, geo_dim=6, cat_num=4, cat_emb=32, roi_dim=256, out_dim=256, ablation='baseline'):
        super().__init__()
        self.ablation = ablation

        if self.ablation == 'no_geom':
            self.id_emb = nn.Embedding(50, 128)
            geom_feat_dim = 128
        else:
            self.geo_mlp = mlp([geo_dim, 128])
            geom_feat_dim = 128

        self.cat_emb = nn.Embedding(cat_num, cat_emb)
        self.node_mlp = mlp([geom_feat_dim + cat_emb + roi_dim, 256, out_dim])

    def forward(self, g, c, f_roi=None):
        if self.ablation == 'no_geom':
            B, N, _ = g.shape
            dev_ids = torch.arange(1, N + 1, device=g.device).unsqueeze(0).expand(B, -1)
            f_geo = self.id_emb(dev_ids)
        else:
            f_geo = self.geo_mlp(g)

        f_cat = self.cat_emb(c)

        if self.ablation != 'with_visual' or f_roi is None:
            B, N, _ = g.shape
            f_roi = torch.zeros(B, N, 256, device=g.device)

        x = torch.cat([f_geo, f_cat, f_roi], dim=-1)
        return self.node_mlp(x)


class DVEdgeEncoder(nn.Module):
    def __init__(self, node_dim=256, act_num=4, cat_num=4, intent_dim=64, out_dim=256, ablation='baseline'):
        super().__init__()
        self.ablation = ablation
        self.act_emb = nn.Embedding(act_num, 32)
        self.cat_emb = nn.Embedding(cat_num, 32)
        self.intent_map = mlp([64, intent_dim])
        self.fuse = mlp([node_dim * 2 + 7 + intent_dim, out_dim])

    def forward(self, h_d, h_v, p_h, p_v, hand_kps, cat, act_logits=None):
        B, N, D = h_v.shape
        dv = p_v - p_h.unsqueeze(1)
        r = safe_norm(dv, dim=-1, keepdim=True)
        dv_dir = dv / (r + 1e-6)

        palm = hand_kps[:, [0, 5, 9, 13, 17]].mean(1)
        wrist = hand_kps[:, 0]
        u_h = palm - wrist
        u_h = u_h / (safe_norm(u_h, dim=-1, keepdim=True) + 1e-6)

        rel = hand_kps - wrist.unsqueeze(1)
        proj = (rel * u_h.unsqueeze(1)).sum(-1)
        k_far = hand_kps[torch.arange(B), proj.argmax(dim=1)]
        dist_hand_dev = safe_norm(p_v - k_far.unsqueeze(1), dim=-1, keepdim=True)
        cos_align = (dv_dir * u_h.unsqueeze(1)).sum(-1, keepdim=True)


        if self.ablation == 'no_pose':
            cos_align = torch.zeros_like(cos_align)
        if self.ablation == 'no_geom':
            dv_dir = torch.zeros_like(dv_dir)
            r = torch.zeros_like(r)
            dist_hand_dev = torch.zeros_like(dist_hand_dev)

        geo = torch.cat([dv_dir, r, r ** 2, dist_hand_dev, cos_align], dim=-1)

        act_id = torch.zeros(B, dtype=torch.long, device=h_d.device) if act_logits is None else act_logits.argmax(
            dim=-1)
        intent = self.intent_map(
            torch.cat([self.act_emb(act_id).unsqueeze(1).expand(-1, N, -1), self.cat_emb(cat)], dim=-1))
        hd = h_d.unsqueeze(1).expand(-1, N, -1)
        return self.fuse(torch.cat([hd, h_v, geo, intent], dim=-1))


class DDEdgeEncoder(nn.Module):
    def __init__(self, out_dim=256, ablation='baseline'):
        super().__init__()
        self.ablation = ablation
        self.shape_map = mlp([3, 64, 32])
        self.mlp = mlp([6, 128, out_dim])

    def forward(self, p, s):
        dp = pairwise_diff(p)
        dist = safe_norm(dp, dim=-1, keepdim=True)
        s_feat = self.shape_map(s)
        rho = torch.exp(-safe_norm(pairwise_diff(s_feat), dim=-1, keepdim=True))


        if self.ablation == 'no_geom':
            dp = torch.zeros_like(dp)
            dist = torch.zeros_like(dist)

        return self.mlp(torch.cat([dp, dist, rho, rho], dim=-1))


class LinkNet(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = mlp([in_dim, hidden, 1])

    def forward(self, e_ij):
        return torch.sigmoid(self.net(e_ij)).squeeze(-1)


class MessageNet(nn.Module):
    def __init__(self, node_dim=256, edge_dim=256, msg_dim=256):
        super().__init__()
        self.mlp = mlp([node_dim + edge_dim, msg_dim])

    def forward(self, h_i, e_ij):
        B, M, D = h_i.shape
        h_i_expand = h_i.unsqueeze(2).expand(-1, -1, M, -1)
        return self.mlp(torch.cat([h_i_expand, e_ij], dim=-1))


class UpdateGRU(nn.Module):
    def __init__(self, node_dim=256, msg_dim=256):
        super().__init__()
        self.gru = nn.GRUCell(msg_dim, node_dim)

    def forward(self, h, m):
        return torch.cat([self.gru(m[b], h[b]).unsqueeze(0) for b in range(h.size(0))], dim=0)


class GPNNCore(nn.Module):
    def __init__(self, node_dim=256, edge_dim=256, msg_dim=256, iters=3):
        super().__init__()
        self.link = LinkNet(edge_dim)
        self.msg = MessageNet(node_dim, edge_dim, msg_dim)
        self.update = UpdateGRU(node_dim, msg_dim)
        self.iters = iters

    def forward(self, H0, E):
        h = H0
        for _ in range(self.iters):
            A = self.link(E)
            A_norm = A / (A.sum(-1, keepdim=True) + 1e-6)
            M_ij = self.msg(h, E)
            m_j = torch.einsum('bij,bijd->bjd', A_norm, M_ij)
            h = self.update(h, m_j)
        return h, A


class DriverHOIModel(nn.Module):
    def __init__(self, num_act=4, num_cat=4, node_dim=256, num_devices=31, ablation='baseline'):
        super().__init__()
        self.ablation = ablation
        self.hand_gcn = HandGCN()
        self.roi_extractor = ROIFeatureExtractor(out_dim=256)

        self.driver_enc = DriverEncoder(ablation=ablation)
        self.device_enc = DeviceEncoder(cat_num=num_cat, ablation=ablation)
        self.dv_edge = DVEdgeEncoder(node_dim=node_dim, ablation=ablation)
        self.dd_edge = DDEdgeEncoder(ablation=ablation)
        self.gpnn = GPNNCore()

        self.act_head = nn.Linear(node_dim, num_act)
        self.num_devices = num_devices
        self.dev_head = mlp([512, 256, num_devices + 1])

    def forward(self, hand_kps, dev_geom, dev_cat, f_hand_roi=None, f_dev_roi=None):
        B, N, _ = dev_geom.shape
        f_hand_gcn = self.hand_gcn(hand_kps)

        if self.ablation == 'no_pose':
            f_hand_gcn = torch.zeros_like(f_hand_gcn)

        h_d = self.driver_enc(f_hand_gcn, f_hand_roi)
        h_v = self.device_enc(dev_geom, dev_cat, f_dev_roi)

        p_h = hand_kps.mean(1)
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

        H_T, A = self.gpnn(H0, E_all)
        h_d_T = H_T[:, 0, :]
        h_v_T = H_T[:, 1:, :]

        act_logits = self.act_head(h_d_T)
        s_all_raw = self.dev_head(
            torch.cat([h_v_T, h_d_T.unsqueeze(1).expand(-1, h_v_T.size(1), -1)], dim=-1)
        )
        dev_logits_raw = s_all_raw.mean(dim=1)
        dev_prob = torch.softmax(dev_logits_raw, dim=-1)

        return {
            "act_logits": act_logits,
            "dev_logits_raw": dev_logits_raw,
            "dev_prob": dev_prob,
            "adjacency": A
        }

    def prepare_inputs(self, sample, device_config):
        device_ids = sample["device_id"]
        keypoints3d = sample["keypoints3d"]["hands_keypoints3d"]
        batch_size = device_ids.shape[0]

        dev_keys_sorted = sorted(device_config.keys(), key=lambda x: int(x))
        hands, dev_geoms, dev_cats = [], [], []

        for b in range(batch_size):
            hand_pair = keypoints3d[b]

            geom_list, cat_list = [], []
            for did_i in dev_keys_sorted:
                info = device_config[did_i]
                x, y, z = info["x"] / 1000.0, info["y"] / 1000.0, info["z"] / 1000.0
                l, w, h = info["l"] / 1000.0, info["w"] / 1000.0, info["h"] / 1000.0
                geom_list.append([x, y, z, l, w, h])
                cat_list.append(info["c"])

            dev_centers = torch.tensor(geom_list)[:, :3].to(hand_pair.device)
            hand0_center = hand_pair[0].mean(dim=0, keepdim=True)
            hand1_center = hand_pair[1].mean(dim=0, keepdim=True)

            dist0 = torch.cdist(hand0_center, dev_centers).min()
            dist1 = torch.cdist(hand1_center, dev_centers).min()

            if dist0 < dist1:
                hand = hand_pair[0]
            else:
                hand = hand_pair[1]

            hands.append(hand)
            dev_geoms.append(geom_list)
            dev_cats.append(cat_list)

        hand_tensor = torch.stack(hands, dim=0)
        dev_geom_tensor = torch.tensor(dev_geoms, dtype=torch.float32)
        dev_cat_tensor = torch.tensor(dev_cats, dtype=torch.long)

        images = sample.get("image", None)
        have_cam = ("camera_intrinsic" in sample) and ("camera_extrinsic" in sample)

        if self.ablation == 'with_visual' and isinstance(images, torch.Tensor) and have_cam:
            device = next(self.parameters()).device
            images = images.float().to(device)

            all_hand_bboxes = []
            all_dev_bboxes = []

            for b in range(batch_size):
                H, W = images.shape[2], images.shape[3]
                K = sample["camera_intrinsic"]["K"][b].cpu().numpy()
                D = sample["camera_intrinsic"]["D"][b].cpu().numpy()
                R = sample["camera_extrinsic"]["R"][b].cpu().numpy()
                T = sample["camera_extrinsic"]["T"][b].cpu().numpy()

                hand3d = hand_tensor[b].cpu().numpy()
                try:
                    hand2d = project_3d_to_2d(hand3d, K, D, R, T)
                    if hand2d.ndim == 3: hand2d = hand2d[0]
                    hand2d = np.asarray(hand2d, dtype=np.float32)
                    mask = np.isfinite(hand2d).all(axis=1)
                    if mask.sum() >= 1:
                        pts = hand2d[mask]
                        x_min, y_min = pts.min(axis=0)
                        x_max, y_max = pts.max(axis=0)
                        pad = 0.15 * max((x_max - x_min + 1), (y_max - y_min + 1))
                        bbox_hand = (x_min - pad, y_min - pad, x_max + pad, y_max + pad)
                    else:
                        bbox_hand = (0, 0, W, H)
                except Exception:
                    bbox_hand = (0, 0, W, H)
                all_hand_bboxes.append([bbox_hand])

                centers = dev_geom_tensor[b, :, :3].cpu().numpy() * 1000.0
                dev_bboxes = []
                try:
                    dev2d_all = project_3d_to_2d(centers, K, D, R, T)
                    if dev2d_all.ndim == 3: dev2d_all = dev2d_all[0]
                    for n in range(dev_geom_tensor.shape[1]):
                        dev2d = dev2d_all[n]
                        if np.isfinite(dev2d).all():
                            x, y = float(dev2d[0]), float(dev2d[1])
                            dev_bboxes.append((x - 20, y - 20, x + 20, y + 20))
                        else:
                            dev_bboxes.append((0, 0, W, H))
                except Exception:
                    dev_bboxes = [(0, 0, W, H)] * dev_geom_tensor.shape[1]
                all_dev_bboxes.append(dev_bboxes)

            f_hand_list = self.roi_extractor(images, all_hand_bboxes)
            f_dev_list = self.roi_extractor(images, all_dev_bboxes)

            f_hand_roi = torch.stack([x.squeeze(0) for x in f_hand_list], dim=0)
            f_dev_roi = torch.stack(f_dev_list, dim=0)
        else:
            f_hand_roi = torch.zeros(batch_size, 256, device=hand_tensor.device)
            f_dev_roi = torch.zeros(batch_size, dev_geom_tensor.shape[1], 256, device=hand_tensor.device)

        return hand_tensor, dev_geom_tensor, dev_cat_tensor, f_hand_roi, f_dev_roi

    def compute_loss(self, out, batch, hand, dev_geom, lambda_act=1.0, lambda_id=3.0, lambda_inter=0.5, lambda_aux=0.2):
        device = hand.device

        act_logits = out["act_logits"]
        act_label = batch["action_label"].to(device)
        loss_act = nn.CrossEntropyLoss()(act_logits, act_label)

        dev_logits_raw = out["dev_logits_raw"]
        device_id = batch["device_id"].to(device).long()
        loss_id = nn.CrossEntropyLoss()(dev_logits_raw, device_id)

        dev_prob = out["dev_prob"].clamp(1e-6, 1 - 1e-6)
        inter_flag = batch["interaction_flag"].float().to(device)
        inter_prob = 1.0 - dev_prob[:, 0]
        loss_inter = nn.BCELoss()(inter_prob, inter_flag)

        if self.ablation == 'no_pose':
            loss_aux = torch.zeros((), device=device)
        else:
            dev_pred = dev_logits_raw.argmax(dim=-1)
            mask = (dev_pred > 0).float()
            if mask.sum() > 0:
                idx = (dev_pred - 1).clamp(min=0)
                p_t = dev_geom[torch.arange(hand.size(0), device=device), idx, :3]
                p_h = hand.mean(dim=1)
                v_finger = hand[:, 9, :] - hand[:, 0, :]

                EPS = 1e-6
                a, b = p_t - p_h, v_finger
                a_n = a / (a.norm(dim=-1, keepdim=True) + EPS)
                b_n = b / (b.norm(dim=-1, keepdim=True) + EPS)

                cos_val = (a_n * b_n).sum(dim=-1).clamp(-1, 1)
                L_dir = 1 - cos_val
                hand_tip = get_tip_point(hand)
                L_dist = F.smooth_l1_loss(hand_tip, p_t, reduction='none').sum(dim=-1)

                loss_aux = ((L_dir + L_dist) * mask).sum() / (mask.sum() + EPS)
            else:
                loss_aux = torch.zeros((), device=device)

        loss_total = (
                lambda_act * loss_act +
                lambda_id * loss_id +
                lambda_inter * loss_inter +
                lambda_aux * loss_aux
        )

        return {
            "total": loss_total,
            "act": loss_act.detach(),
            "id": loss_id.detach(),
            "inter": loss_inter.detach(),
            "aux": loss_aux.detach()
        }
