import os
import json
import argparse
from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from DriverHOIDatasets import DriverHOIDataset
from models import create_model
from utils import save_config_module_as_json
import config as default_cfg


def parse_args():
    p = argparse.ArgumentParser(description="DriverHOI 训练脚本")

    p.add_argument('--exp_name',    type=str, default=default_cfg.EXP_NAME)
    p.add_argument('--job_name',    type=str, default=default_cfg.JOB_NAME)
    p.add_argument('--test_subj',   type=str, default=default_cfg.TEST_SUBJ)
    p.add_argument('--ablation',    type=str, default=default_cfg.ABLATION_MODE,
                   choices=['baseline', 'with_visual', 'no_pose', 'no_geom'])
    p.add_argument('--model_type',  type=str, default=default_cfg.MODEL_TYPE,
                   choices=['DriverHOI', 'MLP-HOI', 'TransHOI', 'SCG-HOI'])

    p.add_argument('--batch_size',  type=int,   default=default_cfg.BATCH_SIZE)
    p.add_argument('--num_epochs',  type=int,   default=default_cfg.NUM_EPOCHS)
    p.add_argument('--lr',          type=float, default=default_cfg.LEARNING_RATE)
    p.add_argument('--num_workers', type=int,   default=default_cfg.NUM_WORKERS)

    args, _ = p.parse_known_args()
    return args


def build_cfg(args):
    cfg = SimpleNamespace(

        EXP_NAME=args.exp_name,
        JOB_NAME=args.job_name,
        TEST_SUBJ=args.test_subj,
        ABLATION_MODE=args.ablation,
        MODEL_TYPE=args.model_type,
        BATCH_SIZE=args.batch_size,
        NUM_EPOCHS=args.num_epochs,
        LEARNING_RATE=args.lr,
        NUM_WORKERS=args.num_workers,

        DATA_ROOT=default_cfg.DATA_ROOT,
        DEVICE_CFG_PATH=default_cfg.DEVICE_CFG_PATH,
        LOG_DIR=default_cfg.LOG_DIR,
        CKPT_DIR=default_cfg.CKPT_DIR,
        DATA_SPLIT=default_cfg.DATA_SPLIT,
        ALL_SUBJECTS=default_cfg.ALL_SUBJECTS,
        CAMERA_VIEWS=default_cfg.CAMERA_VIEWS,
        FRAME_SAMPLING=default_cfg.FRAME_SAMPLING,
        NUM_FRAMES_PER_ACTION=default_cfg.NUM_FRAMES_PER_ACTION,
        ACTION_FRAME_POLICY=default_cfg.ACTION_FRAME_POLICY,
        NUM_DEVICES=default_cfg.NUM_DEVICES,
        NUM_ACT=default_cfg.NUM_ACT,
        NUM_CAT=default_cfg.NUM_CAT,
        NODE_DIM=default_cfg.NODE_DIM,
        LAMBDA_ACT=default_cfg.LAMBDA_ACT,
        LAMBDA_ID=default_cfg.LAMBDA_ID,
        LAMBDA_INTER=default_cfg.LAMBDA_INTER,
        LAMBDA_AUX=default_cfg.LAMBDA_AUX,
        SEED=default_cfg.SEED,
    )

    if cfg.ABLATION_MODE == 'no_pose':
        cfg.LAMBDA_AUX = 0.0
    return cfg


def build_loso_split(cfg):
    all_subs = cfg.ALL_SUBJECTS
    if cfg.TEST_SUBJ:
        if cfg.TEST_SUBJ not in all_subs:
            raise ValueError(f"被试 {cfg.TEST_SUBJ} 不在列表 {all_subs} 中")
        test_idx = all_subs.index(cfg.TEST_SUBJ)
        val_idx = (test_idx - 1) % len(all_subs)
        exclude = {cfg.TEST_SUBJ, all_subs[val_idx]}
        split = {
            'test':  [cfg.TEST_SUBJ],
            'val':   [all_subs[val_idx]],
            'train': [s for s in all_subs if s not in exclude]
        }
        return split
    else:
        return cfg.DATA_SPLIT.copy()


def topk_accuracy(pred_probs, true_labels, k=3):
    if len(true_labels) == 0:
        return 0.0
    topk_preds = np.argsort(pred_probs, axis=1)[:, -k:][:, ::-1]
    correct = sum(t in topk for t, topk in zip(true_labels, topk_preds))
    return correct / len(true_labels)


def train():
    args = parse_args()
    cfg = build_cfg(args)

    print(f"\n{'='*60}")
    print(f" 模型: {cfg.MODEL_TYPE} | 消融: {cfg.ABLATION_MODE}")
    print(f" 实验: {cfg.EXP_NAME} | 归档: {cfg.JOB_NAME}")
    print(f" 测试被试: {cfg.TEST_SUBJ}")
    print(f"{'='*60}")

    ablation_desc = {
        'baseline':    "视觉[关] | 姿态[开] | 几何[开]",
        'with_visual': "视觉[开] | 姿态[开] | 几何[开]",
        'no_pose':     "视觉[关] | 姿态[关] | 几何[开]",
        'no_geom':     "视觉[关] | 姿态[开] | 几何[关(隐式ID)]",
    }
    print(f" 消融配置: {ablation_desc[cfg.ABLATION_MODE]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" 计算设备: {device}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subj_suffix = f"_{cfg.TEST_SUBJ}" if (cfg.TEST_SUBJ and not cfg.JOB_NAME) else ""
    run_name = f"{cfg.EXP_NAME}{subj_suffix}_{timestamp}"

    base_log_dir = cfg.LOG_DIR / cfg.JOB_NAME if cfg.JOB_NAME else cfg.LOG_DIR
    base_ckpt_dir = cfg.CKPT_DIR / cfg.JOB_NAME if cfg.JOB_NAME else cfg.CKPT_DIR

    log_dir = base_log_dir / run_name
    ckpt_dir = base_ckpt_dir / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f" 日志: {log_dir}")
    print(f" 权重: {ckpt_dir}")

    save_config_module_as_json(cfg, log_dir)
    writer = SummaryWriter(log_dir)

    current_split = build_loso_split(cfg)
    if cfg.TEST_SUBJ:
        print(f"\n [LOSO] Test: {current_split['test']} | Val: {current_split['val']} | Train: {current_split['train']} ({len(current_split['train'])} subjects)")
    else:
        print(" [使用默认数据划分]")

    train_set, val_set, _ = DriverHOIDataset.split_dataset(
        data_root=cfg.DATA_ROOT, split_cfg=current_split,
        camera_views=cfg.CAMERA_VIEWS, frame_sampling=cfg.FRAME_SAMPLING,
        num_frames_per_action=cfg.NUM_FRAMES_PER_ACTION,
        action_frame_policy=cfg.ACTION_FRAME_POLICY,
        transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    )
    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=cfg.NUM_WORKERS)
    val_loader   = DataLoader(val_set,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

    with open(cfg.DEVICE_CFG_PATH, "r") as f:
        device_config = json.load(f)

    model = create_model(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    best_val_loss = float("inf")

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        model.train()
        train_loss_sum = {"total": 0.0, "act": 0.0, "id": 0.0, "inter": 0.0, "aux": 0.0}
        all_act_preds, all_act_labels = [], []
        all_dev_preds, all_dev_labels = [], []
        all_int_preds, all_int_labels = [], []
        all_dev_probs = []

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{cfg.NUM_EPOCHS}] Train", ncols=140)
        for batch in pbar:
            hand, dev_geom, dev_cat, f_hand_roi, f_dev_roi = model.prepare_inputs(batch, device_config)
            hand, dev_geom, dev_cat = hand.to(device), dev_geom.to(device), dev_cat.to(device)
            if f_hand_roi is not None: f_hand_roi = f_hand_roi.to(device)
            if f_dev_roi  is not None: f_dev_roi  = f_dev_roi.to(device)

            out = model(hand, dev_geom, dev_cat, f_hand_roi, f_dev_roi)
            losses = model.compute_loss(
                out, batch, hand, dev_geom,
                lambda_act=cfg.LAMBDA_ACT, lambda_id=cfg.LAMBDA_ID,
                lambda_inter=cfg.LAMBDA_INTER, lambda_aux=cfg.LAMBDA_AUX
            )

            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            for k in train_loss_sum: train_loss_sum[k] += losses[k].item()

            dev_prob = out["dev_prob"].detach().cpu().numpy()
            all_act_preds.extend(out["act_logits"].argmax(dim=-1).cpu().numpy())
            all_act_labels.extend(batch["action_label"].cpu().numpy())
            all_dev_preds.extend(out["dev_logits_raw"].argmax(dim=-1).cpu().numpy())
            all_dev_labels.extend(batch["device_id"].cpu().numpy())
            all_int_preds.extend((1.0 - dev_prob[:, 0] > 0.5).astype(int))
            all_int_labels.extend(batch["interaction_flag"].cpu().numpy())
            all_dev_probs.extend(dev_prob[:, 1:])

            pbar.set_postfix(Tot=f"{losses['total'].item():.3f}",
                             Act=f"{losses['act'].item():.3f}",
                             Id=f"{losses['id'].item():.3f}")

        train_loss_avg = {k: v / len(train_loader) for k, v in train_loss_sum.items()}
        act_acc = (np.array(all_act_preds) == np.array(all_act_labels)).mean()
        inter_acc = (np.array(all_int_preds) == np.array(all_int_labels)).mean()
        dev_acc = (np.array(all_dev_preds) == np.array(all_dev_labels)).mean()
        train_overall = ((np.array(all_act_preds) == np.array(all_act_labels)) &
                         (np.array(all_dev_preds) == np.array(all_dev_labels))).mean()

        mask_valid = np.array(all_dev_labels) > 0
        train_top3 = topk_accuracy(np.array(all_dev_probs)[mask_valid],
                                   np.array(all_dev_labels)[mask_valid] - 1, k=3) if mask_valid.sum() > 0 else 0.0

        model.eval()
        val_loss_sum = {"total": 0.0, "act": 0.0, "id": 0.0, "inter": 0.0, "aux": 0.0}
        all_act_preds, all_act_labels = [], []
        all_dev_preds, all_dev_labels = [], []
        all_int_preds, all_int_labels = [], []
        all_dev_probs = []

        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch [{epoch}/{cfg.NUM_EPOCHS}] Val  ", ncols=140)
            for batch in pbar_val:
                hand, dev_geom, dev_cat, f_hand_roi, f_dev_roi = model.prepare_inputs(batch, device_config)
                hand, dev_geom, dev_cat = hand.to(device), dev_geom.to(device), dev_cat.to(device)
                if f_hand_roi is not None: f_hand_roi = f_hand_roi.to(device)
                if f_dev_roi  is not None: f_dev_roi  = f_dev_roi.to(device)

                out = model(hand, dev_geom, dev_cat, f_hand_roi, f_dev_roi)
                losses = model.compute_loss(out, batch, hand, dev_geom,
                                            cfg.LAMBDA_ACT, cfg.LAMBDA_ID,
                                            cfg.LAMBDA_INTER, cfg.LAMBDA_AUX)
                for k in val_loss_sum: val_loss_sum[k] += losses[k].item()

                dev_prob = out["dev_prob"].cpu().numpy()
                all_act_preds.extend(out["act_logits"].argmax(dim=-1).cpu().numpy())
                all_act_labels.extend(batch["action_label"].cpu().numpy())
                all_dev_preds.extend(out["dev_logits_raw"].argmax(dim=-1).cpu().numpy())
                all_dev_labels.extend(batch["device_id"].cpu().numpy())
                all_int_preds.extend((1.0 - dev_prob[:, 0] > 0.5).astype(int))
                all_int_labels.extend(batch["interaction_flag"].cpu().numpy())
                all_dev_probs.extend(dev_prob[:, 1:])

                pbar_val.set_postfix(Tot=f"{losses['total'].item():.3f}",
                                     Act=f"{losses['act'].item():.3f}",
                                     Id=f"{losses['id'].item():.3f}")

        val_loss_avg = {k: v / len(val_loader) for k, v in val_loss_sum.items()}
        val_act  = (np.array(all_act_preds) == np.array(all_act_labels)).mean()
        val_int  = (np.array(all_int_preds) == np.array(all_int_labels)).mean()
        val_dev  = (np.array(all_dev_preds) == np.array(all_dev_labels)).mean()
        val_overall = ((np.array(all_act_preds) == np.array(all_act_labels)) &
                       (np.array(all_dev_preds) == np.array(all_dev_labels))).mean()
        mask_val = np.array(all_dev_labels) > 0
        val_top3 = topk_accuracy(np.array(all_dev_probs)[mask_val],
                                 np.array(all_dev_labels)[mask_val] - 1, k=3) if mask_val.sum() > 0 else 0.0

        print(f"  Valid | Loss: {val_loss_avg['total']:.3f} | Act: {val_act:.3f} | Int: {val_int:.3f} "
              f"| Top-1: {val_dev:.3f} | Top-3: {val_top3:.3f} | Overall: {val_overall:.3f}")

        for tag, val in [("Train_Total", train_loss_avg["total"]), ("Train_Act", train_loss_avg["act"]),
                         ("Train_Id", train_loss_avg["id"]), ("Train_Inter", train_loss_avg["inter"]),
                         ("Train_Aux", train_loss_avg["aux"])]:
            writer.add_scalar(f"Loss/{tag}", val, epoch)
        for tag, val in [("Val_Total", val_loss_avg["total"]), ("Val_Act", val_loss_avg["act"]),
                         ("Val_Id", val_loss_avg["id"]), ("Val_Inter", val_loss_avg["inter"]),
                         ("Val_Aux", val_loss_avg["aux"])]:
            writer.add_scalar(f"Loss/{tag}", val, epoch)
        for tag, val in [("Train_Overall", train_overall), ("Train_Act", act_acc),
                         ("Train_Int", inter_acc), ("Train_Top1", dev_acc), ("Train_Top3", train_top3)]:
            writer.add_scalar(f"Acc/{tag}", val, epoch)
        for tag, val in [("Val_Overall", val_overall), ("Val_Act", val_act),
                         ("Val_Int", val_int), ("Val_Top1", val_dev), ("Val_Top3", val_top3)]:
            writer.add_scalar(f"Acc/{tag}", val, epoch)

        if val_loss_avg["total"] < best_val_loss:
            best_val_loss = val_loss_avg["total"]
            torch.save(model.state_dict(), f"{ckpt_dir}/best_model.pth")
            print(f"  --> 保存最优模型 (val_loss={best_val_loss:.4f})")

    writer.close()
    print("训练完成!")


if __name__ == "__main__":
    train()