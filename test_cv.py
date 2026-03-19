import glob
import json
from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from DriverHOIDatasets import DriverHOIDataset
from models import create_model
import config as default_cfg


MODEL_TYPE = "DriverHOI"        # 'DriverHOI' | 'MLP-HOI' | 'TransHOI' | 'SCG-HOI'
ABLATION_MODE = "baseline"      # 'baseline' | 'with_visual' | 'no_pose' | 'no_geom'
JOB_NAME = "driverhoi_exp02"
BATCH_SIZE = 16
TOP_K = 3

CM_SAVE_DIR = "results"              # such as "results/confusion_matrices"

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 12,
})


def build_cfg():
    return SimpleNamespace(
        MODEL_TYPE=MODEL_TYPE,
        ABLATION_MODE=ABLATION_MODE,
        DATA_ROOT=default_cfg.DATA_ROOT,
        DEVICE_CFG_PATH=default_cfg.DEVICE_CFG_PATH,
        CKPT_DIR=default_cfg.CKPT_DIR,
        ALL_SUBJECTS=default_cfg.ALL_SUBJECTS,
        CAMERA_VIEWS=default_cfg.CAMERA_VIEWS,
        FRAME_SAMPLING=default_cfg.FRAME_SAMPLING,
        NUM_FRAMES_PER_ACTION=default_cfg.NUM_FRAMES_PER_ACTION,
        ACTION_FRAME_POLICY=default_cfg.ACTION_FRAME_POLICY,
        NUM_DEVICES=default_cfg.NUM_DEVICES,
        NUM_ACT=default_cfg.NUM_ACT,
        NUM_CAT=default_cfg.NUM_CAT,
        NODE_DIM=default_cfg.NODE_DIM,
        NUM_WORKERS=default_cfg.NUM_WORKERS,
    )


def build_loso_split(test_subj, all_subs):
    test_idx = all_subs.index(test_subj)
    val_idx = (test_idx - 1) % len(all_subs)
    exclude = {test_subj, all_subs[val_idx]}
    return {
        'test':  [test_subj],
        'val':   [all_subs[val_idx]],
        'train': [s for s in all_subs if s not in exclude]
    }


def topk_accuracy(pred_probs, true_labels, k=3):
    if len(true_labels) == 0:
        return 0.0
    topk_preds = np.argsort(pred_probs, axis=1)[:, -k:][:, ::-1]
    correct = sum(t in topk for t, topk in zip(true_labels, topk_preds))
    return correct / len(true_labels)


def plot_confusion_matrix(y_true, y_pred, labels, title, save_path=None):
    font_config = {'family': 'serif', 'size': 13, 'weight': 'bold'}
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title, fontdict=font_config)
    ax.set_xlabel('Predicted label', fontdict=font_config)
    ax.set_ylabel('True label', fontdict=font_config)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"  混淆矩阵已保存: {save_path}")
    plt.show()
    plt.close()


def run_cv_test():
    cfg = build_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_subjects = cfg.ALL_SUBJECTS
    n_folds = len(test_subjects)

    print(f"\n{'='*85}")
    print(f" {n_folds} 折批量测试 | 模型: {MODEL_TYPE} | 消融: {ABLATION_MODE} | 设备: {device}")
    print(f" 归档目录: {JOB_NAME}")
    print(f" 测试被试: {test_subjects}")
    print(f"{'='*85}")

    with open(cfg.DEVICE_CFG_PATH, "r") as f:
        device_config = json.load(f)

    base_ckpt_dir = cfg.CKPT_DIR / JOB_NAME
    fold_results = []

    global_act_true, global_act_pred = [], []
    global_dev_true, global_dev_pred = [], []

    for idx, subj in enumerate(test_subjects):
        print(f"\n[{idx+1}/{n_folds}] {subj} 正在准备测试...")

        search_pattern = str(base_ckpt_dir / f"{subj}_*" / "best_model.pth")
        found_ckpts = glob.glob(search_pattern)
        if not found_ckpts:
            print(f" 未找到权重, 已跳过 (搜索: {search_pattern})")
            continue

        ckpt_path = found_ckpts[0]
        print(f"  -> 权重: {Path(ckpt_path).parent.name}/best_model.pth")

        current_split = build_loso_split(subj, cfg.ALL_SUBJECTS)

        _, _, test_set = DriverHOIDataset.split_dataset(
            data_root=cfg.DATA_ROOT, split_cfg=current_split,
            camera_views=cfg.CAMERA_VIEWS, frame_sampling=cfg.FRAME_SAMPLING,
            num_frames_per_action=cfg.NUM_FRAMES_PER_ACTION,
            action_frame_policy=cfg.ACTION_FRAME_POLICY,
            transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        )
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

        model = create_model(cfg).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        s_act_t, s_act_p = [], []
        s_dev_t, s_dev_p, s_dev_probs = [], [], []
        s_int_t, s_int_p = [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Testing {subj}", ncols=100):
                hand, dev_geom, dev_cat, f_hand_roi, f_dev_roi = model.prepare_inputs(batch, device_config)
                hand, dev_geom, dev_cat = hand.to(device), dev_geom.to(device), dev_cat.to(device)
                if f_hand_roi is not None: f_hand_roi = f_hand_roi.to(device)
                if f_dev_roi  is not None: f_dev_roi  = f_dev_roi.to(device)

                out = model(hand, dev_geom, dev_cat, f_hand_roi, f_dev_roi)
                dev_prob = torch.softmax(out["dev_logits_raw"], dim=-1).cpu().numpy()

                s_act_t.extend(batch["action_label"].cpu().numpy().astype(int))
                s_act_p.extend(out["act_logits"].argmax(dim=-1).cpu().numpy())
                s_dev_t.extend(batch["device_id"].cpu().numpy().astype(int))
                s_dev_p.extend(dev_prob.argmax(axis=-1))
                s_int_t.extend(batch["interaction_flag"].cpu().numpy().astype(int))
                s_int_p.extend((1.0 - dev_prob[:, 0] > 0.5).astype(int))
                s_dev_probs.extend(dev_prob[:, 1:])

        s_act_t, s_act_p = np.array(s_act_t), np.array(s_act_p)
        s_dev_t, s_dev_p = np.array(s_dev_t), np.array(s_dev_p)
        s_int_t, s_int_p = np.array(s_int_t), np.array(s_int_p)
        s_dev_probs = np.array(s_dev_probs)

        int_acc = (s_int_t == s_int_p).mean()
        act_acc = (s_act_t == s_act_p).mean()
        dev_acc = (s_dev_t == s_dev_p).mean()
        overall = ((s_act_t == s_act_p) & (s_dev_t == s_dev_p)).mean()
        mask = s_dev_t > 0
        topk = topk_accuracy(s_dev_probs[mask], s_dev_t[mask] - 1, k=TOP_K) if mask.sum() > 0 else 0.0

        print(f"  -> Int: {int_acc:.4f} | Act: {act_acc:.4f} | Top-1: {dev_acc:.4f} | Top-{TOP_K}: {topk:.4f} | Overall: {overall:.4f}")
        fold_results.append(dict(Subject=subj, Int=int_acc, Act=act_acc, Top1=dev_acc, Top3=topk, Overall=overall))

        global_act_true.extend(s_act_t)
        global_act_pred.extend(s_act_p)
        global_dev_true.extend(s_dev_t)
        global_dev_pred.extend(s_dev_p)

    if not fold_results:
        print("\n无测试数据, 请检查路径。")
        return

    print(f"\n{'='*85}")
    print(f" 最终报告 | 模型: {MODEL_TYPE} | 消融: {ABLATION_MODE} | {len(fold_results)} 折")
    print(f"{'='*85}")
    print(f"{'Subject':<12} | {'Int':<10} | {'Act':<10} | {'Top-1':<10} | {'Top-3':<10} | {'Overall':<10}")
    print("-" * 85)
    for r in fold_results:
        print(f"{r['Subject']:<12} | {r['Int']:.4f}     | {r['Act']:.4f}     | {r['Top1']:.4f}      | {r['Top3']:.4f}      | {r['Overall']:.4f}")
    print("-" * 85)

    for key in ['Int', 'Act', 'Top1', 'Top3', 'Overall']:
        vals = [r[key] for r in fold_results]
        if key == 'Int':
            print(f"{'Mean':<12} | {np.mean(vals):.4f}     |", end=" ")
        elif key == 'Overall':
            print(f"{np.mean(vals):.4f}")
        else:
            print(f"{np.mean(vals):.4f}      |", end=" ")

    for key in ['Int', 'Act', 'Top1', 'Top3', 'Overall']:
        vals = [r[key] for r in fold_results]
        if key == 'Int':
            print(f"{'Std':<12} | {np.std(vals):.4f}     |", end=" ")
        elif key == 'Overall':
            print(f"{np.std(vals):.4f}")
        else:
            print(f"{np.std(vals):.4f}      |", end=" ")

    print(f"{'='*85}")

    global_act_true = np.array(global_act_true)
    global_act_pred = np.array(global_act_pred)
    global_dev_true = np.array(global_dev_true)
    global_dev_pred = np.array(global_dev_pred)

    n_tested = len(fold_results)
    print(f"\n 混淆矩阵统计 | 共 {n_tested} 折, {len(global_act_true)} 个样本")

    save_dir = Path(CM_SAVE_DIR) if CM_SAVE_DIR else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    act_labels = [str(i) for i in range(cfg.NUM_ACT)]
    dev_labels = [str(i) for i in range(cfg.NUM_DEVICES + 1)]

    plot_confusion_matrix(
        global_act_true, global_act_pred, act_labels,
        title=f"Action Confusion Matrix",
        save_path=save_dir / f"action_cm_{MODEL_TYPE}_{ABLATION_MODE}.pdf" if save_dir else None)

    plot_confusion_matrix(
        global_dev_true, global_dev_pred, dev_labels,
        title=f"Device Confusion Matrix",
        save_path=save_dir / f"device_cm_{MODEL_TYPE}_{ABLATION_MODE}.pdf" if save_dir else None)

    print()


if __name__ == "__main__":
    run_cv_test()