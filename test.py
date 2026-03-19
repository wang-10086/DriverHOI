import json
import time
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from types import SimpleNamespace

from DriverHOIDatasets import DriverHOIDataset
from models import create_model
from utils import (
    visualize_all_on_image,
    visualize_3d_keypoints,
    draw_heatmap_on_image,
    plot_device_probability,
    to_numpy
)
import config as default_cfg

CKPT_PATH = "checkpoints/driverhoi_exp02/subject5_20260310_003518/best_model.pth"
MODEL_TYPE = "DriverHOI"        # 'DriverHOI' | 'MLP-HOI' | 'TransHOI' | 'SCG-HOI'
ABLATION_MODE = "baseline"      # 'baseline' | 'with_visual' | 'no_pose' | 'no_geom'
TEST_SUBJ = "subject3"
BATCH_SIZE = 16
TOP_K = 3
SAVE_JSON = False

SINGLE_SAMPLE_INDEX = 120   # None or test_sample_index

VIS_SAVE_DIR = "results"        # None or save_dir

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 12,
})


def build_loso_split(test_subj):
    all_subs = default_cfg.ALL_SUBJECTS
    if test_subj not in all_subs:
        raise ValueError(f"被试 {test_subj} 不在列表 {all_subs} 中")
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
    plt.show()
    plt.close()


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = build_loso_split(TEST_SUBJ)
    print(f" [LOSO] Test: {split['test']} | Val: {split['val']} | Train: {split['train']}")

    _, _, test_set = DriverHOIDataset.split_dataset(
        data_root=default_cfg.DATA_ROOT, split_cfg=split,
        camera_views=default_cfg.CAMERA_VIEWS, frame_sampling=default_cfg.FRAME_SAMPLING,
        num_frames_per_action=default_cfg.NUM_FRAMES_PER_ACTION,
        action_frame_policy=default_cfg.ACTION_FRAME_POLICY,
        transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    )
    print(f" 测试集样本数: {len(test_set)}\n")

    with open(default_cfg.DEVICE_CFG_PATH, "r") as f:
        device_config = json.load(f)

    cfg = SimpleNamespace(
        MODEL_TYPE=MODEL_TYPE,
        ABLATION_MODE=ABLATION_MODE,
        NUM_ACT=default_cfg.NUM_ACT,
        NUM_CAT=default_cfg.NUM_CAT,
        NODE_DIM=default_cfg.NODE_DIM,
        NUM_DEVICES=default_cfg.NUM_DEVICES,
    )
    model = create_model(cfg).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    if SINGLE_SAMPLE_INDEX is not None:
        batches = [default_collate([test_set[SINGLE_SAMPLE_INDEX]])]
    else:
        batches = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=default_cfg.NUM_WORKERS)

    results = []
    all_act_true, all_act_pred = [], []
    all_dev_true, all_dev_pred, all_dev_probs = [], [], []
    all_int_true, all_int_pred = [], []

    pbar = tqdm(batches, desc=f"Testing {TEST_SUBJ}", ncols=140)
    for batch in pbar:
        with torch.no_grad():
            hand, dev_geom, dev_cat, f_hand_roi, f_dev_roi = model.prepare_inputs(batch, device_config)
            hand, dev_geom, dev_cat = hand.to(device), dev_geom.to(device), dev_cat.to(device)
            if f_hand_roi is not None: f_hand_roi = f_hand_roi.to(device)
            if f_dev_roi  is not None: f_dev_roi  = f_dev_roi.to(device)

            out = model(hand, dev_geom, dev_cat, f_hand_roi, f_dev_roi)

            dev_prob = torch.softmax(out["dev_logits_raw"], dim=-1).cpu().numpy()
            act_true = batch["action_label"].cpu().numpy().astype(int)
            dev_true = batch["device_id"].cpu().numpy().astype(int)
            inter_true = batch["interaction_flag"].cpu().numpy().astype(int)
            act_pred = out["act_logits"].argmax(dim=-1).cpu().numpy()
            dev_pred = dev_prob.argmax(axis=-1)
            inter_pred = (1.0 - dev_prob[:, 0] > 0.5).astype(int)

            all_act_true.extend(act_true)
            all_act_pred.extend(act_pred)
            all_dev_true.extend(dev_true)
            all_dev_pred.extend(dev_pred)
            all_int_true.extend(inter_true)
            all_int_pred.extend(inter_pred)
            all_dev_probs.extend(dev_prob[:, 1:])

            for i in range(len(act_true)):
                results.append({
                    "img_path": batch["img_path"][i],
                    "action_true": int(act_true[i]), "action_pred": int(act_pred[i]),
                    "device_true": int(dev_true[i]), "device_pred": int(dev_pred[i]),
                })

            pbar.set_postfix_str(
                f"Act{{g:{act_true[0]},p:{act_pred[0]}}} "
                f"Dev{{g:{dev_true[0]},p:{dev_pred[0]}}} "
                f"Int{{g:{inter_true[0]},p:{inter_pred[0]}}}")

            if SINGLE_SAMPLE_INDEX is not None:
                save_dir = Path(VIS_SAVE_DIR) if VIS_SAVE_DIR else None
                if save_dir:
                    save_dir.mkdir(parents=True, exist_ok=True)
                prefix = f"{TEST_SUBJ}_sample{SINGLE_SAMPLE_INDEX}"

                visualize_all_on_image(
                    batch["img_path"][0],
                    batch["keypoints3d"]["body_keypoints3d"][0],
                    batch["keypoints3d"]["hands_keypoints3d"][0],
                    default_cfg.DEVICE_CFG_PATH,
                    batch["camera_intrinsic"], batch["camera_extrinsic"],
                    save_path=str(save_dir / f"{prefix}_1_overlay.jpg") if save_dir else None)

                visualize_3d_keypoints(
                    body_keypoints3d=to_numpy(batch["keypoints3d"]["body_keypoints3d"][0]),
                    hands_keypoints3d=[to_numpy(h) for h in batch["keypoints3d"]["hands_keypoints3d"][0]],
                    device_json_path=default_cfg.DEVICE_CFG_PATH,
                    save_path=str(save_dir / f"{prefix}_2_3d.jpg") if save_dir else None)

                draw_heatmap_on_image(
                    batch["img_path"][0], dev_prob[0, 1:],
                    default_cfg.DEVICE_CFG_PATH,
                    batch["camera_intrinsic"], batch["camera_extrinsic"],
                    sigma=35, alpha=0.5,
                    save_path=str(save_dir / f"{prefix}_3_heatmap.jpg") if save_dir else None)

                plot_device_probability(
                    dev_prob[0, 1:], default_cfg.DEVICE_CFG_PATH,
                    int(dev_true[0]), circle_radius=0.03, cmap="Reds", annotate=True,
                    save_path=str(save_dir / f"{prefix}_4_devprob.jpg") if save_dir else None)

    if SINGLE_SAMPLE_INDEX is not None:
        return

    all_act_true, all_act_pred = np.array(all_act_true), np.array(all_act_pred)
    all_dev_true, all_dev_pred = np.array(all_dev_true), np.array(all_dev_pred)
    all_int_true, all_int_pred = np.array(all_int_true), np.array(all_int_pred)

    n = len(all_act_true)
    int_acc = (all_int_true == all_int_pred).mean()
    act_acc = (all_act_true == all_act_pred).mean()
    dev_acc = (all_dev_true == all_dev_pred).mean()
    overall = ((all_act_true == all_act_pred) & (all_dev_true == all_dev_pred)).mean()

    mask_valid = all_dev_true > 0
    topk_acc = topk_accuracy(
        np.array(all_dev_probs)[mask_valid],
        all_dev_true[mask_valid] - 1, k=TOP_K
    ) if mask_valid.sum() > 0 else 0.0

    print(f"\n{'='*60}")
    print(f" 测试结果 | {TEST_SUBJ} | {MODEL_TYPE} ({ABLATION_MODE})")
    print(f"{'='*60}")
    print(f"  样本数:       {n}")
    print(f"  Int ACC:      {int_acc:.4f}")
    print(f"  Action ACC:   {act_acc:.4f}")
    print(f"  Device Top-1: {dev_acc:.4f}")
    print(f"  Device Top-{TOP_K}: {topk_acc:.4f}")
    print(f"  Overall ACC:  {overall:.4f}")
    print(f"{'='*60}\n")

    plot_confusion_matrix(
        all_act_true, all_act_pred,
        [str(i) for i in range(default_cfg.NUM_ACT)],
        title=f"Action Confusion Matrix ({TEST_SUBJ})")
    plot_confusion_matrix(
        all_dev_true, all_dev_pred,
        [str(i) for i in range(default_cfg.NUM_DEVICES + 1)],
        title=f"Device Confusion Matrix ({TEST_SUBJ})")

    if SAVE_JSON:
        out_path = Path(CKPT_PATH).with_name(f"test_results_{TEST_SUBJ}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  结果已保存: {out_path}")


if __name__ == "__main__":
    test()