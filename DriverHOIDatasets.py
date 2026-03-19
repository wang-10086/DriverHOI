import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Subset

from utils import (
    visualize_3d_keypoints_on_image,
    visualize_devices_on_image,
    print_sample_info,
    visualize_3d_keypoints,
    plot_dataset_sankey
)


class DriverHOIDataset(Dataset):
    def __init__(self,
                 data_root: str,
                 camera_views: Optional[List[str]] = None,
                 frame_sampling: str = 'random',
                 num_frames_per_action: int = 1,
                 transform=None,
                 target_size: Tuple[int, int] = (224, 224),
                 action_frame_policy: Optional[Dict[str, int]] = None
                 ):

        self.data_root = Path(data_root)
        self.transform = transform
        self.target_size = target_size
        self.frame_sampling = frame_sampling
        self.num_frames_per_action = num_frames_per_action
        self.action_frame_policy = action_frame_policy or {}

        self.available_views = ['MBP25030012', 'MBP25030014', 'MBP25030016', 'MBP25030017']
        self.camera_views = camera_views if camera_views is not None else self.available_views

        self.action_to_idx = {'point': 0, 'press': 1, 'push': 2, 'swing': 3}
        self.idx_to_action = {v: k for k, v in self.action_to_idx.items()}

        self.camera_intrinsics, self.camera_extrinsics = self._load_calibration_files(self.data_root / "calibration")
        print("已加载相机内外参配置：")
        print(f"  内参相机数: {len(self.camera_intrinsics)}  外参相机数: {len(self.camera_extrinsics)}")

        print("扫描数据文件夹中...")
        self.subjects = self._discover_subjects()
        self.actions = self._discover_actions()
        self.data_index = self._build_data_index()

        print(f"数据集初始化完成")
        print(f"  被试数量: {len(self.subjects)}")
        print(f"  动作类别: {self.actions}")
        print(f"  视角选择: {self.camera_views}")
        print(f"  样本总数: {len(self.data_index)}")

    def _load_calibration_files(self, calib_dir: Path):
        intri_path = calib_dir / "intri.yml"
        extri_path = calib_dir / "extri.yml"
        intrinsics, extrinsics = {}, {}

        if intri_path.exists():
            fs = cv2.FileStorage(str(intri_path), cv2.FILE_STORAGE_READ)
            if not fs.isOpened():
                print(f"无法打开内参文件: {intri_path}")
            else:
                names_node = fs.getNode("names")
                cam_names = []
                if not names_node.empty():
                    for i in range(int(names_node.size())):
                        cam_names.append(names_node.at(i).string())

                for cam in cam_names:
                    K_node = fs.getNode(f"K_{cam}")
                    D_node = fs.getNode(f"dist_{cam}")
                    if K_node.empty():
                        print(f" {cam} 缺少 K_{cam}，跳过该相机内参")
                        continue
                    K = K_node.mat()
                    if K is None:
                        print(f"️ {cam} 的 K_{cam} 读取为空，跳过")
                        continue
                    if D_node.empty() or D_node.mat() is None:
                        D = np.zeros((5,), dtype=np.float32)
                    else:
                        D = D_node.mat().astype(np.float32).flatten()

                    intrinsics[cam] = {
                        "K": np.array(K, dtype=np.float32).reshape(3, 3),
                        "D": D
                    }
            fs.release()
        else:
            print(" 未找到 intri.yml 文件")

        if extri_path.exists():
            fs = cv2.FileStorage(str(extri_path), cv2.FILE_STORAGE_READ)
            if not fs.isOpened():
                print(f" 无法打开外参文件: {extri_path}")
            else:
                names_node = fs.getNode("names")
                cam_names = []
                if not names_node.empty():
                    for i in range(int(names_node.size())):
                        cam_names.append(names_node.at(i).string())

                for cam in cam_names:
                    Rm_node = fs.getNode(f"Rot_{cam}")
                    rvec_node = fs.getNode(f"R_{cam}")
                    T_node = fs.getNode(f"T_{cam}")

                    R = None
                    if not Rm_node.empty() and Rm_node.mat() is not None:
                        R = Rm_node.mat().astype(np.float32).reshape(3, 3)
                    elif not rvec_node.empty() and rvec_node.mat() is not None:
                        rvec = rvec_node.mat().astype(np.float32).reshape(3, 1)
                        R, _ = cv2.Rodrigues(rvec)
                    else:
                        print(f" {cam} 缺少 Rot_{cam} 与 R_{cam}，无法得到旋转矩阵，跳过")
                        continue

                    if T_node.empty() or T_node.mat() is None:
                        print(f" {cam} 缺少 T_{cam}，使用零平移")
                        T = np.zeros((3,), dtype=np.float32)
                    else:
                        T = T_node.mat().astype(np.float32).reshape(-1)
                        if T.shape[0] != 3:
                            print(f" {cam} 的 T_{cam} 形状异常 {T.shape}，尝试扁平化为(3,)")
                            T = T.flatten()[:3].astype(np.float32)

                    extrinsics[cam] = {"R": R, "T": T}
            fs.release()
        else:
            print(" 未找到 extri.yml 文件")

        expected = set(getattr(self, "camera_views", [])) or set(intrinsics.keys()) | set(extrinsics.keys())
        for cam in sorted(expected):
            if cam not in intrinsics:
                print(f" 内参数缺失: {cam}")
            if cam not in extrinsics:
                print(f" 外参数缺失: {cam}")
            else:
                extrinsics[cam]["R"] = extrinsics[cam]["R"].astype(np.float32).reshape(3, 3)
                extrinsics[cam]["T"] = extrinsics[cam]["T"].astype(np.float32).reshape(3, )

        return intrinsics, extrinsics


    def _discover_subjects(self):
        return sorted([d.name for d in self.data_root.iterdir() if d.is_dir() and d.name.startswith("subject")])

    def _discover_actions(self):
        actions = set()
        for subj in self.subjects:
            subj_path = self.data_root / subj
            for act in subj_path.iterdir():
                if act.is_dir() and act.name in self.action_to_idx:
                    actions.add(act.name)
        return sorted(list(actions))

    def _get_frame_indices(self, total_frames: int, action: Optional[str] = None) -> List[int]:
        if action is not None and action in self.action_frame_policy:
            k = max(1, int(self.action_frame_policy[action]))
        else:
            k = max(1, int(self.num_frames_per_action))

        k = min(k, total_frames) if total_frames > 0 else 0
        if k == 0:
            return []

        if self.frame_sampling == 'center':
            c = total_frames // 2
            s = max(0, c - k // 2)
            e = min(total_frames, s + k)
            idxs = list(range(s, e))
            while len(idxs) < k:
                if s > 0:
                    s -= 1
                    idxs.insert(0, s)
                elif e < total_frames:
                    idxs.append(e)
                    e += 1
                else:
                    break
            return sorted(set(idxs))[:k]

        elif self.frame_sampling == 'uniform':
            if k >= total_frames:
                return list(range(total_frames))
            step = total_frames / float(k)
            idxs = [int(i * step) for i in range(k)]
            idxs = sorted({min(total_frames - 1, max(0, x)) for x in idxs})
            while len(idxs) < k:
                cand = np.random.randint(0, total_frames)
                if cand not in idxs:
                    idxs.append(cand)
            return sorted(idxs)

        elif self.frame_sampling == 'random':
            if k >= total_frames:
                return list(range(total_frames))
            idxs = np.random.choice(total_frames, k, replace=False)
            return sorted(idxs.tolist())

        else:
            return [total_frames // 2]

    def _build_data_index(self) -> List[Dict]:
        data_index = []
        for subj in self.subjects:
            subj_path = self.data_root / subj
            for action in self.actions:
                act_path = subj_path / action
                if not act_path.exists():
                    continue
                for dev_dir in act_path.iterdir():
                    if not dev_dir.is_dir() or not dev_dir.name.startswith("device"):
                        continue
                    try:
                        device_id = int(dev_dir.name.replace("device", "").replace("_", ""))
                    except ValueError:
                        continue
                    interaction_flag = 0 if device_id == 0 else 1

                    json_dir = dev_dir / "post_json"

                    for view_name in self.camera_views:
                        view_path = dev_dir / view_name
                        if not view_path.exists():
                            continue
                        img_files = sorted([f for f in view_path.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg"]])
                        if not img_files:
                            continue

                        frame_indices = self._get_frame_indices(len(img_files), action=action)
                        for frame_idx in frame_indices:
                            img_path = img_files[frame_idx]
                            json_name = img_path.stem + ".json"
                            json_path = json_dir / json_name if json_dir.exists() else None

                            data_index.append({
                                "subject": subj,
                                "action": action,
                                "device_id": device_id,
                                "interaction_flag": interaction_flag,
                                "view_name": view_name,
                                "frame_idx": frame_idx,
                                "img_path": img_path,
                                "json_path": json_path
                            })
        return data_index

    def _load_image(self, img_path: Path):
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.new("RGB", self.target_size, (0, 0, 0))
        return img

    def _load_keypoints3d(self, json_path: Optional[Path]) -> Dict[str, torch.Tensor]:
        if json_path is None or not json_path.exists():
            body = np.zeros((25, 3), dtype=np.float32)
            hands = np.zeros((2, 21, 3), dtype=np.float32)
            return {
                "body_keypoints3d": torch.from_numpy(body),
                "hands_keypoints3d": torch.from_numpy(hands)
            }

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            kp = data.get("keypoints3d", {})

            body = np.array(kp.get("BODY", np.zeros((25, 3))), dtype=np.float32)
            left = np.array(kp.get("LeftHand", np.zeros((21, 3))), dtype=np.float32)
            right = np.array(kp.get("RightHand", np.zeros((21, 3))), dtype=np.float32)
            hands = np.stack([left, right], axis=0)  # [2, 21, 3]

            body = body[..., [2, 0, 1]]
            hands = hands[..., [2, 0, 1]]

        except Exception as e:
            print(f" 加载关键点失败: {json_path} ({e})")
            body = np.zeros((25, 3), dtype=np.float32)
            hands = np.zeros((2, 21, 3), dtype=np.float32)

        return {
            "body_keypoints3d": torch.from_numpy(body),
            "hands_keypoints3d": torch.from_numpy(hands)
        }

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx: int):
        info = self.data_index[idx]
        img = self._load_image(info["img_path"])
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        keypoints3d = self._load_keypoints3d(info["json_path"])

        view_name = info["view_name"]
        intri = self.camera_intrinsics.get(view_name, {'K': np.eye(3, dtype=np.float32), 'D': np.zeros(5, np.float32)})
        extri = self.camera_extrinsics.get(view_name, {'R': np.eye(3, dtype=np.float32), 'T': np.zeros(3, np.float32)})
        intri_t = {'K': torch.tensor(intri['K'], dtype=torch.float32),
                   'D': torch.tensor(intri['D'], dtype=torch.float32)}
        extri_t = {'R': torch.tensor(extri['R'], dtype=torch.float32),
                   'T': torch.tensor(extri['T'], dtype=torch.float32)}

        return {
            "image": img,
            "action_label": torch.tensor(self.action_to_idx[info["action"]], dtype=torch.long),
            "interaction_flag": torch.tensor(info["interaction_flag"], dtype=torch.long),
            "device_id": torch.tensor(info["device_id"], dtype=torch.long),
            "subject": info["subject"],
            "action": info["action"],
            "view": view_name,
            "img_path": str(info["img_path"]),
            "keypoints3d": keypoints3d,  # body [25,3], hands [2,21,3]
            "camera_intrinsic": {'K': torch.from_numpy(intri['K']), 'D': torch.from_numpy(intri['D'])},
            "camera_extrinsic": {'R': torch.from_numpy(extri['R']), 'T': torch.from_numpy(extri['T'])}
        }

    @classmethod
    def split_dataset(cls, data_root, split_cfg, **dataset_kwargs):
        full_dataset = cls(data_root=data_root, **dataset_kwargs)

        idxs = {"train": [], "val": [], "test": []}
        for i, entry in enumerate(full_dataset.data_index):
            subj = entry["subject"]
            for split_name, subj_list in split_cfg.items():
                if subj in subj_list:
                    idxs[split_name].append(i)

        subsets = {k: Subset(full_dataset, v) for k, v in idxs.items()}
        return subsets.get("train"), subsets.get("val"), subsets.get("test")

    def get_class_distribution(self) -> Dict:
        action_counts, device_counts, view_counts = {}, {}, {}
        inter_counts = {'non_interaction': 0, 'interaction': 0}
        for s in self.data_index:
            action_counts[s['action']] = action_counts.get(s['action'], 0) + 1
            device_counts[s['device_id']] = device_counts.get(s['device_id'], 0) + 1
            view_counts[s['view_name']] = view_counts.get(s['view_name'], 0) + 1
            if s['interaction_flag']:
                inter_counts['interaction'] += 1
            else:
                inter_counts['non_interaction'] += 1
        return {
            'actions': dict(sorted(action_counts.items())),
            'devices': dict(sorted(device_counts.items())),
            'views': dict(sorted(view_counts.items())),
            'interaction_types': inter_counts
        }

    def plot_distribution(self, save_path: Optional[str] = None):
        stats = self.get_class_distribution()
        fig, axs = plt.subplots(2, 2, figsize=(14, 8))

        ax = axs[0, 0]
        actions = list(stats['actions'].keys())
        counts = list(stats['actions'].values())
        bars = ax.bar(actions, counts, color='skyblue')
        ax.set_title('Action Distribution')
        ax.set_ylabel('Count')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{int(height)}",
                    ha='center', va='bottom', fontsize=9)

        ax = axs[0, 1]
        dev_keys = [str(k) for k in stats['devices'].keys()]
        dev_counts = list(stats['devices'].values())
        bars = ax.bar(dev_keys, dev_counts, color='lightgreen')
        ax.set_title('Device Distribution')
        ax.set_ylabel('Count')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{int(height)}",
                    ha='center', va='bottom', fontsize=8, rotation=90)

        ax = axs[1, 0]
        views = list(stats['views'].keys())
        vcounts = list(stats['views'].values())
        bars = ax.bar(views, vcounts, color='salmon')
        ax.set_title('View Distribution')
        ax.set_ylabel('Count')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{int(height)}",
                    ha='center', va='bottom', fontsize=9)

        ax = axs[1, 1]
        types = list(stats['interaction_types'].keys())
        tcounts = list(stats['interaction_types'].values())
        bars = ax.bar(types, tcounts, color='orange')
        ax.set_title('Interaction Type Distribution')
        ax.set_ylabel('Count')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{int(height)}",
                    ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存至: {save_path}")
        else:
            plt.show()

    def plot_sankey_distribution(self, save_path=None):
        print(f"正在生成桑基图...")
        plot_dataset_sankey(self.data_index, save_path=save_path)
