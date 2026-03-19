import cv2
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tabulate import tabulate
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from pathlib import Path
import plotly.graph_objects as go
import pandas as pd

body_connections = [
    (0, 15), (15, 17), (0, 16), (16, 18), (0, 2), (0, 5),
    (2, 3), (3, 4), (5, 6), (6, 7), (2, 5), (2, 9), (5, 12), (9, 12)
]

hand_connections = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

def to_numpy(x):
    if x is None: return None
    if hasattr(x, 'cpu'): return x.cpu().numpy()
    return np.array(x, dtype=np.float32)

def project_3d_to_2d(points_3d, K, D, R, T):
    points_3d, K, D, R, T = map(lambda x: np.asarray(x, dtype=np.float32), (points_3d, K, D, R, T))
    if points_3d.ndim == 2: points_3d = points_3d[None, ...] 
    if K.ndim == 2: K = K[None, ...]
    if D.ndim == 1: D = D[None, ...]
    if R.ndim == 2: R = R[None, ...]
    if T.ndim == 1: T = T[None, :, None]
    elif T.ndim == 2 and T.shape[-1] != 1: T = T[..., None]

    B, N, _ = points_3d.shape
    pts_2d_all = []

    for b in range(B):
        p3 = points_3d[b]
        Kb, Db, Rb, Tb = K[min(b, len(K) - 1)], D[min(b, len(D) - 1)], R[min(b, len(R) - 1)], T[min(b, len(T) - 1)]

        cam_points = (Rb @ p3.T + Tb).T
        X, Y, Z = cam_points[:, 0], cam_points[:, 1], cam_points[:, 2]
        Z[Z == 0] = 1e-6
        x, y = X / Z, Y / Z

        k1, k2, p1, p2, k3 = (Db.tolist() + [0, 0, 0, 0, 0])[:5]
        r2 = x ** 2 + y ** 2
        radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        x_dist = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
        y_dist = y * radial + p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y

        fx, fy, cx, cy = Kb[0, 0], Kb[1, 1], Kb[0, 2], Kb[1, 2]
        u, v = fx * x_dist + cx, fy * y_dist + cy
        pts_2d_all.append(np.stack([u, v], axis=-1))

    pts_2d_all = np.stack(pts_2d_all, axis=0)
    if pts_2d_all.shape[0] == 1: pts_2d_all = pts_2d_all[0]
    return pts_2d_all

def format_matrix(mat, precision=4):
    return "\n".join(["[" + "  ".join([f"{v:8.{precision}f}" for v in row]) + "]" for row in mat])

def get_tip_point(hand):
    wrist = hand[:, 0, :]
    v_finger = hand[:, 9, :] - wrist
    v_finger_norm = v_finger / (v_finger.norm(dim=-1, keepdim=True) + 1e-6)
    rel = hand - wrist.unsqueeze(1)
    proj = (rel * v_finger_norm.unsqueeze(1)).sum(dim=-1)
    idx = proj.argmax(dim=-1)
    batch_idx = torch.arange(hand.size(0), device=hand.device)
    return hand[batch_idx, idx, :]

def save_config_module_as_json(config_module, save_dir):
    raw_dict = {k: v for k, v in vars(config_module).items() if not k.startswith("_") and not callable(v)}
    final_dict = {}
    for k, v in raw_dict.items():
        try:
            json.dumps(v)
            final_dict[k] = v
        except (TypeError, OverflowError):
            final_dict[k] = str(v)

    save_path = Path(save_dir) / "experiment_config.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_dict, f, indent=4, ensure_ascii=False)

def visualize_3d_keypoints_on_image(image_path, body_keypoints3d, hands_keypoints3d, camera_intrinsic, camera_extrinsic, save_path=None):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]
    K, D = to_numpy(camera_intrinsic["K"]), to_numpy(camera_intrinsic.get("D", np.zeros(5)))
    R, T = to_numpy(camera_extrinsic["R"]), to_numpy(camera_extrinsic["T"])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)

    if body_keypoints3d is not None:
        body_3d = to_numpy(body_keypoints3d)
        body_2d_all = project_3d_to_2d(body_3d, K, D, R, T)
        used_indices = set([i for conn in body_connections for i in conn])
        body_2d = np.array([body_2d_all[i] for i in sorted(used_indices) if i < len(body_2d_all)])
        ax.scatter(body_2d[:, 0], body_2d[:, 1], s=20, color="red")
        for i, j in body_connections:
            if i < len(body_2d_all) and j < len(body_2d_all):
                p1, p2 = body_2d_all[i], body_2d_all[j]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="blue", linewidth=1.5)

    if hands_keypoints3d is not None:
        for hand_3d in hands_keypoints3d:
            hand_2d = project_3d_to_2d(hand_3d[:21, :], K, D, R, T)
            ax.scatter(hand_2d[:, 0], hand_2d[:, 1], s=5, color="red")
            for i, j in hand_connections:
                if i < len(hand_2d) and j < len(hand_2d):
                    p1, p2 = hand_2d[i], hand_2d[j]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="blue", linewidth=1)

    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0); ax.set_position([0, 0, 1, 1])

    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show(); plt.close()

def visualize_devices_on_image(image_path, camera_intrinsic, camera_extrinsic, device_json_path, save_path=None):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]
    K, D = to_numpy(camera_intrinsic['K']), to_numpy(camera_intrinsic.get('D', np.zeros(5)))
    R, T = to_numpy(camera_extrinsic['R']), to_numpy(camera_extrinsic['T'])

    with open(device_json_path, "r", encoding="utf-8") as f: devices = json.load(f)

    ids, coords = [], []
    for dev_id, info in devices.items():
        coords.append([info["x"] / 1000, info["y"] / 1000, info["z"] / 1000])
        ids.append(dev_id)
    pts_2d = project_3d_to_2d(np.array(coords, dtype=np.float32), K, D, R, T)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    ax.scatter(pts_2d[:, 0], pts_2d[:, 1], s=40, c="yellow", edgecolors="black")

    for dev_id, (u, v) in zip(ids, pts_2d): ax.text(u + 5, v - 5, f"ID:{dev_id}", color="white", fontsize=9)

    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0); ax.set_position([0, 0, 1, 1])

    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show(); plt.close()

def visualize_all_on_image(image_path, body_keypoints3d, hands_keypoints3d, device_json_path, camera_intrinsic, camera_extrinsic, save_path=None):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None: return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    K, D = to_numpy(camera_intrinsic['K']), to_numpy(camera_intrinsic.get('D', np.zeros(5)))
    R, T = to_numpy(camera_extrinsic['R']), to_numpy(camera_extrinsic['T'])

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(img_rgb)

    SIZE_DEVICE, SIZE_BODY, SIZE_HAND = 180, 80, 30
    LINE_WIDTH_BODY, LINE_WIDTH_HAND, FONT_SIZE = 3.0, 1.5, 16

    if device_json_path:
        with open(device_json_path, "r", encoding="utf-8") as f: devices = json.load(f)
        ids, coords = [], []
        for dev_id, info in devices.items():
            ids.append(dev_id)
            coords.append([info["x"] / 1000, info["y"] / 1000, info["z"] / 1000])
        if coords:
            dev_2d = project_3d_to_2d(np.array(coords, dtype=np.float32), K, D, R, T)
            ax.scatter(dev_2d[:, 0], dev_2d[:, 1], s=SIZE_DEVICE, c="yellow", edgecolors="black", linewidths=1.5, zorder=2)
            for dev_id, (u, v) in zip(ids, dev_2d):
                if 0 <= u < W and 0 <= v < H:
                    ax.text(u + 8, v - 8, f"{dev_id}", color="white", fontsize=FONT_SIZE, fontweight='bold', zorder=3)

    if body_keypoints3d is not None:
        body_2d_all = project_3d_to_2d(to_numpy(body_keypoints3d), K, D, R, T)
        used_indices = set([i for conn in body_connections for i in conn])
        valid_pts = [body_2d_all[i] for i in sorted(used_indices) if i < len(body_2d_all)]
        if valid_pts:
            ax.scatter(np.array(valid_pts)[:, 0], np.array(valid_pts)[:, 1], s=SIZE_BODY, color="red", edgecolors='white', linewidths=1, zorder=4)
        for i, j in body_connections:
            if i < len(body_2d_all) and j < len(body_2d_all):
                p1, p2 = body_2d_all[i], body_2d_all[j]
                if -100 < p1[0] < W + 100 and -100 < p1[1] < H + 100:
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="blue", linewidth=LINE_WIDTH_BODY, zorder=3)

    if hands_keypoints3d is not None:
        for hand_3d in hands_keypoints3d:
            hand_2d = project_3d_to_2d(to_numpy(hand_3d)[:21, :], K, D, R, T)
            ax.scatter(hand_2d[:, 0], hand_2d[:, 1], s=SIZE_HAND, color="red", zorder=5)
            for i, j in hand_connections:
                if i < len(hand_2d) and j < len(hand_2d):
                    p1, p2 = hand_2d[i], hand_2d[j]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="blue", linewidth=LINE_WIDTH_HAND, zorder=4)

    ax.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0); plt.margins(0, 0)

    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_3d_keypoints(body_keypoints3d=None, hands_keypoints3d=None, device_json_path=None, save_path=None):
    fig = plt.figure(figsize=(18, 16))
    ax = fig.add_subplot(111, projection="3d")
    all_points = []

    if body_keypoints3d is not None:
        body_3d_all = np.asarray(body_keypoints3d)
        used_indices = set(i for conn in body_connections for i in conn)
        body_3d = np.array([body_3d_all[i] for i in sorted(used_indices) if i < len(body_3d_all)])
        ax.scatter(body_3d[:, 0], body_3d[:, 1], body_3d[:, 2], color="blue", s=60)
        all_points.append(body_3d)
        for i, j in body_connections:
            if i < len(body_3d_all) and j < len(body_3d_all):
                p1, p2 = body_3d_all[i], body_3d_all[j]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="grey", linewidth=2.5)

    if hands_keypoints3d is not None:
        for hand_3d in hands_keypoints3d:
            hand_3d = np.asarray(hand_3d)
            ax.scatter(hand_3d[:, 0], hand_3d[:, 1], hand_3d[:, 2], color="blue", s=40)
            all_points.append(hand_3d)
            for i, j in hand_connections:
                if i < len(hand_3d) and j < len(hand_3d):
                    p1, p2 = hand_3d[i], hand_3d[j]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="grey", linewidth=2)

    if device_json_path is not None:
        with open(device_json_path, "r", encoding="utf-8") as f: devices = json.load(f)
        for key, info in devices.items():
            dev_id = int(key)
            x, y, z = info["x"] / 1000, info["y"] / 1000, info["z"] / 1000
            all_points.append([[x, y, z]])
            if 1 <= dev_id <= 25:
                theta = np.linspace(0, 2 * np.pi, 50)
                X, Y = x + 0.015 * np.cos(theta), y + 0.015 * np.sin(theta)
                ax.plot(X, Y, np.full_like(X, 0.0), color="red", alpha=0.6)
            elif 26 <= dev_id <= 31:
                L, Wd = 0.207, 0.15
                corners = np.array([[x, y-L/2, z-Wd/2], [x, y+L/2, z-Wd/2], [x, y+L/2, z+Wd/2], [x, y-L/2, z+Wd/2]])
                rect = Poly3DCollection([corners], facecolors="red", alpha=0.5, edgecolors="k", linewidths=0.5)
                ax.add_collection3d(rect)

    if all_points:
        all_points = np.vstack([np.array(p).reshape(-1, 3) for p in all_points])
        mins, maxs = np.min(all_points, axis=0), np.max(all_points, axis=0)
        centers = (maxs + mins) / 2
        max_range = (maxs - mins).max() / 3
        ax.set_xlim(centers[0] - max_range, centers[0] + max_range)
        ax.set_ylim(centers[1] - max_range, centers[1] + max_range)
        ax.set_zlim(centers[2] - max_range, centers[2] + max_range)
    else:
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=15, azim=-60)
    # ax.grid(False); ax.axis("off")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel("X", fontsize=10, labelpad=5)
    ax.set_ylabel("Y", fontsize=10, labelpad=5)
    ax.set_zlabel("Z", fontsize=10, labelpad=5)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor((0.95, 0.95, 0.95, 0.5))
    ax.yaxis.pane.set_facecolor((0.90, 0.90, 0.90, 0.5))
    ax.zaxis.pane.set_facecolor((0.85, 0.85, 0.85, 0.5))

    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show(); plt.close()

def plot_device_probability(dev_prob_1N, device_json_path, true_device_id=None, circle_radius=0.05, cmap="Reds", annotate=True, save_path=None):
    with open(device_json_path, "r", encoding="utf-8") as f: devices = json.load(f)
    ids, xs, ys = [], [], []
    for dev_id, info in devices.items():
        ids.append(int(dev_id))
        xs.append(info["x"] / 1000.0)
        ys.append(info["y"] / 1000.0)

    order = np.argsort(np.array(ids))
    ids, xs, ys = np.array(ids)[order], np.array(xs)[order], np.array(ys)[order]
    probs = np.asarray(dev_prob_1N, dtype=np.float32)

    cmap_obj = plt.get_cmap(cmap)
    norm_probs = np.clip((probs / (probs.max() + 1e-6)) * 0.8, 0, 1)
    colors = cmap_obj(norm_probs)
    pred_device_id = ids[np.argmax(probs)]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("white")
    ax.set_aspect("equal")

    for i in range(len(ids)):
        device_id = ids[i]
        is_true = (true_device_id is not None and device_id == int(true_device_id))
        is_pred = (device_id == pred_device_id)

        ax.add_patch(Circle((ys[i], xs[i]), radius=circle_radius, facecolor=colors[i], edgecolor="black", linewidth=0.6, alpha=0.9))

        if is_true and is_pred:
            ax.add_patch(Circle((ys[i], xs[i]), radius=circle_radius * 1.08, facecolor="none", edgecolor="green", linewidth=1.5))
            ax.add_patch(Circle((ys[i], xs[i]), radius=circle_radius * 0.95, facecolor="none", edgecolor="blue", linewidth=1.2))
        elif is_true:
            ax.add_patch(Circle((ys[i], xs[i]), radius=circle_radius, facecolor="none", edgecolor="blue", linewidth=1.5))
        elif is_pred:
            ax.add_patch(Circle((ys[i], xs[i]), radius=circle_radius, facecolor="none", edgecolor="green", linewidth=1.5))

        if annotate:
            ax.text(ys[i], xs[i], f"{device_id}", ha="center", va="center", fontsize=8, color="black")

    ax.set_xlim(ys.min() - 0.2, ys.max() + 0.2)
    ax.set_ylim(xs.min() - 0.2, xs.max() + 0.2)
    ax.invert_xaxis()
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='blue', markerfacecolor='none', markersize=10, linewidth=0, label="True Device"),
        Line2D([0], [0], marker='o', color='green', markerfacecolor='none', markersize=10, linewidth=0, label="Predicted Device")
    ], loc="upper right", frameon=True)
    plt.tight_layout()

    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show(); plt.close()

def plot_device_positions(device_json_path, circle_radius=0.05, annotate=True, save_path=None):
    with open(device_json_path, "r", encoding="utf-8") as f: devices = json.load(f)
    ids, xs, ys = [], [], []
    for dev_id, info in devices.items():
        ids.append(int(dev_id))
        xs.append(info["x"] / 1000.0); ys.append(info["y"] / 1000.0)

    order = np.argsort(np.array(ids))
    ids, xs, ys = np.array(ids)[order], np.array(xs)[order], np.array(ys)[order]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("white"); ax.set_aspect("equal")

    for i in range(len(ids)):
        ax.add_patch(Circle((ys[i], xs[i]), radius=circle_radius, facecolor="#cccccc", edgecolor="black", linewidth=0.6, alpha=0.9))
        if annotate: ax.text(ys[i], xs[i], f"{ids[i]}", ha="center", va="center", fontsize=8, color="black")

    ax.set_xlim(ys.min() - 0.2, ys.max() + 0.2)
    ax.set_ylim(xs.min() - 0.2, xs.max() + 0.2)
    ax.invert_xaxis(); plt.tight_layout()

    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show(); plt.close()

def print_sample_info(sample):
    info = [
        ["subject", sample["subject"]], ["action", sample["action"]], ["device_id", sample["device_id"].item()],
        ["view", sample["view"]], ["body_shape", tuple(sample["keypoints3d"]["body_keypoints3d"].shape)],
        ["hands_shape", tuple(sample["keypoints3d"]["hands_keypoints3d"].shape)],
    ]
    print("\n" + "=" * 90)
    print(tabulate(info, headers=["Field", "Value"], tablefmt="fancy_grid"))

def draw_heatmap_on_image(img_path, dev_prob_1N, device_json_path, camera_intrinsic, camera_extrinsic, sigma=30, alpha=0.6, save_path=None):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None: return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    K, D = to_numpy(camera_intrinsic['K']), to_numpy(camera_intrinsic.get('D', np.zeros(5)))
    R, T = to_numpy(camera_extrinsic['R']), to_numpy(camera_extrinsic['T'])

    with open(device_json_path, "r", encoding="utf-8") as f: devices = json.load(f)
    coords = []
    for i in range(1, len(dev_prob_1N) + 1):
        info = devices.get(str(i))
        coords.append([info["x"] / 1000.0, info["y"] / 1000.0, info["z"] / 1000.0] if info else [0, 0, 0])
    pts_2d = project_3d_to_2d(np.array(coords, dtype=np.float32), K, D, R, T)

    heatmap_data = np.zeros((H, W), dtype=np.float32)
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))

    for i, (u, v) in enumerate(pts_2d):
        prob = dev_prob_1N[i]
        if prob < 0.001 or not (-100 <= u < W + 100 and -100 <= v < H + 100): continue
        roi_size = int(sigma * 4)
        x1, x2 = max(0, int(u - roi_size)), min(W, int(u + roi_size))
        y1, y2 = max(0, int(v - roi_size)), min(H, int(v + roi_size))
        if x2 <= x1 or y2 <= y1: continue

        local_x, local_y = grid_x[y1:y2, x1:x2], grid_y[y1:y2, x1:x2]
        dist_sq = (local_x - u) ** 2 + (local_y - v) ** 2
        heatmap_data[y1:y2, x1:x2] += prob * np.exp(-dist_sq / (2 * sigma ** 2))

    max_val = heatmap_data.max()
    heatmap_norm = heatmap_data / max_val if max_val > 0 else heatmap_data

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(img_rgb)
    ax.imshow(heatmap_norm, cmap='jet', alpha=alpha, vmin=0, vmax=1, interpolation='bilinear')
    ax.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0); plt.margins(0, 0)

    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show(); plt.close()

def plot_dataset_sankey(data_index, save_path=None):
    df = pd.DataFrame(data_index)
    view_counts = df['view_name'].value_counts()
    action_counts = df['action'].value_counts()
    device_counts = df['device_id'].value_counts()

    layer0 = sorted(df['view_name'].unique())
    layer1 = action_counts.index.tolist()
    sorted_devices = device_counts.index.tolist()
    if 0 in sorted_devices:
        sorted_devices.remove(0)
        sorted_devices = [0] + sorted_devices
    layer2 = [str(d) for d in sorted_devices]

    all_node_ids = layer0 + layer1 + layer2
    node_map = {str(name): i for i, name in enumerate(all_node_ids)}

    final_labels = []
    for v in layer0: final_labels.append(f"<b>View {v} (N={view_counts.get(v, 0)})<b>")
    for a in layer1: final_labels.append(f"<b>{a} (N={action_counts.get(a, 0)})<b>")
    for d_id in layer2:
        d_name = "No Interaction" if int(d_id) == 0 else f"Dev_{d_id}"
        final_labels.append(f"<b>           {d_name} (N={device_counts.get(int(d_id), 0)})<b>")

    view_colors_hex = ["#8ECAE6", "#219EBC", "#A2D2FF", "#BDE0FE"]
    action_colors_map = {'point': '#D62828', 'press': '#F77F00', 'push': '#FCBF49', 'swing': '#EAE2B7'}
    
    node_colors = []
    for node in all_node_ids:
        if node in layer0: node_colors.append(view_colors_hex[layer0.index(node) % 4])
        elif node in layer1: node_colors.append(action_colors_map.get(node, "#888888"))
        else: node_colors.append("#457B9D" if node == '0' else "#8D99AE")

    def hex_to_rgba(hex_code, alpha=0.5):
        hex_code = hex_code.lstrip('#')
        rgb = tuple(int(hex_code[i:i + 2], 16) for i in (0, 2, 4))
        return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"

    links = {'source': [], 'target': [], 'value': [], 'color': []}
    
    flow1 = df.groupby(['view_name', 'action']).size().reset_index(name='count')
    for _, row in flow1.iterrows():
        links['source'].append(node_map[row['view_name']])
        links['target'].append(node_map[row['action']])
        links['value'].append(row['count'] * 10)
        links['color'].append(hex_to_rgba(view_colors_hex[layer0.index(row['view_name']) % 4], alpha=0.3))

    flow2 = df.groupby(['action', 'device_id']).size().reset_index(name='count')
    for _, row in flow2.iterrows():
        links['source'].append(node_map[row['action']])
        links['target'].append(node_map[str(row['device_id'])])
        links['value'].append(row['count'] * 10)
        links['color'].append(hex_to_rgba(action_colors_map.get(row['action'], "#888888"), alpha=0.5))

    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="white", width=0.5), label=final_labels, color=node_colors, hovertemplate='%{label}<extra></extra>'),
        link=dict(source=links['source'], target=links['target'], value=links['value'], color=links['color']),
        textfont=dict(size=10, family="Times New Roman, serif", color="black")
    )])

    fig.update_layout(font_family="Times New Roman, serif", width=1200, height=800, plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=30, r=30, t=30, b=30))

    if save_path:
        path_obj = Path(save_path)
        fig.write_image(str(path_obj), format=path_obj.suffix.lower().replace('.', ''), width=1000, height=600, scale=1)
    else:
        fig.show()
