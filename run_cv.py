import os
import time
import argparse


def main():
    p = argparse.ArgumentParser(description="LOSO 10折交叉验证调度器")
    p.add_argument('--model_type', type=str, default='DriverHOI',
                   choices=['DriverHOI', 'MLP-HOI', 'TransHOI', 'SCG-HOI'])
    p.add_argument('--ablation',   type=str, default='baseline',
                   choices=['baseline', 'with_visual', 'no_pose', 'no_geom'])
    p.add_argument('--job_name',   type=str, required=True,
                   help='归档目录名, 如 baseline_exp01')
    p.add_argument('--num_epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr',         type=float, default=1e-3)
    args = p.parse_args()

    print(f"\n{'='*60}")
    print(f" 10 折交叉验证 | 模型: {args.model_type} | 消融: {args.ablation}")
    print(f" 归档目录: {args.job_name}")
    print(f"{'='*60}\n")

    for i in range(1, 11):
        subj = f"subject{i}"

        print(f"\n[{i}/10] 正在启动: {subj}")
        print("-" * 50)

        cmd = (
            f"python train.py"
            f" --test_subj {subj}"
            f" --exp_name {subj}"
            f" --job_name {args.job_name}"
            f" --model_type {args.model_type}"
            f" --ablation {args.ablation}"
            f" --num_epochs {args.num_epochs}"
            f" --batch_size {args.batch_size}"
            f" --lr {args.lr}"
        )

        os.system(cmd)
        print(f"  {subj} 完成!")
        time.sleep(2)

    print(f"\n{'='*60}")
    print(f" 全部 10 折训练完成! 归档目录: {args.job_name}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()