#!/usr/bin/env python3
"""
验证 PyTorch 和 ONNX 输出一致性

用法：
    python verify_onnx_consistency.py \
        --pytorch-output outputs_pytorch/20260918_144430_575536/image_20260828153357597007_prob.png \
        --onnx-output outputs_onnx/image_20260828153357597007_prob.png

或者自动比较目录：
    python verify_onnx_consistency.py \
        --pytorch-dir outputs_pytorch/20260918_144430_575536 \
        --onnx-dir outputs_onnx
"""

import argparse
import cv2
import numpy as np
from pathlib import Path


def compare_images(pt_path, onnx_path, name="Image"):
    """比较两张图片的差异"""
    pt_img = cv2.imread(str(pt_path), cv2.IMREAD_GRAYSCALE)
    onnx_img = cv2.imread(str(onnx_path), cv2.IMREAD_GRAYSCALE)
    
    if pt_img is None:
        print(f"❌ 无法读取 PyTorch 图片: {pt_path}")
        return None
    if onnx_img is None:
        print(f"❌ 无法读取 ONNX 图片: {onnx_path}")
        return None
    
    if pt_img.shape != onnx_img.shape:
        print(f"❌ {name} 形状不匹配: PyTorch {pt_img.shape} vs ONNX {onnx_img.shape}")
        return None
    
    diff = np.abs(pt_img.astype(float) - onnx_img.astype(float))
    
    stats = {
        'max_diff': diff.max(),
        'mean_diff': diff.mean(),
        'pixels_diff_gt_1': (diff > 1).sum(),
        'total_pixels': diff.size,
        'pct_diff_gt_1': (diff > 1).mean() * 100,
    }
    
    return stats


def print_comparison(name, stats):
    """打印比较结果"""
    if stats is None:
        return
    
    print(f"\n{name}:")
    print(f"  形状: 正常")
    print(f"  最大差异: {stats['max_diff']:.2f} / 255")
    print(f"  平均差异: {stats['mean_diff']:.4f} / 255")
    print(f"  差异 > 1 的像素: {stats['pixels_diff_gt_1']:,} / {stats['total_pixels']:,} "
          f"({stats['pct_diff_gt_1']:.2f}%)")
    
    # 评估
    if stats['mean_diff'] < 0.5 and stats['pct_diff_gt_1'] < 5.0:
        print(f"  ✅ 一致性: 优秀")
    elif stats['mean_diff'] < 1.0 and stats['pct_diff_gt_1'] < 10.0:
        print(f"  ✅ 一致性: 良好")
    else:
        print(f"  ⚠️  一致性: 需要检查")


def main():
    parser = argparse.ArgumentParser(description="验证 PyTorch 和 ONNX 输出一致性")
    parser.add_argument('--pytorch-output', type=str, help='PyTorch 输出图片路径')
    parser.add_argument('--onnx-output', type=str, help='ONNX 输出图片路径')
    parser.add_argument('--pytorch-dir', type=str, help='PyTorch 输出目录（自动匹配）')
    parser.add_argument('--onnx-dir', type=str, help='ONNX 输出目录（自动匹配）')
    
    args = parser.parse_args()
    
    if args.pytorch_output and args.onnx_output:
        # 单文件比较
        print("=" * 60)
        print("PyTorch vs ONNX 输出一致性验证")
        print("=" * 60)
        
        stats = compare_images(args.pytorch_output, args.onnx_output)
        print_comparison("比较结果", stats)
        
    elif args.pytorch_dir and args.onnx_dir:
        # 目录比较
        pt_dir = Path(args.pytorch_dir)
        onnx_dir = Path(args.onnx_dir)
        
        print("=" * 60)
        print("PyTorch vs ONNX 目录批量比较")
        print("=" * 60)
        print(f"PyTorch 目录: {pt_dir}")
        print(f"ONNX 目录: {onnx_dir}")
        
        # 查找所有 _prob.png 文件
        pt_probs = sorted(pt_dir.glob("*_prob.png"))
        
        if not pt_probs:
            print(f"\n❌ 在 {pt_dir} 中未找到 *_prob.png 文件")
            return
        
        print(f"\n找到 {len(pt_probs)} 个 PyTorch 输出文件")
        
        all_stats = []
        for pt_prob in pt_probs:
            # 构造对应的 ONNX 文件名
            onnx_prob = onnx_dir / pt_prob.name
            
            if not onnx_prob.exists():
                print(f"\n⚠️  跳过 {pt_prob.name}（ONNX 输出不存在）")
                continue
            
            stats = compare_images(pt_prob, onnx_prob, pt_prob.stem)
            if stats:
                all_stats.append((pt_prob.name, stats))
                print_comparison(pt_prob.stem, stats)
        
        # 汇总统计
        if all_stats:
            print("\n" + "=" * 60)
            print("汇总统计")
            print("=" * 60)
            
            avg_max_diff = np.mean([s['max_diff'] for _, s in all_stats])
            avg_mean_diff = np.mean([s['mean_diff'] for _, s in all_stats])
            avg_pct_diff = np.mean([s['pct_diff_gt_1'] for _, s in all_stats])
            
            print(f"文件数量: {len(all_stats)}")
            print(f"平均最大差异: {avg_max_diff:.2f} / 255")
            print(f"平均平均差异: {avg_mean_diff:.4f} / 255")
            print(f"平均差异像素比例: {avg_pct_diff:.2f}%")
            
            if avg_mean_diff < 0.5 and avg_pct_diff < 5.0:
                print("\n✅ 整体一致性: 优秀 - ONNX 部署验证通过")
            elif avg_mean_diff < 1.0 and avg_pct_diff < 10.0:
                print("\n✅ 整体一致性: 良好 - ONNX 部署可用")
            else:
                print("\n⚠️  整体一致性: 需要检查 - 建议重新导出 ONNX")
    
    else:
        parser.print_help()
        print("\n请提供：")
        print("  方式1: --pytorch-output 和 --onnx-output（单文件比较）")
        print("  方式2: --pytorch-dir 和 --onnx-dir（目录批量比较）")


if __name__ == '__main__':
    main()
