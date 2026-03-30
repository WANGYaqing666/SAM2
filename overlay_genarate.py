import os
import cv2
import numpy as np
import imageio

def calculate_iou(mask1, mask2):
    """
    计算两个蒙版之间的 IoU (交并比)。
    
    参数:
    - mask1 (np.ndarray): 第一个蒙版。
    - mask2 (np.ndarray): 第二个蒙版。
    
    返回:
    - iou (float): 两个蒙版的 IoU 值。
    """
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    return intersection / union if union > 0 else 0.0

def apply_green_overlay_with_smooth_edges_and_iou(
    image_folder, pred_mask_folder, gt_mask_folder, save_folder, alpha=0.4, edge_color=(0, 255, 0), edge_thickness=1):
    """
    将绿色透明蒙版叠加到一批原图上，计算 IoU，并保存结果。
    
    参数:
    - image_folder (str): 原图文件夹路径。
    - pred_mask_folder (str): 预测蒙版文件夹路径。
    - gt_mask_folder (str): 原始蒙版文件夹路径。
    - save_folder (str): 保存叠加结果的文件夹路径。
    - alpha (float): 透明度值，0.0表示完全透明，1.0表示完全不透明。
    - edge_color (tuple): 边缘颜色，格式为(B, G, R)。
    - edge_thickness (int): 边缘线的厚度。
    """
    # 创建保存文件夹
    os.makedirs(save_folder, exist_ok=True)
    
    # 遍历原图文件夹中的所有图像文件
    for image_name in os.listdir(image_folder):
        if image_name.endswith('.png'):
            # 构建路径
            image_path = os.path.join(image_folder, image_name)
            pred_mask_path = os.path.join(pred_mask_folder, image_name)
            gt_mask_name = image_name.replace("original_image_", "mask_image_")
            gt_mask_path = os.path.join(gt_mask_folder, gt_mask_name)

            # 检查对应的蒙版是否存在
            if not os.path.exists(pred_mask_path) or not os.path.exists(gt_mask_path):
                print(f"Warning: Missing masks for {image_name}. Skipping.")
                continue

            # 读取原图和蒙版
            image = cv2.imread(image_path)[..., ::-1]  # BGR to RGB
            pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

            # 检查是否成功读取
            if image is None or pred_mask is None or gt_mask is None:
                print(f"Error: Unable to load image or masks for {image_name}. Skipping.")
                continue

            # 调整蒙版大小以匹配原图
            pred_mask_resized = cv2.resize(pred_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            gt_mask_resized = cv2.resize(gt_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            # 计算 IoU
            iou = calculate_iou(pred_mask_resized, gt_mask_resized)
            print(f"IoU for {image_name}: {iou:.4f}")

            # 平滑蒙版边缘 高斯模糊 9-9是模糊程度
            mask_smoothed = cv2.GaussianBlur(pred_mask_resized, (9, 9), 0)

            # 创建绿色叠加图层
            green_overlay = np.zeros_like(image)
            green_overlay[:, :, 1] = 255  # 设置绿色通道为最大值

            # 将蒙版应用为透明叠加
            overlay_image = cv2.addWeighted(image, 1, green_overlay, alpha, 0)
            overlay_image[mask_smoothed == 0] = image[mask_smoothed == 0]  # 没有蒙版的区域保留原图内容

            # 保存叠加图像，文件名包含 IoU
            save_filename = f"{os.path.splitext(image_name)[0]}_iou_{iou:.4f}.png"
            save_path = os.path.join(save_folder, save_filename)
            imageio.imwrite(save_path, overlay_image)
            print(f"Saved overlay image with IoU for {image_name} to {save_path}")

# 使用示例
image_folder = './Micro_Prostate_Dataset/images'
pred_mask_folder = './results/SAM2/large/double/Micro_Prostate_Dataset/masks'
gt_mask_folder = './Micro_Prostate_Dataset/masks'
save_folder = './results/SAM2/large/double/Micro_Prostate_Dataset/overlay_with_iou'

apply_green_overlay_with_smooth_edges_and_iou(image_folder, pred_mask_folder, gt_mask_folder, save_folder)
# 这个是overlay_generated with IoU 这个是讲predictor.py中生成的mask进行高斯模糊 然后在overlay覆盖在原图