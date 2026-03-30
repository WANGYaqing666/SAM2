import os
import torch
import numpy as np
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import imageio
from collections import OrderedDict

# Set this flag to 1 to load fine-tuned weights, or 0 to use the original SAM2 model 
# 0是没有加载torch权重文件的原生文件，1是用torch文件
USE_FINE_TUNED_MODEL = 1

# Paths to model configuration and checkpoint 加载预训练模型 一一对应
sam2_checkpoint = "sam2_hiera_large.pt"  # @param ["sam2_hiera_tiny.pt", "sam2_hiera_small.pt", "sam2_hiera_base_plus.pt", "sam2_hiera_large.pt"]
model_cfg = "configs/sam2/sam2_hiera_l.yaml" # @param ["sam2_hiera_t.yaml", "sam2_hiera_s.yaml", "sam2_hiera_b+.yaml", "sam2_hiera_l.yaml"]


# Load the SAM2 model
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

if USE_FINE_TUNED_MODEL:
    # Path to fine-tuned model weights 这个是最好的学生，是在fine-tuning.ipynb生成的torch文件 Top 10 IoUs and corresponding steps:Rank 1: IoU = 0.9357 at Step = 6600
    FINE_TUNED_MODEL_WEIGHTS = "sam2-fine-tuned-checkpoint/fine_tuned_sam2_1200.torch"
    if os.path.exists(FINE_TUNED_MODEL_WEIGHTS):
        weights = torch.load(FINE_TUNED_MODEL_WEIGHTS)
        new_state_dict = OrderedDict()
        for k, v in weights.items():
            if 'total_ops' not in k and 'total_params' not in k:
                new_state_dict[k] = v
        predictor.model.load_state_dict(new_state_dict)
        print("Loaded fine-tuned model weights.")
    else:
        print(f"Fine-tuned model weights not found at {FINE_TUNED_MODEL_WEIGHTS}. Using original SAM2 model.")
else:
    print("Using original SAM2 model without fine-tuned weights.")
# 加载显卡
predictor.model.cuda()
predictor.model.eval()

# Define dataset and paths 更改路径
dataset_name = 'Micro_Prostate_Dataset'
data_path = f'./{dataset_name}'
# 这两个路径是ft-sam2生成png后保存路径
save_path = f'./results/SAM2/large/4x/{dataset_name}/masks'
overlay_save_path = f'./results/SAM2/large/4x/{dataset_name}/overlays'
os.makedirs(save_path, exist_ok=True)
os.makedirs(overlay_save_path, exist_ok=True)
# 读取路径原数据集的images和masks，这个images和masks是通过，prossing_data.py中对nii文件处理成png后生成的
image_root = os.path.join(data_path, 'images')
gt_root = os.path.join(data_path, 'masks')

# Get all image filenames with "original_image_" prefix 加载Micro—prostate_dataset中的原图，以original_image_'开头，.png结尾
image_files = [f for f in os.listdir(image_root) if f.startswith('original_image_') and f.endswith('.png')]

for image_name in image_files:
    # Construct corresponding mask filename based on the unique part of the image filename 
    # 把images文件夹中的图片名和masks文件夹中的图片名对应起来，其实下面的操作只取用了test_01.nii.gz_0.png这部分
    unique_name = image_name.replace('original_image_', '')
    mask_name = f'mask_image_{unique_name}'

    # Check if corresponding mask exists
    image_path = os.path.join(image_root, image_name)
    mask_path = os.path.join(gt_root, mask_name)
    if not os.path.exists(mask_path):
        print(f"Warning: Mask file not found for {image_name}. Skipping.")
        continue

    # Load image and mask
    image = cv2.imread(image_path)[..., ::-1]  # BGR to RGB
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Check if mask is all black 没有mask的时候，输出原图如果没有这一步的会报错，因为没有兴趣点就无法分割
    if not np.any(mask):
        seg_map = np.zeros((int(image.shape[0]), int(image.shape[1])), dtype=np.uint8)  # 全黑图像、
        # 保存图像的大小(962, 1372)
        seg_map_resized = cv2.resize(seg_map, (3848, 5488), interpolation=cv2.INTER_NEAREST)
        save_image_path = os.path.join(save_path, image_name)
        imageio.imwrite(save_image_path, (seg_map_resized * 255).astype(np.uint8))
        print(f"Output blank image for {image_name} as mask is empty.")
        continue

    # Resize image and mask  # 输入的原始png图像的大小(962, 1372)
    r = np.min([3848 / image.shape[1], 5488 / image.shape[0]])
    image = cv2.resize(image, (int(image.shape[1] * r), int(image.shape[0] * r)))
    mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)

    # Generate input points from mask 根据mask生成兴趣点，如果没有兴趣点，和sam2官网视频中点一下，把衣服分割出来，点一下就是兴趣点
    coords = np.argwhere(mask > 0)
    num_samples = 30
    input_points = []
    for i in range(num_samples):
        yx = np.array(coords[np.random.randint(len(coords))])
        input_points.append([[yx[1], yx[0]]])
    input_points = np.array(input_points)

    # Perform prediction
    with torch.no_grad():
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=np.ones([input_points.shape[0], 1])
        )

    # Process predicted masks and sort by score
    np_masks = np.array(masks[:, 0])
    np_scores = scores[:, 0]
    sorted_masks = np_masks[np.argsort(np_scores)][::-1]

    # Initialize segmentation map and occupancy mask
    seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
    occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)

    # Merge masks to create the final segmentation map
    for i in range(sorted_masks.shape[0]):
        mask = sorted_masks[i]
        if (mask * occupancy_mask).sum() / mask.sum() > 0.15:
            continue

        mask_bool = mask.astype(bool)
        mask_bool[occupancy_mask] = False  # Set overlapping areas to False
        seg_map[mask_bool] = i + 1  # Use boolean mask to index seg_map
        occupancy_mask[mask_bool] = True  # Update occupancy_mask

    # Adjust output image size 最后生成出来的masks大小
    output_size = (3848, 5488)  # Set desired output size
    seg_map_resized = cv2.resize(seg_map, output_size, interpolation=cv2.INTER_NEAREST)

    # Save mask result 保存到你给的路径
    save_image_path = os.path.join(save_path, image_name)
    imageio.imwrite(save_image_path, (seg_map_resized * 255).astype(np.uint8))


print("Prediction and overlay creation completed.")

def calculate_iou(mask1, mask2):
    """ 固定计算公式
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

def apply_green_overlay_batch_with_iou(image_folder, pred_mask_folder, gt_mask_folder, save_folder, alpha=0.4):
    """将黑白mask转为绿色，覆盖在原图上
    将绿色透明蒙版叠加到一批原图上并保存结果，同时计算 IoU 值并将其记录在文件名中。

    参数:
    - image_folder (str): 原图文件夹路径。
    - pred_mask_folder (str): 预测蒙版文件夹路径。
    - gt_mask_folder (str): 原始蒙版（Ground Truth）文件夹路径。
    - save_folder (str): 保存叠加结果的文件夹路径。
    - alpha (float): 透明度值，0.0表示完全透明，1.0表示完全不透明。
    """
    # 创建保存文件夹
    os.makedirs(save_folder, exist_ok=True)
    
    # 遍历原图文件夹中的所有图像文件
    for image_name in os.listdir(image_folder):
        if image_name.endswith('.png') or image_name.endswith('.jpg'):
            # 构建原图和蒙版的路径
            image_path = os.path.join(image_folder, image_name)
            pred_mask_path = os.path.join(pred_mask_folder, image_name)
            gt_mask_name = image_name.replace("original_image_", "mask_image_")
            gt_mask_path = os.path.join(gt_mask_folder, gt_mask_name)

            
            # 检查对应的蒙版是否存在
            if not os.path.exists(pred_mask_path) or not os.path.exists(gt_mask_path):
                print(f"Warning: Mask for {image_name} not found. Skipping.")
                continue

            # 读取原图、预测蒙版和原始蒙版
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
            intersection = np.logical_and(pred_mask_resized > 0, gt_mask_resized > 0).sum()
            union = np.logical_or(pred_mask_resized > 0, gt_mask_resized > 0).sum()
            iou = intersection / union if union > 0 else 0.0

            # 创建绿色叠加图层
            green_overlay = np.zeros_like(image)
            green_overlay[:, :, 1] = 255  # 设置绿色通道为最大值

            # 将预测蒙版应用为透明叠加
            overlay_image = cv2.addWeighted(image, 1, green_overlay, alpha, 0)
            overlay_image[pred_mask_resized == 0] = image[pred_mask_resized == 0]  # 没有蒙版的区域保留原图内容

            # 将 IoU 附加到文件名中
            save_filename = f"{os.path.splitext(image_name)[0]}_iou_{iou:.4f}.png"
            save_path = os.path.join(save_folder, save_filename)

            # 保存叠加图像
            imageio.imwrite(save_path, overlay_image)
            print(f"Saved overlay image for {image_name} with IoU {iou:.4f} to {save_path}")
# image_root原图路径, save_path预测出的mask的路径, gt_root原图mask路径, overlay_save_path overlay以后的路径
# 这个是overlay
apply_green_overlay_batch_with_iou(image_root, save_path, gt_root, overlay_save_path)