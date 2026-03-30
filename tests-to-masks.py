import os
import cv2
import numpy as np

# 定义路径
input_dir = './data/nose/tests'
output_dir = './data/nose/masks'

# 确保输出文件夹存在
os.makedirs(output_dir, exist_ok=True)

# 定义要保留的 RGB 颜色（转换为 BGR，因为 OpenCV 使用 BGR 格式）
target_color = [0, 255, 0]  # 对应 #75fb4c 的 BGR 值

# 遍历 input_dir 下的所有 PNG 文件
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        # 读取图像
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)  # 读取为彩色图像
        
        # 检查图像是否读取成功
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue
        
        # 创建一个黑色的掩码图像
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # 找到目标颜色的位置，将其设为白色（255）
        mask[np.all(img == target_color, axis=-1)] = 255

        # 将处理后的图像保存到 output_dir
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, mask)
        print(f"已保存处理后的图像到: {output_path}")

print("所有图像处理完成！")