import os
import csv

# 定义文件夹路径
masks_folder = './Micro_Prostate_Dataset/masks'  # 这是您的 masks 文件夹路径
images_folder = './Micro_Prostate_Dataset/images'

# 获取masks文件夹中所有的png文件
mask_files = [f for f in os.listdir(masks_folder) if f.endswith('png')]

# 创建CSV文件并写入数据
csv_file_path = './data/nose/TrainDataset.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ImageId', 'MaskId'])  # 写入表头
    
    # 遍历masks文件夹中的文件
    for mask_file in mask_files:
        # 提取frame部分以匹配对应的jpg文件
        base_name = mask_file.replace('mask', 'original')
        image_name = f"{base_name}"
        
        image_path = os.path.join(images_folder, image_name)
        mask_path = os.path.join(masks_folder, mask_file)

        # 确认images文件夹中也存在该文件
        if os.path.exists(image_path):
            writer.writerow([image_path, mask_path])

print(f"CSV文件已生成，路径为：{csv_file_path}")