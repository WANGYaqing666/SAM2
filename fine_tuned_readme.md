- nii转png用的是ni_to_masks用的是expert_annotations生成黑白的mask然后原图+红色maks是自己叠加的
- 和ni_to_images用的micro——ultrasound——scans生成原始图像
- 1. 数据集
- 
  首先有一个images labels tests三个文件夹，tests是事先已经用sam2自己分割的，下面是做图像处理

用到的代码文件： tests-to-masks.py
/media/wang/Ubuntu/sam2-fine-tuned/sam2-main/data
test的数据集是自己选的 然后用sam2分割成绿色的mask然后转黑白mask

用到的代码文件： csv_generate.py
将对应的mask黑白的和原图想images一一对应

2. fine-tuning
   微调
   fine-tuning.ipynb
需要修改的为
更改文件夹位置
# data_dir = os.path.join(current_dir, "data/prostate")
更改验证数据集站比例 现在是0.2
train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)
统一照片大小
 r = np.min([352 / Img.shape[1], 352 / Img.shape[0]])  # Scaling factor
更改用的哪个模型
 # sam2_checkpoint = "sam2_hiera_small.pt"  # @param ["sam2_hiera_tiny.pt", "sam2_hiera_small.pt", "sam2_hiera_base_plus.pt", "sam2_hiera_large.pt"]
sam2_checkpoint = "sam2_hiera_tiny.pt"
# model_cfg = "configs/sam2/sam2_hiera_s.yaml" # @param ["sam2_hiera_t.yaml", "sam2_hiera_s.yaml", "sam2_hiera_b+.yaml", "sam2_hiera_l.yaml"]
model_cfg = "configs/sam2/sam2_hiera_t.yaml" 

3.predictor.py运行程序
original_image_microUS_test_01.nii.gz_0.png

microUS_test_01.nii.gz_0.png

mask_image_microUS_test_01.nii.gz_0.png