import os
import random
import shutil

# 设定参数为适当值
is_skip_step_1 = False
use_custom_dataset = True  # 确保这一项为 True
dataset_is_split = False

def split_into_train_test(images_origin, images_dest, test_split):
    _, dirs, _ = next(os.walk(images_origin))
    TRAIN_DIR = os.path.join(images_dest, 'yoga_train')
    TEST_DIR = os.path.join(images_dest, 'yoga_test')
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    for dir in dirs:
        filenames = os.listdir(os.path.join(images_origin, dir))
        filenames = [os.path.join(images_origin, dir, f) for f in filenames if (
            f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.bmp'))]
        
        # 打乱并划分数据
        filenames.sort()
        random.seed(42)
        random.shuffle(filenames)
        
        os.makedirs(os.path.join(TEST_DIR, dir), exist_ok=True)
        os.makedirs(os.path.join(TRAIN_DIR, dir), exist_ok=True)
        
        test_count = int(len(filenames) * test_split)
        for i, file in enumerate(filenames):
            if i < test_count:
                destination = os.path.join(TEST_DIR, dir, os.path.split(file)[1])
            else:
                destination = os.path.join(TRAIN_DIR, dir, os.path.split(file)[1])
            shutil.copyfile(file, destination)
        print(f'Moved {test_count} of {len(filenames)} from class \"{dir}\" into test.')
    print(f'Your split dataset is in \"{images_dest}\"')

# 确保 dataset_in 是有效目录
dataset_in = r'F:\unsw\COMP9444\Assignment2\dataset\yoga_data_png'

# 正确创建 dataset_out 的路径，避免重复定义
dataset_out = os.path.join(os.path.dirname(dataset_in), 'split_' + os.path.basename(dataset_in))

# 确保生成的路径是合法的并打印出来检查
print("生成的 dataset_out 路径:", dataset_out)

# 检查生成的路径是否有效
if not os.path.isdir(dataset_in):
    raise Exception("dataset_in is not a有效目录")

# 使用自定义数据集的条件检查
if use_custom_dataset:
    if not os.path.isdir(dataset_in):
        raise Exception("dataset_in is not a valid directory")
    if dataset_is_split:
        IMAGES_ROOT = dataset_in
    else:
        # 使用前面定义的 dataset_out
        split_into_train_test(dataset_in, dataset_out, test_split=0.2)
        IMAGES_ROOT = dataset_out

