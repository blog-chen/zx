import os
import json
import torch
from PIL import Image
from torchvision import transforms
from model import resnet34

# 配置参数（注意修正路径中的目录名称）
CONFIG = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "batch_size": 8,
    "imgs_root": r"C:\Users\86135\Desktop\bishe\ResNet-main1\data_set\food_data\test",  # 注意改为-main1
    "json_path": "./class_indices.json",
    "weights_path": r"C:\Users\86135\Desktop\bishe\ResNet-main1\resNet34.pth",  # 同步修正权重路径
    "result_file": "./prediction_results.csv"
}


def validate_class_names(class_indict):
    """验证类别名称是否包含预期缺陷类型"""
    expected_classes = {'错口', '破裂', '沉积', '障碍物'}
    actual_classes = set(class_indict.values())
    if not expected_classes.issubset(actual_classes):
        missing = expected_classes - actual_classes
        raise ValueError(f"缺失类别: {missing}")


def main():
    # 数据预处理
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载类别索引
    assert os.path.exists(CONFIG['json_path']), f"文件不存在: {CONFIG['json_path']}"
    with open(CONFIG['json_path'], "r", encoding="utf-8") as f:
        class_indict = json.load(f)
    validate_class_names(class_indict)

    # 创建模型
    model = resnet34(num_classes=len(class_indict)).to(CONFIG['device'])  # 自动适配类别数
    assert os.path.exists(CONFIG['weights_path']), f"权重文件不存在: {CONFIG['weights_path']}"
    model.load_state_dict(torch.load(CONFIG['weights_path'], map_location=CONFIG['device']))

    # 获取所有测试图片路径（关键修改部分）
    img_path_list = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

    for root, _, files in os.walk(CONFIG['imgs_root']):
        for file in files:
            if file.lower().endswith(valid_extensions):
                full_path = os.path.join(root, file)
                img_path_list.append(full_path)

    # 排序确保顺序一致
    img_path_list = sorted(img_path_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))

    print(f"找到测试图片数量: {len(img_path_list)}")  # 调试输出

    # 批量预测
    model.eval()
    with torch.no_grad(), open(CONFIG['result_file'], 'w', encoding='utf-8') as f:
        f.write("image_path,predicted_class,probability,class_index\n")

        total_images = len(img_path_list)
        processed = 0

        for idx in range(0, total_images, CONFIG['batch_size']):
            batch_paths = img_path_list[idx: idx + CONFIG['batch_size']]
            batch_images = []

            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    batch_images.append(data_transform(img))
                except Exception as e:
                    print(f"⚠️ 跳过损坏文件: {img_path} | 错误: {str(e)}")
                    continue

            if not batch_images:
                continue

            # 执行预测
            batch_tensor = torch.stack(batch_images).to(CONFIG['device'])
            outputs = model(batch_tensor)
            probs, indices = torch.max(torch.softmax(outputs, dim=1), dim=1)

            # 写入结果
            for img_path, prob, index in zip(batch_paths, probs.cpu().numpy(), indices.cpu().numpy()):
                class_name = class_indict[str(index)]
                f.write(f"{img_path},{class_name},{prob:.4f},{index}\n")
                processed += 1
                print(
                    f"进度: {processed}/{total_images} | 当前预测: {os.path.basename(img_path)} => {class_name} ({prob:.2%})")

    print(f"\n预测完成！结果已保存至: {CONFIG['result_file']}")


if __name__ == '__main__':
    main()