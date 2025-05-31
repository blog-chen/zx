import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import resnet34

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 预期缺陷类别（根据实际类别字典调整）
TARGET_CLASSES = ['错口', '破裂', '沉积', '障碍物']


def format_probability(prob):
    """将概率值格式化为百分比字符串"""
    return f"{prob * 100:.2f}%"


def main():
    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载图像
    img_path = r"C:\Users\86135\Desktop\bishe\ResNet-main1\data_set\food_data\test\img_0.jpeg"
    assert os.path.exists(img_path), f"文件不存在: {img_path}"

    img = Image.open(img_path)
    plt_img = img.copy()  # 保存用于显示的副本
    img = data_transform(img).unsqueeze(0).to(device)

    # 加载类别字典
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"类别文件不存在: {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        class_indict = json.load(f)

    # 验证目标类别
    missing_classes = set(TARGET_CLASSES) - set(class_indict.values())
    if missing_classes:
        raise ValueError(f"类别字典中缺少以下目标类别: {missing_classes}")

    # 创建模型并加载权重
    model = resnet34(num_classes=10).to(device)
    weights_path = "./resNet34.pth"
    assert os.path.exists(weights_path), f"权重文件不存在: {weights_path}"
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # 执行预测
    model.eval()
    with torch.no_grad():
        output = model(img).squeeze()
        probabilities = torch.softmax(output, dim=0)
        pred_class_idx = torch.argmax(probabilities).item()

    # 获取预测结果
    pred_class = class_indict[str(pred_class_idx)]
    pred_prob = probabilities[pred_class_idx].item()

    # 可视化设置
    plt.figure(figsize=(4, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(plt_img)
    plt.axis('off')
    plt.title(f"预测结果: {pred_class}\n置信度: {format_probability(pred_prob)}")

    # 创建过滤后的概率分布条形图
    plt.subplot(1, 2, 2)

    # 过滤非目标类别
    filtered_probs = [
        (prob.item(), class_indict[str(i)])
        for i, prob in enumerate(probabilities)
        if class_indict[str(i)] in TARGET_CLASSES
    ]

    # 按概率排序
    sorted_probs, sorted_labels = zip(*sorted(
        filtered_probs,
        key=lambda x: x[0],
        reverse=True
    ))

    # 生成颜色列表（预测类别高亮）
    colors = ['#2ecc71' if label == pred_class else '#3498db' for label in sorted_labels]

    # 绘制水平条形图
    plt.barh(range(len(sorted_probs)), sorted_probs[::-1],
             color=colors[::-1], alpha=0.8, height=0.6)
    plt.yticks(range(len(sorted_probs)), sorted_labels[::-1])
    plt.xlim(0, 1.0)
    plt.xlabel('概率值')
    plt.title('目标类别概率分布')
    plt.tight_layout()

    # 控制台输出
    print("=" * 40)
    print(f"图像路径: {img_path}")
    print(f"最终预测: {pred_class} ({format_probability(pred_prob)})")
    print("-" * 40)
    print("详细概率分布：")
    for label in TARGET_CLASSES:
        idx = list(class_indict.values()).index(label)
        prob = probabilities[idx].item()
        print(f"▏{label:　<6} → {format_probability(prob):<8} [类别索引: {idx}]")
    print("=" * 40)

    plt.show()


if __name__ == '__main__':
    main()