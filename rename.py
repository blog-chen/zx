import os
import uuid
import argparse


def rename_images(folder):
    image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    files = []

    # 收集所有图片文件
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_exts:
                files.append(filename)

    # 按文件名排序（可根据需求修改排序方式）
    files.sort()

    if not files:
        print("未找到图片文件。")
        return

    # 使用临时名称避免覆盖冲突
    temp_files = []
    for filename in files:
        ext = os.path.splitext(filename)[1]
        src = os.path.join(folder, filename)
        temp_name = f"__temp_{uuid.uuid4()}{ext}"
        dst = os.path.join(folder, temp_name)
        os.rename(src, dst)
        temp_files.append((dst, ext))

    # 正式重命名
    for idx, (temp_path, ext) in enumerate(temp_files):
        new_name = f"img_{idx}{ext}"
        os.rename(temp_path, os.path.join(folder, new_name))
        print(f"重命名为: {new_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量重命名图片文件')
    parser.add_argument('--folder', type=str, default='.', help='目标文件夹路径（默认为当前目录）')
    args = parser.parse_args()

    target_folder = os.path.abspath(args.folder)
    rename_images(target_folder)