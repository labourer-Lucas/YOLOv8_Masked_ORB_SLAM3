import os
import cv2
import numpy as np
import argparse
import sys
def dilate_images_in_folder(folder_path,output_path, kernel_size=10, iterations=1):
    # 遍历文件夹里的所有图像文件
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # 假设只处理png、jpg和jpeg格式的图像文件
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            # 确认图像已正确读取
            if image is None:
                print(f"图像文件 {filename} 未找到或无法读取。")
            else:
                # 创建膨胀用的核
                kernel = np.ones((kernel_size, kernel_size), np.uint8)

                # 应用膨胀操作
                dilated_image = cv2.dilate(image, kernel, iterations=iterations)
                # 保存膨胀后的图像
                dilated_image_path = os.path.join(output_path, f'{filename}')
                cv2.imwrite(dilated_image_path, dilated_image)
                print(f"膨胀后的图像已保存为 {dilated_image_path}")

if __name__ == "__main__":
    #initializing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--maskpath", type=str, default="mask")#mask folder
    parser.add_argument("--outpath", type=str, default="delated")#output folder
    parser.add_argument("--kernelSize", type=str, default=10)#output folder
    parser.add_argument("--iterations", type=str, default=1)#output folder
    args = parser.parse_args()
    print(f"Resulting masks will be saved to {args.outpath}/")
    confirm = input("Are you sure? (y/n)\n")
    if (confirm.lower().find("y") != -1 or confirm.lower().find("yes") != -1):
        print("Confirmed")
    else:
        sys.exit("User rejected")
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
    dilate_images_in_folder(args.maskpath,args.outpath,args.kernelSize,args.iterations)