import cv2
from ultralytics import YOLO
import numpy as np
import torch
import os
import argparse
import sys

model = YOLO('yolov8x-seg.pt')
device = "cuda:0" if torch.cuda.is_available() else "cpu"
def generate_mask(source,maskpath):#YOLOv8 to segmant dynamic objects and generate masks
    results = model.predict(source, save=False, classes=[0], show_labels=False,stream=True, show_boxes=False, show_conf=False)
    for result in results:
        if len(result.boxes)==0:
            input_path = result.path  # Your input path string
            print(input_path)
            output_path = os.path.join(os.path.basename(input_path))
            print(output_path)
            print("no detections for" + output_path)
            cv2.imwrite(str(maskpath+'/'+output_path), np.zeros((480, 640, 3), np.uint8))
        else : 
        # get array results
            masks = result.masks.data
            boxes = result.boxes.data
            clss = boxes[:, 5]
            # extract classes
            # get indices of results where class is 0 (people in COCO)
            people_indices = torch.where(clss == 0)
            # use these indices to extract the relevant masks
            people_masks = masks[people_indices]
            # scale for visualizing results
            people_mask = torch.any(people_masks, dim=0).int() * 255
            # save to file
            input_path = result.path  # Your input path string
            print(input_path)
            output_path = os.path.join(os.path.basename(input_path))
            print(output_path)
            cv2.imwrite(str(maskpath+'/'+output_path), people_mask.cpu().numpy())

def remove_dyna(rgb_folder,maskpath,output_folder):#remove the dynamnic objects in the frames
    # Iterate over all files in the rgb folder
    for filename in os.listdir(rgb_folder):
        if filename.endswith(".png"):
            # Construct the paths for RGB image and corresponding mask image
            rgb_path = os.path.join(rgb_folder, filename)
            mask_path = os.path.join(maskpath, filename)  # Assuming mask images are in the "output" folder

            # Check if the mask image exists
            if os.path.exists(mask_path):
                # Read the RGB image
                rgb_image = cv2.imread(rgb_path)

                # Read the binary mask image
                mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                # Invert the mask (assuming white represents people)
                mask_image = cv2.bitwise_not(mask_image)

                # Apply the mask to the RGB image
                result_image = cv2.bitwise_and(rgb_image, rgb_image, mask=mask_image)

                # Output path for the resulting image
                output_path = os.path.join(output_folder, filename)

                # Save the resulting image
                cv2.imwrite(output_path, result_image)

                print("People removed from", filename, "and saved as:", output_path)
            else:
                print("No corresponding mask found for", filename)

if __name__ == "__main__":
    #initializing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="rgb")
    parser.add_argument("--maskpath", type=str, default="mask")
    parser.add_argument("--removedpath", type=str, default="removed")
    args = parser.parse_args()
    print(f"Resulting masks will be saved to {args.maskpath}/, and frames with dynamic objects removed will be saved to {args.removedpath}/n")
    confirm = input("Are you sure? (y/n)\n")
    if (confirm.lower().find("y") != -1 or confirm.lower().find("yes") != -1):
        print("Confirmed")
    else:
        sys.exit("User rejected")
    if not os.path.exists(args.maskpath):
        os.makedirs(args.maskpath)
    if not os.path.exists(args.removedpath):
        os.makedirs(args.removedpath)
    generate_mask(args.source,args.maskpath)
    remove_dyna(args.source,args.maskpath,args.removedpath)
