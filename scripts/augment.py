import albumentations as A
import cv2
import os

#Input
IMG_INPUT = 'data/images/train'
LBL_INPUT = 'data/labels/train'

#Output
IMG_OUTPUT = 'data/augmented_images'
LBL_OUTPUT = 'data/augmented_labels'
os.makedirs(IMG_OUTPUT, exist_ok=True)
os.makedirs(LBL_OUTPUT, exist_ok=True)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def load_yolo_labels(label_path):
    boxes = []
    classes = []
    if not os.path.exists(label_path):
        return boxes, classes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, x, y, w, h = map(float, parts)
                boxes.append([x, y, w, h])
                classes.append(int(cls))
    return boxes, classes

def save_yolo_labels(label_path, boxes, classes):
    with open(label_path, 'w') as f:
        for box, cls in zip(boxes, classes):
            f.write(f"{cls} {' '.join(f'{coord:.6f}' for coord in box)}\n")

for img_file in os.listdir(IMG_INPUT):
    if not img_file.endswith(('.jpg', '.png')):
        continue

    img_path = os.path.join(IMG_INPUT, img_file)
    label_path = os.path.join(LBL_INPUT, os.path.splitext(img_file)[0] + '.txt')

    image = cv2.imread(img_path)
    if image is None:
        continue

    bboxes, class_labels = load_yolo_labels(label_path)
    if not bboxes:
        continue

    for i in range(3):
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_img = transformed['image']
        aug_bboxes = transformed['bboxes']
        aug_classes = transformed['class_labels']

        if not aug_bboxes:
            continue

        base_name = os.path.splitext(img_file)[0]
        aug_img_name = f"{base_name}_aug{i}.jpg"
        aug_lbl_name = f"{base_name}_aug{i}.txt"

        cv2.imwrite(os.path.join(IMG_OUTPUT, aug_img_name), aug_img)
        save_yolo_labels(os.path.join(LBL_OUTPUT, aug_lbl_name), aug_bboxes, aug_classes)

print("Augmented images and updated labels saved in 'data/augmented_images' and 'data/augmented_labels'.")
