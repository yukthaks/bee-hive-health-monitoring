import os

base_dirs = ['train', 'val', 'test']
image_base = 'data/images'
label_base = 'data/labels'

for split in base_dirs:
    img_dir = os.path.join(image_base, split)
    lbl_dir = os.path.join(label_base, split)
    
    missing = []
    for img_file in os.listdir(img_dir):
        if not img_file.endswith(('.jpg', '.png')):
            continue
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + ".txt"
        if not os.path.exists(os.path.join(lbl_dir, label_file)):
            missing.append(img_file)
    
    if missing:
        print(f"Missing labels in {split}: {missing}")
    else:
        print(f"All images in {split} have matching label files.")
