from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import os, glob

# example transform giống train
img_size = 224
transform = A.Compose([
    A.Resize(img_size + 60, img_size + 60),
    A.RandomCrop(img_size, img_size),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])

def test_and_remove(folder):
    removed = 0
    exts = ["*.jpg","*.jpeg","*.png"]
    for ext in exts:
        for path in glob.glob(os.path.join(folder,"**",ext), recursive=True):
            try:
                img = np.array(Image.open(path).convert("RGB"))
                # thử transform
                transformed = transform(image=img)
            except Exception:
                print("Removing corrupt:", path)
                os.remove(path)
                removed += 1
    return removed

train_removed = test_and_remove("/home/triuq/projects/detect-waste/classifier/images_square/train")
test_removed  = test_and_remove("/home/triuq/projects/detect-waste/classifier/images_square/test")
print("Removed train:", train_removed)
print("Removed test:", test_removed)