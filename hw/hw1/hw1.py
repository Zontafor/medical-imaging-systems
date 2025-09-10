import os, cv2
import pydicom
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Directories
input_dir = "/Users/mlwu/Documents/Academia/USC/BME/527/hw/hw1/code/data/dicom_images"
output_dir = "/Users/mlwu/Documents/Academia/USC/BME/527/hw/hw1/code/data_out/extracted_figs"

os.makedirs(output_dir, exist_ok=True)

# Loop through images 01–20
for i in range(1, 21):
    filename = f"image{i:02d}.dcm"
    filepath = os.path.join(input_dir, filename)
    
    # Read DICOM file
    dicom_data = pydicom.dcmread(filepath)
    img_array = dicom_data.pixel_array

    # Display image
    plt.imshow(img_array, cmap="gray")
    plt.axis("off")
    plt.show()

    # Save as JPG
    output_file = os.path.join(output_dir, f"image{i:02d}.jpg")
    img = Image.fromarray((img_array / np.max(img_array) * 255).astype(np.uint8))
    img.save(output_file)
    print(f"Saved {output_file}")

# Reload original image to avoid cumulative borders
img_path = os.path.join(output_dir, "image01.jpg")
img = cv2.imread(img_path)

# Define label positions (adjusted for thoracic slice)
h, w = img.shape[:2]
labels = {
    "Trachea": (w//2, h//3),
    "Right lung": (int(w*0.25), int(h*0.45)),
    "Left lung": (int(w*0.75), int(h*0.45)),
    "Right clavicle": (int(w*0.25), int(h*0.2)),
    "Left clavicle": (int(w*0.75), int(h*0.2)),
    "Vertebral body": (w//2, int(h*0.55)),
    "Spinal canal": (w//2, int(h*0.6)),
    "Subclavian artery": (int(w*0.45), int(h*0.35)),
    "Pectoralis muscle": (int(w*0.15), int(h*0.4)),
    "Paraspinal muscles": (int(w*0.65), int(h*0.55))
}

# Overlay labels
for label, (x, y) in labels.items():
    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 255, 0), 1, cv2.LINE_AA)

# Save labeled output
output_labeled = os.path.join(output_dir, "image01_labeled.jpg")
cv2.imwrite(output_labeled, img)
print(f"Saved labeled image at {output_labeled}")

# Display with no borders/padding
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.show()