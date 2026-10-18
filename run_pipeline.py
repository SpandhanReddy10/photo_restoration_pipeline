import os
import sys
import torch
import numpy as np
from PIL import Image

# --- SETUP PATHS ---
BASE_DIR = r"C:\photo_restoration"
INPUT_DIR = os.path.join(BASE_DIR, "input_photos")
UPSCALED_DIR = os.path.join(BASE_DIR, "upscaled_photos")
COLORIZED_DIR = os.path.join(BASE_DIR, "colorized_photos")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

os.makedirs(UPSCALED_DIR, exist_ok=True)
os.makedirs(COLORIZED_DIR, exist_ok=True)

# --- IMPORT REAL-ESRGAN ---
# No need to append Real-ESRGAN path since we installed it via pip
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# --- IMPORT DEOLDIFY (CRITICAL FIX) ---
# Change directory FIRST so Python finds the bundled fastai
DEOLDIFY_DIR = os.path.join(BASE_DIR, "DeOldify")
os.chdir(DEOLDIFY_DIR) 
sys.path.insert(0, DEOLDIFY_DIR)

from deoldify import device
from deoldify.device_id import DeviceId
from deoldify.visualize import get_image_colorizer

def main():
    print("🚀 Starting Photo Restoration Pipeline...")
    
    print("⏳ Loading Real-ESRGAN model...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    # Updated path to look in C:\photo_restoration\weights\
    realesrgan_weights = os.path.join(WEIGHTS_DIR, "RealESRGAN_x4plus.pth")
    
    upsampler = RealESRGANer(
        scale=4,
        model_path=realesrgan_weights,
        model=model,
        tile=256,      
        tile_pad=10,
        pre_pad=0,
        half=False
    )
    print("✅ Real-ESRGAN loaded.")

    print("⏳ Loading DeOldify model...")
    device.set(device=DeviceId.GPU0)
    colorizer = get_image_colorizer(artistic=True)
    render_factor = 20 
    print("✅ DeOldify loaded.")

    valid_extensions = ('.png', '.jpg', '.jpeg')
    photos = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]
    
    if not photos:
        print(f"❌ No photos found in {INPUT_DIR}")
        print("Please drop some .jpg or .png files in there and run again.")
        return

    print(f"✅ Found {len(photos)} photos to process.\n")

    for filename in photos:
        print(f"🖼️ Processing: {filename}")
        
        input_path = os.path.join(INPUT_DIR, filename)
        upscaled_path = os.path.join(UPSCALED_DIR, filename)
        colorized_path = os.path.join(COLORIZED_DIR, f"colorized_{filename}")

        print("   -> Upscaling...")
        img = Image.open(input_path).convert("RGB")
        sr_image, _ = upsampler.enhance(np.array(img), outscale=2)
        Image.fromarray(sr_image).save(upscaled_path)

        print("   -> Colorizing...")
        result = colorizer.get_transformed_image(upscaled_path, render_factor=render_factor)
        result.save(colorized_path)
        
        print(f"   ✅ Done! Saved to colorized_photos\n")

    print("🎉 All photos processed successfully!")

if __name__ == "__main__":
    main()