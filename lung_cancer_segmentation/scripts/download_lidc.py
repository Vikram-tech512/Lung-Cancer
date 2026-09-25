import os
import pydicom
from pathlib import Path
from torchvision.transforms import ToPILImage

def convert_dicom_to_png(dicom_path: Path, out_dir: Path):
    ds = pydicom.dcmread(str(dicom_path))
    img = ds.pixel_array.astype('float32')
    # Normalize to 0-255 (HU windowing could be added later)
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    img = img.astype('uint8')
    pil = ToPILImage()(img)
    out_path = out_dir / (dicom_path.stem + ".png")
    pil.save(out_path)

def main():
    # Expect raw DICOM files under 'raw_dicom/' directory.
    raw_dir = Path('raw_dicom')
    img_out = Path('data/images')
    mask_out = Path('data/masks')
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)
    for dcm_file in raw_dir.rglob('*.dcm'):
        convert_dicom_to_png(dcm_file, img_out)
        # Placeholder for mask conversion – user can add logic if masks are provided.

if __name__ == "__main__":
    main()
