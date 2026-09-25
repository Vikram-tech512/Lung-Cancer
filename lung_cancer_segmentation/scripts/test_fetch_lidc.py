import urllib.request
import json
import io
import pydicom
import numpy as np
from PIL import Image

def test():
    series_uid = "1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192"
    sop_url = f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getSOPInstanceUIDs?SeriesInstanceUID={series_uid}"
    req = urllib.request.Request(sop_url, headers={"User-Agent": "Mozilla/5.0"})
    res = urllib.request.urlopen(req, timeout=15)
    sops = json.loads(res.read())
    print(f"Total slices in series: {len(sops)}")

    # Let's inspect a slice near the middle (e.g. index 65)
    target_sop = sops[len(sops)//2]["SOPInstanceUID"]
    img_url = f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getSingleImage?SeriesInstanceUID={series_uid}&SOPInstanceUID={target_sop}"
    print(f"Downloading slice: {target_sop}...")
    req2 = urllib.request.Request(img_url, headers={"User-Agent": "Mozilla/5.0"})
    dcm_bytes = urllib.request.urlopen(req2, timeout=20).read()
    
    ds = pydicom.dcmread(io.BytesIO(dcm_bytes))
    print(f"PatientID: {ds.PatientID}")
    print(f"Modality: {ds.Modality}, Rows: {ds.Rows}, Cols: {ds.Columns}")
    print(f"Pixel Spacing: {getattr(ds, 'PixelSpacing', 'N/A')}")
    print(f"Slice Location: {getattr(ds, 'SliceLocation', 'N/A')}")
    print(f"Rescale Intercept: {getattr(ds, 'RescaleIntercept', 0)}, Rescale Slope: {getattr(ds, 'RescaleSlope', 1)}")
    
    pixel_array = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = pixel_array * slope + intercept
    print(f"HU Range: min={hu.min():.1f}, max={hu.max():.1f}, mean={hu.mean():.1f}")
    
    # Apply standard clinical lung window (WL = -600, WW = 1500)
    wl, ww = -600, 1500
    lower = wl - ww / 2
    upper = wl + ww / 2
    windowed = np.clip((hu - lower) / (upper - lower) * 255.0, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(windowed)
    img.save("data/test_real_slice.png")
    print("Saved test_real_slice.png successfully!")

if __name__ == "__main__":
    test()
