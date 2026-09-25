# scratch/test_upload.py
import requests

file_path = r'lung_cancer_segmentation/data/real_world_dataset/malignant_large_cell/large_cell_006.png'
with open(file_path, 'rb') as f:
    files = {'file': ('large_cell_006.png', f, 'image/png')}
    data = {'architecture': 'hybrid'}
    try:
        r = requests.post('http://localhost:8000/api/analyze-ct', files=files, data=data)
        print("Status code:", r.status_code)
        if r.status_code != 200:
            print("Response:", r.text)
        else:
            j = r.json()
            print("Prediction:", j.get('classification'))
    except Exception as e:
        print("Exception:", e)
