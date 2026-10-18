import sys; sys.path.insert(0, '.')
pairs = [
    ('src.inference.engine',         'InferenceEngine'),
    ('src.inference.service',        'InferenceService'),
    ('src.explainability.gradcam',   'GradCAM'),
    ('src.explainability.attention_viz', 'AttentionVisualizer'),
    ('src.robustness.analyzer',      'RobustnessAnalyzer'),
    ('src.robustness.perturbations', 'apply_perturbation'),
    ('src.dataset.lung_dataset',     'LungDataset'),
]
for mod, cls in pairs:
    try:
        m = __import__(mod, fromlist=[cls])
        getattr(m, cls)
        print('[PASS]', mod + '.' + cls)
    except Exception as e:
        print('[FAIL]', mod, str(e))
import pathlib
jsons = list(pathlib.Path('.').rglob('*.json'))
print('[INFO] Result JSON files found:', len(jsons))
for j in sorted(jsons)[:10]:
    print('  ', j)
