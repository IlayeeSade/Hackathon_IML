# resnet50 — fraction=0.8, epochs=100

## Training config
- batch_size: 128
- lr: 0.001
- weight_decay: 0.0001
- seed: 42
- best val_acc (training): 0.9516
- training time: 49.1 min
- weights: resnet50_weights.joblib
- training curves: resnet50_curves.png

## Self-evaluation (stress-test suites)
- Base (Clean): 0.9537
- Geometric: 0.7906
- Color/Photo: 0.6669
- Noise/Occlude: 0.8081
- AutoAugment: 0.9313
- Ultimate Combo: 0.3331
- **Avg Stress Acc**: 0.7060
- **Final Score**: 0.8299
