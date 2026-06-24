# resnet50 — fraction=0.02, epochs=1

## Training config
- batch_size: 32
- lr: 0.001
- weight_decay: 0.0001
- seed: 42
- best val_acc (training): 0.0499
- training time: 0.5 min
- weights: resnet50_weights.joblib
- training curves: resnet50_curves.png

## Self-evaluation (stress-test suites)
- Base (Clean): 0.0500
- Geometric: 0.0500
- Color/Photo: 0.0500
- Noise/Occlude: 0.0500
- AutoAugment: 0.0500
- Ultimate Combo: 0.0500
- **Avg Stress Acc**: 0.0500
- **Final Score**: 0.0500
