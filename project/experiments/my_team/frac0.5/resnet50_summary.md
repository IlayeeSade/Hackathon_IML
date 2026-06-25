# resnet50 — fraction=0.5, epochs=100

## Training config
- batch_size: 128
- lr: 0.001
- weight_decay: 0.0001
- seed: 42
- best val_acc (training): 0.8814
- training time: 37.2 min
- weights: resnet50_weights.joblib
- training curves: resnet50_curves.png

## Self-evaluation (stress-test suites)
- Base (Clean): 0.8875
- Geometric: 0.7031
- Color/Photo: 0.5437
- Noise/Occlude: 0.7231
- AutoAugment: 0.8606
- Ultimate Combo: 0.2669
- **Avg Stress Acc**: 0.6195
- **Final Score**: 0.7535
