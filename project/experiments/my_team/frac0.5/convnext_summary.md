# convnext — fraction=0.5, epochs=100

## Training config
- batch_size: 128
- lr: 0.001
- weight_decay: 0.0001
- seed: 42
- best val_acc (training): 0.8641
- training time: 40.8 min
- weights: convnext_weights.joblib
- training curves: convnext_curves.png

## Self-evaluation (stress-test suites)
- Base (Clean): 0.8700
- Geometric: 0.7294
- Color/Photo: 0.4356
- Noise/Occlude: 0.8219
- AutoAugment: 0.8287
- Ultimate Combo: 0.2794
- **Avg Stress Acc**: 0.6190
- **Final Score**: 0.7445
