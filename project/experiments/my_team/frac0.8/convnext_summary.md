# convnext — fraction=0.8, epochs=100

## Training config
- batch_size: 128
- lr: 0.001
- weight_decay: 0.0001
- seed: 42
- best val_acc (training): 0.9497
- training time: 53.8 min
- weights: convnext_weights.joblib
- training curves: convnext_curves.png

## Self-evaluation (stress-test suites)
- Base (Clean): 0.9544
- Geometric: 0.8194
- Color/Photo: 0.5887
- Noise/Occlude: 0.9213
- AutoAugment: 0.9306
- Ultimate Combo: 0.3406
- **Avg Stress Acc**: 0.7201
- **Final Score**: 0.8373
