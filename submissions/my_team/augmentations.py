import numpy as np
import torch
import torchvision.transforms as transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def rand_bbox(size, lam):
    """Generates random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def apply_cutmix(images, labels, alpha=1.0):
    """Applies CutMix augmentation on a batch of tensors."""
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(images.size()[0]).to(images.device)
    
    target_a = labels
    target_b = labels[rand_index]
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
    
    return images, target_a, target_b, lam

def build_auto_transforms():
    """Applies automatic augmentation policy (RandAugment)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def build_geometric_stress_transforms():
    """Spatial and geometric augmentations."""
    return transforms.Compose([
        transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(degrees=45),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def build_color_stress_transforms():
    """Photometric and color augmentations."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
        transforms.RandomGrayscale(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def build_noise_stress_transforms():
    """Corruption, noise, and occlusion augmentations."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=1.0, scale=(0.05, 0.3), ratio=(0.3, 3.3)),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def build_train_transforms():
    """Standard combined training augmentations."""
    return transforms.Compose([
        transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def build_eval_transforms():
    """Clean validation transforms."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def build_stress_transforms():
    """The ultimate combined stress test."""
    return transforms.Compose([
        transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
        transforms.RandomGrayscale(p=0.5),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.05, 0.3), ratio=(0.3, 3.3)),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])