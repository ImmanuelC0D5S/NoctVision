"""
Custom Albumentations Augmentations for NoctVision.
Applies low-light simulation, synthetic fog, synthetic rain, and weather effects
specifically tailored for visible camera images in night-time/low-light pedestrian detection.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


def get_noctvision_transforms(img_size: int = 640, is_train: bool = True) -> A.Compose:
    """
    Returns Albumentations transform pipeline for training or evaluation.

    Includes low-light simulation, synthetic fog, rain effects,
    RandomBrightnessContrast, Gamma, Gaussian Noise, and Blur.
    """
    if is_train:
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.OneOf(
                    [
                        A.RandomGamma(gamma_limit=(30, 80), p=0.7),
                        A.RandomBrightnessContrast(
                            brightness_limit=(-0.5, -0.1),
                            contrast_limit=(-0.3, 0.2),
                            p=0.7,
                        ),
                    ],
                    p=0.8,
                ),
                A.OneOf(
                    [
                        A.RandomRain(drop_length=15, drop_width=1, drop_color=(200, 200, 200), blur_value=3, p=0.5),
                        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.5),
                        A.MotionBlur(blur_limit=(3, 7), p=0.3),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
                    ],
                    p=0.4,
                ),
                A.HorizontalFlip(p=0.5),
            ],
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_visibility=0.2,
            ),
        )
    else:
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
            ],
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
            ),
        )


def get_offline_augmentations(img_size: int = 640, num_augmentations: int = 2) -> list[A.Compose]:
    """
    Returns a list of augmentation pipelines for offline (pre-baked) augmentation.

    Each pipeline applies a different combination of low-light / weather / noise effects
    so that multiple augmented copies of each image are generated with varied conditions.
    """
    pipelines = []

    for i in range(num_augmentations):
        transforms = [
            A.Resize(height=img_size, width=img_size),
        ]

        # Vary low-light simulation per pipeline
        gamma_range = (30 + i * 5, 80 + i * 5)
        brightness = (-0.5, -0.1 - i * 0.05)
        transforms.append(
            A.OneOf(
                [
                    A.RandomGamma(gamma_limit=gamma_range, p=0.7),
                    A.RandomBrightnessContrast(
                        brightness_limit=brightness,
                        contrast_limit=(-0.3, 0.2),
                        p=0.7,
                    ),
                ],
                p=0.8,
            )
        )

        # Alternate weather effects across pipelines
        weather_choices = [
            A.RandomRain(drop_length=15, drop_width=1, drop_color=(200, 200, 200), blur_value=3, p=0.5),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.5),
            A.MotionBlur(blur_limit=(3, 7), p=0.3),
        ]
        # Pick weather effect based on pipeline index
        selected_weather = [weather_choices[i % len(weather_choices)]]
        transforms.append(A.OneOf(selected_weather, p=0.5))

        # Vary noise levels
        noise_var = (10.0 + i * 5, 50.0 + i * 10)
        transforms.append(
            A.OneOf(
                [
                    A.GaussNoise(var_limit=noise_var, p=0.5),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
                ],
                p=0.4,
            )
        )

        transforms.append(A.HorizontalFlip(p=0.5))

        pipelines.append(
            A.Compose(
                transforms,
                bbox_params=A.BboxParams(
                    format="yolo",
                    label_fields=["class_labels"],
                    min_visibility=0.2,
                ),
            )
        )

    return pipelines


if __name__ == "__main__":
    print("Testing Albumentations pipeline for NoctVision...")
    transforms = get_noctvision_transforms(img_size=640, is_train=True)
    print("Augmentation pipeline initialized successfully.")

    offline = get_offline_augmentations(img_size=640, num_augmentations=3)
    print(f"Offline augmentation pipelines: {len(offline)}")
