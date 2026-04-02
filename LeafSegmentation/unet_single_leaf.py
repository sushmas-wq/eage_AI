"""
U-Net for Single Leaf Segmentation
Specialized for extracting and segmenting individual leaves
Features:
- Instance segmentation (separate each leaf)
- Individual leaf extraction
- Post-processing for clean boundaries
- Leaf-by-leaf analysis
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict, Optional

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from scipy import ndimage


# ============================================================================
# CUSTOM LOSS FUNCTIONS FOR LEAF SEGMENTATION
# ============================================================================

def dice_loss(y_true, y_pred, smooth=1.0):
    """Dice loss - better for imbalanced segmentation"""
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def dice_coefficient(y_true, y_pred, smooth=1.0):
    """Dice coefficient metric"""
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def iou_metric(y_true, y_pred, smooth=1.0):
    """Intersection over Union metric"""
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(tf.maximum(y_true_flat, y_pred_flat))
    
    iou = (intersection + smooth) / (union + smooth)
    return iou


# ============================================================================
# U-NET ARCHITECTURE FOR LEAF SEGMENTATION
# ============================================================================

def conv_block(inputs, filters, kernel_size=3, activation='relu', dropout_rate=0.2):
    """Convolution block: Conv2D → BatchNorm → ReLU → Dropout"""
    x = layers.Conv2D(filters, kernel_size, padding='same', 
                      kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    
    return x


def create_unet_for_leaves(img_size=256, num_classes=1, filters_start=32, dropout_rate=0.2):
    """
    Create U-Net model optimized for leaf segmentation.
    
    Args:
        img_size: Input image size (256, 512, etc.)
        num_classes: 1 for binary, 2+ for multi-class
        filters_start: Base number of filters
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Compiled Keras U-Net model
    """
    
    inputs = keras.Input(shape=(img_size, img_size, 3), name='input')
    
    # ========== ENCODER (Contracting Path) ==========
    # Block 1
    c1 = conv_block(inputs, filters_start, dropout_rate=dropout_rate)
    p1 = layers.MaxPooling2D(pool_size=2, strides=2)(c1)
    
    # Block 2
    c2 = conv_block(p1, filters_start * 2, dropout_rate=dropout_rate)
    p2 = layers.MaxPooling2D(pool_size=2, strides=2)(c2)
    
    # Block 3
    c3 = conv_block(p2, filters_start * 4, dropout_rate=dropout_rate)
    p3 = layers.MaxPooling2D(pool_size=2, strides=2)(c3)
    
    # Block 4
    c4 = conv_block(p3, filters_start * 8, dropout_rate=dropout_rate)
    p4 = layers.MaxPooling2D(pool_size=2, strides=2)(c4)
    
    # ========== BOTTLENECK ==========
    c5 = conv_block(p4, filters_start * 16, dropout_rate=dropout_rate)
    
    # ========== DECODER (Expanding Path) ==========
    # Block 6
    u6 = layers.UpSampling2D(size=2)(c5)
    u6 = layers.concatenate([u6, c4], axis=3)
    c6 = conv_block(u6, filters_start * 8, dropout_rate=dropout_rate)
    
    # Block 7
    u7 = layers.UpSampling2D(size=2)(c6)
    u7 = layers.concatenate([u7, c3], axis=3)
    c7 = conv_block(u7, filters_start * 4, dropout_rate=dropout_rate)
    
    # Block 8
    u8 = layers.UpSampling2D(size=2)(c7)
    u8 = layers.concatenate([u8, c2], axis=3)
    c8 = conv_block(u8, filters_start * 2, dropout_rate=dropout_rate)
    
    # Block 9
    u9 = layers.UpSampling2D(size=2)(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = conv_block(u9, filters_start, dropout_rate=dropout_rate)
    
    # ========== OUTPUT ==========
    outputs = layers.Conv2D(num_classes, kernel_size=1, activation='sigmoid', 
                           name='output')(c9)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='UNet_Leaf_Segmentation')
    
    # Compile with appropriate loss for leaf segmentation
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=dice_loss,  # Better than BCE for segmentation
        metrics=[dice_coefficient, iou_metric, 'binary_accuracy']
    )
    
    return model


# ============================================================================
# DATA PREPARATION FOR LEAF SEGMENTATION
# ============================================================================

class LeafDataLoader:
    """Load and prepare training data for leaf segmentation"""
    
    def __init__(self, img_size=256):
        self.img_size = img_size
    
    def load_image_mask_pair(self, img_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load image and mask"""
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=-1)
        
        return img, mask
    
    def load_dataset(self, img_dir: str, mask_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load all images and masks"""
        images = []
        masks = []
        
        img_files = sorted([f for f in os.listdir(img_dir) 
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        for img_file in img_files:
            img_path = os.path.join(img_dir, img_file)
            mask_path = os.path.join(mask_dir, img_file)
            
            if not os.path.exists(mask_path):
                continue
            
            try:
                img, mask = self.load_image_mask_pair(img_path, mask_path)
                images.append(img)
                masks.append(mask)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
                continue
        
        return np.array(images), np.array(masks)
    
    def get_train_val_split(self, images: np.ndarray, masks: np.ndarray, 
                           val_split: float = 0.2) -> Dict:
        """Split into train/validation"""
        X_train, X_val, y_train, y_val = train_test_split(
            images, masks, test_size=val_split, random_state=42
        )
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val
        }


class LeafDataAugmentation:
    """Augmentation pipeline for leaf images"""
    
    def __init__(self):
        self.image_augmenter = ImageDataGenerator(
            rotation_range=45,
            width_shift_range=0.15,
            height_shift_range=0.15,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='reflect'
        )
    
    def augmented_generator(self, images, masks, batch_size=16, seed=42):
        """Generate augmented batches"""
        n_samples = len(images)
        indices = np.arange(n_samples)
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            X_batch = images[batch_indices]
            y_batch = masks[batch_indices]
            
            # Apply augmentation
            aug_images = []
            aug_masks = []
            
            for X, y in zip(X_batch, y_batch):
                X_aug = self.image_augmenter.random_transform(X)
                y_aug = self.image_augmenter.random_transform(y)
                
                aug_images.append(X_aug)
                aug_masks.append(y_aug)
            
            yield np.array(aug_images), np.array(aug_masks)


# ============================================================================
# TRAINING
# ============================================================================

class LeafUNetTrainer:
    """Train U-Net for leaf segmentation"""
    
    def __init__(self, model, model_name='leaf_unet'):
        self.model = model
        self.model_name = model_name
        self.history = None
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=16, use_augmentation=True):
        """Train the model"""
        
        print(f"\n{'='*70}")
        print(f"Training Leaf U-Net")
        print(f"{'='*70}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print(f"{'='*70}\n")
        
        if use_augmentation:
            augmentor = LeafDataAugmentation()
            steps_per_epoch = len(X_train) // batch_size
            
            self.history = self.model.fit(
                augmentor.augmented_generator(X_train, y_train, batch_size),
                validation_data=(X_val, y_val),
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                verbose=1,
                callbacks=self._get_callbacks()
            )
        else:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                callbacks=self._get_callbacks()
            )
        
        return self.history
    
    def _get_callbacks(self):
        """Create training callbacks"""
        return [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                f'{self.model_name}_best.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
    
    def plot_history(self, save_path=None):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Dice Coefficient
        axes[0, 1].plot(self.history.history['dice_coefficient'], label='Train Dice')
        axes[0, 1].plot(self.history.history['val_dice_coefficient'], label='Val Dice')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Coefficient')
        axes[0, 1].set_title('Dice Coefficient')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # IoU
        axes[1, 0].plot(self.history.history['iou_metric'], label='Train IoU')
        axes[1, 0].plot(self.history.history['val_iou_metric'], label='Val IoU')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU')
        axes[1, 0].set_title('Intersection over Union')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1, 1].plot(self.history.history['binary_accuracy'], label='Train Acc')
        axes[1, 1].plot(self.history.history['val_binary_accuracy'], label='Val Acc')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Binary Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ History plot saved: {save_path}")
        
        plt.show()


# ============================================================================
# INFERENCE & SINGLE LEAF EXTRACTION
# ============================================================================

class LeafUNetPredictor:
    """Make predictions and extract individual leaves"""
    
    def __init__(self, model_path=None, model=None, img_size=256):
        """Initialize predictor"""
        if model_path:
            self.model = keras.models.load_model(
                model_path,
                custom_objects={
                    'dice_loss': dice_loss,
                    'dice_coefficient': dice_coefficient,
                    'iou_metric': iou_metric
                }
            )
        else:
            self.model = model
        
        self.img_size = img_size
    
    def predict_leaf_mask(self, image_path: str, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict leaf segmentation mask.
        
        Returns:
            (binary_mask, probability_map)
        """
        # Load and preprocess
        original_img = cv2.imread(image_path)
        original_size = original_img.shape[:2]
        
        img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        
        # Predict
        prediction = self.model.predict(np.expand_dims(img, 0), verbose=0)[0]
        
        # Threshold
        binary_mask = (prediction > threshold).astype(np.uint8) * 255
        
        # Resize back to original
        binary_mask = cv2.resize(binary_mask, (original_size[1], original_size[0]))
        prediction_resized = cv2.resize(
            prediction[:, :, 0], 
            (original_size[1], original_size[0])
        )
        
        return binary_mask, prediction_resized
    
    def extract_individual_leaves(self, binary_mask: np.ndarray, min_area: int = 500) -> List[Dict]:
        """
        Extract individual leaves from mask.
        
        Returns:
            List of leaf info dicts
        """
        # Find connected components
        ret, labels = cv2.connectedComponents(binary_mask, connectivity=8)
        
        leaves = []
        
        for label in range(1, ret):  # Skip background
            leaf_mask = (labels == label).astype(np.uint8) * 255
            
            area = cv2.countNonZero(leaf_mask)
            if area < min_area:
                continue
            
            # Find contour
            contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate properties
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            leaves.append({
                'label': label,
                'mask': leaf_mask,
                'contour': contour,
                'area': area,
                'perimeter': perimeter,
                'circularity': circularity,
                'solidity': solidity,
                'bounding_box': (x, y, w, h)
            })
        
        return leaves
    
    def predict_and_extract(self, image_path: str, threshold: float = 0.5, 
                           min_area: int = 500) -> Tuple[np.ndarray, List[Dict]]:
        """
        Complete pipeline: predict → extract individual leaves.
        
        Returns:
            (probability_map, list_of_leaves)
        """
        binary_mask, prob_map = self.predict_leaf_mask(image_path, threshold)
        leaves = self.extract_individual_leaves(binary_mask, min_area)
        
        return prob_map, leaves
    
    def visualize_prediction(self, image_path: str, threshold: float = 0.5, 
                            save_path: str = None):
        """Visualize prediction results"""
        original_img = cv2.imread(image_path)
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        binary_mask, prob_map = self.predict_leaf_mask(image_path, threshold)
        leaves = self.extract_individual_leaves(binary_mask)
        
        # Create overlay
        overlay = original_img_rgb.copy()
        overlay[binary_mask > 0] = overlay[binary_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].imshow(original_img_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(prob_map, cmap='hot')
        axes[0, 1].set_title('Prediction Probability')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(binary_mask, cmap='gray')
        axes[1, 0].set_title(f'Binary Mask ({len(leaves)} leaves)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Segmentation Overlay')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved: {save_path}")
        
        plt.show()
        
        return leaves


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n🌿 U-Net for Single Leaf Segmentation\n")
    
    # Paths
    IMG_DIR = r"D:/data/images"
    MASK_DIR = r"D:/data/masks"
    IMG_SIZE = 256
    
    # Step 1: Create model
    print("[1/4] Creating U-Net model...")
    model = create_unet_for_leaves(img_size=IMG_SIZE, filters_start=32, dropout_rate=0.2)
    model.summary()
    
    # Step 2: Load data (if available)
    try:
        print("\n[2/4] Loading training data...")
        loader = LeafDataLoader(img_size=IMG_SIZE)
        images, masks = loader.load_dataset(IMG_DIR, MASK_DIR)
        print(f"Loaded {len(images)} image-mask pairs")
        
        # Split data
        data = loader.get_train_val_split(images, masks, val_split=0.2)
        
        # Step 3: Train model
        print("\n[3/4] Training model...")
        trainer = LeafUNetTrainer(model, model_name='leaf_unet')
        history = trainer.train(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            epochs=50,
            batch_size=16,
            use_augmentation=True
        )
        
        # Plot training history
        trainer.plot_history('training_history.png')
        
        # Save model
        model.save('leaf_unet_model.h5')
        print("✓ Model saved: leaf_unet_model.h5")
        
        # Step 4: Make predictions
        print("\n[4/4] Making predictions...")
        predictor = LeafUNetPredictor(model=model, img_size=IMG_SIZE)
        
        # Test on validation set
        test_image_path = f"{IMG_DIR}/{os.listdir(IMG_DIR)[0]}"
        prob_map, leaves = predictor.predict_and_extract(test_image_path)
        
        print(f"✓ Found {len(leaves)} individual leaves")
        
        # Visualize
        predictor.visualize_prediction(test_image_path)
        
    except FileNotFoundError:
        print(f"Data directories not found: {IMG_DIR}, {MASK_DIR}")
        print("\nTo train, create:")
        print(f"  {IMG_DIR}/ - RGB leaf images")
        print(f"  {MASK_DIR}/ - Grayscale segmentation masks")
        print("\nFor now, the model is created and ready for training.")
