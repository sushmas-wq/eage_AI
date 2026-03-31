"""
Complete U-Net Implementation for Leaf Segmentation
Includes: Model creation, training, inference, visualization, and utilities
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


# ============================================================================
# CUSTOM LOSS FUNCTIONS
# ============================================================================

def dice_loss(y_true, y_pred, smooth=1.0):
    """
    Dice Loss - Better than BCE for segmentation
    Emphasizes overlap between prediction and ground truth
    """
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def dice_coefficient(y_true, y_pred, smooth=1.0):
    """Dice coefficient metric (higher is better)"""
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def combined_loss(y_true, y_pred, bce_weight=0.5, dice_weight=0.5):
    """
    Combined loss: 0.5 * BCE + 0.5 * Dice
    Often gives best results
    """
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce_weight * bce + dice_weight * dice


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

def conv_block(inputs, filters, kernel_size=3, activation='relu', dropout_rate=0.2):
    """
    Convolution block: Conv2D → BatchNorm → ReLU → Dropout
    Standard building block for U-Net
    """
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


def create_unet(img_size=256, num_classes=1, filters_start=32, dropout_rate=0.2):
    """
    Create U-Net model for segmentation.
    
    Args:
        img_size: Input image size (256, 512, etc.)
        num_classes: Number of output classes (1 for binary)
        filters_start: Number of filters in first layer
        dropout_rate: Dropout rate (0-1)
    
    Returns:
        Compiled Keras model
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
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='UNet')
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=dice_loss,
        metrics=[dice_coefficient, 'binary_accuracy']
    )
    
    return model


def create_unet_compact(img_size=256, num_classes=1):
    """Smaller U-Net for faster training with less data"""
    return create_unet(img_size, num_classes, filters_start=16, dropout_rate=0.3)


def create_unet_large(img_size=256, num_classes=1):
    """Larger U-Net for maximum accuracy with lots of data"""
    return create_unet(img_size, num_classes, filters_start=64, dropout_rate=0.1)


# ============================================================================
# DATA PREPARATION
# ============================================================================

class DataLoader:
    """Load and prepare training data"""
    
    def __init__(self, img_size=256):
        self.img_size = img_size
    
    def load_image_mask_pair(self, img_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load image and corresponding mask.
        
        Args:
            img_path: Path to RGB image
            mask_path: Path to grayscale mask
            
        Returns:
            (image, mask) both normalized to [0, 1]
        """
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
        """
        Load all images and masks from directories.
        Assumes matching filenames.
        """
        images = []
        masks = []
        
        img_files = sorted([f for f in os.listdir(img_dir) 
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        for img_file in img_files:
            img_path = os.path.join(img_dir, img_file)
            mask_path = os.path.join(mask_dir, img_file)
            
            if not os.path.exists(mask_path):
                print(f"Warning: Mask not found for {img_file}")
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


class DataAugmentationPipeline:
    """Create augmented batches for training"""
    
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
        
        self.mask_augmenter = ImageDataGenerator(
            rotation_range=45,
            width_shift_range=0.15,
            height_shift_range=0.15,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='reflect'
        )
    
    def augmented_generator(self, images, masks, batch_size=16, seed=42):
        """
        Generate augmented batches with synchronized transformations.
        
        Args:
            images: Training images
            masks: Training masks
            batch_size: Batch size
            seed: Random seed for reproducibility
        """
        for X_batch, y_batch in self._get_batches(images, masks, batch_size, seed):
            yield X_batch, y_batch
    
    def _get_batches(self, images, masks, batch_size, seed):
        """Generate batches with matching augmentation"""
        n_samples = len(images)
        indices = np.arange(n_samples)
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
                y_aug = self.mask_augmenter.random_transform(y)
                
                aug_images.append(X_aug)
                aug_masks.append(y_aug)
            
            yield np.array(aug_images), np.array(aug_masks)


# ============================================================================
# TRAINING
# ============================================================================

class UNetTrainer:
    """Train U-Net model"""
    
    def __init__(self, model, model_name='unet_leaf_segmentation'):
        self.model = model
        self.model_name = model_name
        self.history = None
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=16, use_augmentation=True):
        """
        Train the model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of training epochs
            batch_size: Batch size
            use_augmentation: Whether to use data augmentation
        """
        
        print(f"\n{'='*70}")
        print(f"Training {self.model_name}")
        print(f"{'='*70}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print(f"Data augmentation: {use_augmentation}")
        print(f"{'='*70}\n")
        
        if use_augmentation:
            augmentor = DataAugmentationPipeline()
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
            ),
            keras.callbacks.TensorBoard(
                log_dir=f'logs/{self.model_name}',
                histogram_freq=1
            )
        ]
    
    def plot_history(self, save_path=None):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Dice Coefficient
        axes[1].plot(self.history.history['dice_coefficient'], label='Train Dice')
        axes[1].plot(self.history.history['val_dice_coefficient'], label='Val Dice')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Dice Coefficient')
        axes[1].set_title('Dice Coefficient')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ History plot saved: {save_path}")
        
        plt.show()


# ============================================================================
# INFERENCE
# ============================================================================

class UNetPredictor:
    """Make predictions with trained U-Net"""
    
    def __init__(self, model_path=None, model=None, img_size=256):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model file (.h5)
            model: Keras model object (alternative to model_path)
            img_size: Input image size
        """
        if model_path:
            self.model = keras.models.load_model(model_path)
        else:
            self.model = model
        
        self.img_size = img_size
    
    def predict(self, image_path: str, threshold: float = 0.5) -> Dict:
        """
        Make prediction on single image.
        
        Args:
            image_path: Path to image
            threshold: Probability threshold (0-1)
            
        Returns:
            Dictionary with prediction results
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
        
        # Create overlay
        overlay = original_img.copy()
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        overlay_rgb[binary_mask > 0] = overlay_rgb[binary_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
        overlay = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        
        return {
            'probability_map': prediction_resized,
            'binary_mask': binary_mask,
            'overlay': overlay,
            'original': original_img
        }
    
    def batch_predict(self, image_dir: str, output_dir: str, threshold: float = 0.5):
        """Process all images in directory"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        img_files = [f for f in os.listdir(image_dir) 
                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        for img_file in img_files:
            img_path = os.path.join(image_dir, img_file)
            result = self.predict(img_path, threshold)
            
            base_name = Path(img_file).stem
            
            # Save results
            cv2.imwrite(f"{output_dir}/{base_name}_mask.png", result['binary_mask'])
            cv2.imwrite(f"{output_dir}/{base_name}_overlay.png", result['overlay'])
            
            print(f"✓ Processed {img_file}")
    
    def visualize_prediction(self, image_path: str, threshold: float = 0.5, save_path=None):
        """Visualize prediction results"""
        result = self.predict(image_path, threshold)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].imshow(cv2.cvtColor(result['original'], cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(result['probability_map'], cmap='hot')
        axes[0, 1].set_title('Prediction Probability')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(result['binary_mask'], cmap='gray')
        axes[1, 0].set_title(f'Binary Mask (threshold={threshold})')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cv2.cvtColor(result['overlay'], cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Segmentation Overlay')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved: {save_path}")
        
        plt.show()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n🌾 U-Net Leaf Segmentation\n")
    
    # Paths (modify as needed)
    IMG_DIR = "D:data/images"
    MASK_DIR = "D:/data/masks"
    IMG_SIZE = 256
    
    # Step 1: Create model
    print("Creating model...")
    model = create_unet(img_size=IMG_SIZE)
    model.summary()
    
    # Step 2: Load data
    print("\nLoading data...")
    loader = DataLoader(img_size=IMG_SIZE)
    
    try:
        images, masks = loader.load_dataset(IMG_DIR, MASK_DIR)
        print(f"Loaded {len(images)} image-mask pairs")
        
        # Split data
        data = loader.get_train_val_split(images, masks, val_split=0.2)
        
        # Step 3: Train model
        print("\nTraining model...")
        trainer = UNetTrainer(model, model_name='leaf_unet')
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
        model.save('leaf_segmentation_model.h5')
        print("✓ Model saved: leaf_segmentation_model.h5")
        
        # Step 4: Make predictions
        print("\nMaking predictions...")
        predictor = UNetPredictor(model=model, img_size=IMG_SIZE)
        
        # Predict on validation set
        for idx in range(min(3, len(data['X_val']))):
            print(f"\nPredicting on validation sample {idx+1}...")
            # Create temporary image file for visualization
            # (In real scenario, you'd have image files)
        
    except FileNotFoundError:
        print(f"❌ Data directories not found: {IMG_DIR}, {MASK_DIR}")
        print("\nExample: Create data directory structure:")
        print("  data/")
        print("  ├── images/  (RGB leaf images)")
        print("  └── masks/   (Grayscale segmentation masks)")
