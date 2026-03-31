import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, List
from scipy import ndimage


class LeafSegmenter:
    """
    A comprehensive leaf segmentation tool supporting multiple algorithms:
    - Color-based segmentation (HSV)
    - Morphological operations
    - Watershed algorithm
    - Contour-based detection
    """

    def __init__(self, image_path: str):
        """
        Initialize the segmenter with an image.
        
        Args:
            image_path: Path to the image file
        """
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.hsv_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        self.height, self.width = self.original_image.shape[:2]

    def segment_by_color(self, lower_hue: Tuple[int, int, int] = (35, 40, 40),
                        upper_hue: Tuple[int, int, int] = (85, 255, 255)) -> np.ndarray:
        """
        Segment leaves using HSV color range (detects green leaves).
        
        Args:
            lower_hue: Lower HSV bounds (H, S, V)
            upper_hue: Upper HSV bounds (H, S, V)
            
        Returns:
            Binary mask of segmented leaves
        """
        # Create mask for green color range
        mask = cv2.inRange(self.hsv_image, lower_hue, upper_hue)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return mask

    def segment_by_watershed(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment leaves using the Watershed algorithm.
        
        Returns:
            Tuple of (markers, segmented_image)
        """
        # Start with color-based mask
        mask = self.segment_by_color()
        
        # Find sure background and foreground
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sure_bg = cv2.dilate(mask, kernel, iterations=3)
        
        # Finding sure foreground
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Finding unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Label connected components
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(self.original_image, markers)
        
        # Create segmented image
        segmented = self.original_image.copy()
        segmented[markers == -1] = [0, 0, 255]  # Mark boundaries in red
        
        return markers, segmented

    def segment_by_contours(self, min_area: int = 500) -> Tuple[np.ndarray, List]:
        """
        Segment leaves by detecting and analyzing contours.
        
        Args:
            min_area: Minimum area to consider as a leaf
            
        Returns:
            Tuple of (mask, contours)
        """
        # Get initial mask
        mask = self.segment_by_color()
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        # Create output mask with filtered contours
        output_mask = np.zeros_like(mask)
        cv2.drawContours(output_mask, valid_contours, -1, 255, -1)
        
        return output_mask, valid_contours

    def apply_morphological_operations(self, mask: np.ndarray,
                                      open_kernel: int = 5,
                                      close_kernel: int = 5) -> np.ndarray:
        """
        Apply morphological operations to clean up the mask.
        
        Args:
            mask: Input binary mask
            open_kernel: Size of kernel for opening operation
            close_kernel: Size of kernel for closing operation
            
        Returns:
            Cleaned mask
        """
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        
        # Opening: remove small noise
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        # Closing: fill small holes
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        return cleaned

    def extract_leaf_properties(self, contours: List) -> List[dict]:
        """
        Extract properties of detected leaves.
        
        Args:
            contours: List of contours
            
        Returns:
            List of dictionaries containing leaf properties
        """
        properties = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Circularity
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Aspect ratio
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Convex hull
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            properties.append({
                'id': i,
                'area': area,
                'perimeter': perimeter,
                'circularity': circularity,
                'bounding_box': (x, y, w, h),
                'aspect_ratio': aspect_ratio,
                'solidity': solidity,
                'contour': contour
            })
        
        return properties

    def visualize_results(self, mask: np.ndarray, output_path: str = None):
        """
        Visualize segmentation results.
        
        Args:
            mask: Binary mask
            output_path: Path to save visualization
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].imshow(self.rgb_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Segmented Leaves')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        plt.show()

    def segment_and_label(self) -> Tuple[np.ndarray, int]:
        """
        Segment leaves and label individual leaves.
        
        Returns:
            Tuple of (labeled_image, number_of_leaves)
        """
        mask = self.segment_by_color()
        labeled_array, num_features = ndimage.label(mask)
        
        return labeled_array, num_features


# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "D:/test/image.jpg"
    
    # Initialize segmenter
    segmenter = LeafSegmenter(image_path)
    
    # Method 1: Color-based segmentation
    print("Performing color-based segmentation...")
    color_mask = segmenter.segment_by_color()
    cleaned_mask = segmenter.apply_morphological_operations(color_mask)
    
    # Method 2: Contour-based segmentation
    print("Performing contour-based segmentation...")
    contour_mask, contours = segmenter.segment_by_contours(min_area=500)
    
    # Method 3: Watershed segmentation
    print("Performing watershed segmentation...")
    markers, watershed_result = segmenter.segment_by_watershed()
    
    # Extract properties
    properties = segmenter.extract_leaf_properties(contours)
    print(f"Detected {len(properties)} leaves")
    
    for prop in properties:
        print(f"Leaf {prop['id']}: Area={prop['area']:.0f}, "
              f"Circularity={prop['circularity']:.3f}, "
              f"Solidity={prop['solidity']:.3f}")
    
    # Visualize results
    segmenter.visualize_results(cleaned_mask, "segmentation_result.png")
    
    # Label individual leaves
    labeled_image, num_leaves = segmenter.segment_and_label()
    print(f"Total leaves labeled: {num_leaves}")