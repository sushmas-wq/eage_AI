"""
Watershed Segmentation - Practical Implementation
Complete with visualization and parameter tuning
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os


class WatershedSegmenter:
    """
    Flexible watershed segmentation for image analysis.
    Supports different distance transforms and peak detection strategies.
    """
    
    def __init__(self, image_path, debug=False):
        """
        Initialize segmenter.
        
        Args:
            image_path: Path to input image
            debug: Enable debug visualization
        """
        self.image_path = image_path
        self.debug = debug
        
        self.image_bgr = cv2.imread(image_path)
        if self.image_bgr is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.image_bgr.shape[:2]
        
    def create_foreground_mask(self, lower_hsv, upper_hsv):
        """
        Create binary mask of foreground objects.
        
        Args:
            lower_hsv: Lower HSV threshold
            upper_hsv: Upper HSV threshold
            
        Returns:
            Binary mask
        """
        hsv = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # Clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return mask
    
    def compute_distance_transform(self, mask, method='L2'):
        """
        Compute distance transform (topographic map).
        
        Args:
            mask: Binary foreground mask
            method: 'L2' (Euclidean), 'L1' (Manhattan), 'C' (Chessboard)
            
        Returns:
            Distance map
        """
        method_map = {
            'L2': cv2.DIST_L2,
            'L1': cv2.DIST_L1,
            'C': cv2.DIST_C
        }
        
        dist = cv2.distanceTransform(mask, method_map[method], cv2.DIST_MASK_PRECISE)
        return dist
    
    def detect_peaks(self, distance_map, threshold_ratio=0.7, min_size=3):
        """
        Detect peaks in distance map (object centers).
        
        Args:
            distance_map: Output from compute_distance_transform
            threshold_ratio: Threshold as ratio of max (0-1)
            min_size: Size of erosion kernel
            
        Returns:
            Binary map of peaks
        """
        max_dist = distance_map.max()
        if max_dist == 0:
            return np.zeros_like(distance_map, dtype=np.uint8)
        
        threshold = threshold_ratio * max_dist
        _, peaks = cv2.threshold(distance_map, threshold, 255, 0)
        peaks = np.uint8(peaks)
        
        # Erode to separate closely-packed peaks
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_size, min_size))
        peaks = cv2.erode(peaks, kernel, iterations=1)
        
        return peaks
    
    def label_markers(self, peaks, mask):
        """
        Create labeled markers for watershed.
        
        Args:
            peaks: Binary peak map
            mask: Original foreground mask
            
        Returns:
            Labeled marker image (background=0, markers=1,2,3,...)
        """
        # Label connected components in peaks
        ret, markers = cv2.connectedComponents(peaks.astype(np.uint8))
        
        # Background stays 0, markers start at 1
        # But we need to shift so watershed can mark boundaries as -1
        # Standard: background=0, objects=1,2,3,..., watershed boundary=-1
        
        return markers
    
    def apply_watershed(self, markers):
        """
        Apply watershed algorithm.
        
        Args:
            markers: Labeled marker image from label_markers
            
        Returns:
            Segmented image with boundaries marked
        """
        markers = cv2.watershed(self.image_bgr, markers.copy())
        return markers
    
    def segment(self, lower_hsv=np.array([25, 20, 20]), 
                upper_hsv=np.array([95, 255, 255]),
                threshold_ratio=0.7, min_area=500,
                distance_method='L2', peak_min_size=3):
        """
        Complete segmentation pipeline.
        
        Args:
            lower_hsv, upper_hsv: HSV color range
            threshold_ratio: Peak detection threshold (0-1)
            min_area: Minimum object area to keep
            distance_method: Distance transform method
            peak_min_size: Size of erosion kernel for peak detection
            
        Returns:
            Dictionary with results
        """
        print("[1/5] Creating foreground mask...")
        mask = self.create_foreground_mask(lower_hsv, upper_hsv)
        
        print("[2/5] Computing distance transform...")
        dist = self.compute_distance_transform(mask, method=distance_method)
        dist_normalized = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        print("[3/5] Detecting peaks...")
        peaks = self.detect_peaks(dist, threshold_ratio, peak_min_size)
        
        print("[4/5] Labeling markers...")
        markers = self.label_markers(peaks, mask)
        
        print("[5/5] Applying watershed...")
        result_markers = self.apply_watershed(markers)
        
        # Extract results
        result = {
            'markers': result_markers,
            'mask': mask,
            'distance_map': dist,
            'distance_normalized': dist_normalized,
            'peaks': peaks,
            'num_objects': len(np.unique(result_markers[result_markers > 1]))
        }
        
        # Extract individual objects
        result['objects'] = self.extract_objects(result_markers, min_area)
        
        if self.debug:
            self.save_debug_images(result)
        
        return result
    
    def extract_objects(self, markers, min_area=500):
        """
        Extract individual objects from markers.
        
        Args:
            markers: Output from watershed
            min_area: Minimum object size
            
        Returns:
            List of object dictionaries
        """
        objects = []
        unique_labels = np.unique(markers)
        
        for label in unique_labels:
            if label <= 0:  # Skip background and boundaries
                continue
            
            # Create object mask
            obj_mask = (markers == label).astype(np.uint8) * 255
            area = cv2.countNonZero(obj_mask)
            
            if area < min_area:
                continue
            
            # Find contour
            contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            contour = max(contours, key=cv2.contourArea)
            
            # Extract properties
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Circularity
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            # Convexity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            objects.append({
                'label': label,
                'mask': obj_mask,
                'contour': contour,
                'area': area,
                'perimeter': perimeter,
                'circularity': circularity,
                'solidity': solidity,
                'bounding_box': (x, y, w, h)
            })
        
        return objects
    
    def visualize_watershed_steps(self, result, output_path=None):
        """
        Visualize all watershed processing steps.
        
        Args:
            result: Output from segment()
            output_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Watershed Segmentation Steps', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(self.image_rgb)
        axes[0, 0].set_title('1. Original Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Foreground mask
        axes[0, 1].imshow(result['mask'], cmap='gray')
        axes[0, 1].set_title('2. Foreground Mask', fontweight='bold')
        axes[0, 1].axis('off')
        
        # Distance transform
        axes[0, 2].imshow(result['distance_normalized'], cmap='hot')
        axes[0, 2].set_title('3. Distance Transform (Topography)', fontweight='bold')
        axes[0, 2].axis('off')
        
        # Peaks
        axes[1, 0].imshow(result['peaks'], cmap='gray')
        axes[1, 0].set_title('4. Detected Peaks (Seeds)', fontweight='bold')
        axes[1, 0].axis('off')
        
        # Watershed result (colored)
        colored_markers = self._color_markers(result['markers'])
        axes[1, 1].imshow(colored_markers)
        axes[1, 1].set_title(f'5. Watershed Result ({result["num_objects"]} objects)', 
                            fontweight='bold')
        axes[1, 1].axis('off')
        
        # Objects with contours
        obj_image = self.image_rgb.copy()
        for obj in result['objects']:
            cv2.drawContours(obj_image, [obj['contour']], 0, (0, 255, 0), 2)
            x, y, w, h = obj['bounding_box']
            cv2.putText(obj_image, str(obj['label']), (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        axes[1, 2].imshow(obj_image)
        axes[1, 2].set_title('6. Labeled Objects', fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved: {output_path}")
        
        plt.show()
    
    def _color_markers(self, markers):
        """Color different marker regions for visualization"""
        unique_labels = np.unique(markers)
        colored = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
        
        # Use hsv colormap for variety
        from matplotlib import cm
        cmap = cm.get_cmap('hsv')
        
        for label in unique_labels:
            if label <= 0:  # Background or boundaries
                continue
            
            # Get color from colormap
            color_val = cmap(label / unique_labels.max())[:3]
            color_bgr = tuple([int(c * 255) for c in color_val[::-1]])
            
            colored[markers == label] = color_bgr
        
        # Boundaries in red
        colored[markers == -1] = [0, 0, 255]
        
        return colored
    
    def save_results(self, result, output_prefix="watershed_result"):
        """
        Save all segmentation results.
        
        Args:
            result: Output from segment()
            output_prefix: Prefix for output files
        """
        # Save visualizations
        self.visualize_watershed_steps(result, f"{output_prefix}_steps.png")
        
        # Save colored markers
        colored = self._color_markers(result['markers'])
        cv2.imwrite(f"{output_prefix}_colored.png", colored)
        
        # Save binary mask
        cv2.imwrite(f"{output_prefix}_mask.png", result['mask'])
        
        # Save distance map
        cv2.imwrite(f"{output_prefix}_distance.png", result['distance_normalized'])
        
        # Save peaks
        cv2.imwrite(f"{output_prefix}_peaks.png", result['peaks'])
        
        # Save labeled image
        labeled = self.image_bgr.copy()
        for obj in result['objects']:
            cv2.drawContours(labeled, [obj['contour']], 0, (0, 255, 0), 2)
        cv2.imwrite(f"{output_prefix}_labeled.png", labeled)
        
        # Save statistics
        self._save_statistics(result, f"{output_prefix}_stats.txt")
        
        print(f"✓ Results saved with prefix: {output_prefix}")
    
    def _save_statistics(self, result, filepath):
        """Save object statistics to file"""
        with open(filepath, 'w') as f:
            f.write(f"Image: {self.image_path}\n")
            f.write(f"Image size: {self.width} x {self.height}\n")
            f.write(f"Total objects: {result['num_objects']}\n")
            f.write(f"Distance method: L2 (Euclidean)\n\n")
            
            f.write("Objects:\n")
            f.write("-" * 90 + "\n")
            f.write(f"{'ID':<5} {'Area':<10} {'Perim':<10} {'Circ':<8} {'Solid':<8} {'Aspect':<8}\n")
            f.write("-" * 90 + "\n")
            
            for obj in result['objects']:
                x, y, w, h = obj['bounding_box']
                aspect = w / h if h > 0 else 0
                
                f.write(f"{obj['label']:<5} {obj['area']:<10.0f} {obj['perimeter']:<10.1f} "
                       f"{obj['circularity']:<8.3f} {obj['solidity']:<8.3f} {aspect:<8.3f}\n")
            
            f.write("\nSummary:\n")
            areas = [o['area'] for o in result['objects']]
            f.write(f"Mean area: {np.mean(areas):.0f}\n")
            f.write(f"Median area: {np.median(areas):.0f}\n")
            f.write(f"Total area: {np.sum(areas):.0f}\n")
    
    def save_debug_images(self, result):
        """Save debug images"""
        cv2.imwrite("debug_mask.png", result['mask'])
        cv2.imwrite("debug_distance.png", result['distance_normalized'])
        cv2.imwrite("debug_peaks.png", result['peaks'])
        print("✓ Debug images saved")


# ============================================================================
# EXAMPLE APPLICATIONS
# ============================================================================

def example_1_basic_watershed():
    """Example 1: Basic watershed on leaf image"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Watershed Segmentation")
    print("="*70)
    
    image_path = "field_image.jpg"
    
    try:
        segmenter = WatershedSegmenter(image_path)
        
        # Standard parameters for green leaves
        result = segmenter.segment(
            lower_hsv=np.array([25, 20, 20]),
            upper_hsv=np.array([95, 255, 255]),
            threshold_ratio=0.7,
            min_area=500
        )
        
        print(f"\n✓ Found {result['num_objects']} objects")
        
        segmenter.visualize_watershed_steps(result, "example1_watershed.png")
        segmenter.save_results(result, "example1")
        
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")


def example_2_parameter_tuning():
    """Example 2: Test different parameters"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Parameter Sensitivity Analysis")
    print("="*70)
    
    image_path = "field_image.jpg"
    thresholds = [0.5, 0.6, 0.7, 0.8]
    
    try:
        segmenter = WatershedSegmenter(image_path)
        
        results = {}
        for threshold in thresholds:
            print(f"\nTesting threshold ratio: {threshold}")
            result = segmenter.segment(threshold_ratio=threshold)
            results[threshold] = result['num_objects']
            print(f"  Objects found: {result['num_objects']}")
        
        # Plot results
        plt.figure(figsize=(8, 5))
        plt.plot(list(results.keys()), list(results.values()), marker='o', linewidth=2)
        plt.xlabel('Threshold Ratio')
        plt.ylabel('Number of Objects')
        plt.title('Effect of Threshold on Segmentation')
        plt.grid(True, alpha=0.3)
        plt.savefig('parameter_sensitivity.png', dpi=150, bbox_inches='tight')
        print("\n✓ Sensitivity analysis saved: parameter_sensitivity.png")
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")


def example_3_distance_methods():
    """Example 3: Compare distance transform methods"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Distance Method Comparison")
    print("="*70)
    
    image_path = "field_image.jpg"
    methods = ['L2', 'L1', 'C']
    
    try:
        segmenter = WatershedSegmenter(image_path)
        
        fig, axes = plt.subplots(1, len(methods), figsize=(15, 4))
        
        for idx, method in enumerate(methods):
            print(f"\nTesting distance method: {method}")
            result = segmenter.segment(distance_method=method)
            
            # Visualize distance map
            axes[idx].imshow(result['distance_normalized'], cmap='hot')
            axes[idx].set_title(f'{method} Distance\n({result["num_objects"]} objects)')
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('distance_methods_comparison.png', dpi=150, bbox_inches='tight')
        print("\n✓ Distance method comparison saved")
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")


if __name__ == "__main__":
    print("\n🌾 Watershed Segmentation Examples\n")
    
    # Get image path from user
    image_path = input("Enter image path (or press Enter for 'field_image.jpg'): ").strip()
    if not image_path:
        image_path = "field_image.jpg"
    
    # Run basic example
    try:
        print(f"\n📸 Loading image: {image_path}")
        segmenter = WatershedSegmenter(image_path, debug=False)
        
        print("\n🔄 Running watershed segmentation...")
        result = segmenter.segment(
            lower_hsv=np.array([25, 20, 20]),
            upper_hsv=np.array([95, 255, 255]),
            threshold_ratio=0.7,
            min_area=500
        )
        
        print(f"\n✅ Segmentation complete!")
        print(f"   Objects found: {result['num_objects']}")
        
        # Visualize
        print("\n📊 Generating visualizations...")
        segmenter.visualize_watershed_steps(result)
        
        # Save
        print("\n💾 Saving results...")
        segmenter.save_results(result, image_path.split('.')[0])
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
