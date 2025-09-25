"""
Professional Image Processing Application
Fixed version of the garbage code
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Callable
import functools


# ==================== DATA CLASSES ====================

@dataclass
class FilterParameters:
    """Immutable filter parameters"""
    brightness_r: int = 0
    brightness_g: int = 0
    brightness_b: int = 0
    contrast: int = 0
    is_negative: bool = False
    channel_order: Tuple[int, int, int] = (0, 1, 2)
    flip_horizontal: bool = False
    flip_vertical: bool = False

    @property
    def brightness_array(self) -> np.ndarray:
        return np.array([self.brightness_b, self.brightness_g, self.brightness_r], dtype=np.int16)


@dataclass
class Rectangle:
    """Pure data structure for rectangle"""
    center_x: int
    center_y: int
    size: int = 6


@dataclass
class ImageStats:
    """Statistics for image region"""
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    position: Tuple[int, int]


# ==================== IMAGE PROCESSOR ====================

class ImageProcessor:
    """
    Pure image processing operations
    No state, only functions
    """

    @staticmethod
    def apply_contrast(image: np.ndarray, contrast: int) -> np.ndarray:
        """Apply contrast adjustment"""
        if contrast == 0:
            return image

        contrast_factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
        image_float = image.astype(np.float32)
        adjusted = contrast_factor * (image_float - 128) + 128
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_brightness(image: np.ndarray, brightness: np.ndarray) -> np.ndarray:
        """Apply brightness adjustment"""
        if np.all(brightness == 0):
            return image

        return np.clip(image.astype(np.int16) + brightness, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_negative(image: np.ndarray) -> np.ndarray:
        """Apply negative filter"""
        return 255 - image

    @staticmethod
    def apply_channel_swap(image: np.ndarray, channel_order: Tuple[int, int, int]) -> np.ndarray:
        """Swap color channels"""
        return image[..., channel_order]

    @staticmethod
    def apply_flip(image: np.ndarray, flip_h: bool, flip_v: bool) -> np.ndarray:
        """Apply flip operations"""
        if flip_h:
            image = image[:, ::-1]
        if flip_v:
            image = image[::-1, :]
        return image

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert to grayscale using standard weights"""
        return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    @staticmethod
    def extract_region(image: np.ndarray, center_x: int, center_y: int, size: int) -> Optional[np.ndarray]:
        """Extract square region from image"""
        height, width = image.shape[:2]

        x1 = max(0, center_x - size)
        x2 = min(width, center_x + size + 1)
        y1 = max(0, center_y - size)
        y2 = min(height, center_y + size + 1)

        if x2 <= x1 or y2 <= y1:
            return None

        return image[y1:y2, x1:x2]

    @staticmethod
    def calculate_region_stats(image: np.ndarray, center_x: int, center_y: int,
                               window_size: int = 5) -> Optional[ImageStats]:
        """Calculate statistics for image region"""
        region = ImageProcessor.extract_region(image, center_x, center_y, window_size // 2)
        if region is None or region.size == 0:
            return None

        if len(region.shape) == 3:  # Color image
            means = [np.mean(region[..., i]) for i in range(3)]
            stds = [np.std(region[..., i]) for i in range(3)]
        else:  # Grayscale
            means = [np.mean(region)] * 3
            stds = [np.std(region)] * 3

        return ImageStats(
            mean=tuple(means),
            std=tuple(stds),
            position=(center_x, center_y)
        )


# ==================== IMAGE MODEL ====================

class ImageModel:
    """
    Manages image data and filter parameters
    Separated from processing logic
    """

    def __init__(self, image_path: str):
        self.original_image = self._load_image(image_path)
        if self.original_image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        self.current_image = self.original_image.copy()
        self.params = FilterParameters()
        self._apply_filters()

    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image with error handling"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load image from {image_path}")
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def _apply_filters(self):
        """Apply all filters to current image"""
        self.current_image = self.original_image.copy()

        # Apply filters in optimal order
        self.current_image = ImageProcessor.apply_channel_swap(
            self.current_image, self.params.channel_order)
        self.current_image = ImageProcessor.apply_flip(
            self.current_image, self.params.flip_horizontal, self.params.flip_vertical)
        self.current_image = ImageProcessor.apply_contrast(
            self.current_image, self.params.contrast)

        if self.params.is_negative:
            self.current_image = ImageProcessor.apply_negative(self.current_image)

        self.current_image = ImageProcessor.apply_brightness(
            self.current_image, self.params.brightness_array)

    def update_params(self, **kwargs):
        """Update parameters and reapply filters"""
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)

        self._apply_filters()

    def get_pixel_value(self, x: int, y: int) -> Tuple[int, int, int]:
        """Get RGB value at specified position"""
        if 0 <= x < self.width and 0 <= y < self.height:
            # OpenCV uses BGR, convert to RGB for display
            b, g, r = self.current_image[y, x]
            return (r, g, b)
        return (0, 0, 0)

    @property
    def height(self) -> int:
        return self.current_image.shape[0]

    @property
    def width(self) -> int:
        return self.current_image.shape[1]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.current_image.shape


# ==================== RECTANGLE DRAWER ====================

class RectangleDrawer:
    """
    Handles rectangle drawing operations
    Separated from image model
    """

    def __init__(self, image_shape: Tuple[int, int, int]):
        self.image_shape = image_shape
        self.rectangle = Rectangle(0, 0)
        self.underlying_data = None
        self.last_position = None

    def update_rectangle(self, center_x: int, center_y: int):
        """Update rectangle position"""
        self._restore_underlying()
        self.rectangle = Rectangle(center_x, center_y)
        self._save_underlying()

    def draw_on_image(self, image: np.ndarray) -> np.ndarray:
        """Draw rectangle on image"""
        result = image.copy()

        x1, x2, y1, y2 = self._calculate_bounds()

        if y1 >= 0:
            result[y1, x1:x2] = [0, 0, 255]  # Red color
        if y2 < self.image_shape[0]:
            result[y2 - 1, x1:x2] = [0, 0, 255]
        if x1 >= 0:
            result[y1:y2, x1] = [0, 0, 255]
        if x2 < self.image_shape[1]:
            result[y1:y2, x2 - 1] = [0, 0, 255]

        return result

    def _calculate_bounds(self) -> Tuple[int, int, int, int]:
        """Calculate rectangle bounds"""
        y1 = max(0, self.rectangle.center_y - self.rectangle.size)
        y2 = min(self.image_shape[0], self.rectangle.center_y + self.rectangle.size + 1)
        x1 = max(0, self.rectangle.center_x - self.rectangle.size)
        x2 = min(self.image_shape[1], self.rectangle.center_x + self.rectangle.size + 1)
        return x1, x2, y1, y2

    def _save_underlying(self):
        """Save image data under rectangle"""
        x1, x2, y1, y2 = self._calculate_bounds()
        if hasattr(self, 'current_image'):  # Reference to current image
            self.underlying_data = self.current_image[y1:y2, x1:x2].copy()
            self.last_position = (y1, y2, x1, x2)

    def _restore_underlying(self):
        """Restore image data under rectangle"""
        if self.underlying_data is not None and self.last_position is not None:
            y1, y2, x1, x2 = self.last_position
            self.current_image[y1:y2, x1:x2] = self.underlying_data


# ==================== ANALYSIS MANAGER ====================

class AnalysisManager:
    """
    Handles image analysis and visualization
    """

    def __init__(self, image_model: ImageModel):
        self.image_model = image_model
        self.is_analysis_active = False
        self.analysis_windows = [
            "1. Grayscale", "2. Red Channel", "3. Green Channel",
            "4. Blue Channel", "5. Histogram"
        ]

    def toggle_analysis(self):
        """Toggle analysis windows"""
        self.is_analysis_active = not self.is_analysis_active
        if self.is_analysis_active:
            self._show_analysis()
        else:
            self._hide_analysis()

    def _show_analysis(self):
        """Show analysis windows"""
        grayscale = ImageProcessor.to_grayscale(self.image_model.current_image)
        b_channel = self.image_model.current_image[:, :, 0]
        g_channel = self.image_model.current_image[:, :, 1]
        r_channel = self.image_model.current_image[:, :, 2]
        histogram = self._create_histogram()

        cv2.imshow("1. Grayscale", grayscale)
        cv2.imshow("2. Red Channel", r_channel)
        cv2.imshow("3. Green Channel", g_channel)
        cv2.imshow("4. Blue Channel", b_channel)
        cv2.imshow("5. Histogram", histogram)

    def _hide_analysis(self):
        """Hide analysis windows"""
        for window in self.analysis_windows:
            cv2.destroyWindow(window)

    def _create_histogram(self) -> np.ndarray:
        """Create combined histogram visualization"""
        hist_height, hist_width = 600, 800
        combined_hist = np.ones((hist_height, hist_width, 3), dtype=np.uint8) * 50

        grayscale = ImageProcessor.to_grayscale(self.image_model.current_image)
        channels = [
            self.image_model.current_image[:, :, 2],  # R
            self.image_model.current_image[:, :, 1],  # G
            self.image_model.current_image[:, :, 0],  # B
            grayscale
        ]

        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)]
        labels = ['Red', 'Green', 'Blue', 'Grayscale']

        regions = [
            (0, 0, hist_width // 2, hist_height // 2),
            (hist_width // 2, 0, hist_width, hist_height // 2),
            (0, hist_height // 2, hist_width // 2, hist_height),
            (hist_width // 2, hist_height // 2, hist_width, hist_height)
        ]

        for idx, (channel, color, label) in enumerate(zip(channels, colors, labels)):
            self._draw_single_histogram(combined_hist, regions[idx], channel, color, label)

        return combined_hist

    def _draw_single_histogram(self, canvas: np.ndarray, region: Tuple[int, int, int, int],
                               channel: np.ndarray, color: Tuple[int, int, int], label: str):
        """Draw single histogram on canvas"""
        x1, y1, x2, y2 = region
        region_width = x2 - x1
        region_height = y2 - y1

        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        if hist.max() > 0:
            hist = (hist / hist.max()) * (region_height - 40)

        axis_margin = 30
        cv2.line(canvas, (x1 + axis_margin, y2 - axis_margin),
                 (x2 - 10, y2 - axis_margin), color, 2)
        cv2.line(canvas, (x1 + axis_margin, y1 + 10),
                 (x1 + axis_margin, y2 - axis_margin), color, 2)

        for i in range(256):
            x_pos = x1 + axis_margin + int(i * (region_width - axis_margin - 20) / 256)
            height = int(hist[i])
            if height > 0:
                cv2.line(canvas, (x_pos, y2 - axis_margin),
                         (x_pos, y2 - axis_margin - height), color, 1)

        cv2.putText(canvas, label, (x1 + 10, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


# ==================== MAIN APPLICATION ====================

class ImageViewer:
    """
    Main application controller
    """

    def __init__(self, image_path: str):
        self.image_model = ImageModel(image_path)
        self.rectangle_drawer = RectangleDrawer(self.image_model.shape)
        self.analysis_manager = AnalysisManager(self.image_model)

        self.window_name = 'Professional Image Viewer'
        self.mouse_x, self.mouse_y = 0, 0
        self.zoom_window = None

        self._setup_ui()

    def _setup_ui(self):
        """Initialize UI components"""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        self._print_help()

    def _print_help(self):
        """Print control instructions"""
        help_text = """
        Controls:
        =/- - Brightness    C/c - Contrast      n - Negative
        R/r - Red channel   G/g - Green channel B/b - Blue channel  
        s/S - Swap channels h - Flip horizontal v - Flip vertical
        a - Toggle analysis w - Save image      0 - Zoom window
        ESC - Exit
        """
        print(help_text)

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x, self.mouse_y = x, y

    def _create_display_image(self) -> np.ndarray:
        """Create image with overlay information"""
        display_image = self.rectangle_drawer.draw_on_image(self.image_model.current_image)

        # Add information overlay
        r, g, b = self.image_model.get_pixel_value(self.mouse_x, self.mouse_y)
        stats = ImageProcessor.calculate_region_stats(
            self.image_model.current_image, self.mouse_x, self.mouse_y)

        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 0)  # Green text

        texts = [
            f"Pos: ({self.mouse_x}, {self.mouse_y})",
            f"RGB: ({r}, {g}, {b})",
            f"Intensity: {(r + g + b) / 3:.1f}"
        ]

        if stats:
            texts.extend([
                f"Mean: R{stats.mean[0]:.1f} G{stats.mean[1]:.1f} B{stats.mean[2]:.1f}",
                f"Std:  R{stats.std[0]:.1f} G{stats.std[1]:.1f} B{stats.std[2]:.1f}"
            ])

        for i, text in enumerate(texts):
            cv2.putText(display_image, text, (10, 30 + i * 20), font, 0.5, color, 1)

        return display_image

    def _update_zoom_window(self):
        """Update zoom window if active"""
        if self.zoom_window is not None:
            region = ImageProcessor.extract_region(
                self.image_model.current_image, self.mouse_x, self.mouse_y, 5)
            if region is not None:
                zoomed = cv2.resize(region, None, fx=20, fy=20, interpolation=cv2.INTER_NEAREST)
                cv2.imshow(self.zoom_window, zoomed)

    def _handle_keypress(self, key: int) -> bool:
        """Handle keyboard input"""
        key_actions = {
            ord('='): lambda: self.image_model.update_params(
                brightness_r=self.image_model.params.brightness_r + 10,
                brightness_g=self.image_model.params.brightness_g + 10,
                brightness_b=self.image_model.params.brightness_b + 10
            ),
            ord('-'): lambda: self.image_model.update_params(
                brightness_r=self.image_model.params.brightness_r - 10,
                brightness_g=self.image_model.params.brightness_g - 10,
                brightness_b=self.image_model.params.brightness_b - 10
            ),
            ord('n'): lambda: self.image_model.update_params(
                is_negative=not self.image_model.params.is_negative
            ),
            ord('C'): lambda: self.image_model.update_params(
                contrast=min(self.image_model.params.contrast + 10, 254)
            ),
            ord('c'): lambda: self.image_model.update_params(
                contrast=max(self.image_model.params.contrast - 10, -254)
            ),
            ord('R'): lambda: self.image_model.update_params(
                brightness_r=self.image_model.params.brightness_r + 10
            ),
            ord('r'): lambda: self.image_model.update_params(
                brightness_r=self.image_model.params.brightness_r - 10
            ),
            ord('G'): lambda: self.image_model.update_params(
                brightness_g=self.image_model.params.brightness_g + 10
            ),
            ord('g'): lambda: self.image_model.update_params(
                brightness_g=self.image_model.params.brightness_g - 10
            ),
            ord('B'): lambda: self.image_model.update_params(
                brightness_b=self.image_model.params.brightness_b + 10
            ),
            ord('b'): lambda: self.image_model.update_params(
                brightness_b=self.image_model.params.brightness_b - 10
            ),
            ord('s'): lambda: self.image_model.update_params(
                channel_order=self.image_model.params.channel_order[1:] +
                              (self.image_model.params.channel_order[0],)
            ),
            ord('S'): lambda: self.image_model.update_params(
                channel_order=(self.image_model.params.channel_order[0],) +
                              self.image_model.params.channel_order[2:] +
                              (self.image_model.params.channel_order[1],)
            ),
            ord('h'): lambda: self.image_model.update_params(
                flip_horizontal=not self.image_model.params.flip_horizontal
            ),
            ord('v'): lambda: self.image_model.update_params(
                flip_vertical=not self.image_model.params.flip_vertical
            ),
            ord('a'): self.analysis_manager.toggle_analysis,
            ord('w'): lambda: cv2.imwrite('processed_image.png', self.image_model.current_image),
            ord('0'): self._toggle_zoom_window,
            27: lambda: None  # ESC handler
        }

        if key in key_actions:
            if key == 27:  # ESC
                return False
            key_actions[key]()
            return True

        return True

    def _toggle_zoom_window(self):
        """Toggle zoom window"""
        if self.zoom_window is None:
            self.zoom_window = "Zoom"
            cv2.namedWindow(self.zoom_window)
        else:
            cv2.destroyWindow(self.zoom_window)
            self.zoom_window = None

    def run(self):
        """Main application loop"""
        try:
            while True:
                # Update display
                display_image = self._create_display_image()
                cv2.imshow(self.window_name, display_image)

                # Update zoom window
                self._update_zoom_window()

                # Update rectangle
                self.rectangle_drawer.update_rectangle(self.mouse_x, self.mouse_y)

                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_keypress(key):
                    break

        except KeyboardInterrupt:
            print("\nApplication interrupted")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        cv2.destroyAllWindows()
        if self.zoom_window:
            cv2.destroyWindow(self.zoom_window)


# ==================== MAIN EXECUTION ====================

def main():
    """Application entry point"""
    try:
        image_path = 'data/rinat.jpg'  # Change this to your image path

        print("Starting Professional Image Viewer...")
        viewer = ImageViewer(image_path)
        viewer.run()

    except Exception as e:
        print(f"Application error: {e}")
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()