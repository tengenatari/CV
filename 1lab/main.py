import numpy
import cv2
import numpy as np


class Image:
    def __init__(self, image: numpy.ndarray):
        self.image = image
        self.height, self.width, self.channels = self.image.shape
        self.brightness = np.array([0, 0, 0])
        self.contrast =  0
        self.original = 1
        self.is_negative = False
        self.channel = (0, 1, 2)
        self.copy_image = image.copy()
        self.rectangle = Rectangle(self)
        self.is_flipped_h = False
        self.is_flipped_v = False
        self.is_show_analysis = False
        self.is_opened = False
    def apply_negative(self):
        if self.is_negative:
            self.copy_image = 255 - self.copy_image
    def apply_brightness(self):
        self.copy_image = np.clip(self.brightness + self.copy_image.astype(np.int16), 0, 255).astype(np.uint8)

    def apply_contrast(self):
        contrast_factor = (259 * (self.contrast + 255)) / (255 * (259 - self.contrast))
        self.copy_image = np.clip(contrast_factor * (self.copy_image.astype(np.float32) - 128) + 128, 0, 255).astype(
            np.uint8)
    def write(self):
        cv2.imwrite('modified_image.png', self.copy_image)
    def apply_flip(self):
        if self.is_flipped_h:
            self.copy_image = self.copy_image[:, ::-1, :]
        if self.is_flipped_v:
            self.copy_image = self.copy_image[::-1, :, :]
    def apply_analysis(self):
        if self.is_show_analysis:
            self.show_analysis()
            self.is_opened = True
        else:
            if self.is_opened:
                analysis_windows = [
                    "1. Grayscale", "2. Red Channel", "3. Green Channel", "4. Blue Channel",
                "5. Histogram"
                ]

                for window in analysis_windows:
                    cv2.destroyWindow(window)
    def apply_filters(self):

        self.copy_image = self.image.copy()
        self.apply_switching()
        self.apply_flip()
        self.apply_contrast()
        self.apply_negative()
        self.apply_brightness()
        self.apply_analysis()
    def apply_switching(self):
        self.copy_image = self.copy_image[..., self.channel]
    def get_image(self):
        self.apply_filters()
        return self.copy_image
    def get_image_without_filters(self):
        return self.copy_image
    def to_grayscale(self):
        return np.dot(self.get_image_without_filters(), [0.299, 0.587, 0.114]).astype(np.uint8)

    def get_window_stats(self, center_x, center_y, window_size=5):
        img = self.get_image_without_filters()


        half_size = window_size // 2
        y1 = max(0, center_y - half_size)
        y2 = min(self.height, center_y + half_size + 1)
        x1 = max(0, center_x - half_size)
        x2 = min(self.width, center_x + half_size + 1)

        # Вырезаем окно
        window = img[y1:y2, x1:x2]

        if window.size == 0:
            return (0, 0, 0), (0, 0, 0)


        mean_b, std_b = np.mean(window[:, :, 0]), np.std(window[:, :, 0])
        mean_g, std_g = np.mean(window[:, :, 1]), np.std(window[:, :, 1])
        mean_r, std_r = np.mean(window[:, :, 2]), np.std(window[:, :, 2])

        return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)
    def increase_brightness(self):
        self.brightness += 10
    def decrease_brightness(self):
        self.brightness -= 10
    def increase_contrast(self):
        self.contrast = min(self.contrast + 10, 254)
    def decrease_contrast(self):
        self.contrast = max(self.contrast - 10, -254)
    def redraw_rectangle(self, center_x, center_y):
        self.rectangle.redraw(center_x, center_y)
    def get_rgb_pos(self, y, x):
        return self.image[x, y]
    def set_negative(self):
        self.is_negative = not self.is_negative
    def increase_brightness_channel(self, channel):
        self.brightness[channel] += 10
    def decrease_brightness_channel(self, channel):
        self.brightness[channel] -= 10
    def increase_brightness_r(self):
        self.increase_brightness_channel(2)
    def decrease_brightness_r(self):
        self.decrease_brightness_channel(2)
    def increase_brightness_g(self):
        self.increase_brightness_channel(1)
    def decrease_brightness_g(self):
        self.decrease_brightness_channel(1)
    def increase_brightness_b(self):
        self.increase_brightness_channel(0)
    def decrease_brightness_b(self):
        self.decrease_brightness_channel(0)
    def change_channels(self):
        self.channel = self.channel[1], self.channel[0], self.channel[2]
    def change_channels_2(self):
        self.channel = self.channel[0], self.channel[2], self.channel[1]
    def flip_horizontal(self):
        self.is_flipped_h = not self.is_flipped_h
    def flip_vertical(self):
        self.is_flipped_v = not self.is_flipped_v
    def toggle_show_analysis(self):
        self.is_show_analysis = not self.is_show_analysis
    def get_channel_images(self):
        img = self.copy_image

        b, g, r = img[:,:, 0], img[:,:, 1], img[:,:, 2]

        return b, g, r

    def get_histograms(self):
        """4 гистограммы в одном окне на разных осях"""
        gray = self.to_grayscale()
        b, g, r = self.get_channel_images()

        hist_height, hist_width = 600, 800
        combined_hist = np.ones((hist_height, hist_width, 3), dtype=np.uint8) * 50

        total_pixels = self.height * self.width
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)]  # BGR: R, G, B, Gray
        channels = [r, g, b, gray]
        labels = ['Red Channel', 'Green Channel', 'Blue Channel', 'Grayscale']

        #Разделение на 4 области
        regions = [
            (0, 0, hist_width // 2, hist_height // 2),  # Верх-лево
            (hist_width // 2, 0, hist_width, hist_height // 2),  # Верх-право
            (0, hist_height // 2, hist_width // 2, hist_height),  # Низ-лево
            (hist_width // 2, hist_height // 2, hist_width, hist_height)  # Низ-право
        ]

        for idx, (channel, color, label) in enumerate(zip(channels, colors, labels)):

            x1, y1, x2, y2 = regions[idx]
            region_width = x2 - x1
            region_height = y2 - y1


            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])

            if total_pixels > 0:
                hist = hist / total_pixels


            if hist.max() > 0:
                hist = hist * (region_height - 40) / hist.max()

            axis_margin = 30 # отступы осей
            cv2.line(combined_hist,
                     (x1 + axis_margin, y2 - axis_margin),
                     (x2 - 10, y2 - axis_margin), color, 2)  # ось X
            cv2.line(combined_hist,
                     (x1 + axis_margin, y1 + 10),
                     (x1 + axis_margin, y2 - axis_margin), color, 2)  # ось Y


            for i in range(256):
                height = int(hist[i])
                if height > 0:

                    x_pos = x1 + axis_margin + int(i * (region_width - axis_margin - 20) / 256)

                    cv2.line(combined_hist,
                             (x_pos, y2 - axis_margin),
                             (x_pos, y2 - axis_margin - height),
                             color, 2)


            cv2.putText(combined_hist, label,
                        (x1 + 10, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return combined_hist
    def show_analysis(self):
        gray = self.to_grayscale()
        b_ch, g_ch, r_ch = self.get_channel_images()
        hist = self.get_histograms()


        cv2.imshow("1. Grayscale", gray)
        cv2.imshow("2. Red Channel", r_ch)
        cv2.imshow("3. Green Channel", g_ch)
        cv2.imshow("4. Blue Channel", b_ch)
        cv2.imshow("5. Histogram", hist)



class Window:
    def __init__(self, image: Image, name='lab1'):
        self.image = image
        self.name = name
        cv2.namedWindow(name)
        cv2.imshow(name, image.get_image())
        self.mouse = Mouse()
        self.half_win_size = 5
        self.subwindow = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.color = (0, 0, 255)  # Зеленый цвет
        self.thickness = 1

    def update(self):
        self.image.redraw_rectangle(self.mouse.current_x, self.mouse.current_y)
        image = self.image.get_image_without_filters().copy()
        if self.subwindow:
            y1, y2 = max(0, self.mouse.current_y - self.half_win_size), min(self.image.height,
                                                                            self.mouse.current_y + self.half_win_size + 1)
            x1, x2 = max(0, self.mouse.current_x - self.half_win_size), min(self.image.width,
                                                                            self.mouse.current_x + self.half_win_size + 1)
            resized_image = cv2.resize(image[y1:y2, x1:x2], None, fx=20, fy=20,
                                       interpolation=cv2.INTER_LINEAR)
            cv2.imshow(self.subwindow.name, resized_image)
        r, g, b = self.image.get_rgb_pos(self.mouse.current_x, self.mouse.current_y)
        coord_text = f"({self.mouse.current_x}, {self.mouse.current_y})"
        rgb_text = f"R:{r} G:{g} B:{b}"
        intensity_text = f"I:{(r+g+b)/3:.1f}"
        cv2.putText(image, coord_text, (10, 30), self.font, self.font_scale, self.color, self.thickness)
        cv2.putText(image, rgb_text, (10, 50), self.font, self.font_scale, self.color, self.thickness)
        cv2.putText(image, intensity_text, (10, 70), self.font, self.font_scale, self.color, self.thickness)

        mean_rgb, std_rgb = self.image.get_window_stats(self.mouse.current_x, self.mouse.current_y, 5)
        cv2.putText(image, "RGB Window Stats:", (10, 90),
                    self.font, self.font_scale, self.color, self.thickness)
        cv2.putText(image, f"Mean R:{mean_rgb[0]:.1f} G:{mean_rgb[1]:.1f} B:{mean_rgb[2]:.1f}", (10, 110),
                    self.font, self.font_scale, self.color, self.thickness)
        cv2.putText(image, f"Std  R:{std_rgb[0]:.1f} G:{std_rgb[1]:.1f} B:{std_rgb[2]:.1f}", (10, 130),
                    self.font, self.font_scale, self.color, self.thickness)
        cv2.imshow(self.name, image)

    def full_update(self):

        self.image.apply_filters()
        self.image.rectangle.save_underlying_region(self.image.copy_image)

    def start(self):
        cv2.setMouseCallback(self.name, self.mouse.mouse_callback)

        while True:
            key = cv2.waitKey(1) & 0xFF
            self.process_key(key)
            self.update()



    def process_key(self, key):

        key_function = {
            ord("="): self.image.increase_brightness,
            ord("-"): self.image.decrease_brightness,
            27: exit,
            ord("0"): self.add_window,
            ord("n"): self.image.set_negative,
            ord("R"): self.image.increase_brightness_r,
            ord("G"): self.image.increase_brightness_g,
            ord("B"): self.image.increase_brightness_b,
            ord("r"): self.image.decrease_brightness_r,
            ord("g"): self.image.decrease_brightness_g,
            ord("b"): self.image.decrease_brightness_b,
            ord("C"): self.image.increase_contrast,
            ord("c"): self.image.decrease_contrast,
            ord("s"): self.image.change_channels,
            ord("S"): self.image.change_channels_2,
            ord("h"): self.image.flip_horizontal,
            ord("v"): self.image.flip_vertical,
            ord("a"): self.image.toggle_show_analysis,
            ord("w"): self.image.write

        }
        if key in key_function:
            key_function[key]()
            self.full_update()


    def add_window(self):
        self.subwindow = Window(self.image, 'window')


class Mouse:
    def __init__(self):
        self.current_x = 0
        self.current_y = 0

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.current_x, self.current_y = x, y


class Rectangle:
    def __init__(self, image):
        self.image = image
        self.center_x = 0
        self.center_y = 0
        self.half_size = 6
        self.color = (0, 0, 255)
        self.last_drawn_region = None
        self.last_position = None
        self.height = image.height
        self.width = image.width
        self.draw()

    def set_position(self, x, y):
        self.center_x = x
        self.center_y = y

    def set_size(self, half_size):
        self.half_size = half_size

    def set_color(self, color):
        self.color = color

    def draw(self):
        self.save_underlying_region(self.image.copy_image)
        x1,x2,y1,y2 = self.calc_rectangle()
        # Верхняя граница
        if y1 >= 0:
            self.image.copy_image[y1, x1:x2] = self.color

        # Нижняя граница
        if y2 < self.height:
            self.image.copy_image[y2 - 1, x1:x2] = self.color

        # Левая граница
        if x1 >= 0:
            self.image.copy_image[y1:y2, x1] = self.color

        # Правая граница
        if x2 < self.width:
            self.image.copy_image[y1:y2, x2 - 1] = self.color

    def calc_rectangle(self):
        y1 = max(0, self.center_y - self.half_size)
        y2 = min(self.height, self.center_y + self.half_size + 1)
        x1 = max(0, self.center_x - self.half_size)
        x2 = min(self.width, self.center_x + self.half_size + 1)
        return x1,x2,y1,y2

    def save_underlying_region(self, image):

        x1, x2, y1, y2 = self.calc_rectangle()


        if y2 >= y1 and x2 >= x1:
            self.last_drawn_region = image[y1:y2, x1:x2].copy()
            self.last_position = (y1, y2, x1, x2)

    def erase(self):
        if self.last_drawn_region is not None and self.last_position is not None:
            y1, y2, x1, x2 = self.last_position
            self.image.copy_image[y1:y2, x1:x2] = self.last_drawn_region
    def redraw(self, new_x, new_y):
        self.erase()
        self.set_position(new_x, new_y)

        self.draw()

if __name__ == '__main__':
    Window(Image(np.array(cv2.imread('data/rinat.jpg')))).start()
