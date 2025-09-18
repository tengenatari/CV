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
        self.copy_image = image.copy()
        self.rectangle = Rectangle(self)
    def apply_negative(self):
        if self.is_negative:
            self.copy_image = 255 - self.copy_image
    def apply_brightness(self):
        self.copy_image = np.clip(self.brightness + self.copy_image.astype(np.int16), 0, 255).astype(np.uint8)

    def apply_contrast(self):
        self.copy_image = np.clip(self.copy_image.astype(np.float32) / (1 + np.exp(-self.contrast)), 0, 255).astype(np.uint8)
    def apply_filters(self):

        self.copy_image = self.image.copy()
        self.apply_contrast()
        self.apply_negative()
        self.apply_brightness()

    def get_image(self):
        self.apply_filters()
        return self.copy_image
    def get_image_without_filters(self):
        return self.copy_image
    def to_grayscale(self):
        return np.dot(self.get_image(), [0.299, 0.587, 0.114]).astype(np.uint8)

    def increase_brightness(self):
        self.brightness += 10
    def decrease_brightness(self):
        self.brightness -= 10
    def increase_contrast(self):
        self.contrast += 1
    def decrease_contrast(self):
        self.contrast -= 1
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
        cv2.imshow(self.name, image)

    def full_update(self):
        self.image.apply_filters()

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
            ord("2"): self.image.set_negative,
            ord("R"): self.image.increase_brightness_r,
            ord("G"): self.image.increase_brightness_g,
            ord("B"): self.image.increase_brightness_b,
            ord("r"): self.image.decrease_brightness_r,
            ord("g"): self.image.decrease_brightness_g,
            ord("b"): self.image.decrease_brightness_b,
            ord("C"): self.image.increase_contrast,
            ord("c"): self.image.decrease_contrast,
        }
        if key in key_function:
            self.full_update()
            key_function[key]()
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
        self._save_underlying_region(self.image.copy_image)
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

    def _save_underlying_region(self, image):

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
