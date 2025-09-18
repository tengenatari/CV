import cv2
import numpy as np


original_image = None
current_image = None
window_name = 'Lab 1 - Basic Computer Vision'
current_x, current_y = 0, 0

# -- Функция для обработки событий мыши --
def mouse_callback(event, x, y, flags, param):
    global current_x, current_y
    if event == cv2.EVENT_MOUSEMOVE:
        current_x, current_y = x, y
        update_display()

def update_display():
    display_image = current_image.copy()
    
    # Рисуем рамку
    half_border_size = 6
    top_left = (current_x - half_border_size, current_y - half_border_size)
    bottom_right = (current_x + half_border_size, current_y + half_border_size)
    cv2.rectangle(display_image, top_left, bottom_right, (0, 0, 255), thickness=1)
    
    # Получаем значения пикселя
    b, g, r = current_image[current_y, current_x]
    intensity = (int(b) + int(g) + int(r)) / 3.0
    
    # Формируем текст для отображения
    coord_text = f"({current_x}, {current_y})"
    rgb_text = f"R:{r} G:{g} B:{b}"
    intensity_text = f"I:{intensity:.1f}"
    
    # Настройки шрифта
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 0, 255)  # Зеленый цвет
    thickness = 1
    
    # Отображаем текст на изображении
    cv2.putText(display_image, coord_text, (10, 30), font, font_scale, color, thickness)
    cv2.putText(display_image, rgb_text, (10, 50), font, font_scale, color, thickness)
    cv2.putText(display_image, intensity_text, (10, 70), font, font_scale, color, thickness)
    
    cv2.imshow(window_name, display_image)
    half_win_size = 5
    y1, y2 = max(0, current_y - half_win_size), min(current_image.shape[0], current_y + half_win_size + 1)
    x1, x2 = max(0, current_x - half_win_size), min(current_image.shape[1], current_x + half_win_size + 1)
    roi = current_image[y1:y2, x1:x2]
    zoom_factor = 20
    if roi.size > 0:
        mean_val = np.mean(roi)
        std_val = np.std(roi)
        roi_zoom = cv2.resize(roi, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('11x11 Window', roi_zoom)
        # Добавляем статистику на изображение
        stats_text = f"Mean: {mean_val:.2f}, Std: {std_val:.2f}"
        cv2.putText(display_image, stats_text, (10, 90), font, font_scale, (0, 0, 255), thickness)  # Желтый цвет
    
    cv2.imshow(window_name, display_image)

# -- Функции для обработки изображения (п.5 задания) --
def adjust_brightness(value):
    global current_image
    # value - дельта, которую нужно прибавить к каждому пикселю
    current_image = cv2.add(original_image, value)

def adjust_contrast(value):
    global current_image
    # value - множитель контрастности (1.0 - без изменений)
    current_image = np.clip(original_image.astype(np.float32)/(1+np.exp(-value)), 0, 255).astype(np.uint8)

def create_negative():
    global current_image
    current_image = 255 - original_image

def swap_channels(channel1, channel2):
    # Например, channel1=0 (B), channel2=1 (G). Меняем синий и зеленый каналы.
    global current_image
    current_image = original_image.copy()
    current_image[:, :, channel1], current_image[:, :, channel2] = original_image[:, :, channel2], original_image[:, :, channel1]

def flip_image(direction):
    # direction: 0 - вертикально, 1 - горизонтально
    global current_image
    current_image = cv2.flip(original_image, direction)

# -- Главная функция --
def main():
    global original_image, current_image
    
    # 1. Загрузка изображения (замените 'your_image.bmp' на путь к вашему файлу)
    image_path = 'data/rinat.jpg'
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print(f"Ошибка: Не удалось загрузить изображение {image_path}")
        return
    
    current_image = original_image.copy()
    
    # 2. Создание окон
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    

    print("Управление:")
    print("'+'/-: Яркость")
    print("'c'/'C': Контрастность (+/-)")
    print("'n': Негатив")
    print("'s': Поменять местами каналы B и G")
    print("'h': Отразить по горизонтали")
    print("'v': Отразить по вертикали")
    print("'w': Сохранить изображение")
    print("'r': Сбросить к оригиналу")
    print("ESC: Выход")
    
    update_display() # Первоначальная отрисовка
    
    # 3. Главный цикл обработки нажатий клавиш
    brightness_delta = 0
    contrast_factor = 0
    
    while True:
        key = cv2.waitKey(100) & 0xFF # Ждем нажатия клавиши 100 мс
        
        if key == 27: # ESC - выход
            break
        elif key == ord('='):
            brightness_delta += 10
            adjust_brightness(brightness_delta)
        elif key == ord('-'):
            brightness_delta -= 10
            adjust_brightness(brightness_delta)
        elif key == ord('c'):
            contrast_factor += 1
            adjust_contrast(contrast_factor)
        elif key == ord('C'):
            contrast_factor -= 1
            adjust_contrast(contrast_factor)
        elif key == ord('n'):
            create_negative()
        elif key == ord('s'):
            swap_channels(0, 1) # Меняем B и G
        elif key == ord('h'):
            flip_image(1)
        elif key == ord('v'):
            flip_image(0)
        elif key == ord('r'): # Reset
            current_image = original_image.copy()
            brightness_delta = 0
            contrast_factor = 1.0
        elif key == ord('w'): # Save
            cv2.imwrite('modified_image.png', current_image)
            print("Изображение сохранено как 'modified_image.png'")
        
        update_display()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()