import cv2
import os
import numpy as np


# Обрабатываем изображение в папке image
def stylize_image(input_path, output_path):
    # загружаем
    image = cv2.imread(input_path)
    if image is None:
        print(f"Ошибка: невозможно загрузить изображение {input_path}.")
        return

    # Осветляем
    bright_image = cv2.convertScaleAbs(image, alpha=1.2, beta=30)  # Увеличиваем яркость (alpha) и контраст (beta)

    # Размываем
    blurred_image = cv2.bilateralFilter(bright_image, d=25, sigmaColor=100, sigmaSpace=100)

    # Делаем чОтче
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    sharpened_image = cv2.filter2D(blurred_image, -1, kernel_sharpening)

    #  Сохраняем
    cv2.imwrite(output_path, sharpened_image)
    print(f"Сохранено в  {output_path}.")


def main():
    # Директория с изображением
    images_dir = "images"

    # Проверка наличия директории
    if not os.path.exists(images_dir):
        print(f"Директория '{images_dir}' не существует. Создайте её и поместите туда изображения.")
        return

    # Список изображений (с заданным расширением)
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not images:
        print("В папке 'images' нет изображений, добавьте")
        return

    # Самое первое изображение (тут можно цикл по всем, но для проверки пойдет)
    input_image_path = os.path.join(images_dir, images[0])
    file_name, file_ext = os.path.splitext(images[0])
    output_image_path = os.path.join(images_dir, f"{file_name}_styled{file_ext}")

    # преобразуем
    stylize_image(input_image_path, output_image_path)


if __name__ == "__main__":
    main()
