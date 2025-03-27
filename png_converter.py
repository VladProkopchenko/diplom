import os
import pydicom as dicom
import numpy as np
from PIL import Image, ImageEnhance

brightness_factor = 1.1  
contrast_factor = 1.1  

def clear_output_folder(output_folder):
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Не удалось удалить {file_path}: {e}")

def process_files(file_list, input_folder, output_folder):
    # Очищаем папку перед началом
    clear_output_folder(output_folder)

    for filename in file_list:
        dicom_path = os.path.join(input_folder, filename)
        x = dicom.dcmread(dicom_path)
        img_array = x.pixel_array

        img_array = img_array - np.min(img_array)
        img_array = img_array / np.ptp(img_array)
        img_array = (img_array * 255).astype(np.uint8)

        img = Image.fromarray(img_array).convert('L')

        # Коррекция яркости и контрастности
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)

        png_filename = os.path.splitext(filename)[0] + '.png'
        png_path = os.path.join(output_folder, png_filename)

        try:
            img.save(png_path)
            print(f"Сохранено: {png_path}")
        except Exception as e:
            print(f"Ошибка при сохранении {png_path}: {e}")
