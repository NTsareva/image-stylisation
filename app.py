import cv2
import numpy as np
import dlib
from PIL import Image
from rembg import remove
import mediapipe as mp

#увеличиваем насыщенность
def increase_color_intensity(image, sat_scale=1.3, val_scale=1.0):
      #sat_scale  : во сколько раз увеличить насыщенность (1.0 = без изменений)
      #val_scale  : во сколько раз увеличить яркость 

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    s = s.astype(np.float32)
    s = np.clip(s * sat_scale, 0, 255).astype(np.uint8)

    v = v.astype(np.float32)
    v = np.clip(v * val_scale, 0, 255).astype(np.uint8)

    hsv_new = cv2.merge([h, s, v])
    result = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
    return result

#удаляем бэкграунд с сохранением в файл
def replace_background(input_image_path, bg_image_path, output_path):

    with Image.open(input_image_path) as input_image:
        image_without_bg = remove(input_image)
    
    with Image.open(bg_image_path) as bg_image:
        bg_image = bg_image.convert("RGBA")
        bg_image = bg_image.resize(image_without_bg.size)
        combined_image = Image.alpha_composite(bg_image, image_without_bg)
    
    combined_image = combined_image.convert("RGB") 
    combined_image.save(output_path, format="JPEG")

    print(f"Фон заменён. Результат: {output_path}")

#та же функция, но без вывода в файл, работает чаще хуже
def replace_image_background(input_image, bg_image):
    image_without_bg = remove(input_image)
    
    bg_image = bg_image.convert("RGBA")
    bg_image = bg_image.resize(image_without_bg.size)
    combined_image = Image.alpha_composite(bg_image, image_without_bg)
    
    combined_image = combined_image.convert("RGB")  
    return combined_image

#удаление фона через сегментацию
def remove_background_selfie_segmentation(image):
    mp_selfie = mp.solutions.selfie_segmentation
    with mp_selfie.SelfieSegmentation(model_selection=1) as segmentor:

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = segmentor.process(rgb)
        
        # маска от 0.0 до 1.0, где ближе к 1.0 - человек
        mask = results.segmentation_mask
        # МаскаL 255 = человек, 0 = фон)
        # Порог: 0.5 или 0.7
        person_mask = (mask > 0.7).astype(np.uint8) * 255

        result = cv2.bitwise_and(image, image, mask=person_mask)
        return result
    
# еще одна функция удаления фона
def remove_bg_rembg(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    result_pil = remove(pil_img)  
    result_pil = result_pil.convert("RGB")
    result_np = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
    return result_np

# Увеличение резкости (Unsharp Mask)
def unsharp_mask(image, ksize=(4,4), sigma=1.0, amount=3.5, threshold=0):
    blurred = cv2.GaussianBlur(image, ksize, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.abs(image - blurred) < threshold
        sharpened[low_contrast_mask] = image[low_contrast_mask]
    return sharpened

# Усиление теней
def enhance_shadows(image, shadow_threshold=80, shadow_strength=0.6):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shadow_mask = (gray < shadow_threshold).astype(np.uint8) * 255
    img_float = image.astype(np.float32)
    darkened = img_float * (1.0 - shadow_strength)
    mask_3ch = cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    out_float = img_float * (1 - mask_3ch) + darkened * mask_3ch
    shadowed = np.clip(out_float, 0, 255).astype(np.uint8)
    return shadowed

# Удаление покраснений 
def remove_red_tint_small_areas(image_bgr, factor=0.2, area_threshold=4):
      #factor - на сколько, на 0.2, уменьшить красный на 20%
      #area_threshold - манксимальая плозать соседних пикселей

    b, g, r = cv2.split(image_bgr)
    red_mask = ((r > g) & (r > b)).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(red_mask, connectivity=8)

    r_float = r.astype(np.float32)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area <= area_threshold:
            component_mask = (labels == label)
            r_float[component_mask] = r_float[component_mask] * (1.0 - factor)

    r_new = np.clip(r_float, 0, 255).astype(np.uint8)

    result = cv2.merge((b, g, r_new))
    return result

# Регулировка уровней (приведение черного и белого к заданным порогам)
def _levels_channel(ch, black_cut, white_cut):
    ch_float = ch.astype(np.float32)
    ch_float = (ch_float - black_cut) * (255.0 / (white_cut - black_cut))
    ch_float = np.clip(ch_float, 0, 255)
    return ch_float.astype(np.uint8)

#другафф функция регулировки уровней
def adjust_whites_blacks(image_bgr, black_cut=20, white_cut=235):
    (b, g, r) = cv2.split(image_bgr)
    b = _levels_channel(b, black_cut, white_cut)
    g = _levels_channel(g, black_cut, white_cut)
    r = _levels_channel(r, black_cut, white_cut)
    return cv2.merge([b, g, r])

# усиливаем белые (если пиксели близки к белому, делаем их ещё ярче)
def adjust_whites(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    mask = (L > 235) #вот тут считаем, что считаем былым
    image[mask] = [255, 255, 255]
    return image

# Регулировка яркости и контраста
def increase_brightness_contrast(img, alpha=1.3, beta=10):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

#yfvvf rjhhtrwbz
def gamma_correction(img, gamma=0.9):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

#бустим сатурацию
def boost_saturation_hsv(img_bgr, sat_scale=1.2, val_scale=1.0):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s.astype(np.float32) * sat_scale, 0, 255).astype(np.uint8)
    v = np.clip(v.astype(np.float32) * val_scale, 0, 255).astype(np.uint8)
    hsv_boosted = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)

#регулируем яркость
def adjust_brightness(image, alpha=1.3, beta=10, gamma=0.9, sat_scale=1.2, val_scale=1.0):
    #img_lin = increase_brightness_contrast(image, alpha=1.3, beta=10)
    img_lin = increase_brightness_contrast(image, alpha=1.3, beta=10)
    img_gamma = gamma_correction(img_lin, gamma=0.9)
    result = boost_saturation_hsv(img_gamma, sat_scale=1.2, val_scale=1.0)
    return result

# Удаление шумов
def replaceNoise(image, d =9, x = 100, y = 100 ):
    #return cv2.bilateralFilter(image, 5, 30, 40) # для светлых, вот тут размер кисти - первое, затем учет разницы в цвете, а затем учет в пространстве
    return cv2.bilateralFilter(image, 9, 100, 100) # вот тут размер кисти - первое, затем учет разницы в цвете, а затем учет в пространстве

# Простой шарпинг (альтернатива unsharp_mask)
def sharpen_filter(image):
    kernel = np.array([[ 0, -1, 0], #тут я копировала, понятия не имею, что эо
                       [-1, 5, -1],
                       [ 0, -1, 0]], np.float32)
    return cv2.filter2D(image, -1, kernel)

# Функция для глобальной предварительной обработки всего изображения, не актуальна
def process_entire_image(image):
    processed = image.copy()
    processed = replace_background_color_pil(processed, background_color=(150,255,0))
    processed = sharpen_filter(processed)
    processed = remove_red_tint_bgr(processed, factor=0.2)
    processed = replaceNoise(processed)
    processed = adjust_whites(processed)
    processed = enhance_shadows(processed, shadow_threshold=80, shadow_strength=0.3)
    processed = adjust_brightness(processed)
    return processed

# (Обёртка для работы с массивом, если нужен rembg-подобный эффект уже в массиве)
def replace_background_color_pil(image_bgr, background_color=(150,255,0)):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    img_no_bg = remove(pil_img)
    bg = Image.new("RGBA", img_no_bg.size, background_color + (255,))
    comp = Image.alpha_composite(bg, img_no_bg).convert("RGB")
    comp_np = np.array(comp)
    return cv2.cvtColor(comp_np, cv2.COLOR_RGB2BGR)

#Замена фона на хромакей (с помощью rembg)
def replace_background_color_file(input_image_path, output_path, background_color=(150,255,0)):
    with Image.open(input_image_path) as input_image:
        image_no_bg = remove(input_image)
    bg = Image.new("RGBA", image_no_bg.size, background_color + (255,))
    combined = Image.alpha_composite(bg, image_no_bg).convert("RGB")
    combined.save(output_path, format="JPEG")
    print(f"Фон заменён. Результат: {output_path}")

# Получение маски для лиц с увеличенным выделением (в 2 раза), не очень актуально
def get_face_mask(image, predictor_path, scale=2.0):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        center = ((x1+x2)//2, (y1+y2)//2)
        axes = (int(((x2-x1)/2)*scale), int(((y2-y1)/2)*scale))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask, faces

# Разделение изображения на область лица и фон 
def separate_face_background(image, mask):
    face_part = cv2.bitwise_and(image, image, mask=mask)
    background_part = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    return face_part, background_part

# Сегментация области лица центральная часть (голова с волосами) , внешнее кольцо (оставляем без обработки)
def segment_face_region(face_mask):
    ksize = (max(3, face_mask.shape[1]//50), max(3, face_mask.shape[0]//50))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
    region3 = cv2.erode(face_mask, kernel, iterations=1)
    region4 = cv2.subtract(face_mask, region3)
    return region3, region4

# получение маски для глаз 
def get_eye_mask(image, faces, predictor_path):
    predictor = dlib.shape_predictor(predictor_path)
    eye_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for face in faces:
        shape = predictor(gray, face)
        right_eye = np.array([[shape.part(i).x, shape.part(i).y] for i in range(36,42)])
        left_eye = np.array([[shape.part(i).x, shape.part(i).y] for i in range(42,48)])
        cv2.fillPoly(eye_mask, [right_eye], 255)
        cv2.fillPoly(eye_mask, [left_eye], 255)
    return eye_mask

# Разделение глаза и лицо без глаз
def segment_region3(region3_mask, eye_mask):
    area5 = cv2.bitwise_and(region3_mask, eye_mask)
    area6 = cv2.subtract(region3_mask, area5)
    return area5, area6

#  заменяем светлые оттенки  в глазаъ на белый (оставляем тёмные)
def process_eye_region(eye_img):
    processed = eye_img.copy()
    
    hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v.astype(np.int32) + 30, 0, 255).astype(np.uint8)
    hsv_light = cv2.merge([h, s, v])
    processed = cv2.cvtColor(hsv_light, cv2.COLOR_HSV2BGR)
    
    # дляем красные оттенки: уменьшаем интенсивность красного канала на 20%
    b, g, r = cv2.split(processed)
    r = np.clip(r.astype(np.float32) * 0.7, 0, 255).astype(np.uint8)
    processed = cv2.merge([b, g, r])
    
    # Выделяем белые области белков
    hsv2 = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
    lower_white = (0, 0, 150)      # низкая насыщенность, яркость от 30
    upper_white = (150, 100, 255)   # насыщенность до 30, яркость до 255
    white_mask = cv2.inRange(hsv2, lower_white, upper_white)
    
    # Заменяем белые области на чисто белый цвет
    processed[white_mask == 255] = [255, 255, 255]
    
    # Осветляем зрачок: в HSV создаём маску для очень темных областей (яркость < 40)
    #hsv3 = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
    #h3, s3, v3 = cv2.split(hsv3)
    #_, pupil_mask = cv2.threshold(v3, 40, 255, cv2.THRESH_BINARY_INV)
    #v3_new = np.clip(v3.astype(np.int32) + 50, 0, 255).astype(np.uint8)
    # Только для пикселей, где pupil_mask == 255, увеличиваем яркость
    #v3 = np.where(pupil_mask == 255, v3_new, v3).astype(np.uint8)
    #hsv3 = cv2.merge([h3, s3, v3])
    #processed = cv2.cvtColor(hsv3, cv2.COLOR_HSV2BGR)
    
    # Создаем размытую версию обработанного изображения
    blurred = cv2.GaussianBlur(processed, (5, 5), 0)
    
    # Заменяем в исходном обработанном изображении пиксели белков (white_mask) и зрачка (pupil_mask),
    processed[white_mask == 255] = blurred[white_mask == 255]
    #processed[pupil_mask == 255] = blurred[pupil_mask == 255] - надо поиграться
    
    return processed

#Обработка области кожи  заменяем оттенки на 5 бежевых тонов и размываем, это как пример, тут надо сортировать и сильнее размывать
def process_face_skin(region6_img, region6_mask):
    beige_palette = np.array([
        [241, 162, 185], #массив цветов
        [246, 136, 170],
        [235, 185, 204],
        [247, 198, 219],
        [211, 117, 150],
        [253, 210, 225],
        [247, 141, 172],
        [245, 184, 199],
    ], dtype=np.uint8)
    processed = replace_skin_colors(region6_img.copy(), region6_mask, beige_palette)
    processed = cv2.GaussianBlur(processed, (5,5), 0)
    return processed

def replace_skin_colors(image, mask, palette):
    indices = np.where(mask == 255)
    if indices[0].size == 0:
        return image

    region_pixels = image[indices].astype(np.int32)
    diff = region_pixels[:, np.newaxis, :] - palette[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)

    nearest_idx = np.argmin(dist_sq, axis=1)
    new_colors = palette[nearest_idx]
    
    image[indices] = new_colors
    return image

#то же
def process_face_skin(region6_img, region6_mask):
    beige_palette = np.array([
        [200, 190, 180],
        [210, 200, 190],
        [220, 210, 200],
        [230, 220, 210],
        [240, 230, 220]
    ], dtype=np.uint8)
    processed = replace_skin_colors(region6_img.copy(), region6_mask, beige_palette)
    processed = cv2.GaussianBlur(processed, (5,5), 0)
    return processed

# Объединение обработанных областей:
def merge_all(background, face_original, region3_mask, processed_region3):
    merged = background.copy()
    indices = np.where(region3_mask==255)
    merged[indices] = processed_region3[indices]
    face_combined = face_original.copy()
    inv = cv2.bitwise_not(region3_mask)
    face_combined[inv==255] = face_original[inv==255]
    final = background.copy()
    indices2 = np.where(face_original>0)
    final[indices2] = face_combined[indices2]
    final = cv2.add(background, face_combined)
    return final

#отделяем коу от остального
def extract_skin(image):

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Определяем пороги для кожи (эти значения можно корректировать в зависимости от условий освещения и типа кожи)
    #lower = np.array([0,   130,  70],  dtype=np.uint8)
    #upper = np.array([255, 180, 140], dtype=np.uint8) # эти пороги для выкосонтрастных изображений

    lower = np.array([0,   133,  77],  dtype=np.uint8)
    upper = np.array([255, 173, 127],  dtype=np.uint8) #для низкоконтрастных изображений
    
    # Создаем бинарную маску: пиксели в диапазоне оттенков кожи получат значение 255, остальные – 0
    skin_mask = cv2.inRange(ycrcb, lower, upper)
    
    # не обязательно: применяем морф обработку для удаления шума
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
    
    skin = cv2.bitwise_and(image, image, mask=skin_mask)
    return skin, skin_mask

# получение человека (0\255 обасть 255 человеку принадлежит)
def get_person_mask(image):

    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmentor:
    
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = segmentor.process(image_rgb)
        # Результирующая маска имеет значения от 0 до 1
        mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
    return mask

#выдкление кожи
def extract_skin_mask(image):

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    lower = np.array([60,   130,  70],  dtype=np.uint8)
    upper = np.array([255, 180, 140], dtype=np.uint8)
    #lower = np.array([0,   133,  77],  dtype=np.uint8)
    #upper = np.array([255, 173, 127],  dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower, upper)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)
    return skin_mask

# выделение одежды
def segment_clothing(image):

    person_mask = get_person_mask(image)
    skin_mask = extract_skin_mask(image)
    
    # Для получения одежды вычитаем маску кожи из маски человека
    clothing_mask = cv2.subtract(person_mask, skin_mask)
    
    #  можно применить морфологические операции для сглаживания маски одежды
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Извлекаем одежду из изображения
    clothing = cv2.bitwise_and(image, image, mask=clothing_mask)
    return clothing, clothing_mask

#выделение кожи маской
def replace_skin_with_palette_by_mask(image, user_mask, lower, upper, palette_bgr):
    # Использует user_mask (255 = область, 0 = вне области) 
    # кожа по диапазону (lower..upper) в YCrCb.

    # что user_mask – одноканальная (0/255)
    if len(user_mask.shape) != 2:
        raise ValueError("user_mask должна иметь один канал (grayscale).")

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    skin_mask = cv2.inRange(ycrcb, lower, upper)  # 0/255
    

    combined_mask = cv2.bitwise_and(skin_mask, user_mask)
    # ombined_mask = 255 только там, где user_mask=255 И skin_mask=255.
    

    indices = np.where(combined_mask == 255)
    if indices[0].size == 0:
        return image
  
    region_pixels = image[indices].astype(np.int32)
    palette_array = np.array(palette_bgr, dtype=np.int32)
    
    diff = region_pixels[:, np.newaxis, :] - palette_array[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    nearest_idx = np.argmin(dist_sq, axis=1)
    
    #  Заменяем пиксели на соответствующий цвет из палитры
    new_colors = palette_array[nearest_idx]
    image[indices] = new_colors
    
    return image

# замещение кожи из палеки по градиенту
def replace_skin_with_palette_by_mask_gradient(
    image,        # Исходное BGR-изображение
    user_mask,    # Бинарная маска (0/255), внутри которой обрабатываем
    lower, upper, # Пороги YCrCb (skin detection)
    palette_bgr,  # Палитра дискретных цветов (BGR - тут внимание, это не RGB)
    blur_ksize=5, # Размер размытия (например, 5,7,9)
    alpha=0.5 #степень смешения размытой версии (0..1). 0.5 = 50% blur, 50% жёсткая заливка.
):
     # [162, 185, 241], # - присланная палитра
       # [136, 170, 246],
       # [185, 204, 235],
       # [198, 219, 247],  # 3064 skin, eyebrows
        #[210, 225, 253], # 3770 skin
        #[141, 172, 247], # 3771 lips
       # [184, 199, 245], # 3779 skin, lips

    #
    replaced_img = replace_skin_with_palette_by_mask(
        image.copy(), user_mask, lower, upper, palette_bgr
    )
    # replaced_img теперь содержит дискретные цвета внутри mask, 
    # и исходное изображение вне mask.

    # Создаём изображение, где за пределами mask = (0,0,0), чтобы размытие не тянуло цвета извне
    masked_img = np.zeros_like(replaced_img)
    masked_idx = np.where(user_mask == 255)
    masked_img[masked_idx] = replaced_img[masked_idx]

    # Размываем masked_img
    blurred = cv2.GaussianBlur(masked_img, (blur_ksize, blur_ksize), 0)

    # смешиваем blurred и replaced_img внутри mask 
    replaced_f = replaced_img.astype(np.float32)
    blurred_f  = blurred.astype(np.float32)
    out_f      = replaced_f.copy()


    inside_idx = (masked_idx[0], masked_idx[1])  # те же arrays
    out_f[inside_idx] = alpha * blurred_f[inside_idx] + (1 - alpha) * replaced_f[inside_idx]

    out = np.clip(out_f, 0, 255).astype(np.uint8)
    return out


def replace_skin_with_palette_by_mask_soft(image, user_mask, lower, upper, palette_bgr, blur_ksize=5, alpha=0.5):

      #image      исходное BGR-изображение
      #user_mask  бинарная маска (0/255),  255 - область для обработки
      #lower,upper пороги YCrCb для определения кожи
      ##palette_bgr список (N,3) с цветами палитры (BGR - важно)
      #blur_ksize  размер ядра размытия (нечётное, напр. 5,7,9)
      #alpha     доля «замеса» размытой версии (0..1), напр. 0.5 = 50% размытой, 50% исходной.

    replaced_image = replace_skin_with_palette_by_mask(
        image.copy(),  # важно копировать, чтобы не портить original
        user_mask,
        lower, upper,
        palette_bgr
    )
    
    # елаем размытую копию, но размытие — по всему изображению, но эффект внутри маски

    blurred_image = cv2.GaussianBlur(replaced_image, (blur_ksize, blur_ksize), 0)
    
    # Смешиваем канаты (alpha-blend) replaced_image и blurred_image ТОЛЬКО в области user_mask
    mask_3ch = cv2.merge([user_mask, user_mask, user_mask]) #?
    
    replaced_f = replaced_image.astype(np.float32)
    blurred_f = blurred_image.astype(np.float32)
    
    # alpha * blurred + (1-alpha)* replaced, mask_3ch == 255
    out_f = replaced_f.copy()
    
    # Индексы пикселей внутри маски
    inside_idx = np.where(user_mask == 255)
    
    # Смешение
    out_f[inside_idx] = alpha * blurred_f[inside_idx] + (1.0 - alpha) * replaced_f[inside_idx]
    
   
    out = np.clip(out_f, 0, 255).astype(np.uint8)
    
    return out

def replace_colors_with_gradient_no_mask(image, palette, shadow_color, alpha=0.6):
    
    h, w = image.shape[:2]
    flat_pixels = image.reshape((-1, 3)).astype(np.int32)  # (H*W, 3)

    palette_arr = np.array(palette, dtype=np.int32)

    diff = flat_pixels[:, np.newaxis, :] - palette_arr[np.newaxis, :, :]  
    dist_sq = np.sum(diff**2, axis=2) 
    nearest_idx = np.argmin(dist_sq, axis=1)  

    new_colors = palette_arr[nearest_idx]
    orig_pixels = flat_pixels.copy()

    #  Применяем градиент
    for i in range(flat_pixels.shape[0]):
        new_c = new_colors[i]
        old_c = orig_pixels[i]

        if np.array_equal(new_c, shadow_color):
            # т.е. если наш «ближайший цвет» == shadow_color  -  без смешения
            final_c = new_c
        else:
            # final = alpha*new + (1-alpha)*old
            final_c = alpha * new_c + (1.0 - alpha) * old_c

        flat_pixels[i] = np.clip(final_c, 0, 255).astype(np.int32)

    result = flat_pixels.astype(np.uint8).reshape((h, w, 3))
    return result



def replace_colors_with_gradient(image, mask, palette, shadow_color, alpha=0.6):

      #mask   маска (0 или 255), где 255 – пиксели для обработки
      #palette   список или массив цветов (BGR) формы (N,3)
      #shadow_color цвет (BGR - важно), который применяется без градиента
      #alpha       коэффициент смешения (0..1).  Например, alpha=0.6 => 60% новый цвет, 40% исходный.

    # Находим индексы пикселей, где mask == 255
    indices = np.where(mask == 255)
    if indices[0].size == 0:
        return image  # Нет пикселей для обработки

    region_pixels = image[indices].astype(np.int32)
    
    palette_arr = np.array(palette, dtype=np.int32)
    
    diff = region_pixels[:, np.newaxis, :] - palette_arr[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    # Для каждого пикселя - индекс ближайшего цвета из палитры
    nearest_idx = np.argmin(dist_sq, axis=1)
    
    #и сходные пиксели (для смешения)
    orig_pixels = region_pixels.copy()

    # с учетом градиента
    for i in range(len(nearest_idx)):
        new_c = palette_arr[nearest_idx[i]]  # ближайший цвет из палитры
        old_c = orig_pixels[i]
        
        # если ближайший цвет рвен shadow_color - замена напрямую
        if np.array_equal(new_c, shadow_color):
            final_c = new_c
        else:
            # final = alpha*new + (1-alpha)*old
            final_c = alpha * new_c + (1.0 - alpha) * old_c

        region_pixels[i] = np.clip(final_c, 0, 255).astype(np.int32)


    image[indices] = region_pixels.astype(np.uint8)

    return image

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

def create_lips_mask(image, faces, predictor):
    """
    Возвращает бинарную маску (0/255), где 255 = область губ.
    faces: список обнаруженных лиц (dlib.rectangle)
    predictor: dlib.shape_predictor
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    for face in faces:
        shape = predictor(gray, face)
        # Собираем точки губ: внешний контур [48..59], внутренний контур [60..67]
        lips_points = []
        
        # Внешний контур (48–59)
        for i in range(48, 60):
            x = shape.part(i).x
            y = shape.part(i).y
            lips_points.append((x, y))
        
        # Внутренний контур (60–67)
        for i in range(60, 68):
            x = shape.part(i).x
            y = shape.part(i).y
            lips_points.append((x, y))
        
        lips_array = np.array(lips_points, dtype=np.int32)
        # fillPoly заполняет многоугольник
        cv2.fillPoly(mask, [lips_array], 255)
    
    return mask

def grayscale_with_colored_shadows(image, shadow_threshold=60, use_original_color=True, custom_color=(210, 225, 253), alpha=0.6):

    # Превращает всё изображение в grayscale
    # Находит тени, где яркость < shadow_threshold (по каналу V (HSV) или просто по grayscale)
    # Заменяет эти области (тени) либо на цвет из original, либо заливает custom_color (с альфа-смешением)
    #shadow_threshol  (0..255) чем выше порог, тем больше область тени
      #use_original_color True, если хотим брать цвет из оригинала; False, если закрашивать custom_color
      #custom_color (0,0,255) -  чисто красный
      #alpha: доля смешения (0..1).  0.0 - чисто grayscale, 1.0 - чисто цвет

       #        palette_bgr = [
       # [162, 185, 241], #
       # [136, 170, 246],
       # [185, 204, 235],
       # [198, 219, 247],  # 3064 skin, eyebrows
       # [210, 225, 253], # 3770 skin
        #[141, 172, 247], # 3771 lips
       # [184, 199, 245], # 3779 skin, lips
    #]

    h, w = image.shape[:2]
    
    # grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    #  gray менье shadow_threshold - тень
    shadow_mask = (gray > shadow_threshold).astype(np.uint8) * 255
    
    # 
    if use_original_color:
        # пиксели shadow_mask=255 возьмём из original

        shadow_color_img = image.astype(np.float32)
    else:
        # Создаём заливку выбранным цветом
        fill = np.full_like(image, custom_color, dtype=np.uint8).astype(np.float32)
        shadow_color_img = fill
    
    gray_f = gray_bgr.astype(np.float32)
    
    #  alpha=1.0 - чистый color; при alpha=0.0 - чистый grayscale
    # Смешиваем в тенях
    out_f = gray_f.copy()
    mask_idx = np.where(shadow_mask == 255)
    
    # Формула: final = alpha*color + (1-alpha)*gray
    out_f[mask_idx] = alpha * shadow_color_img[mask_idx] + (1 - alpha) * gray_f[mask_idx]

    out = np.clip(out_f, 0, 255).astype(np.uint8)
    return out

# маска бровей
def create_eyebrows_mask(image, faces, predictor):

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    for face in faces:
        shape = predictor(gray, face)
        
        # Левая бровь (17..21)
        left_eyebrow_pts = []
        for i in range(17, 22):
            x = shape.part(i).x
            y = shape.part(i).y
            left_eyebrow_pts.append((x, y))
        
        # Правая бровь (22..26)
        right_eyebrow_pts = []
        for i in range(22, 27):
            x = shape.part(i).x
            y = shape.part(i).y
            right_eyebrow_pts.append((x, y))
        
        left_array = np.array(left_eyebrow_pts, dtype=np.int32)
        right_array = np.array(right_eyebrow_pts, dtype=np.int32)
        
        # fillPoly или fillConvexPoly
        cv2.fillPoly(mask, [left_array], 255)
        cv2.fillPoly(mask, [right_array], 255)
    
    return mask

#разделение лица по частям
def segment_face_into_parts(image, face_mask, lips_mask, eyes_mask):

    #def segment_face_into_parts(image, face_mask, lips_mask, eyebrows_mask, eyes_mask):
    # вне лица - face_mask == 0
    inv_face_mask = cv2.bitwise_not(face_mask)
    outside_face = cv2.bitwise_and(image, image, mask=inv_face_mask)  

    # внутри
    inside_face = cv2.bitwise_and(image, image, mask=face_mask)

  
    lips_only = cv2.bitwise_and(inside_face, inside_face, mask=lips_mask)  # B
    # Остаток (внутри лица, но не губы)
    inv_lips_mask = cv2.bitwise_not(lips_mask)
    outside_lips = cv2.bitwise_and(inside_face, inside_face, mask=inv_lips_mask)

    # Из outside_lips реем брови
    # Брови = eyebrows_mask & outside_lips
    #brows_only = cv2.bitwise_and(outside_lips, outside_lips, mask=eyebrows_mask)  # C
    #inv_brows_mask = cv2.bitwise_not(eyebrows_mask)
    #outside_brows = cv2.bitwise_and(outside_lips, outside_lips, mask=inv_brows_mask)

    # Из outside_brows режем глаза
    eyes_only = cv2.bitwise_and(outside_lips, outside_lips, mask=eyes_mask)  # region D
    inv_eyes_mask = cv2.bitwise_not(eyes_mask)
    skin_only = cv2.bitwise_and(outside_lips, outside_lips, mask=inv_eyes_mask)  # region E (остальная кожа)

    return outside_face, lips_only, eyes_only, skin_only, outside_face
    #return outside_face, lips_only, brows_only, eyes_only, skin_only

#соединение всех частей
def combine_face_parts(outside_face, lips, eyes, skin):
   #def combine_face_parts(outside_face, lips, brows, eyes, skin):
    result = cv2.add(outside_face, lips)
    result = cv2.add(result, eyes)
    result = cv2.add(result, skin)
    return result


##############################
# вот тут прописываем
##############################
def main_pipeline(input_image_path, predictor_path, backgroud_image_path, background_output_image_path, output_path):
  
    #replace_background(input_image_path, backgroud_image_path, background_output_image_path)
    # 
    image1 = cv2.imread(input_image_path)
    if image1 is None:
        print("Ошибка загрузки изображения")
        return
    
    #remove_background_selfie_segmentation(image1)
    
    #remove_background_selfie_segmentation(image1)

    bg_image = cv2.imread(backgroud_image_path)
    if bg_image is None:
        print("Ошибка загрузки изображения")
    bg_image = cv2.imread(backgroud_image_path)
    if bg_image is None:
        print("Ошибка загрузки изображения")
        return

    # Применяем глобальные фильтры:
    #cv2.imshow(image1)
    #image = sharpen_filter(image1)
    
    #image = replace_background_color_pil(image, background_color=(150,255,0))
    #image = remove_red_tint_small_areas(image, factor=0.2)
   # image = replaceNoise(image1,9, 100, 100)
    #image = unsharp_mask(image, ksize=(5,5), sigma=1.0, amount=1.9, threshold=0)
    #image = adjust_whites(image)
    #image = enhance_shadows(image, shadow_threshold=80, shadow_strength=0.4)
   # image = adjust_brightness(image)
    #image = adjust_whites_blacks(image, 5, 210) #хорошо для контрастных снимков
    #image = adjust_whites_blacks(image, 10, 215) #хорошо для светлых неконтрастных снимков
    image = increase_color_intensity(image1, sat_scale=1.4,  val_scale=1.0) # хорошо для блеклых малоконтрастных снимков
    #image = increase_color_intensity(image, sat_scale=1.2, val_scale=1.0) # хорошо для блеклых малоконтрастных снимков
    #image = replace_background_color_pil(image, background_color=(150,255,0))
    #dimage = replace_image_background(image, bg_image)
    #image = remove_background_selfie_segmentation(image)

    #image = remove_bg_rembg(image)
    #cv2.imshow("final", image) #чтобы смотреть обработанное изображение

    # Разделение лица на маски
    predictor = dlib.shape_predictor(predictor_path)
    face_mask, faces = get_face_mask(image, predictor_path, scale=3.0)
    face_mask = extract_skin_mask(image)
    lips_mask = create_lips_mask(image, faces, predictor)
   # eyebrows_mask = create_eyebrows_mask(image, faces, predictor)
    eyes_mask = get_eye_mask(image, faces, predictor_path)

    outside_face, lips_only, eyes_only, skin_only, face_mask= segment_face_into_parts(image, face_mask, lips_mask, eyes_mask)

    processed_eyes = process_eye_region(eyes_only)  # тут снова испортила светлые оттенки

    
    new_skin = grayscale_with_colored_shadows(skin_only, shadow_threshold=20, use_original_color=False, custom_color=(136, 170, 246), alpha=0.5)
    palette_bgr = [
       # [162, 185, 241], #
       # [136, 170, 246],
       # [185, 204, 235],
       # [198, 219, 247],  # 3064 skin, eyebrows
        [210, 225, 253], # 3770 skin
        #[141, 172, 247], # 3771 lips
       # [184, 199, 245], # 3779 skin, lips
    ]

    #shawow_color = np.array([117, 150, 211], dtype=np.int32)
    #new_skin = replace_colors_with_gradient(image, face_mask, palette_bgr, shawow_color, alpha=0.6)
    #new_skin = replace_colors_with_gradient_no_mask(skin_only, palette_bgr, shawow_color, alpha=0.6)
    
    cv2.imshow("eyes", processed_eyes)
    cv2.imshow("face", outside_face)
    cv2.imshow("lips", lips_only)
   # cv2.imshow("brows", brows_only)
    cv2.imshow("skin",  new_skin )


    #skin_img, skin_mask = extract_skin()

    #clothes, clothes_mask = segment_clothing(image)
    #cv2.imshow("clothes", clothes)

    #face_mask, faces = get_face_mask(image, predictor_path, scale=3.0)
   # #eye_mask = get_eye_mask(image, faces, predictor_path)
    #eyes_img =cv2.bitwise_and(image, image, mask=eye_mask)
    #processed_eyes = process_eye_region(eyes_img)  
    #cv2.imshow("eyes", processed_eyes)

    out = combine_face_parts(outside_face, lips_only, processed_eyes, new_skin)
    cv2.imshow("out", out)

    #eyebrows_mask 

    #skin_mask = extract_skin_mask(image)
    #skin_img = cv2.bitwise_and(image, image, mask=skin_mask)
    #cv2.imshow("skin", skin_img)
    #processed_skin = process_face_skin(skin_img, skin_mask)
    #cv2.imshow("skin", processed_skin)



    lower = np.array([60, 135, 80], dtype=np.uint8)
    upper = np.array([230, 175, 135], dtype=np.uint8)

    palette_bgr = [
       # [162, 185, 241], #
       # [136, 170, 246],
       # [185, 204, 235],
       # [198, 219, 247],  # 3064 skin, eyebrows
        [210, 225, 253], # 3770 skin
        #[141, 172, 247], # 3771 lips
       # [184, 199, 245], # 3779 skin, lips
    ]

    shawow_color = np.array([117, 150, 211], dtype=np.int32)

    skin_mask = extract_skin_mask(image)
    skin_img = cv2.bitwise_and(image, image, mask=skin_mask)
   # cv2.imshow("clothes", skin_img)

    replaced_color_skin = replace_skin_with_palette_by_mask(skin_img, skin_mask,lower, upper, palette_bgr)
    #replaced_color_skin = replace_skin_with_palette_by_mask_soft(image, skin_mask, lower, upper, palette_bgr, blur_ksize=5, alpha=0.5)
    #replaced_color_skin = replace_skin_with_palette_by_mask_gradient(
    #    image, skin_mask, lower, upper, palette_bgr,
    #    blur_ksize=9,
    #    alpha=1
    #)
    #cv2.imshow("changed_skin", replaced_color_skin)


    

    

    """ area_no_skin_no_eyes = cv2.bitwise_and(image, image, mask=area_other_mask)
    area_skin = cv2.bitwise_and(image, image, mask=area_other_mask)
    #clothing, clothing_mask = segment_clothing(image1)
    # 
    #face_img, background_img = separate_face_background(image, face_mask)
    #skin_img, skin_mask = extract_skin(face_mask)
    # 
    #region3_mask, region4_mask = segment_face_region(face_mask)
    #region3_img = cv2.bitwise_and(face_img, face_img, mask=region3_mask)
    #region4_img = cv2.bitwise_and(face_img, face_img, mask=region4_mask)  # запоминаем без изменений
    #cv2.imshow("region", clothing)
    #
    
    # 
    area5_mask, area6_mask = segment_region3(region3_mask, eye_mask)
    area5_img = cv2.bitwise_and(region3_img, region3_img, mask=area5_mask)
    area6_img = cv2.bitwise_and(region3_img, region3_img, mask=area6_mask)

    skin_mask = extract_skin_mask(area6_img)

    areaw_img= cv2.bitwise_and(area6_img,area6_img,mask = skin_mask)
    #cv2.imshow("skino",areaw_img)
    # 
    areax_img  = remove_red_tint_small_areas(area5_img)
    
    processed_area5 = process_eye_region(areax_img)  
    #cv2.imshow("eyes", processed_area5)
    # cv2.imshow("eyes1", area5_img)
    #cv2.imshow("eyes2", processed_area5)  
    #cv2.imshow("Area5 Mask", area5_mask)
    #cv2.imshow("Area6 Mask", area6_mask)
    #cv2.imshow("Area5 Img", area5_img)
    #cv2.imshow("Processed Area5", processed_area5)
    #cv2.waitKey(0)      # Область глаз: светлое -> белое
    processed_area6 = process_face_skin(area6_img, area6_mask)   # Обработка кожи: 5 бежевых тонов + размытие
    # Объединяем Area 5 и Area 6 в обработанную центральную область (Area 3) ---
    #cv2.imshow("skin", processed_area6)   
    processed_region3 = region3_img.copy()
    idx6 = np.where(area6_mask==255)
    processed_region3[idx6] = processed_area6[idx6]
    idx5 = np.where(area5_mask==255)
    processed_region3[idx5] = processed_area5[idx5]
    # и не изменённое внешнее кольцо (Area 4)
    processed_face = face_img.copy()
    idx3 = np.where(region3_mask==255)
    processed_face[idx3] = processed_region3[idx3]
    #Объединяем обработанное лицо (Area 2) с неизменённым фоном (Area 1) ---
    final_image = cv2.add(background_img, processed_face) """
    
   # cv2.imwrite(output_path, final_image)
    print(f"Финальный результат сохранён: {output_path}")
    #cv2.imshow("Final Image", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ================= Запуск пайплайна =================
if __name__ == '__main__':
    input_image_path = "1.jpg"  # Путь к исходному изображению
    backgroud_image_path = "background.jpg"
    predictor_path = "shape_predictor_68_face_landmarks.dat"  # Путь к модели dlib
    output_image_path = "final_result.jpg"  # Путь для сохранения результата
    background_output_image_path = "wo_background.jpg"
    main_pipeline(input_image_path, predictor_path, backgroud_image_path, background_output_image_path, output_image_path)
