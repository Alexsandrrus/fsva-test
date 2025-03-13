from gettext import install
from winreg import REG_RESOURCE_REQUIREMENTS_LIST
import pip


import cv2
import os
import numpy as np
from ultralytics import YOLO

# Инициализация модели YOLO
model = YOLO('yolov5s.pt')  

def detect_and_crop(video_path, output_frames_dir, margin=10):
    """Детектирует фигуру спортсмена и обрезает кадры."""
    os.makedirs(output_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    cropped_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Детекция объектов
        results = model(frame)
        boxes = results[0].boxes.xyxy  # Получение координат bounding box'ов

        for box in boxes:
            # Выводим содержимое box для отладки
            print("Box:", box)

            # Проверяем количество значений в box
            if len(box) == 6:
                x1, y1, x2, y2, conf, cls = box
            elif len(box) == 4:
                x1, y1, x2, y2 = box
                conf = None  # Устанавливаем confidence в None, если он недоступен
                cls = None   # Устанавливаем класс в None, если он недоступен
            else:
                print("Unexpected box format:", box)
                continue  # Пропускаем, если формат не соответствует ожиданиям

            # Обрезка кадра с учетом отступа
            x1 = max(0, int(x1) - margin)
            y1 = max(0, int(y1) - margin)
            x2 = min(frame.shape[1], int(x2) + margin)
            y2 = min(frame.shape[0], int(y2) + margin)

            cropped_frame = frame[y1:y2, x1:x2]
            cropped_frames.append(cropped_frame)

            # Сохранение обрезанного кадра
            cv2.imwrite(os.path.join(output_frames_dir, f'frame_{frame_count}.jpg'), cropped_frame)
            frame_count += 1

    cap.release()
    return cropped_frames
def create_video(cropped_frames, output_video_path):
    """Создает видео из обрезанных кадров."""
    if cropped_frames:
        height, width, _ = cropped_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))

        for cropped_frame in cropped_frames:
            out.write(cropped_frame)

        out.release()

if __name__ == "__main__":
    video_path = 'input/input_video.mp4'
    output_frames_dir = 'output/frames/'
    output_video_path = 'output/processed_video.mp4'
    
    cropped_frames = detect_and_crop(video_path, output_frames_dir)
    create_video(cropped_frames, output_video_path)
    print("✅ Обработка завершена.")