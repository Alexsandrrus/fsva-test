import cv2
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Инициализация модели YOLO
model = YOLO('yolov5s.pt')  # Убедитесь, что модель yolov5s.pt доступна

def detect_and_crop(video_path, output_frames_dir, margin=10):
    """Детектирует фигуру спортсмена и обрезает кадры."""
    os.makedirs(output_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {video_path}")
        return [], []

    frame_count = 0
    cropped_frames = []
    box_sizes = []  # Список для хранения размеров боксов

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Детекция объектов
        results = model(frame)
        boxes = results[0].boxes.xyxy  # Получение координат bounding box'ов

        if len(boxes) == 0:
            print("No objects detected in this frame.")
            continue  # Пропускаем кадр, если нет обнаруженных объектов

        for box in boxes:
            if len(box) == 6:
                x1, y1, x2, y2, conf, cls = box
            elif len(box) == 4:
                x1, y1, x2, y2 = box
                conf = None
                cls = None
            else:
                print("Unexpected box format:", box)
                continue

            # Обрезка кадра с учетом отступа
            x1 = max(0, int(x1) - margin)
            y1 = max(0, int(y1) - margin)
            x2 = min(frame.shape[1], int(x2) + margin)
            y2 = min(frame.shape[0], int(y2) + margin)

            # Проверка на корректность координат
            if x1 < x2 and y1 < y2:
                cropped_frame = frame[y1:y2, x1:x2]
                cropped_frames.append(cropped_frame)

                # Сохранение обрезанного кадра
                cv2.imwrite(os.path.join(output_frames_dir, f'frame_{frame_count}.jpg'), cropped_frame)
                frame_count += 1

                # Сохранение размеров боксов
                box_sizes.append((x2 - x1, y2 - y1))  # (ширина, высота)

    cap.release()
    return cropped_frames, box_sizes

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
    
    cropped_frames, box_sizes = detect_and_crop(video_path, output_frames_dir)
    box_sizes_df = pd.DataFrame(box_sizes, columns=['width', 'height'])
    box_sizes_df.to_csv('output/box_sizes.csv', index=False)
    create_video(cropped_frames, output_video_path)
    print("✅ Обработка завершена.")