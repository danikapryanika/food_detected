from ultralytics import YOLO
import cv2
import dxcam
import time

# Инициализация модели детекции (замените на свою обученную модель)
model = YOLO('best.pt')
model.to('cuda')

  # Используем модель детекции
food_classes = ['french_fries', 'hamburger', 'ice_cream', 'pizza']

# Настройка захвата экрана
cam = dxcam.create(output_idx=0, output_color="BGR")
target_fps = 10
frame_time = 10.0 / target_fps

def detect_food(frame):
    """Детекция еды на кадре"""
    results = model(frame, verbose=False)  # Детекция объектов
    
    annotated_frame = frame.copy()
    
    for result in results:
        boxes = result.boxes  # Получаем ограничивающие рамки
        for box in boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            
            # Проверяем, что это один из наших классов еды
            if class_name in food_classes:
                confidence = box.conf.item()
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Рисуем рамку и подпись
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                print(f"Обнаружено: {label}")
    
    return annotated_frame

def main():
    cam.start(target_fps=target_fps)
    cv2.namedWindow('Food Detection', cv2.WINDOW_NORMAL)
    
    try:
        while True:
            start_time = time.time()
            
            # Захват кадра
            frame = cam.get_latest_frame()
            if frame is None:
                continue
            
            # Детекция и отображение
            processed_frame = detect_food(frame)
            cv2.imshow('Food Detection', processed_frame)
            
            # Выход по 'q'
            if cv2.waitKey(1) == ord('q'):
                break
                
            # Контроль FPS
            elapsed = time.time() - start_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
                
    finally:
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()