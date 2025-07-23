import os
import torch
from ultralytics import YOLO

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

import os

def main():
    model = YOLO('yolov8n.pt') 
    
    results = model.train(data='data.yaml', epochs=100, imgsz=640, batch=4)

if __name__ == '__main__':
    main()