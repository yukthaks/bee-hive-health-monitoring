from ultralytics import YOLO

def main():
    model = YOLO('trains/run3/weights/best.pt')
    metrics = model.val(data='config.yaml', imgsz=416, batch=2, device='0')
    print(metrics)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support() 
    main()
