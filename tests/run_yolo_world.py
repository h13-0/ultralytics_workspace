from ultralytics import YOLO
 
 
if __name__ == '__main__':
 
    # Initialize a YOLO-World model
    model = YOLO(r'./weights/yolov8/yolov8l-worldv2.pt')  # or choose yolov8m/l-world.pt
    #model = YOLO(r'./ultralytics/cfg/models/v8/yolov8s-worldv2.yaml')  # or choose yolov8m/l-world.pt

    # Define custom classes
    model.set_classes(["person", "building", "bubble"])
 
    # Execute prediction for specified categories on an image
    results = model.predict(r'E:/h13/workspace/datasets/MISC/1.jpg')
 
    # Show results
    results[0].show()

