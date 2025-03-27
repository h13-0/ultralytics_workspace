from ultralytics import YOLO
from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
from ultralytics import YOLOWorld


def main():
    model = YOLO(r'.\ultralytics\cfg\models\v8\yolov8s-worldv2.yaml')

    data = dict(
        train=dict(
            yolo_data=["Objects365.yaml"],
            grounding_data=[
                dict(
                    img_path="../datasets/flickr30k/images",
                    json_file="../datasets/flickr30k/final_flickr_separateGT_train.json",
                ),
                #dict(
                #    img_path="../datasets/GQA/images",
                #    json_file="../datasets/GQA/final_mixed_train_no_coco.json",
                #),
            ],
        ),
        val=dict(yolo_data=["lvis.yaml"]),
    )

    model = YOLOWorld("yolov8s-worldv2.yaml")
    model.train(data=data, trainer=WorldTrainerFromScratch)


if __name__ == '__main__':
    main()