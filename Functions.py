from ultralytics import YOLOv10
import supervision as sv
import cv2


model = YOLOv10('best.pt')


def Draw(Image_path):
    image = cv2.imread(Image_path)
    results = model(image)[0]
    detections = sv.Detections.from_ultralytics(results)

    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)

    cv2.imwrite("Output.jpg", annotated_image)


#results = Run_Image("Images/Image3.jpg")
Draw("Images_test/image.png")

























