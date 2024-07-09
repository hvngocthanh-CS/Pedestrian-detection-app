import streamlit as st
from PIL import Image
import cv2
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.utils import draw_bounding_boxes

res_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
mobile_weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT

res_categories = res_weights.meta["categories"] 
res_img_preprocess = res_weights.transforms()

mobile_categories = mobile_weights.meta["categories"] 
mobile_img_preprocess = mobile_weights.transforms()

from ultralytics import YOLO

yolo_model = None
def load_yolo_model():
    global yolo_model
    if yolo_model is None:
        yolo_model = YOLO('yolov8n.pt')  # Pre-trained YOLOv8n model
    return yolo_model

def load_res_model(thresh):
    model = fasterrcnn_resnet50_fpn_v2(weights=res_weights, box_score_thresh = thresh)
    model.eval()
    return model

def load_mobile_model(thresh):
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=mobile_weights, box_score_thresh = thresh)
    model.eval()
    return model

def make_prediction(img, model, categories, img_preprocess): 
    img_processed = img_preprocess(img) 
    prediction = model(img_processed.unsqueeze(0))
    prediction = prediction[0]                       ## Dictionary with keys "boxes", "labels", "scores".
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

def create_image_with_bboxes(img, prediction): ## Adds Bounding Boxes around original Image.
    img_tensor = torch.tensor(img) ## Transpose
    count=0
    img_with_bboxes = draw_bounding_boxes(img_tensor,
                                          boxes=prediction["boxes"], 
                                          labels=prediction["labels"],
                                          colors = ["red" if label == "person" else "green" if label == "bird" else "blue" for label in prediction["labels"]],
                                        width=2)
    for i in prediction["labels"]:
        if i=='person':
            count = count+1
    
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1,2,0) 
    st.write('=> ', count, 'pedestrians')
    return img_with_bboxes_np

def main():
    st.title("Object Detection")
    st.write("Huỳnh Võ Ngọc Thanh - 21520449")
    pick = st.sidebar.selectbox("Model", ("YOLOv8", "Faster-RCNN ResNet50"))   
    upload = st.file_uploader(label="Upload Image Here:", type=["png", "jpg", "jpeg"])

    if upload:
        img = Image.open(upload)
        if pick == 'YOLOv8':
            # Sử dụng mô hình YOLO
            model = load_yolo_model() 
            yolo_start_time = time.time()
            results = model(img)  # Dự đoán trên ảnh
            yolo_time = time.time() - yolo_start_time
            
            for r in results:
                #r.show()
                r.save(filename=f'result_{upload.name}')
                # Show the saved image with bounding boxes
                result_img = Image.open(f'result_{upload.name}')
                st.image(result_img, caption=f"Result Image", use_column_width=True)
            st.write(f"Time: {yolo_time:.2f} seconds")
            
        else:
            thresh = st.slider('Thresh', min_value=0.1, max_value=1.0, value=0.55, step=0.01)
            res_start_time = time.time()
            prediction = make_prediction(img, load_res_model(thresh), res_categories, res_img_preprocess)
            resnet50_time = time.time() - res_start_time
            img_with_bbox = create_image_with_bboxes(np.array(img).transpose(2,0,1), prediction)
            # Display image with bounding boxes and confidence scores
            fig, ax = plt.subplots()
            ax.imshow(img_with_bbox)
            for box, score in zip(prediction["boxes"], prediction["scores"]):
                    label = f"{score:.2f}"
                    ax.text(box[0], box[1], label, color="white", fontsize=8, ha="left", va="bottom", bbox=dict(facecolor="purple", alpha=0.8, pad=0.5))
            plt.axis('off')
            plt.tight_layout()

            st.pyplot(fig, use_container_width=True)

            # Display time taken for Faster-RCNN ResNet50
            st.write(f"Time: {resnet50_time:.2f} seconds")
            
            # Remove boxes from prediction dictionary for cleaner display
            del prediction["boxes"]
            st.header("Predicted Probabilities")
            st.write(prediction)
            
if __name__ == "__main__":
    main()