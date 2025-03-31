import streamlit as st
import time
import os
import cv2
import glob
import shutil
import zipfile
import pandas as pd
from ultralytics import YOLO
import supervision as sv
from PIL import Image

# Set up Streamlit UI
st.set_page_config(page_title="AgriSavant", layout="wide")
st.title("\U0001F33F AgriSavant")

st.write(" ")
st.markdown("---")  # Horizontal Separator
st.write(" ")

# File Upload Section
col1, col2 = st.columns([1, 1])
image_file = col1.file_uploader("\U0001F4F8 Select an Image", type=["jpg", "jpeg", "png"], key="image")
folder_file = col2.file_uploader("\U0001F4C1 Select a Folder (Zip)", type=["zip"], key="folder")

# Initialize model
model = YOLO("best.pt")

def process_image(image_path):
    results = model(image_path)
    detections = sv.Detections.from_ultralytics(results[0])
    
    class_names = [model.names[int(cls)] for cls in detections.class_id]
    pest_counts = {name: class_names.count(name) for name in set(class_names)}
    
    image = cv2.imread(image_path)
    if image is None:
        return None, {}, "Error reading image"
    
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    annotated_image = box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    
    return annotated_image, pest_counts, None

processed_data = []

# Processing uploaded image
if image_file:
    image_path = f"temp_{image_file.name}"
    with open(image_path, "wb") as f:
        f.write(image_file.getbuffer())
    
    with st.spinner("⏳ Processing Image..."):
        annotated_image, pest_counts, error = process_image(image_path)
        if error:
            st.error(error)
        else:
            st.success("✅ Processing Completed!")
            processed_data.append([image_file.name, str(pest_counts)])

# Processing ZIP file
if folder_file:
    extract_path = "temp_extracted"
    unprocessed_path = "Unprocessed_Images"
    
    # Create folders
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)
    os.makedirs(extract_path)
    if os.path.exists(unprocessed_path):
        shutil.rmtree(unprocessed_path)
    os.makedirs(unprocessed_path)
    
    # Extract ZIP
    with zipfile.ZipFile(folder_file, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    image_files = glob.glob(os.path.join(extract_path, "*.jpg")) + \
                  glob.glob(os.path.join(extract_path, "*.jpeg")) + \
                  glob.glob(os.path.join(extract_path, "*.png"))
    
    total_images = len(image_files)
    processed_images = 0
    failed_images = []
    
    st.write(f"Total images found: {total_images}")
    
    with st.spinner("⏳ Processing Images..."):
        for img_path in image_files:
            annotated_image, pest_counts, error = process_image(img_path)
            if error:
                failed_images.append(img_path)
                shutil.move(img_path, os.path.join(unprocessed_path, os.path.basename(img_path)))
            else:
                processed_images += 1
                processed_data.append([os.path.basename(img_path), str(pest_counts)])
    
    st.success(f"✅ Processed {processed_images}/{total_images} images successfully!")
    if failed_images:
        st.warning(f"⚠️ {len(failed_images)} images failed to process and were moved to 'Unprocessed_Images' folder.")
        
        # Create ZIP of failed images
        failed_zip_path = "Unprocessed_Images.zip"
        shutil.make_archive("Unprocessed_Images", 'zip', unprocessed_path)
        
        with open(failed_zip_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Unprocessed Images",
                data=f,
                file_name="Unprocessed_Images.zip",
                mime="application/zip"
            )

# Download Processed Data as Excel
if processed_data:
    df = pd.DataFrame(processed_data, columns=["Image Name", "Pest Count"])
    excel_path = "Processed_Data.xlsx"
    df.to_excel(excel_path, index=False)
    
    with open(excel_path, "rb") as f:
        st.download_button(
            label="⬇️ Download Processed Data (Excel)",
            data=f,
            file_name="Processed_Data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ---------- Analysis Section ----------
st.markdown("---")
st.markdown("#### 📌 Select an Analysis Type")

# Analysis Cards
cards = {
    "Pest Detection Name": "🐛 Pest Detected: ",
    "Pest Count with Names": "📊 Pest Count: ",
    "Leaf Extraction": "🍃 Leaf Extraction: Extracted leaf details will appear here.",
    "Color Analysis": "🎨 Color Analysis: Insights about color will be shown.",
    "Nutrient Availability": "💧 Nutrient Availability: Distribution insights will be displayed."
}

col1, col2, col3 = st.columns([1, 0.05, 2])

# Left Section: Card Selection
with col1:
    for card in cards.keys():
        if st.button(card, key=card):
            st.session_state["selected_card"] = card

# Right Section: Display Output
with col3:
    if "selected_card" in st.session_state:
        selected_card = st.session_state["selected_card"]
        output_text = cards[selected_card]
        
        if selected_card == "Pest Detection Name" and image_file:
            output_text += ", ".join(pest_counts.keys()) if pest_counts else "No pests detected."
        
        elif selected_card == "Pest Count with Names" and image_file:
            output_text += str(pest_counts) if pest_counts else "No pests detected."
        
        st.markdown(
            f"""
            <div style="
                border: 2px solid ; 
                border-radius: 10px; 
                padding: 15px; 
                background-color: white; 
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                font-size: 16px;
                text-align: left;
                color: black;">
                🔍 {selected_card} <br><br>
                {output_text}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if selected_card in ["Pest Detection Name", "Pest Count with Names"] and image_file and annotated_image is not None:
            st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Annotated Image", use_container_width=True)
