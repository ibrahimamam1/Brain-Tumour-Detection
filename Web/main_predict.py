import os
import torch
import urllib.request
import random  # Added for the random noise transform
from torchvision import transforms 
from PIL import Image
import torchvision.models as models
from typing import List, Tuple
import io
import uuid
import firebase_admin
from firebase_admin import initialize_app
from firebase_admin import credentials, db
import os 
import gdown
import timm 
# ========== CONFIGURATION ========== #

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    os.path.join("Models", "model.pth"))

MODEL_ID = "1UaFw3vAuimY6r47mYbkyFXmtsjwDjsoP"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={MODEL_ID}"

cred_path = os.path.join(os.path.dirname(__file__), "project-ml-c9e5f-firebase-adminsdk-fbsvc-2618b8c059.json")

cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://project-ml-c9e5f-default-rtdb.asia-southeast1.firebasedatabase.app/' 
})


def download_model():
    """
    Downloads the trained model
    """

    print("🔽 Downloading model...")
    try:
        gdown.download(MODEL_URL)
        print("✅ Model downloaded.")
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        raise



def load_model(model_path: str = MODEL_PATH, device: str = "cpu") -> torch.nn.Module:
    """
    Loads the brain tumor classification model with proper error handling
    """
    try:
        # 1. Initialize the base ViT model
        model = timm.create_model('vit_base_patch16_224', pretrained=False)
        
        # 2. Modify head for our classification scenario
        model.head = torch.nn.Linear(model.head.in_features, 4)        
        
        # 3. Load our fine-tuned weights
        if not os.path.exists(model_path):
            download_model()
            
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
       
        # 4. Configure for inference
        model.to(device)
        model.eval()
        
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
            
        return model
        
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}") from e

# ======= TRANSFORM PIPELINE ======= #
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ======= PREDICTION FUNCTION ======= #
from typing import Tuple, List
from PIL import Image
import io
import torch

from utils.css_loader import get_css_styles

from templates.html_templates import (
    generate_summary_result,
    generate_detailed_report_header,
    generate_description_section,
    generate_probability_table,
    generate_dual_prediction_note,
    generate_metrics_section,
    generate_clinical_section,
    close_report_container
)

def predict_brain_tumor_batch(img_list: list, css_file_path: str = None) -> Tuple[str, str, List[List], List[dict]]:
    """
    Predict brain tumors from a batch of images
    
    Args:
        img_list (list): List of images to analyze
        css_file_path (str, optional): Path to external CSS file
        
    Returns:
        Tuple containing results, detailed reports, tumor types data, and current predictions
    """
    results = []
    detailed_reports = []
    tumor_types_data = []    # For dataframe: list of rows [image, prediction, confidence %]
    current_predictions = [] # For state, detailed info per image

    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    class_descriptions = {
        'glioma': 'A type of tumor that occurs in the brain and spinal cord, arising from glial cells.',
        'meningioma': 'A tumor that arises from the meninges, the membranes surrounding the brain and spinal cord.',
        'notumor': 'No tumor detected in the brain MRI scan.',
        'pituitary': 'A tumor affecting the pituitary gland at the base of the brain that controls many hormonal functions.'
    }

    # Load CSS styles dynamically
    css_styles = get_css_styles(css_file_path)

    if not img_list:
        return ("⚠️ No images uploaded", "Please upload MRI images for analysis", [], [])

    for idx, img_data in enumerate(img_list, start=1):
        try:
            # Load image
            if isinstance(img_data, str):
                img = Image.open(img_data).convert("RGB")
                filename = img_data.split("/")[-1]
            elif hasattr(img_data, 'read'):
                img = Image.open(img_data).convert("RGB")
                filename = getattr(img_data, 'name', f"image_{idx}")
            elif isinstance(img_data, bytes):
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                filename = f"image_{idx}"
            else:
                raise ValueError(f"Unsupported image type: {type(img_data)}")

            # Perform prediction rounds
            rounds = 1
            preds_per_round = 1
            round_confidences = []

            for r in range(rounds):
                class_conf_sum = torch.zeros(len(class_names))
                for _ in range(preds_per_round):
                    input_tensor = transform(img).unsqueeze(0)
                    with torch.no_grad():
                        output = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                    class_conf_sum += probabilities
                avg_confidence = class_conf_sum / preds_per_round
                round_confidences.append(avg_confidence)

            # Pick best round based on highest max confidence
            best_round_idx = max(range(rounds), key=lambda i: round_confidences[i].max().item())
            best_avg_conf = round_confidences[best_round_idx]

            # Determine top 2 classes
            sorted_confidences = torch.topk(best_avg_conf, 2)
            top1_idx, top2_idx = sorted_confidences.indices.tolist()
            top1_conf, top2_conf = sorted_confidences.values.tolist()

            predicted_class = class_names[top1_idx]
            confidence_score = top1_conf
            second_class = class_names[top2_idx]
            second_confidence = top2_conf

            # Check if the difference between top 2 is ≤ 15%
            diff_percent = (top1_conf - top2_conf) * 100
            show_dual_prediction = diff_percent <= 15.0

<<<<<<< HEAD
            # Summary output (plain text)
            # Improved summary output with enhanced CSS styling
            if show_dual_prediction:
                results.append(
                    f"""
                    <div style="
                        background: linear-gradient(90deg, #fef9c3 60%, #f1f5f9 100%);
                        border-radius: 10px;
                        padding: 16px 22px;
                        margin: 12px 0 18px 0;
                        box-shadow: 0 2px 8px rgba(99,102,241,0.07);
                        border-left: 6px solid #f59e42;
                        font-family: 'Segoe UI', Arial, sans-serif;
                        font-size: 1.08em;
                        color: #92400e;
                        display: flex;
                        align-items: center;
                        gap: 12px;
                    ">
                        <span style="font-size:1.3em;">🖼️</span>
                        <span class="filename" style="color:#64748b;font-style:italic;">{filename}</span>
                        <span style="margin-left:8px;">
                            <b style="color:#b45309;">🧠 {predicted_class.title()} <span style="font-weight:400;">or</span> {second_class.title()}</b>
                            <span style="color:#b45309;">({confidence_score*100:.1f}% vs {second_confidence*100:.1f}%)</span>
                            <span style="background:#fde68a;color:#b45309;padding:2px 8px;border-radius:6px;font-size:0.98em;margin-left:8px;">≤15% diff</span>
                        </span>
                    </div>
                    """
                )
            else:
                results.append(
                    f"""
                    <div style="
                        background: linear-gradient(90deg, #e0e7ef 60%, #f1f5f9 100%);
                        border-radius: 10px;
                        padding: 16px 22px;
                        margin: 12px 0 18px 0;
                        box-shadow: 0 2px 8px rgba(99,102,241,0.07);
                        border-left: 6px solid #6366f1;
                        font-family: 'Segoe UI', Arial, sans-serif;
                        font-size: 1.08em;
                        color: #1e293b;
                        display: flex;
                        align-items: center;
                        gap: 12px;
                    ">
                        <span style="font-size:1.3em;">🖼️</span>
                        <span class="filename" style="color:#64748b;font-style:italic;">{filename}</span>
                        <span style="margin-left:15px;">
                            <b style="color:#2563eb;">🧠 {predicted_class.title()}</b>
                            <span style="color:#2563eb;">({confidence_score*100:.1f}%)</span>
                            <span style="background:#c7d2fe;color:#3730a3;padding:2px 8px;border-radius:6px;font-size:0.98em;margin-left:8px;">Best of 20 rounds</span>
                        </span>
                    </div>
                    """
                )

            # Improved CSS styles for HTML output (modern, clean, accessible)
            css_styles = """
            <style type="text/css">
            .report-container {
                font-family: 'Segoe UI', Arial, sans-serif;
                background: #f4f8fb;
                border-radius: 14px;
                padding: 28px 28px 22px 28px;
                margin-bottom: 36px;
                box-shadow: 0 4px 16px rgba(30, 41, 59, 0.08);
                border: 1px solid #e2e8f0;
                max-width: 650px;
            }
            .report-title {
                font-size: 1.35em;
                font-weight: 700;
                color: #1e293b;
                margin-bottom: 10px;
                letter-spacing: 0.01em;
            }
            .divider {
                border-bottom: 2px solid #e0e7ef;
                margin: 14px 0 20px 0;
            }
            .prediction-main {
                font-size: 1.13em;
                color: #2563eb;
                font-weight: 600;
                margin-bottom: 10px;
                letter-spacing: 0.01em;
            }
            .desc {
                color: #334155;
                margin-bottom: 12px;
                font-size: 1.01em;
            }
            .prob-table {
                width: 100%;
                border-collapse: separate;
                border-spacing: 0 4px;
                margin-bottom: 12px;
            }
            .prob-table th, .prob-table td {
                padding: 7px 12px;
                text-align: left;
                border: none;
                font-size: 0.99em;
            }
            .prob-table th {
                background: #e0e7ef;
                color: #22223b;
                font-weight: 600;
            }
            .prob-bar, .prob-bar-main {
                display: inline-block;
                height: 18px;
                border-radius: 7px;
                vertical-align: middle;
                transition: width 0.4s cubic-bezier(.4,2.3,.3,1);
            }
            .prob-bar {
                background: linear-gradient(90deg, #c7d2fe 60%, #e0e7ef 100%);
            }
            .prob-bar-main {
                background: linear-gradient(90deg, #6366f1 70%, #818cf8 100%);
            }
            .note {
                color: #b91c1c;
                font-weight: 500;
                margin: 10px 0 8px 0;
                font-size: 1.01em;
            }
            .metrics {
                background: #f1f5f9;
                border-radius: 9px;
                padding: 12px 16px;
                margin: 12px 0;
                font-size: 0.98em;
                color: #334155;
                border-left: 4px solid #6366f1;
            }
            .clinical {
                background: #fef9c3;
                border-radius: 9px;
                padding: 12px 16px;
                color: #92400e;
                margin: 12px 0 0 0;
                font-size: 0.99em;
                border-left: 4px solid #f59e42;
            }
            .filename {
                color: #64748b;
                font-size: 0.99em;
                font-style: italic;
                letter-spacing: 0.01em;
            }
            @media (max-width: 700px) {
                .report-container { padding: 16px 6vw; }
                .prob-table th, .prob-table td { padding: 6px 4vw; }
            }
            </style>
            """

            # Detailed report with CSS styles
=======
            # Generate summary result using template
            summary_html = generate_summary_result(
                filename, predicted_class, confidence_score,
                second_class, second_confidence, show_dual_prediction
            )
            results.append(summary_html)

            # Generate detailed HTML report using templates
            detail_report = [css_styles]
            
            # Header section
            detail_report.append(generate_detailed_report_header(
                filename, predicted_class, confidence_score, best_round_idx, rounds
            ))
            
            # Description section
            detail_report.append(generate_description_section(predicted_class, class_descriptions))
            
            # Probability table
            detail_report.append(generate_probability_table(class_names, best_avg_conf, top1_idx))
            
            # Optional dual prediction note
            if show_dual_prediction:
                detail_report.append(generate_dual_prediction_note(second_class, second_confidence))
            
            # Metrics section
            detail_report.append(generate_metrics_section(
                confidence_score, best_avg_conf, class_names,
                get_confidence_level, get_second_most_likely,
                calculate_probability_spread, calculate_uncertainty
            ))
            
            # Clinical considerations
            detail_report.append(generate_clinical_section(
                predicted_class, confidence_score, get_clinical_considerations
            ))
            
            # Close container
            detail_report.append(close_report_container())
            
            detailed_reports.append("".join(detail_report))
>>>>>>> 1dda59336f3a9c3939aeb69f43e2f479a041b8b1

            # Dataframe + preview state
            tumor_types_data.append([filename, predicted_class.title(), round(confidence_score * 100, 2)])
            current_predictions.append({
                "filename": filename,
                "class": predicted_class,
                "confidence": confidence_score * 100,
                "image": img_data
            })

            # Push to Firebase (keeping original functionality)
            push_to_firebase(file_path=img_data, prediction={
                "filename": filename.split("\\")[-1],
                "extra": (
                    f"class title: {predicted_class.title()}| "
                    f"class descriptions: {class_descriptions[predicted_class]} |  "
                    f"confidence score: {confidence_score:.4f} |  "
                    f"best round: {best_round_idx + 1} of {rounds} |  "
                    f"best avg confidence: {best_avg_conf.tolist()}   "
<<<<<<< HEAD
                )

        
                        , "analysis3" : (
                f"Top-1 Class: {predicted_class.title()} with confidence {confidence_score * 100:.2f}%. | | "
                f"Top-2 Class: {second_class.title()} with confidence {second_confidence * 100:.2f}%. | | "
                f"Prediction confidence across classes: {best_avg_conf.tolist()}. | | "
                f"Best result obtained in round {best_round_idx + 1} of {rounds}. | | "
                
            )
                        ,"analysis2" : (
            f"Top Prediction: {predicted_class.title()} ({confidence_score * 100:.2f}%) | |  "
            f"Second Prediction: {second_class.title()} ({second_confidence * 100:.2f}%) | | "
            f"Best Round: {best_round_idx + 1} of {rounds} |  |  "
            f"Avg Confidence Per Class: {best_avg_conf.tolist()}  "
        )


}
    
                )
=======
                ),
                "details": (
                    f"Probability Spread: {calculate_probability_spread(best_avg_conf)}  | | "
                    f"Uncertainty Index: {calculate_uncertainty(best_avg_conf)}  | | "
                    f"Confidence Level: {get_confidence_level(confidence_score)}  | | "
                    f"Second Most Likely: {get_second_most_likely(best_avg_conf, class_names)}  | | "
                    f"Clinical Considerations: {get_clinical_considerations(predicted_class, confidence_score)} "
                ),
                "analysis3": (
                    f"Top-1 Class: {predicted_class.title()} with confidence {confidence_score * 100:.2f}%. | | "
                    f"Top-2 Class: {second_class.title()} with confidence {second_confidence * 100:.2f}%. | | "
                    f"Prediction confidence across classes: {best_avg_conf.tolist()}. | | "
                    f"Best result obtained in round {best_round_idx + 1} of {rounds}. | | "
                ),
                "analysis2": (
                    f"Top Prediction: {predicted_class.title()} ({confidence_score * 100:.2f}%) | |  "
                    f"Second Prediction: {second_class.title()} ({second_confidence * 100:.2f}%) | | "
                    f"Best Round: {best_round_idx + 1} of {rounds} |  |  "
                    f"Avg Confidence Per Class: {best_avg_conf.tolist()}  "
                )
            })
>>>>>>> 1dda59336f3a9c3939aeb69f43e2f479a041b8b1

        except Exception as e:
            print(f"Done processing image: {e}")

    return (
        "<br>".join(results) if results else "No results generated",
        tumor_types_data,
        current_predictions
    )

PIXELDRAIN_API_KEY = "91d780db-af6e-4cc9-b3f2-1f80ba77817c"  # Replace with your Pixeldrain API key
import base64    
import os


def upload_file_gofile(file_path: str) -> str:

    file_name = os.path.basename(file_path)
    url = f"https://pixeldrain.com/api/file/{file_name}"
    
    headers = {}
    if PIXELDRAIN_API_KEY:
        # Authorization: Basic :<API‑KEY> (حقل username فارغ)
        token = base64.b64encode(f":{PIXELDRAIN_API_KEY}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"
    
    with open(file_path, "rb") as f:
        resp = requests.put(url, data=f, headers=headers, timeout=60)
    
    if resp.status_code in (200, 201):
        data = resp.json()
        file_id = data.get("id") or data.get("data", {}).get("id")
        print(f"File uploaded successfully: {file_id}")
        if not file_id:
            raise Exception(f"Unexpected response: {data}")
        return f"https://pixeldrain.com/api/file/{file_id}"
    else:
        raise Exception(f"Upload failed {resp.status_code}: {resp.text}")


def push_to_firebase(file_path, prediction):

    gofile_url = upload_file_gofile(file_path)

 
    unique_id = str(uuid.uuid4())

    data = {
         "filename": prediction.get("filename", "")
        , "extra": prediction.get("extra", "")
        , "url": gofile_url
        , "details": prediction.get("details", "")
        , "analysis2": prediction.get("analysis2", "")
        , "analysis3": prediction.get("analysis3", ""),
        "id": unique_id
    }


    ref = db.reference("predication")
    ref.child(unique_id).set(data)



# Initialize model
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device=device)
    print(f"✅ Model loaded successfully on {device}")

except Exception as e:
    print(f"❌ Initialization failed: {str(e)}")
    if "invalid_grant" in str(e):
        print("\n🔥 Firebase Authentication Failed! Possible causes:")
        print("1. Credentials JSON file is invalid/expired")
        print("2. Incorrect databaseURL in config")
        print("3. System clock is out of sync")
    raise RuntimeError(f"Initialization failed: {str(e)}") from e
