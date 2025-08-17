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
from assets.templates.html_templates import generate_batch_overview, generate_detailed_report
from attention_map import ViTAttentionMap, get_overlaid_image
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_DESCRIPTIONS = {
    'glioma': 'Tumor arising from glial cells in brain/spinal cord.',
    'meningioma': 'Tumor from meninges surrounding brain/spinal cord.',
    'notumor': 'No tumor detected.',
    'pituitary': 'Tumor affecting the pituitary gland.',
}

def predict_image(img: Image.Image) -> Tuple[str, float, List[Tuple[float, str]]]:
    """Run model on single image and return top prediction + confidences"""
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]

    sorted_vals, sorted_idx = torch.topk(probs, len(CLASS_NAMES))
    confidences = [(v.item(), CLASS_NAMES[i]) for v,i in zip(sorted_vals, sorted_idx)]
    return confidences[0][1], confidences[0][0], confidences


def predict_brain_tumor_batch(img_list: list) -> Tuple[str, str, List[List], List[dict]]:
    results, df_data, state = [], [], []
    if not img_list:
        return ("<div>⚠️ No images uploaded</div>", "", [], [])

    # Store original image data for detailed view
    image_data_store = {}
    overlaid_images_store = {}
    model = load_model()
    vit_attn = ViTAttentionMap(model)


    for idx, img_data in enumerate(img_list, 1):
        try:
            if isinstance(img_data, str):
                img = Image.open(img_data).convert("RGB")
                fname = img_data.split("/")[-1]
            elif hasattr(img_data, 'read'):
                img = Image.open(img_data).convert("RGB")
                fname = getattr(img_data, 'name', f"image_{idx}")
            elif isinstance(img_data, bytes):
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                fname = f"image_{idx}"
            else:
                raise ValueError("Unsupported input type")

            # Store original image temporarily
            original_img = img.copy()  # Keep a copy if you want to preserve original

            pred_class, conf, all_conf = predict_image(img)

            # Preprocess for model input
            input_tensor = transform(img).unsqueeze(0).to(device)
            # Compute attention map
            attn_map = vit_attn(input_tensor)

            # Apply overlay
            overlaid_img = get_overlaid_image(original_img, attn_map)

            image_data_store[idx-1] = original_img
            overlaid_images_store[idx-1] = overlaid_img

            df_data.append([fname, pred_class, f"{conf*100:.1f}%"])
            state.append({
                "filename": fname,
                "predicted_class": pred_class,
                "confidence": conf,
                "all_predictions": all_conf,
                "description": CLASS_DESCRIPTIONS[pred_class],
                "image_idx": idx-1
            })
        except Exception as e:
            print(f"failed to predict image{e}")
            fname = f"image_{idx}"
            df_data.append([fname, "Error", "0%"])
            state.append({
                "filename": fname,
                "predicted_class": "Error",
                "confidence": 0,
                "all_predictions": [],
                "description": str(e),
                "image_idx": idx-1
            })

    batch_html = generate_batch_overview(state)

    # Generate detailed view for first image by default
    detailed_html = ""
    if state:
        detailed_html = generate_detailed_report(
            state[0], state,  # Pass all predictions for navigation
            0,  # Current index
            overlaid_images_store.get(0)  # Now passes the overlaid image
        )

    return batch_html, detailed_html, df_data, state

    # tumor_types_data.append([filename, predicted_class.title(), round(confidence_score * 100, 2)])
           # current_predictions.append({
           #     "filename": filename,
           #     "class": predicted_class,
           #     "confidence": confidence_score * 100,
           #     "image": img_data
           # })
#
           # # Push to Firebase (keeping original functionality)
           # push_to_firebase(file_path=img_data, prediction={
           #     "filename": filename.split("\\")[-1],
           #     "extra": (
           #         f"class title: {predicted_class.title()}| "
           #         f"class descriptions: {class_descriptions[predicted_class]} |  "
           #         f"confidence score: {confidence_score:.4f} |  "
           #         f"best round: {best_round_idx + 1} of {rounds} |  "
           #         f"best avg confidence: {best_avg_conf.tolist()}   "
           #     ),
           #     "details": (
           #         f"Probability Spread: {calculate_probability_spread(best_avg_conf)}  | | "
           #         f"Uncertainty Index: {calculate_uncertainty(best_avg_conf)}  | | "
           #         f"Confidence Level: {get_confidence_level(confidence_score)}  | | "
           #         f"Second Most Likely: {get_second_most_likely(best_avg_conf, class_names)}  | | "
           #         f"Clinical Considerations: {get_clinical_considerations(predicted_class, confidence_score)} "
           #     ),
           #     "analysis3": (
           #         f"Top-1 Class: {predicted_class.title()} with confidence {confidence_score * 100:.2f}%. | | "
           #         f"Top-2 Class: {second_class.title()} with confidence {second_confidence * 100:.2f}%. | | "
           #         f"Prediction confidence across classes: {best_avg_conf.tolist()}. | | "
           #         f"Best result obtained in round {best_round_idx + 1} of {rounds}. | | "
           #     ),
           #     "analysis2": (
           #         f"Top Prediction: {predicted_class.title()} ({confidence_score * 100:.2f}%) | |  "
           #         f"Second Prediction: {second_class.title()} ({second_confidence * 100:.2f}%) | | "
           #         f"Best Round: {best_round_idx + 1} of {rounds} |  |  "
           #         f"Avg Confidence Per Class: {best_avg_conf.tolist()}  "
           #     )
           # })

       # except Exception as e:
       #     print(f"Done processing image: {e}")

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
    
    #with open(file_path, "rb") as f:
        #resp = requests.put(url, data=f, headers=headers, timeout=60)
    
   # if resp.status_code in (200, 201):
   #     data = resp.json()
   #     file_id = data.get("id") or data.get("data", {}).get("id")
   #     print(f"File uploaded successfully: {file_id}")
   #     if not file_id:
   #         raise Exception(f"Unexpected response: {data}")
   #     return f"https://pixeldrain.com/api/file/{file_id}"
   # else:
   #     raise Exception(f"Upload failed {resp.status_code}: {resp.text}")



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


def get_clinical_considerations(pred_class, confidence) -> str:
    considerations = {
        'glioma': [
            "• Gliomas can be aggressive and require prompt attention.",
            "• Recommend follow-up with neurologist and MRI spectroscopy.",
            "• Consider grading evaluation (low-grade vs high-grade)."
        ],
        'meningioma': [
            "• Most meningiomas are benign (WHO Grade I).",
            "• Recommend monitoring growth rate if asymptomatic.",
            "• Surgical resection may be indicated for symptomatic cases."
        ],
        'notumor': [
            "• No immediate intervention needed.",
            "• Recommend routine follow-up if clinically indicated.",
            "• Consider alternative diagnoses if symptoms persist."
        ],
        'pituitary': [
            "• Endocrine evaluation recommended.",
            "• Assess for hormonal hypersecretion syndromes.",
            "• Monitor for visual field defects if macroadenoma."
        ]
    }
    base = "CLINICAL CONSIDERATIONS:\n" + "\n".join(considerations.get(pred_class, []))
    if confidence < 0.7:
        base += (
            "\n\n⚠️ NOTE: Due to lower confidence in prediction, consider:\n"
            "• Additional imaging (contrast-enhanced MRI).\n"
            "• Second opinion from neuroradiologist.\n"
            "• Clinical correlation with patient symptoms."
        )
    return base


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
