import os
import torch
from torchvision import transforms 
from PIL import Image
from typing import List, Tuple
import io
import uuid
import firebase_admin
from firebase_admin import credentials, db
import gdown
import timm 


# ========== CONFIGURATION ========== #
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    os.path.join("Models", "model.pth"))

MODEL_ID = "1UaFw3vAuimY6r47mYbkyFXmtsjwDjsoP"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={MODEL_ID}"

cred_path = os.path.join(os.path.dirname(__file__), "firebase.json")

cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://brain-tumour-detection-8f253-default-rtdb.asia-southeast1.firebasedatabase.app/' 
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
from assets.templates.html_templates import generate_batch_overview, generate_detailed_report, generate_tumour_types
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

            push_to_firebase(fname, pred_class, conf)
            
            # Generate Attention Map 
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
    
    #Get stats from firebase 
    stats = get_stats_from_firebase()
    tumour_types = generate_tumour_types(stats)
    return batch_html, detailed_html, tumour_types, state


def push_to_firebase(fname, class_name, confidence):

    unique_id = str(uuid.uuid4())
    data = {
        "id": unique_id
        , "filename": fname
        , "class": class_name
        , "confidence": confidence
    }

    ref = db.reference("predictions")
    ref.child(unique_id).set(data)

def get_stats_from_firebase():
    ref = db.reference("predictions")
    
    # Get all predictions data
    predictions = ref.get()
    
    # Initialize statistics dictionary
    stats = {
        'total_count': 0,
        'class_stats': {
            'glioma': {'count': 0, 'total_confidence': 0, 'avg_confidence': 0},
            'meningioma': {'count': 0, 'total_confidence': 0, 'avg_confidence': 0},
            'notumor': {'count': 0, 'total_confidence': 0, 'avg_confidence': 0},
            'pituitary': {'count': 0, 'total_confidence': 0, 'avg_confidence': 0}
        }
    }
    
    if not predictions:
        return stats
    
    for prediction_id, prediction_data in predictions.items():
        if not prediction_data:
            continue
            
        # Get the predicted class and confidence from details
        predicted_class = prediction_data.get('class', '')
        confidence = float(prediction_data.get('confidence', 0))
        
        # Update overall count
        stats['total_count'] += 1
        
        # Update class-specific stats if class is valid
        if predicted_class in stats['class_stats']:
            stats['class_stats'][predicted_class]['count'] += 1
            stats['class_stats'][predicted_class]['total_confidence'] += confidence
    
    # Calculate average confidence for each class
    for class_name, class_stat in stats['class_stats'].items():
        if class_stat['count'] > 0:
            class_stat['avg_confidence'] = class_stat['total_confidence'] / class_stat['count'] * 100
        else:
            class_stat['avg_confidence'] = 0
    
    return stats


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
