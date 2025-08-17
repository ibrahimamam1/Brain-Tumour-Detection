import gradio as gr
from PIL import Image
import io
import os
from pathlib import Path
from main_predict import predict_brain_tumor_batch

class GradioApp:
    def __init__(self, assets_dir="assets"):
        self.assets_dir = Path(assets_dir)
        self.css = self._load_css()
        self.templates = self._load_templates()
    
    def _load_file(self, filepath):
        """Load content from a file with error handling"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Warning: {filepath} not found.")
            return ""
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return ""
    
    def _load_css(self):
        """Load CSS from external file"""
        css_path = self.assets_dir / "styles/style.css"
        return self._load_file(css_path)
    
    def _load_templates(self):
        """Load HTML templates from external files"""
        templates_dir = self.assets_dir / "templates"
        return {
            'header': self._load_file(templates_dir / "header.html"),
            'description': self._load_file(templates_dir / "description.html"),
            'image_details': self._load_file(templates_dir / "image_details.html")
        }
    
    def clear_all(self):
        """Clear all outputs and reset the interface"""
        return (
            None,  # summary_output
            None,  # detailed_report_output
            gr.update(value=None),  # tumor_types_output
            [],    # current_predictions
            None,  # preview_image
            "",    # preview_text
            None   # image_input
        )
    
    def show_image_details(self, evt: gr.SelectData, predictions):
        """Show detailed information about a selected image"""
        idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        
        if idx >= len(predictions):
            return None, "No prediction data available"
        
        pred = predictions[idx]
        img = pred["image"]
        
        try:
            # Handle different image formats
            if hasattr(img, 'read'):
                img.seek(0)
                image = Image.open(img).convert("RGB")
            elif isinstance(img, bytes):
                image = Image.open(io.BytesIO(img)).convert("RGB")
            elif isinstance(img, str):
                image = Image.open(img).convert("RGB")
            else:
                return None, "Invalid image data"
                
        except Exception as e:
            return None, f"Error loading image: {str(e)}"
        
        # Use template if available, otherwise fallback to inline HTML
        if self.templates['image_details']:
            details_html = self.templates['image_details'].format(
                filename=pred['filename'],
                prediction=pred['class'].upper(),
                confidence=f"{pred['confidence']:.2f}"
            )
        else:
            details_html = f"""
            <div style="font-size:1.18em;">
                <b>🖼️ Image:</b> {pred['filename']}<br>
                <b>🧠 Prediction:</b> <span style="color:#6366f1;">{pred['class'].upper()}</span><br>
                <b>🔍 Confidence:</b> <span style="color:#16a34a;">{pred['confidence']:.2f}%</span>
            </div>
            """
        
        return image, details_html
    
    def create_interface(self):
        """Create and return the Gradio interface"""
        with gr.Blocks(css=self.css) as app:
            # Header and description
            gr.HTML(self.templates['header'])
            gr.HTML(self.templates['description'])
            
            # Main input section
            with gr.Row():
                with gr.Column(scale=2):
                    image_input = gr.File(
                        file_types=[
                            ".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG",
                            ".bmp", ".BMP", ".gif", ".GIF", ".tiff", ".TIFF",
                            ".webp", ".WEBP"
                        ],
                        file_count="multiple",
                        label="📤 Upload MRI Images",
                        elem_id="image-upload"
                    )
                with gr.Column(scale=1, min_width=180):
                    predict_button = gr.Button("🚀 Predict", elem_id="predict-btn")
                    clear_button = gr.Button("🧹 Clear All", elem_id="clear-btn")
            
            # Results tabs
            with gr.Tabs():
                with gr.TabItem("📋 Summary"):
                    summary_output = gr.HTML(elem_id="summary-md")
                with gr.TabItem("📄 Detailed Reports"):
                    detailed_report_output = gr.HTML(elem_id="detailed-html")
                with gr.TabItem("🧬 Tumor Types"):
                    tumor_types_output = gr.Dataframe(
                        headers=["Image", "Prediction", "Confidence %"],
                        datatype=["str", "str", "number"],
                        interactive=True,
                        elem_id="tumor-types-df",
                        wrap=True,
                        row_count=5,
                        col_count=(3, "fixed"),
                        visible=True,
                        label="Tumor Type Predictions"
                    )
            
            # State for storing predictions
            current_predictions = gr.State([])
            
            # Preview section (hidden by default)
            with gr.Row(visible=False) as preview_row:
                with gr.Column(scale=1):
                    preview_image = gr.Image(
                        label="Selected Image",
                        show_label=True,
                        elem_id="preview-img"
                    )
                with gr.Column(scale=2):
                    preview_text = gr.Markdown(elem_id="preview-md")
            
            # Event handlers
            clear_button.click(
                fn=self.clear_all,
                inputs=[],
                outputs=[
                    summary_output, detailed_report_output, tumor_types_output,
                    current_predictions, preview_image, preview_text, image_input
                ]
            )
            
            predict_button.click(
                fn=predict_brain_tumor_batch,
                inputs=image_input,
                outputs=[summary_output, detailed_report_output, tumor_types_output, current_predictions]
            )
            
            tumor_types_output.select(
                fn=self.show_image_details,
                inputs=current_predictions,
                outputs=[preview_image, preview_text]
            ).then(
                fn=lambda: gr.update(visible=True),
                outputs=preview_row
            )
        
        return app

# Usage
if __name__ == "__main__":
    # Create the app instance
    brain_tumor_app = GradioApp(assets_dir="Web/assets/")
    
    
    # Create and launch the interface
    app = brain_tumor_app.create_interface()
    app.launch()
