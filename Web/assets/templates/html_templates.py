"""
HTML template functions for brain tumor analysis reports
"""

def severity_class(predicted_class: str) -> str:
    """Map tumor class to CSS severity class"""
    mapping = {
        'glioma': 'severity-high',
        'meningioma': 'severity-medium',
        'pituitary': 'severity-medium',
        'notumor': 'severity-normal',
    }
    return mapping.get(predicted_class.lower(), 'severity-unknown')

def severity_icon(predicted_class: str) -> str:
    """Map tumor class to icon"""
    mapping = {
        'glioma': '🔴',
        'meningioma': '🟠',
        'pituitary': '🟠',
        'notumor': '🟢',
    }
    return mapping.get(predicted_class.lower(), '⚪')
def generate_result_card(pred: dict, index: int) -> str:
    """Generate result card with click handler to switch to detailed tab"""
    return f"""
    <div class="result-card {severity_class(pred['predicted_class'])}" 
         onclick="switchToDetailedView({index})" data-index="{index}">
        <div class="card-header">
            <span>{severity_icon(pred['predicted_class'])}</span>
            <span>{pred['filename']}</span>
        </div>
        <div class="card-prediction">
            <span>{pred['predicted_class'].title()}</span>
            <span class="card-confidence">{pred['confidence']*100:.1f}%</span>
        </div>
    </div>
    """


def generate_batch_overview(predictions: list) -> str:
    """Generate batch overview with enhanced interactivity"""
    total = len(predictions)
    tumors = sum(1 for p in predictions if p['predicted_class'] != 'notumor')
    normals = total - tumors
    cards = "".join(generate_result_card(p, i) for i,p in enumerate(predictions))
    
    return f"""
    <div class="batch-results-overview">
        <div class="batch-summary">
            <div class="stat-item">
                <span class="stat-number">{total}</span>
                <span class="stat-label">Images</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">{tumors}</span>
                <span class="stat-label">Tumors</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">{normals}</span>
                <span class="stat-label">Normal</span>
            </div>
        </div>
        <div class="results-grid">{cards}</div>
        
        <script>
        // Store predictions data globally for cross-tab communication
        window.currentPredictions = {predictions};
        window.currentSelectedIndex = 0;
        
        function switchToDetailedView(index) {{
            // Update global selection
            window.currentSelectedIndex = index;
            console.log("switch to detailed for " + index); 
            // Switch to detailed tab (Gradio specific)
            const detailedTab = document.querySelector('button[data-testid="tab-detailed-analysis"]') || 
                               document.querySelector('button:contains("Detailed Analysis")') ||
                               document.querySelectorAll('.tab-nav button')[1]; // fallback to second tab
            
            if (detailedTab) {{
                detailedTab.click();
            }}
            
            // Trigger update of detailed view
            setTimeout(() => {{
                if (window.updateDetailedView) {{
                    window.updateDetailedView(index);
                }}
            }}, 100);
        }}
        
        function showDetailedResult(index) {{
            const cards = document.querySelectorAll('.result-card');
            cards.forEach(c => c.classList.remove('selected'));
            if (cards[index]) {{
                cards[index].classList.add('selected');
            }}
        }}
        </script>
        
        <style>
        .result-card {{
            cursor: pointer;
            transition: all 0.3s ease;
            border: 2px solid transparent;
        }}
        
        .result-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        
        .result-card.selected {{
            border-color: #3498db;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
        }}
        </style>
    </div>
    """


def generate_navigation_panel(predictions: list, current_index: int = 0) -> str:
    """Generate navigation panel for detailed view"""
    nav_items = ""
    for i, pred in enumerate(predictions):
        is_active = i == current_index
        active_class = "nav-item-active" if is_active else ""
        
        nav_items += f"""
        <div class="nav-item {active_class}" onclick="selectPrediction({i})" data-index="{i}">
            <div class="nav-item-icon">{severity_icon(pred['predicted_class'])}</div>
            <div class="nav-item-content">
                <div class="nav-item-name">{pred['filename']}</div>
                <div class="nav-item-prediction">{pred['predicted_class'].title()}</div>
                <div class="nav-item-confidence">{pred['confidence']*100:.1f}%</div>
            </div>
        </div>
        """
    
    return f"""
    <div class="navigation-panel">
        <h3>All Predictions</h3>
        <div class="nav-items-container">
            {nav_items}
        </div>
    </div>
    """


def generate_detailed_report(prediction: dict, predictions_list: list, current_index: int = 0, img_data=None) -> str:
    """Generate detailed report with navigation panel"""
    
    # Generate probability bars for all classes
    prob_bars = ""
    for prob, class_name in prediction['all_predictions']:
        percentage = prob * 100
        bar_class = severity_class(class_name)
        is_predicted = class_name == prediction['predicted_class']
        
        prob_bars += f"""
        <div class="probability-bar {'predicted-class' if is_predicted else ''}">
            <div class="prob-label">
                <span class="class-name">{class_name.title()}</span>
                <span class="prob-value">{percentage:.1f}%</span>
            </div>
            <div class="prob-bar-container">
                <div class="prob-bar-fill {bar_class}" style="width: {percentage}%"></div>
            </div>
        </div>
        """
    
    # Image display section
    image_section = ""
    if img_data:
        try:
            import base64
            from io import BytesIO
            
            if hasattr(img_data, 'save'):  # PIL Image
                buffer = BytesIO()
                img_data.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
            elif isinstance(img_data, bytes):
                img_str = base64.b64encode(img_data).decode()
            else:
                img_str = None
                
            if img_str:
                image_section = f"""
                <div class="image-section">
                    <h3>Medical Image</h3>
                    <div class="image-container">
                        <img src="data:image/png;base64,{img_str}" 
                             alt="{prediction['filename']}" 
                             class="medical-image" 
                             onclick="zoomImage(this)">
                        <div class="zoom-hint">Click to zoom</div>
                    </div>
                </div>
                """
        except Exception as e:
            image_section = f"""
            <div class="image-section">
                <h3>Medical Image</h3>
                <div class="image-placeholder">
                    <span>Image not available: {str(e)}</span>
                </div>
            </div>
            """
    else:
        image_section = """
        <div class="image-section">
            <h3>Medical Image</h3>
            <div class="image-placeholder">
                <span>Image data not provided</span>
            </div>
        </div>
        """
    
    # Navigation panel
    nav_panel = generate_navigation_panel(predictions_list, current_index)
    
    # Generate the complete detailed report
    report_html = f"""
    <div class="detailed-report-container">
        <div class="detailed-report-main">
            <div class="report-header">
                <h2 id="report-title">{severity_icon(prediction['predicted_class'])} Detailed Analysis</h2>
                <div class="image-info">
                    <div class="info-item">
                        <label>Filename:</label>
                        <span id="report-filename">{prediction['filename']}</span>
                    </div>
                    <div class="info-item">
                        <label>Prediction:</label>
                        <span id="report-prediction" class="{severity_class(prediction['predicted_class'])}">{prediction['predicted_class'].title()}</span>
                    </div>
                    <div class="info-item">
                        <label>Confidence:</label>
                        <span id="report-confidence" class="confidence-value">{prediction['confidence']*100:.1f}%</span>
                    </div>
                </div>
            </div>
            
            <div class="report-content">
                <div class="left-content">
                    <div class="predictions-section">
                        <h3>Class Probabilities</h3>
                        <div id="probabilities-container" class="probabilities-container">
                            {prob_bars}
                        </div>
                    </div>
                    
                    <div id="image-section" class="image-section-container">
                        {image_section}
                    </div>
                    
                    <div class="description-section">
                        <h3>Medical Information</h3>
                        <div id="description-content" class="description-content {severity_class(prediction['predicted_class'])}">
                            <p>{prediction['description']}</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="navigation-sidebar">
            {nav_panel}
        </div>
        
        <!-- Image zoom modal -->
        <div id="imageModal" class="image-modal" onclick="closeModal()">
            <span class="modal-close">&times;</span>
            <img class="modal-content" id="modalImage">
            <div class="modal-caption" id="modalCaption"></div>
        </div>
        
        <script>
        // Store all predictions data and images
        window.allPredictions = {predictions_list};
        window.currentDetailedIndex = {current_index};
        window.imageDataStore = {{}};
        
        // Function to update detailed view smoothly
        window.updateDetailedView = function(index) {{
            if (index < 0 || index >= window.allPredictions.length) return;
            
            const prediction = window.allPredictions[index];
            window.currentDetailedIndex = index;
            
            // Add loading animation
            const mainContent = document.querySelector('.detailed-report-main');
            if (mainContent) {{
                mainContent.style.opacity = '0.7';
                mainContent.style.transform = 'translateY(10px)';
            }}
            
            setTimeout(() => {{
                // Update header info
                document.getElementById('report-title').innerHTML = `${{getSeverityIcon(prediction.predicted_class)}} Detailed Analysis`;
                document.getElementById('report-filename').textContent = prediction.filename;
                document.getElementById('report-prediction').textContent = prediction.predicted_class.charAt(0).toUpperCase() + prediction.predicted_class.slice(1);
                document.getElementById('report-prediction').className = getSeverityClass(prediction.predicted_class);
                document.getElementById('report-confidence').textContent = `${{(prediction.confidence * 100).toFixed(1)}}%`;
                
                // Update probabilities
                updateProbabilityBars(prediction.all_predictions);
                
                // Update description
                document.getElementById('description-content').innerHTML = `<p>${{prediction.description}}</p>`;
                document.getElementById('description-content').className = `description-content ${{getSeverityClass(prediction.predicted_class)}}`;
                
                // Update navigation
                updateNavigationPanel(index);
                
                // Restore animation
                if (mainContent) {{
                    mainContent.style.opacity = '1';
                    mainContent.style.transform = 'translateY(0)';
                }}
            }}, 200);
        }};
        
        function selectPrediction(index) {{
            window.updateDetailedView(index);
        }}
        
        function updateProbabilityBars(predictions) {{
            const container = document.getElementById('probabilities-container');
            let barsHtml = '';
            
            predictions.forEach(([prob, className]) => {{
                const percentage = prob * 100;
                const barClass = getSeverityClass(className);
                const isPredicted = className === window.allPredictions[window.currentDetailedIndex].predicted_class;
                
                barsHtml += `
                <div class="probability-bar ${{isPredicted ? 'predicted-class' : ''}}">
                    <div class="prob-label">
                        <span class="class-name">${{className.charAt(0).toUpperCase() + className.slice(1)}}</span>
                        <span class="prob-value">${{percentage.toFixed(1)}}%</span>
                    </div>
                    <div class="prob-bar-container">
                        <div class="prob-bar-fill ${{barClass}}" style="width: ${{percentage}}%"></div>
                    </div>
                </div>
                `;
            }});
            
            container.innerHTML = barsHtml;
        }}
        
        function updateNavigationPanel(activeIndex) {{
            const navItems = document.querySelectorAll('.nav-item');
            navItems.forEach((item, index) => {{
                if (index === activeIndex) {{
                    item.classList.add('nav-item-active');
                }} else {{
                    item.classList.remove('nav-item-active');
                }}
            }});
        }}
        
        function getSeverityClass(predictedClass) {{
            const mapping = {{
                'glioma': 'severity-high',
                'meningioma': 'severity-medium',
                'pituitary': 'severity-medium',
                'notumor': 'severity-normal',
            }};
            return mapping[predictedClass.toLowerCase()] || 'severity-unknown';
        }}
        
        function getSeverityIcon(predictedClass) {{
            const mapping = {{
                'glioma': '🔴',
                'meningioma': '🟠',
                'pituitary': '🟠',
                'notumor': '🟢',
            }};
            return mapping[predictedClass.toLowerCase()] || '⚪';
        }}
        
        function zoomImage(img) {{
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImage');
            const caption = document.getElementById('modalCaption');
            
            modal.style.display = 'block';
            modalImg.src = img.src;
            caption.innerHTML = img.alt;
        }}
        
        function closeModal() {{
            document.getElementById('imageModal').style.display = 'none';
        }}
        
        // Initialize on load
        document.addEventListener('DOMContentLoaded', function() {{
            if (window.currentSelectedIndex !== undefined) {{
                window.updateDetailedView(window.currentSelectedIndex);
            }}
        }});
        
        // Close modal with escape key
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape') {{
                closeModal();
            }}
        }});
        </script>
    </div>
    """
    
    return report_html
