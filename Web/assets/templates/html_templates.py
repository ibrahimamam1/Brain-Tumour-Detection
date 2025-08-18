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
    """Generate result card with data attributes instead of onclick"""
    return f"""
    <div class="result-card {severity_class(pred['predicted_class'])}" 
         data-index="{index}" data-action="switch-to-detailed">
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
    """Generate batch overview with event delegation"""
    total = len(predictions)
    tumors = sum(1 for p in predictions if p['predicted_class'] != 'notumor')
    normals = total - tumors
    cards = "".join(generate_result_card(p, i) for i,p in enumerate(predictions))
    
    # Embed predictions data in a hidden div
    predictions_json = str(predictions).replace("'", '"')
    
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
        
        <!-- Hidden data storage -->
        <div id="predictions-data" style="display: none;">{predictions_json}</div>
        
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
    
    <script>
    // Use setTimeout to ensure this runs after the DOM is ready
    setTimeout(function() {{
        // Set up event delegation for result cards
        document.addEventListener('click', function(e) {{
            const card = e.target.closest('[data-action="switch-to-detailed"]');
            if (card) {{
                const index = parseInt(card.getAttribute('data-index'));
                
                // Store selection globally
                window.selectedPredictionIndex = index;
                localStorage.setItem('selectedPredictionIndex', index);
                
                // Visual feedback
                document.querySelectorAll('.result-card').forEach(c => c.classList.remove('selected'));
                card.classList.add('selected');
                
                // Try to switch tabs
                switchToDetailedTabMultiple();
            }}
        }});
        
        // Store predictions data globally with multiple fallbacks
        try {{
            const dataEl = document.getElementById('predictions-data');
            if (dataEl) {{
                window.currentPredictions = JSON.parse(dataEl.textContent);
            }}
        }} catch (e) {{
            console.log('Could not parse predictions data');
        }}
        
        // Multiple strategies to find and click detailed tab
        function switchToDetailedTabMultiple() {{
            const strategies = [
                () => document.querySelector('button[data-testid*="detailed"]'),
                () => document.querySelector('button[data-testid*="analysis"]'),
                () => document.querySelector('button:nth-child(2)[role="tab"]'),
                () => document.querySelectorAll('button[role="tab"]')[1],
                () => document.querySelectorAll('.tab-nav button')[1],
                () => Array.from(document.querySelectorAll('button')).find(b => b.textContent.toLowerCase().includes('detailed')),
                () => Array.from(document.querySelectorAll('button')).find(b => b.textContent.toLowerCase().includes('analysis'))
            ];
            
            for (let strategy of strategies) {{
                try {{
                    const tab = strategy();
                    if (tab) {{
                        tab.click();
                        console.log('Successfully clicked detailed tab');
                        return;
                    }}
                }} catch (e) {{
                    continue;
                }}
            }}
            
            console.warn('Could not find detailed tab');
        }}
    }}, 500);
    </script>
    """


def generate_navigation_panel(predictions: list, current_index: int = 0) -> str:
    """Generate navigation panel with data attributes"""
    nav_items = ""
    for i, pred in enumerate(predictions):
        is_active = i == current_index
        active_class = "nav-item-active" if is_active else ""
        
        nav_items += f"""
        <div class="nav-item {active_class}" data-index="{i}" data-action="select-prediction">
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
    """Generate detailed report with event delegation"""
    
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
                             data-action="zoom-image">
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
    
    # Embed all predictions data
    predictions_json = str(predictions_list).replace("'", '"')
    
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
        <div id="imageModal" class="image-modal" data-action="close-modal">
            <span class="modal-close" data-action="close-modal">&times;</span>
            <img class="modal-content" id="modalImage">
            <div class="modal-caption" id="modalCaption"></div>
        </div>
        
        <!-- Hidden data storage -->
        <div id="detailed-predictions-data" style="display: none;">{predictions_json}</div>
    </div>
    
    <script>
    setTimeout(function() {{
        // Load predictions data
        try {{
            const dataEl = document.getElementById('detailed-predictions-data');
            if (dataEl) {{
                window.allPredictions = JSON.parse(dataEl.textContent);
                window.currentDetailedIndex = {current_index};
            }}
        }} catch (e) {{
            console.log('Could not parse detailed predictions data');
        }}
        
        // Set up event delegation for detailed view
        document.addEventListener('click', function(e) {{
            const target = e.target;
            const action = target.getAttribute('data-action') || target.closest('[data-action]')?.getAttribute('data-action');
            
            switch(action) {{
                case 'select-prediction':
                    const index = parseInt(target.closest('[data-index]').getAttribute('data-index'));
                    updateDetailedView(index);
                    break;
                    
                case 'zoom-image':
                    zoomImage(target);
                    break;
                    
                case 'close-modal':
                    closeModal();
                    break;
            }}
        }});
        
        // Check for pending selection from overview
        const pendingIndex = localStorage.getItem('selectedPredictionIndex');
        if (pendingIndex !== null) {{
            updateDetailedView(parseInt(pendingIndex));
            localStorage.removeItem('selectedPredictionIndex');
        }}
        
        function updateDetailedView(index) {{
            if (!window.allPredictions || index < 0 || index >= window.allPredictions.length) return;
            
            const prediction = window.allPredictions[index];
            window.currentDetailedIndex = index;
            
            // Add loading animation
            const mainContent = document.querySelector('.detailed-report-main');
            if (mainContent) {{
                mainContent.style.opacity = '0.7';
                mainContent.style.transform = 'translateY(10px)';
            }}
            
            setTimeout(() => {{
                // Update header info safely
                const titleEl = document.getElementById('report-title');
                const filenameEl = document.getElementById('report-filename');
                const predictionEl = document.getElementById('report-prediction');
                const confidenceEl = document.getElementById('report-confidence');
                
                if (titleEl) titleEl.innerHTML = getSeverityIcon(prediction.predicted_class) + ' Detailed Analysis';
                if (filenameEl) filenameEl.textContent = prediction.filename;
                if (predictionEl) {{
                    predictionEl.textContent = prediction.predicted_class.charAt(0).toUpperCase() + prediction.predicted_class.slice(1);
                    predictionEl.className = getSeverityClass(prediction.predicted_class);
                }}
                if (confidenceEl) confidenceEl.textContent = (prediction.confidence * 100).toFixed(1) + '%';
                
                // Update probabilities
                updateProbabilityBars(prediction.all_predictions, prediction.predicted_class);
                
                // Update description
                const descEl = document.getElementById('description-content');
                if (descEl) {{
                    descEl.innerHTML = '<p>' + prediction.description + '</p>';
                    descEl.className = 'description-content ' + getSeverityClass(prediction.predicted_class);
                }}
                
                // Update navigation
                updateNavigationPanel(index);
                
                // Restore animation
                if (mainContent) {{
                    mainContent.style.opacity = '1';
                    mainContent.style.transform = 'translateY(0)';
                }}
            }}, 200);
        }}
        
        function updateProbabilityBars(predictions, currentPrediction) {{
            const container = document.getElementById('probabilities-container');
            if (!container) return;
            
            let barsHtml = '';
            
            predictions.forEach(function(pred) {{
                const prob = pred[0];
                const className = pred[1];
                const percentage = prob * 100;
                const barClass = getSeverityClass(className);
                const isPredicted = className === currentPrediction;
                
                barsHtml += '<div class="probability-bar ' + (isPredicted ? 'predicted-class' : '') + '">' +
                    '<div class="prob-label">' +
                        '<span class="class-name">' + className.charAt(0).toUpperCase() + className.slice(1) + '</span>' +
                        '<span class="prob-value">' + percentage.toFixed(1) + '%</span>' +
                    '</div>' +
                    '<div class="prob-bar-container">' +
                        '<div class="prob-bar-fill ' + barClass + '" style="width: ' + percentage + '%"></div>' +
                    '</div>' +
                '</div>';
            }});
            
            container.innerHTML = barsHtml;
        }}
        
        function updateNavigationPanel(activeIndex) {{
            const navItems = document.querySelectorAll('.nav-item');
            navItems.forEach(function(item, index) {{
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
            
            if (modal && modalImg && caption) {{
                modal.style.display = 'block';
                modalImg.src = img.src;
                caption.innerHTML = img.alt;
            }}
        }}
        
        function closeModal() {{
            const modal = document.getElementById('imageModal');
            if (modal) {{
                modal.style.display = 'none';
            }}
        }}
        
        // Close modal with escape key
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape') {{
                closeModal();
            }}
        }});
    }}, 500);
    </script>
    """
    
    return report_html

def generate_tumour_types(stats):

    # Educational information for each tumor type
    tumor_info = {
        'glioma': {
            'title': 'Glioma',
            'description': 'Gliomas are tumors that arise from glial cells in the brain and spinal cord. They are the most common type of primary brain tumor in adults.',
            'characteristics': 'Can range from low-grade (slow-growing) to high-grade (aggressive). Symptoms may include headaches, seizures, and neurological deficits.',
            'treatment': 'Treatment typically involves surgery, radiation therapy, and/or chemotherapy depending on the grade and location.',
            'severity': 'high',
            'icon': '🧠'
        },
        'meningioma': {
            'title': 'Meningioma',
            'description': 'Meningiomas are tumors that develop from the meninges, the protective membranes surrounding the brain and spinal cord.',
            'characteristics': 'Usually slow-growing and benign (90-95% of cases). More common in women and typically diagnosed in middle age.',
            'treatment': 'Treatment may include observation, surgery, or radiation therapy. Many small meningiomas can be monitored without immediate treatment.',
            'severity': 'medium',
            'icon': '🛡️'
        },
        'notumor': {
            'title': 'No Tumor Detected',
            'description': 'No tumor or abnormal growth detected in the brain scan. This indicates normal brain tissue without signs of malignancy.',
            'characteristics': 'Normal brain anatomy with no evidence of tumor formation or abnormal cell growth.',
            'treatment': 'No treatment required. Regular follow-up may be recommended based on clinical symptoms or risk factors.',
            'severity': 'low',
            'icon': '✅'
        },
        'pituitary': {
            'title': 'Pituitary Tumor',
            'description': 'Pituitary tumors develop in the pituitary gland, a small organ at the base of the brain that controls hormone production.',
            'characteristics': 'Most are benign adenomas. Can be functioning (hormone-producing) or non-functioning. May cause hormonal imbalances or vision problems.',
            'treatment': 'Treatment options include medication, surgery, or radiation therapy depending on size, type, and symptoms.',
            'severity': 'medium',
            'icon': '⚡'
        }
    }
    
    # Generate HTML for tumor type cards
    html_content = '<div class="tumor-types-container">'
    html_content += '<h2 class="section-title">Brain Tumor Types & Statistics</h2>'
    
    for class_name in ['glioma', 'meningioma', 'notumor', 'pituitary']:
        info = tumor_info[class_name]
        class_stats = stats['class_stats'][class_name]
        
        # Calculate percentage of total predictions
        percentage = (class_stats['count'] / stats['total_count'] * 100) if stats['total_count'] > 0 else 0
        
        # Determine severity class for styling
        severity_class_name = f"severity-{info['severity']}"
        
        html_content += f"""
        <div class="tumor-type-card {severity_class_name}">
            <div class="tumor-header">
                <span class="tumor-icon">{info['icon']}</span>
                <h3 class="tumor-title">{info['title']}</h3>
            </div>
            
            <div class="tumor-content">
                <div class="tumor-description">
                    <p><strong>Description:</strong> {info['description']}</p>
                    <p><strong>Characteristics:</strong> {info['characteristics']}</p>
                    <p><strong>Treatment:</strong> {info['treatment']}</p>
                </div>
                
                <div class="tumor-stats">
                    <h4>Prediction Statistics</h4>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <span class="stat-label">Count:</span>
                            <span class="stat-value">{class_stats['count']}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Percentage:</span>
                            <span class="stat-value">{percentage:.1f}%</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Avg Confidence:</span>
                            <span class="stat-value">{class_stats['avg_confidence']:.1f}%</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
    
    html_content += '</div>'
    return html_content
