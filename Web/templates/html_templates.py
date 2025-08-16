"""
HTML template functions for brain tumor analysis reports
"""

def generate_summary_result(filename, predicted_class, confidence_score, 
                          second_class=None, second_confidence=None, 
                          show_dual_prediction=False):
    """Generate summary result HTML"""
    if show_dual_prediction:
        return f"""
        <div class="summary-dual-prediction">
            <span class="summary-icon">🖼️</span>
            <span class="summary-filename">{filename}</span>
            <span class="summary-prediction">
                <b>🧠 {predicted_class.title()} <span style="font-weight:400;">or</span> {second_class.title()}</b>
                <span class="summary-confidence">({confidence_score*100:.1f}% vs {second_confidence*100:.1f}%)</span>
                <span class="summary-badge">≤15% diff</span>
            </span>
        </div>
        """
    else:
        return f"""
        <div class="summary-single-prediction">
            <span class="summary-icon">🖼️</span>
            <span class="summary-filename">{filename}</span>
            <span class="summary-prediction">
                <b>🧠 {predicted_class.title()}</b>
                <span class="summary-confidence">({confidence_score*100:.1f}%)</span>
                <span class="summary-badge">Best of 20 rounds</span>
            </span>
        </div>
        """

def generate_detailed_report_header(filename, predicted_class, confidence_score, 
                                  best_round_idx, rounds):
    """Generate detailed report header HTML"""
    return f"""
    <div class="report-container">
        <div class="report-title">📄 Comprehensive Analysis Report: <span class="filename">{filename}</span></div>
        <div class="divider"></div>
        <div class="prediction-main">🏆 Final Prediction: <b>{predicted_class.upper()}</b> 
            ({confidence_score*100:.2f}% confidence) 
            <span style="font-size:0.95em;color:#64748b;">(Best of 20 rounds)</span>
        </div>
        <div class="filename">🏅 Best Round: {best_round_idx+1} / {rounds}</div>
    """

def generate_description_section(predicted_class, class_descriptions):
    """Generate description section HTML"""
    return f"""
    <div class="desc"><b>📖 Description:</b> {class_descriptions[predicted_class]}</div>
    """

def generate_probability_table(class_names, best_avg_conf, top1_idx):
    """Generate probability table HTML"""
    table_html = ["<table class='prob-table'><tr><th>Class</th><th>Probability</th><th>Bar</th></tr>"]
    max_len = 180
    
    for i, class_name in enumerate(class_names):
        percentage = best_avg_conf[i].item() * 100
        bar_len = int(percentage / 100 * max_len)
        bar_class = "prob-bar-main" if i == top1_idx else "prob-bar"
        
        table_html.append(
            f"<tr><td>{'→' if i == top1_idx else ''} {class_name.title()}</td>"
            f"<td>{percentage:5.2f}%</td>"
            f"<td><div class='{bar_class}' style='width:{bar_len}px'></div></td></tr>"
        )
    
    table_html.append("</table>")
    return "".join(table_html)

def generate_dual_prediction_note(second_class, second_confidence):
    """Generate dual prediction warning note HTML"""
    return f"""
    <div class='note'>⚠️ The second most likely class is within 15% confidence margin.<br>
        - Alternative Diagnosis: <b>{second_class.title()}</b> ({second_confidence*100:.2f}%)
    </div>
    """

def generate_metrics_section(confidence_score, best_avg_conf, class_names,
                           get_confidence_level, get_second_most_likely,
                           calculate_probability_spread, calculate_uncertainty):
    """Generate analysis metrics section HTML"""
    return f"""
    <div class="metrics">
        <b>🔍 Confidence Analysis:</b><br>
        - Prediction Confidence Score: <b>{confidence_score:.4f}</b><br>
        - Confidence Level: {get_confidence_level(confidence_score)}<br>
        - Second Most Likely Class: {get_second_most_likely(best_avg_conf, class_names)}<br>
        <br><b>📊 Prediction Reliability Indicators:</b><br>
        - Probability Spread: <b>{calculate_probability_spread(best_avg_conf):.3f}</b> (higher is better)<br>
        - Uncertainty Index: <b>{calculate_uncertainty(best_avg_conf):.3f}</b> (lower is better)
    </div>
    """

def generate_clinical_section(predicted_class, confidence_score, get_clinical_considerations):
    """Generate clinical considerations section HTML"""
    clinical_text = get_clinical_considerations(predicted_class, confidence_score).replace('\n', '<br>')
    return f"""
    <div class="clinical">
        <b>💡 Clinical Considerations:</b><br>
        {clinical_text}
    </div>
    """

def close_report_container():
    """Close the report container div"""
    return "</div>"  # Close report-container
