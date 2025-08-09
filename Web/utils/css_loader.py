"""
CSS loading utility for brain tumor analysis reports
"""
import os
from typing import Optional

def load_css_file(css_file_path: str) -> str:
    """
    Load CSS from a file and return it wrapped in <style> tags
    
    Args:
        css_file_path (str): Path to the CSS file
        
    Returns:
        str: CSS content wrapped in <style> tags
    """
    try:
        with open(css_file_path, 'r', encoding='utf-8') as file:
            css_content = file.read()
        return f"<style type='text/css'>\n{css_content}\n</style>"
    except FileNotFoundError:
        print(f"Warning: CSS file not found at {css_file_path}")
        return get_fallback_css()
    except Exception as e:
        print(f"Error loading CSS file: {e}")
        return get_fallback_css()

def get_fallback_css() -> str:
    """
    Return a minimal fallback CSS in case the external file cannot be loaded
    """
    return """
    <style type="text/css">
    .report-container {
        font-family: Arial, sans-serif;
        background: #f9fafb;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
        max-width: 650px;
    }
    .report-title {
        font-size: 1.3em;
        font-weight: bold;
        color: #1e293b;
        margin-bottom: 10px;
    }
    .prediction-main {
        font-size: 1.1em;
        color: #2563eb;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .prob-table {
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
    }
    .prob-table th, .prob-table td {
        padding: 8px 12px;
        text-align: left;
        border-bottom: 1px solid #e5e7eb;
    }
    .prob-table th {
        background: #f1f5f9;
        font-weight: 600;
    }
    .prob-bar, .prob-bar-main {
        display: inline-block;
        height: 16px;
        border-radius: 4px;
    }
    .prob-bar {
        background: #c7d2fe;
    }
    .prob-bar-main {
        background: #6366f1;
    }
    .metrics, .clinical, .note {
        background: #f8fafc;
        border-radius: 6px;
        padding: 12px;
        margin: 10px 0;
        border-left: 4px solid #6366f1;
    }
    .clinical {
        background: #fef3c7;
        border-left-color: #f59e0b;
        color: #92400e;
    }
    .note {
        background: #fee2e2;
        border-left-color: #f87171;
        color: #b91c1c;
    }
    </style>
    """

def get_css_styles(css_file_path: Optional[str] = None) -> str:
    """
    Get CSS styles either from file or fallback
    
    Args:
        css_file_path (str, optional): Path to CSS file. If None, uses fallback.
        
    Returns:
        str: CSS content wrapped in <style> tags
    """
    if css_file_path and os.path.exists(css_file_path):
        return load_css_file(css_file_path)
    else:
        return get_fallback_css()
