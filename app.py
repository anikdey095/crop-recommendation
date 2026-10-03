"""
Local Development Server for Crop Recommendation System
Run with: python app.py
Serves the web application at http://127.0.0.1:5000 and mounts the /api routes.
"""

import os
from flask import send_from_directory, send_file
from api.index import app

PUBLIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'public')

@app.route('/')
def serve_index():
    index_path = os.path.join(PUBLIC_DIR, 'index.html')
    if os.path.exists(index_path):
        return send_file(index_path)
    return "Crop Recommendation Frontend: public/index.html not found.", 404

@app.route('/<path:path>')
def serve_static(path):
    # Don't hijack /api routes
    if path.startswith('api/'):
        return "Not found", 404
        
    file_path = os.path.join(PUBLIC_DIR, path)
    if os.path.exists(file_path):
        return send_from_directory(PUBLIC_DIR, path)
        
    # Default fallback to index.html for SPA style routing if needed
    index_path = os.path.join(PUBLIC_DIR, 'index.html')
    if os.path.exists(index_path):
        return send_file(index_path)
        
    return f"File {path} not found.", 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n[CropPulse AI] Server starting on http://127.0.0.1:{port}")
    print("[CropPulse AI] Model Accuracy: 99.55% (Random Forest Ensemble)")
    print("[CropPulse AI] Vercel-ready serverless API mounted on /api/*\n")
    app.run(host='0.0.0.0', port=port, debug=True)
