#!/usr/bin/env python3

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import threading
import time
from datetime import datetime
import os
import sys
import glob
import re

# Import your existing bot class
# Your script is saved as 'final_searcher.py'
from final_searcher import EnhancedUniMiBot

app = Flask(__name__)
CORS(app)

# Global bot instance
bot = None
current_status = {"status": "idle", "message": "", "progress": 0}

def init_bot():
    global bot
    try:
        print("Initializing UniMi bot...")
        bot = EnhancedUniMiBot()
        print("Bot initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing bot: {e}")
        return False

def find_latest_response_file(query):
    """Find the latest LaTeX response file for a given query"""
    try:
        # Clean query for filename matching (similar to what the bot does)
        clean_query = re.sub(r'[^\w\s-]', '', query)[:50]
        clean_query = re.sub(r'\s+', '_', clean_query).lower()
        
        # Look for files matching the pattern
        pattern = f"unimi_enhanced_response_{clean_query}_*.tex"
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            # Try a broader search in case the query cleaning differs
            pattern = "unimi_enhanced_response_*_*.tex"
            all_files = glob.glob(pattern)
            
            # Get files from the last 5 minutes (recent files)
            recent_files = []
            current_time = time.time()
            for file_path in all_files:
                try:
                    if os.path.getctime(file_path) > current_time - 300:  # 5 minutes
                        recent_files.append(file_path)
                except OSError:
                    continue
            
            if recent_files:
                # Return the most recent file
                matching_files = [max(recent_files, key=lambda x: os.path.getctime(x))]
        
        if matching_files:
            # Return the most recent file
            latest_file = max(matching_files, key=lambda x: os.path.getctime(x))
            return latest_file
        
        return None
        
    except Exception as e:
        print(f"Error finding response file: {e}")
        return None

def read_latex_file(filepath):
    """Read the LaTeX file content"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading LaTeX file {filepath}: {e}")
        return None

def process_query_async(query, request_id):
    global current_status
    
    try:
        current_status = {"status": "processing", "message": f"Processing query: {query}", "progress": 10, "request_id": request_id}
        
        # Load database if not loaded
        if not bot.database:
            current_status["message"] = "Loading database..."
            current_status["progress"] = 30
            if not bot.load_database():
                current_status = {"status": "error", "message": "Failed to load database", "progress": 0, "request_id": request_id}
                return
        
        current_status["message"] = "Performing semantic search..."
        current_status["progress"] = 50
        
        # Process the query - this will create the LaTeX file
        result = bot.process_query(query, top_k=10)
        
        current_status["message"] = "Reading generated LaTeX file..."
        current_status["progress"] = 80
        
        # Wait a moment for file to be written
        time.sleep(1)
        
        # Find and read the generated LaTeX file
        latex_file_path = find_latest_response_file(query)
        
        if latex_file_path and os.path.exists(latex_file_path):
            latex_content = read_latex_file(latex_file_path)
            if latex_content:
                current_status = {
                    "status": "completed", 
                    "message": "Query processed successfully", 
                    "progress": 100,
                    "request_id": request_id,
                    "result": {
                        "latex_content": latex_content,
                        "latex_file_path": latex_file_path,
                        "original_query": getattr(bot, 'current_query', query),
                        "cleaned_query": getattr(bot, 'cleaned_query', query),
                        "pages_found": len(getattr(bot, 'search_results', [])),
                        "pages_processed": len(getattr(bot, 'scraped_pages', [])),
                        "timestamp": datetime.now().isoformat()
                    }
                }
                return
        
        # Fallback to in-memory result if file not found
        if result:
            current_status = {
                "status": "completed", 
                "message": "Query processed (using in-memory result)", 
                "progress": 100,
                "request_id": request_id,
                "result": {
                    "latex_content": result,
                    "latex_file_path": "in-memory",
                    "original_query": getattr(bot, 'current_query', query),
                    "cleaned_query": getattr(bot, 'cleaned_query', query),
                    "pages_found": len(getattr(bot, 'search_results', [])),
                    "pages_processed": len(getattr(bot, 'scraped_pages', [])),
                    "timestamp": datetime.now().isoformat()
                }
            }
        else:
            current_status = {"status": "error", "message": "No results found and no LaTeX file generated", "progress": 0, "request_id": request_id}
    
    except Exception as e:
        print(f"Error in process_query_async: {e}")
        current_status = {"status": "error", "message": f"Processing error: {str(e)}", "progress": 0, "request_id": request_id}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    return jsonify(current_status)

@app.route('/api/query', methods=['POST'])
def process_query():
    global current_status
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        if current_status.get("status") == "processing":
            return jsonify({"error": "Another query is already being processed"}), 409
        
        # Check if bot is initialized
        if not bot:
            return jsonify({"error": "Bot not initialized"}), 500
        
        # Generate unique request ID
        request_id = f"req_{int(time.time())}"
        
        # Start processing in background
        thread = threading.Thread(target=process_query_async, args=(query, request_id))
        thread.daemon = True
        thread.start()
        
        return jsonify({"message": "Query processing started", "request_id": request_id})
    
    except Exception as e:
        print(f"Error in process_query: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/health')
def health_check():
    try:
        return jsonify({
            "status": "healthy",
            "bot_initialized": bot is not None,
            "database_loaded": bot.database is not None if bot else False,
            "database_pages": len(bot.database) if (bot and bot.database) else 0
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    
    # Initialize bot
    if init_bot():
        print("Starting Flask app on http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    else:
        print("Failed to initialize bot. Exiting.")
        sys.exit(1)
