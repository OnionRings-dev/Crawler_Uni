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
from flask import send_file

from final_searcher import MultiDomainSearchBot, get_all_domains

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

bot = None
current_status = {"status": "idle", "message": "", "progress": 0}

def init_bot():
    global bot
    try:
        print("Initializing Multi-Domain bot...")
        bot = MultiDomainSearchBot()
        print("Bot initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing bot: {e}")
        return False

def find_latest_response_file(query, domain):
    try:
        clean_query = re.sub(r'[^\w\s-]', '', query)[:50]
        clean_query = re.sub(r'\s+', '_', clean_query).lower()
        domain_clean = domain.replace('.', '_')
        
        pattern = f"{domain_clean}_response_{clean_query}_*.tex"
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            pattern = f"{domain_clean}_response_*_*.tex"
            all_files = glob.glob(pattern)
            
            recent_files = []
            current_time = time.time()
            for file_path in all_files:
                try:
                    if os.path.getctime(file_path) > current_time - 300:
                        recent_files.append(file_path)
                except OSError:
                    continue
            
            if recent_files:
                matching_files = [max(recent_files, key=lambda x: os.path.getctime(x))]
        
        if matching_files:
            latest_file = max(matching_files, key=lambda x: os.path.getctime(x))
            return latest_file
        
        return None
        
    except Exception as e:
        print(f"Error finding response file: {e}")
        return None

def read_latex_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading LaTeX file {filepath}: {e}")
        return None

def process_query_async(query, domain, request_id):
    global current_status
    
    try:
        current_status = {
            "status": "processing", 
            "message": f"Processing query for domain {domain}: {query}", 
            "progress": 10, 
            "request_id": request_id
        }
        
        if not bot.database or bot.current_domain != domain:
            current_status["message"] = f"Loading database for domain {domain}..."
            current_status["progress"] = 30
            current_status["request_id"] = request_id
            if not bot.load_database_by_domain(domain):
                current_status = {
                    "status": "error", 
                    "message": f"Failed to load database for domain {domain}", 
                    "progress": 0, 
                    "request_id": request_id
                }
                return
        
        current_status["message"] = "Performing semantic search..."
        current_status["progress"] = 50
        current_status["request_id"] = request_id
        
        result = bot.process_query(query, domain, top_k=15)
        
        current_status["message"] = "Reading generated LaTeX file..."
        current_status["progress"] = 80
        current_status["request_id"] = request_id
        
        time.sleep(2)  # Aumenta da 1 a 2 secondi
        
        print(f"Current directory: {os.getcwd()}")
        print(f"Looking for LaTeX files matching domain: {domain}")
        
        latex_file_path = find_latest_response_file(query, domain)
        
        print(f"Found LaTeX file: {latex_file_path}")
        
        if latex_file_path and os.path.exists(latex_file_path):
            print(f"File exists: True")
            print(f"Full path: {os.path.abspath(latex_file_path)}")
            
            latex_content = read_latex_file(latex_file_path)
            
            if latex_content:
                print(f"LaTeX content loaded: {len(latex_content)} characters")
                current_status = {
                    "status": "completed", 
                    "message": "Query processed successfully", 
                    "progress": 100,
                    "request_id": request_id,
                    "result": {
                        "latex_content": latex_content,
                        "latex_file_path": latex_file_path,
                        "domain": domain,
                        "original_query": getattr(bot, 'current_query', query),
                        "cleaned_query": getattr(bot, 'cleaned_query', query),
                        "pages_found": len(getattr(bot, 'search_results', [])),
                        "pages_processed": len(getattr(bot, 'scraped_pages', [])),
                        "timestamp": datetime.now().isoformat()
                    }
                }
                print("Status updated to completed with LaTeX content")
                return
            else:
                print("Failed to read LaTeX content from file")
        else:
            print(f"LaTeX file not found or doesn't exist")
        
        # Fallback se il file non viene trovato
        if result:
            print("Using in-memory result as fallback")
            current_status = {
                "status": "completed", 
                "message": "Query processed (using in-memory result)", 
                "progress": 100,
                "request_id": request_id,
                "result": {
                    "latex_content": result,
                    "latex_file_path": "in-memory",
                    "domain": domain,
                    "original_query": getattr(bot, 'current_query', query),
                    "cleaned_query": getattr(bot, 'cleaned_query', query),
                    "pages_found": len(getattr(bot, 'search_results', [])),
                    "pages_processed": len(getattr(bot, 'scraped_pages', [])),
                    "timestamp": datetime.now().isoformat()
                }
            }
        else:
            current_status = {
                "status": "error", 
                "message": "No results found and no LaTeX file generated", 
                "progress": 0, 
                "request_id": request_id
            }
    
    except Exception as e:
        print(f"Error in process_query_async: {e}")
        import traceback
        traceback.print_exc()
        current_status = {
            "status": "error", 
            "message": f"Processing error: {str(e)}", 
            "progress": 0, 
            "request_id": request_id
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/domains', methods=['GET'])
def get_domains():
    try:
        domains = get_all_domains()
        return jsonify({"domains": domains})
    except Exception as e:
        print(f"Error getting domains: {e}")
        return jsonify({"error": str(e), "domains": []}), 500

@app.route('/api/status')
def get_status():
    global current_status
    has_result = 'result' in current_status
    latex_length = len(current_status.get('result', {}).get('latex_content', '')) if has_result else 0
    
    print(f"[STATUS] status={current_status.get('status')}, request_id={current_status.get('request_id')}, has_result={has_result}, latex_length={latex_length}")
    
    return jsonify(current_status)

@app.route('/api/query', methods=['POST'])
def process_query():
    global current_status
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        query = data.get('query', '').strip()
        domain = data.get('domain', '').strip()
        
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        if not domain:
            return jsonify({"error": "Domain cannot be empty"}), 400
        
        if current_status.get("status") == "processing":
            return jsonify({"error": "Another query is already being processed"}), 409
        
        if not bot:
            return jsonify({"error": "Bot not initialized"}), 500
        
        available_domains = bot.get_available_domains()
        if domain not in available_domains:
            return jsonify({"error": f"Domain {domain} not found. Available: {', '.join(available_domains)}"}), 400
        
        request_id = f"req_{int(time.time())}"
        
        thread = threading.Thread(target=process_query_async, args=(query, domain, request_id))
        thread.daemon = True
        thread.start()
        
        return jsonify({"message": "Query processing started", "request_id": request_id, "domain": domain})
    
    except Exception as e:
        print(f"Error in process_query: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/download/<filename>')
def download_file(filename):
    try:
        if not (filename.endswith('.tex') or filename.endswith('.txt')):
            return jsonify({"error": "Invalid file type"}), 400
        
        filepath = os.path.join(os.getcwd(), filename)
        
        if not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404
        
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health')
def health_check():
    try:
        available_domains = bot.get_available_domains() if bot else []
        return jsonify({
            "status": "healthy",
            "bot_initialized": bot is not None,
            "database_loaded": bot.database is not None if bot else False,
            "current_domain": bot.current_domain if bot else None,
            "database_pages": len(bot.database) if (bot and bot.database) else 0,
            "available_domains": available_domains
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    
    if init_bot():
        print("Starting Flask app on http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    else:
        print("Failed to initialize bot. Exiting.")
        sys.exit(1)#!/usr/bin/env python3
