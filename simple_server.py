#!/usr/bin/env python3
"""
Simple HTTP server without FastAPI dependencies
For shared hosting compatibility
"""
import sys
import os
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from simple_tokocrypto_screener import TokocryptoScreener
    SCREENER_AVAILABLE = True
    print("✅ Simple screener loaded")
except ImportError as e:
    SCREENER_AVAILABLE = False
    print(f"❌ Screener not available: {e}")

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Parse URL
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        query_params = urllib.parse.parse_qs(parsed_path.query)
        
        # Route handling
        if path == '/':
            self.serve_home()
        elif path == '/health':
            self.serve_health()
        elif path == '/screen':
            self.serve_screen(query_params)
        elif path.startswith('/static/'):
            self.serve_static(path)
        else:
            self.serve_404()
    
    def serve_home(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        response = {
            "message": "🚀 Crypto Screener API - Simple Version",
            "status": "running",
            "version": "2.0.0-simple",
            "endpoints": {
                "/": "API info",
                "/health": "Health check",
                "/screen": "Crypto screening",
                "/static/crypto_screening.html": "Web interface"
            },
            "screener_available": SCREENER_AVAILABLE
        }
        
        self.wfile.write(json.dumps(response, indent=2).encode())
    
    def serve_health(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        response = {
            "status": "healthy",
            "service": "crypto_screener_simple",
            "python_version": sys.version,
            "screener_available": SCREENER_AVAILABLE
        }
        
        self.wfile.write(json.dumps(response).encode())
    
    def serve_screen(self, query_params):
        if not SCREENER_AVAILABLE:
            self.send_error(503, "Screener not available")
            return
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        # Get parameters
        quote_currency = query_params.get('quote_currency', ['USDT'])[0]
        limit = int(query_params.get('limit', ['20'])[0])
        
        try:
            screener = TokocryptoScreener()
            result = screener.screen_bullish_candidates(quote_currency, limit)
            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            error_response = {
                "status": "error",
                "message": str(e),
                "data": []
            }
            self.wfile.write(json.dumps(error_response).encode())
    
    def serve_static(self, path):
        # Remove /static/ prefix
        file_path = path[8:]
        full_path = os.path.join(current_dir, 'static', file_path)
        
        if os.path.exists(full_path) and os.path.isfile(full_path):
            self.send_response(200)
            
            # Determine content type
            if file_path.endswith('.html'):
                content_type = 'text/html'
            elif file_path.endswith('.css'):
                content_type = 'text/css'
            elif file_path.endswith('.js'):
                content_type = 'application/javascript'
            else:
                content_type = 'application/octet-stream'
            
            self.send_header('Content-type', content_type)
            self.end_headers()
            
            with open(full_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.serve_404()
    
    def serve_404(self):
        self.send_response(404)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        html = """
        <!DOCTYPE html>
        <html>
        <head><title>404 Not Found</title></head>
        <body>
            <h1>404 - Not Found</h1>
            <p>The requested resource was not found.</p>
            <p><a href="/">Go to API home</a></p>
        </body>
        </html>
        """.encode()
        
        self.wfile.write(html)

def application(environ, start_response):
    """WSGI interface for cPanel"""
    from wsgiref.simple_server import make_server
    
    # Simple WSGI app
    method = environ.get('REQUEST_METHOD', 'GET')
    path = environ.get('PATH_INFO', '/')
    query_string = environ.get('QUERY_STRING', '')
    
    if path == '/':
        status = '200 OK'
        headers = [('Content-Type', 'application/json')]
        start_response(status, headers)
        
        response = {
            "message": "🚀 Crypto Screener API - Simple WSGI Version",
            "status": "running",
            "method": method,
            "path": path,
            "screener_available": SCREENER_AVAILABLE
        }
        
        return [json.dumps(response).encode()]
    
    elif path == '/health':
        status = '200 OK'
        headers = [('Content-Type', 'application/json')]
        start_response(status, headers)
        
        response = {
            "status": "healthy",
            "service": "crypto_screener_wsgi"
        }
        
        return [json.dumps(response).encode()]
    
    else:
        status = '404 Not Found'
        headers = [('Content-Type', 'text/html')]
        start_response(status, headers)
        
        return [b"<h1>404 Not Found</h1><p><a href='/'>Go to API home</a></p>"]

if __name__ == "__main__":
    print("🚀 Starting simple HTTP server...")
    server = HTTPServer(('localhost', 8000), SimpleHandler)
    print("Server running on http://localhost:8000")
    server.serve_forever()
