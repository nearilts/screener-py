#!/usr/bin/env python3
"""
Pure WSGI application without FastAPI dependency
"""
import sys
import os
import json

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print(f"Pure WSGI starting - Python {sys.version}")

# Check if screener is available
try:
    from simple_tokocrypto_screener import TokocryptoScreener
    SCREENER_AVAILABLE = True
    print("✅ Simple screener loaded")
except ImportError:
    SCREENER_AVAILABLE = False
    print("❌ Screener not available")

def application(environ, start_response):
    """Pure WSGI application without FastAPI"""
    method = environ.get('REQUEST_METHOD', 'GET')
    path = environ.get('PATH_INFO', '/')
    query_string = environ.get('QUERY_STRING', '')
    
    # Parse query parameters
    query_params = {}
    if query_string:
        for param in query_string.split('&'):
            if '=' in param:
                key, value = param.split('=', 1)
                query_params[key] = value
    
    # Route handling
    if path == '/':
        status = '200 OK'
        headers = [('Content-Type', 'application/json')]
        start_response(status, headers)
        
        response = {
            "message": "🚀 Crypto Screener API - Pure WSGI",
            "status": "running",
            "version": "2.0.0",
            "screener_available": SCREENER_AVAILABLE,
            "endpoints": ["/", "/health", "/screen", "/static/crypto_screening.html"]
        }
        return [json.dumps(response, indent=2).encode()]
    
    elif path == '/health':
        status = '200 OK'
        headers = [('Content-Type', 'application/json')]
        start_response(status, headers)
        
        response = {
            "status": "healthy",
            "service": "crypto_screener_wsgi",
            "screener_available": SCREENER_AVAILABLE,
            "python_version": sys.version
        }
        return [json.dumps(response).encode()]
    
    elif path == '/screen':
        if not SCREENER_AVAILABLE:
            status = '503 Service Unavailable'
            headers = [('Content-Type', 'application/json')]
            start_response(status, headers)
            return [json.dumps({"error": "Screener not available"}).encode()]
        
        status = '200 OK'
        headers = [('Content-Type', 'application/json')]
        start_response(status, headers)
        
        # Get parameters
        quote_currency = query_params.get('quote_currency', 'USDT')
        limit = int(query_params.get('limit', '20'))
        
        try:
            screener = TokocryptoScreener()
            result = screener.screen_bullish_candidates(quote_currency, limit)
            return [json.dumps(result).encode()]
        except Exception as e:
            error_response = {
                "status": "error",
                "message": str(e),
                "data": []
            }
            return [json.dumps(error_response).encode()]
    
    elif path.startswith('/static/'):
        # Serve static files
        file_path = path[8:]  # Remove /static/
        full_path = os.path.join(current_dir, 'static', file_path)
        
        if os.path.exists(full_path) and os.path.isfile(full_path):
            status = '200 OK'
            
            # Determine content type
            if file_path.endswith('.html'):
                content_type = 'text/html'
            elif file_path.endswith('.css'):
                content_type = 'text/css'
            elif file_path.endswith('.js'):
                content_type = 'application/javascript'
            else:
                content_type = 'application/octet-stream'
            
            headers = [('Content-Type', content_type)]
            start_response(status, headers)
            
            with open(full_path, 'rb') as f:
                return [f.read()]
        else:
            status = '404 Not Found'
            headers = [('Content-Type', 'text/html')]
            start_response(status, headers)
            return [b"<h1>404 - File Not Found</h1>"]
    
    else:
        # 404 for other paths
        status = '404 Not Found'
        headers = [('Content-Type', 'application/json')]
        start_response(status, headers)
        
        response = {
            "error": "Not found",
            "available_endpoints": ["/", "/health", "/screen", "/static/crypto_screening.html"]
        }
        return [json.dumps(response).encode()]

if __name__ == "__main__":
    print("✅ Pure WSGI loaded successfully")
    print(f"Screener available: {SCREENER_AVAILABLE}")
