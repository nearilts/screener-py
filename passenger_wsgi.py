#!/usr/bin/env python3
"""
Passenger WSGI file for cPanel deployment - Simplified to avoid segfault
"""
import sys
import os
import json

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print(f"WSGI starting - Python {sys.version}")

# Simple fallback WSGI application
def application(environ, start_response):
    try:
        # Try to import and use FastAPI app
        from app import app as fastapi_app
        
        # Use FastAPI's ASGI-to-WSGI adapter
        try:
            from fastapi.middleware.wsgi import WSGIMiddleware
            return fastapi_app(environ, start_response)
        except:
            # Manual WSGI handling
            method = environ.get('REQUEST_METHOD', 'GET')
            path = environ.get('PATH_INFO', '/')
            
            if path == '/':
                status = '200 OK'
                headers = [('Content-Type', 'application/json')]
                start_response(status, headers)
                
                response = {
                    "message": "� Crypto Screener API",
                    "status": "running via WSGI",
                    "version": "2.0.0"
                }
                return [json.dumps(response).encode()]
            
            elif path == '/health':
                status = '200 OK'
                headers = [('Content-Type', 'application/json')]
                start_response(status, headers)
                return [json.dumps({"status": "healthy"}).encode()]
            
            else:
                status = '404 Not Found'
                headers = [('Content-Type', 'text/html')]
                start_response(status, headers)
                return [b"<h1>404 Not Found</h1>"]
                
    except Exception as e:
        # Complete fallback
        status = '200 OK'
        headers = [('Content-Type', 'application/json')]
        start_response(status, headers)
        
        response = {
            "message": "Crypto Screener - Fallback Mode",
            "status": "limited functionality",
            "error": str(e)
        }
        return [json.dumps(response).encode()]

# Alternative application entry point for different cPanel configurations
app = application

# Entry point function for some cPanel setups
def create_app():
    return application

# Debug information (will be logged by cPanel)
if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"Current directory: {current_dir}")
    print(f"Python path: {sys.path}")
    print("Passenger WSGI file loaded successfully")
    
    # Test import
    try:
        from app import app
        print("✅ Main application imported successfully")
    except Exception as e:
        print(f"❌ Failed to import main application: {e}")
