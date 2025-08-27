#!/usr/bin/env python3
"""
Passenger WSGI file for cPanel deployment - Using pure WSGI implementation
"""
import sys
import os
import json

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Change working directory to project directory
os.chdir(current_dir)

# Main WSGI application
def application(environ, start_response):
    try:
        # Use pure WSGI implementation (no FastAPI dependency)
        from pure_wsgi import application as pure_app
        return pure_app(environ, start_response)
        
    except ImportError as import_err:
        # Fallback if pure_wsgi not available
        status = '500 Internal Server Error'
        headers = [('Content-Type', 'application/json')]
        start_response(status, headers)
        
        response = {
            "error": "Import Error",
            "message": f"Could not import pure_wsgi: {str(import_err)}",
            "python_version": sys.version,
            "current_directory": current_dir,
            "files_available": os.listdir('.') if os.path.exists('.') else [],
            "path": sys.path
        }
        return [json.dumps(response, indent=2).encode()]
        
    except Exception as e:
        # General error fallback
        status = '500 Internal Server Error'
        headers = [('Content-Type', 'application/json')]
        start_response(status, headers)
        
        import traceback
        response = {
            "error": "Application Error",
            "message": str(e),
            "traceback": traceback.format_exc(),
            "python_version": sys.version,
            "current_directory": current_dir
        }
        return [json.dumps(response, indent=2).encode()]

# Alternative entry points for different cPanel configurations
app = application

def create_app():
    return application

# Debug information (will be logged by cPanel)
if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"Current directory: {current_dir}")
    print(f"Working directory: {os.getcwd()}")
    print("Passenger WSGI file loaded successfully")
    
    # Test import
    try:
        from pure_wsgi import application as pure_app
        print("✅ Pure WSGI application imported successfully")
    except Exception as e:
        print(f"❌ Failed to import pure WSGI application: {e}")
        
    try:
        from simple_tokocrypto_screener import TokocryptoScreener
        print("✅ Screener module imported successfully")
    except Exception as e:
        print(f"❌ Failed to import screener module: {e}")
