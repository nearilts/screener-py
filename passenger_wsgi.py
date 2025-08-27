#!/usr/bin/env python3
"""
Passenger WSGI file for cPanel deployment
This file is required by cPanel's Python App system
"""
import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Set up environment for cPanel
os.environ.setdefault('PYTHONPATH', current_dir)

try:
    from app import app as application
    
    # cPanel specific configuration
    if hasattr(application, 'mount'):
        # Ensure static files work in cPanel environment
        static_dir = os.path.join(current_dir, 'static')
        if os.path.exists(static_dir):
            from fastapi.staticfiles import StaticFiles
            application.mount("/static", StaticFiles(directory=static_dir), name="static")
    
except ImportError as e:
    # Fallback WSGI application if main app fails to import
    def application(environ, start_response):
        status = '500 Internal Server Error'
        headers = [('Content-Type', 'text/html')]
        start_response(status, headers)
        
        error_msg = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Application Error</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .error {{ background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .error h1 {{ color: #d32f2f; }}
                .details {{ background: #f5f5f5; padding: 10px; border-radius: 3px; margin: 10px 0; font-family: monospace; }}
            </style>
        </head>
        <body>
            <div class="error">
                <h1>🚨 Application Import Error</h1>
                <p>Failed to import the main application.</p>
                
                <h3>Error Details:</h3>
                <div class="details">{str(e)}</div>
                
                <h3>Possible Solutions:</h3>
                <ul>
                    <li>Ensure all required packages are installed: <code>pip install -r requirements.txt</code></li>
                    <li>Check Python version compatibility (requires Python 3.7+)</li>
                    <li>Verify all files are uploaded correctly</li>
                    <li>Check cPanel error logs for more details</li>
                </ul>
                
                <h3>Python Environment:</h3>
                <div class="details">
                    Python version: {sys.version}<br>
                    Python path: {sys.path[0]}<br>
                    Current directory: {current_dir}<br>
                    Files in directory: {', '.join(os.listdir(current_dir)) if os.path.exists(current_dir) else 'Directory not accessible'}
                </div>
                
                <h3>Next Steps:</h3>
                <ol>
                    <li>Check cPanel Python App logs</li>
                    <li>Verify all dependencies are installed</li>
                    <li>Contact hosting support if issue persists</li>
                </ol>
            </div>
        </body>
        </html>
        """.encode('utf-8')
        
        return [error_msg]

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
