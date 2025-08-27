#!/usr/bin/env python3
"""
Minimal WSGI for debugging segmentation fault
"""
import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print(f"Minimal WSGI starting...")
print(f"Python version: {sys.version}")
print(f"Current directory: {current_dir}")

def application(environ, start_response):
    """Minimal WSGI application"""
    status = '200 OK'
    headers = [('Content-Type', 'text/html')]
    start_response(status, headers)
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Minimal WSGI Test</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f0f8ff; }
            .container { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .success { color: #28a745; font-size: 24px; }
            .info { background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="success">✅ Minimal WSGI Working!</h1>
            <div class="info">
                <h3>🔧 Debug Information:</h3>
                <p><strong>Status:</strong> WSGI application is responding</p>
                <p><strong>Python Version:</strong> """ + sys.version + """</p>
                <p><strong>Directory:</strong> """ + current_dir + """</p>
                <p><strong>Files:</strong> """ + ", ".join(os.listdir(current_dir)[:10]) + """</p>
            </div>
            
            <div class="info">
                <h3>🚀 Next Steps:</h3>
                <ul>
                    <li>✅ Basic WSGI is working</li>
                    <li>🔄 Now test with FastAPI import</li>
                    <li>📊 Then test crypto screening functionality</li>
                </ul>
            </div>
            
            <div class="info">
                <h3>🔗 Test Links:</h3>
                <ul>
                    <li><a href="/">Home (this page)</a></li>
                    <li><a href="/test">Test endpoint (will 404 for now)</a></li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """.encode('utf-8')
    
    return [html]

if __name__ == "__main__":
    print("✅ Minimal WSGI loaded successfully")
    print("This file can be used as passenger_wsgi.py for testing")
