#!/usr/bin/env python3
"""
Debug WSGI - Minimal test to identify the exact error
"""
import sys
import os
import json
import traceback

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def application(environ, start_response):
    """Debug WSGI to show exact error"""
    try:
        status = '200 OK'
        headers = [('Content-Type', 'text/html')]
        start_response(status, headers)
        
        # Get environment info
        python_version = sys.version
        python_path = sys.path
        current_dir = os.getcwd()
        files_in_dir = os.listdir(current_dir)
        
        # Test imports one by one
        import_tests = []
        
        # Test 1: Basic imports
        try:
            import json
            import_tests.append("✅ json: OK")
        except Exception as e:
            import_tests.append(f"❌ json: {e}")
        
        # Test 2: Simple screener
        try:
            from simple_tokocrypto_screener import TokocryptoScreener
            import_tests.append("✅ simple_tokocrypto_screener: OK")
            
            # Test screener initialization
            screener = TokocryptoScreener()
            import_tests.append("✅ TokocryptoScreener init: OK")
        except Exception as e:
            import_tests.append(f"❌ simple_tokocrypto_screener: {e}")
            import_tests.append(f"❌ Traceback: {traceback.format_exc()}")
        
        # Test 3: HTTP request capability
        try:
            import urllib.request
            import_tests.append("✅ urllib.request: OK")
        except Exception as e:
            import_tests.append(f"❌ urllib.request: {e}")
        
        # Create debug HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Debug WSGI - Error Analysis</title>
            <style>
                body {{ font-family: monospace; margin: 20px; background: #f5f5f5; }}
                .container {{ background: white; padding: 20px; border-radius: 10px; }}
                .success {{ color: #28a745; }}
                .error {{ color: #dc3545; }}
                .info {{ background: #e3f2fd; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                pre {{ background: #f8f9fa; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔧 Debug WSGI - Error Analysis</h1>
                
                <div class="info">
                    <h3>📊 System Information:</h3>
                    <p><strong>Python Version:</strong> {python_version}</p>
                    <p><strong>Current Directory:</strong> {current_dir}</p>
                    <p><strong>Files in Directory:</strong> {', '.join(files_in_dir[:20])}</p>
                </div>
                
                <div class="info">
                    <h3>🧪 Import Tests:</h3>
                    <pre>{'<br>'.join(import_tests)}</pre>
                </div>
                
                <div class="info">
                    <h3>🛣️ Python Path:</h3>
                    <pre>{'<br>'.join(python_path[:10])}</pre>
                </div>
                
                <div class="info">
                    <h3>🌐 WSGI Environment:</h3>
                    <p><strong>REQUEST_METHOD:</strong> {environ.get('REQUEST_METHOD', 'N/A')}</p>
                    <p><strong>PATH_INFO:</strong> {environ.get('PATH_INFO', 'N/A')}</p>
                    <p><strong>QUERY_STRING:</strong> {environ.get('QUERY_STRING', 'N/A')}</p>
                    <p><strong>SERVER_NAME:</strong> {environ.get('SERVER_NAME', 'N/A')}</p>
                </div>
                
                <div class="info">
                    <h3>✅ WSGI Status:</h3>
                    <p class="success">✅ WSGI application is responding!</p>
                    <p>✅ Python execution is working</p>
                    <p>✅ File system access is working</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return [html.encode('utf-8')]
        
    except Exception as e:
        # Even if everything fails, return basic error info
        status = '500 Internal Server Error'
        headers = [('Content-Type', 'text/html')]
        start_response(status, headers)
        
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Critical Error</title></head>
        <body>
            <h1>🚨 Critical Error in Debug WSGI</h1>
            <p><strong>Error:</strong> {str(e)}</p>
            <p><strong>Traceback:</strong></p>
            <pre>{traceback.format_exc()}</pre>
            <p><strong>Python Version:</strong> {sys.version}</p>
            <p><strong>Working Directory:</strong> {os.getcwd()}</p>
        </body>
        </html>
        """.encode('utf-8')
        
        return [error_html]

if __name__ == "__main__":
    print("Debug WSGI loaded")
    
    # Test the application function locally
    class MockStartResponse:
        def __call__(self, status, headers):
            print(f"Status: {status}")
            print(f"Headers: {headers}")
    
    mock_environ = {
        'REQUEST_METHOD': 'GET',
        'PATH_INFO': '/',
        'QUERY_STRING': '',
        'SERVER_NAME': 'localhost'
    }
    
    mock_start_response = MockStartResponse()
    result = application(mock_environ, mock_start_response)
    print("Debug WSGI test completed")
