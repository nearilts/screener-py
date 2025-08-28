#!/usr/bin/env python3
"""
Pure WSGI application - cPanel compatible
"""
import sys
import os
import json

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Ensure we're in the right directory
os.chdir(current_dir)

def application(environ, start_response):
    """Pure WSGI application for cPanel"""
    try:
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
        
        # Test screener availability here (inside the request)
        try:
            from simple_tokocrypto_screener import TokocryptoScreener
            SCREENER_AVAILABLE = True
        except ImportError as e:
            SCREENER_AVAILABLE = False
        
        # Route handling
        if path == '/':
            status = '200 OK'
            headers = [('Content-Type', 'application/json')]
            start_response(status, headers)
            
            response = {
                "message": "🚀 Crypto Screener API - cPanel WSGI",
                "status": "running",
                "version": "2.0.0",
                "python_version": sys.version,
                "screener_available": SCREENER_AVAILABLE,
                "current_directory": current_dir,
                "endpoints": [
                    "/", 
                    "/health", 
                    "/symbols", 
                    "/debug?symbol=BTCUSDT (Test single symbol analysis)",
                    "/screen (Basic analysis)",
                    "/screen?type=advanced (Advanced bearish-to-bullish analysis)",
                    "/accurate-analysis (Enhanced accurate bullish analysis)",
                    "/static/crypto_screening.html"
                ]
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
                "python_version": sys.version,
                "working_directory": os.getcwd(),
                "files_count": len(os.listdir('.'))
            }
            return [json.dumps(response).encode()]
        
        elif path == '/symbols':
            status = '200 OK'
            headers = [('Content-Type', 'application/json')]
            start_response(status, headers)
            
            if not SCREENER_AVAILABLE:
                return [json.dumps({"error": "Screener not available"}).encode()]
            
            try:
                screener = TokocryptoScreener()
                symbols = screener.get_symbols()
                
                # Get unique quote assets
                quote_assets = set()
                symbol_count = {}
                
                for symbol in symbols:
                    quote_asset = symbol.get('quoteAsset', 'UNKNOWN')
                    quote_assets.add(quote_asset)
                    symbol_count[quote_asset] = symbol_count.get(quote_asset, 0) + 1
                
                response = {
                    "total_symbols": len(symbols),
                    "available_quote_assets": sorted(list(quote_assets)),
                    "symbol_count_by_quote": symbol_count,
                    "sample_symbols": symbols[:10] if symbols else []
                }
                return [json.dumps(response, indent=2).encode()]
            except Exception as e:
                return [json.dumps({"error": str(e)}).encode()]
        
        elif path == '/debug':
            # Debug endpoint to test single symbol analysis
            if not SCREENER_AVAILABLE:
                status = '503 Service Unavailable'
                headers = [('Content-Type', 'application/json')]
                start_response(status, headers)
                return [json.dumps({"error": "Screener not available"}).encode()]
            
            status = '200 OK'
            headers = [('Content-Type', 'application/json')]
            start_response(status, headers)
            
            symbol = query_params.get('symbol', 'BTCUSDT')
            
            try:
                screener = TokocryptoScreener()
                
                # Test both basic and advanced analysis
                basic_result = screener.analyze_symbol({'symbol': symbol})
                advanced_result = screener.analyze_symbol_advanced(symbol)
                
                response = {
                    "symbol": symbol,
                    "basic_analysis": basic_result,
                    "advanced_analysis": advanced_result,
                    "debug_info": "Use this to see why symbols pass/fail filters"
                }
                return [json.dumps(response, indent=2).encode()]
            except Exception as e:
                return [json.dumps({"error": str(e), "symbol": symbol}).encode()]
        
        elif path == '/screen':
            if not SCREENER_AVAILABLE:
                status = '503 Service Unavailable'
                headers = [('Content-Type', 'application/json')]
                start_response(status, headers)
                return [json.dumps({"error": "Screener not available", "path": current_dir}).encode()]
            
            status = '200 OK'
            headers = [('Content-Type', 'application/json')]
            start_response(status, headers)
            
            # Get parameters
            quote_currency = query_params.get('quote_currency', 'USDT')
            limit = int(query_params.get('limit', '999'))  # Default to all coins
            analysis_type = query_params.get('type', 'advanced')  # Default to advanced analysis
            
            # Allow unlimited analysis but warn about performance
            if limit > 500:
                limit = 999  # Use 999 as "all" indicator
            
            try:
                screener = TokocryptoScreener()
                
                # Use advanced analysis by default, but allow basic fallback
                if analysis_type == 'advanced':
                    result = screener.screen_bearish_to_bullish_advanced(quote_currency, limit)
                    
                    # If advanced returns no results, try basic analysis as fallback
                    if result.get('high_quality_candidates', 0) == 0:
                        print("⚠️ Advanced analysis found no results, trying basic analysis...")
                        basic_result = screener.screen_bullish_candidates(quote_currency, limit)
                        if basic_result.get('bullish_candidates', 0) > 0:
                            result = {
                                **basic_result,
                                'analysis_type': 'basic_fallback',
                                'note': 'Advanced criteria too strict, showing basic analysis results'
                            }
                else:
                    result = screener.screen_bullish_candidates(quote_currency, limit)
                    
                return [json.dumps(result).encode()]
            except Exception as e:
                error_response = {
                    "status": "error",
                    "message": str(e),
                    "data": [],
                    "debug_info": {
                        "python_version": sys.version,
                        "current_dir": os.getcwd(),
                        "screener_available": SCREENER_AVAILABLE
                    }
                }
                return [json.dumps(error_response).encode()]
        
        elif path == '/accurate-analysis':
            if not SCREENER_AVAILABLE:
                status = '503 Service Unavailable'
                headers = [('Content-Type', 'application/json')]
                start_response(status, headers)
                return [json.dumps({"error": "Screener not available", "path": current_dir}).encode()]
            
            status = '200 OK'
            headers = [('Content-Type', 'application/json')]
            start_response(status, headers)
            
            # Get parameters
            quote_currency = query_params.get('quote_currency', 'USDT')
            limit = int(query_params.get('limit', '50'))  # Default 50 for accurate analysis
            
            try:
                screener = TokocryptoScreener()
                result = screener.accurate_bullish_analysis(quote_currency, limit)
                return [json.dumps(result).encode()]
            except Exception as e:
                error_response = {
                    "status": "error",
                    "message": str(e),
                    "data": [],
                    "debug_info": {
                        "python_version": sys.version,
                        "current_dir": os.getcwd(),
                        "screener_available": SCREENER_AVAILABLE,
                        "endpoint": "/accurate-analysis"
                    }
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
                    content_type = 'text/html; charset=utf-8'
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
                return [f"<h1>404 - File Not Found</h1><p>Looking for: {full_path}</p>".encode()]
        
        else:
            # 404 for other paths
            status = '404 Not Found'
            headers = [('Content-Type', 'application/json')]
            start_response(status, headers)
            
            response = {
                "error": "Not found",
                "path": path,
                "available_endpoints": ["/", "/health", "/symbols", "/debug", "/screen", "/accurate-analysis", "/static/crypto_screening.html"]
            }
            return [json.dumps(response).encode()]
    
    except Exception as e:
        # Ultimate fallback
        status = '500 Internal Server Error'
        headers = [('Content-Type', 'application/json')]
        start_response(status, headers)
        
        import traceback
        error_response = {
            "error": "Internal server error",
            "message": str(e),
            "traceback": traceback.format_exc(),
            "python_version": sys.version,
            "current_directory": os.getcwd()
        }
        return [json.dumps(error_response).encode()]
