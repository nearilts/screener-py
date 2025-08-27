#!/usr/bin/env python3
"""
cPanel-compatible FastAPI application
Optimized for shared hosting environments
"""
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
# Removed StaticFiles to avoid aiofiles dependency
import asyncio
import logging
from datetime import datetime

# Reduced imports for cPanel compatibility
try:
    from tokocrypto_screener import TokocryptoScreener
    SCREENER_AVAILABLE = True
    print("✅ Using original TokocryptoScreener")
except ImportError:
    try:
        from simple_tokocrypto_screener import TokocryptoScreener
        SCREENER_AVAILABLE = True
        print("✅ Using simple TokocryptoScreener")
    except ImportError:
        SCREENER_AVAILABLE = False
        logging.warning("No screener available, using fallback mode")

# Setup logging for cPanel
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app with cPanel optimizations
app = FastAPI(
    title="Tokocrypto Screener - cPanel Edition",
    description="Cryptocurrency screening optimized for shared hosting",
    version="2.0.0",
    docs_url="/docs",  # API documentation
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files manually without StaticFiles to avoid aiofiles dependency
@app.get("/static/{filename:path}")
async def serve_static(filename: str):
    """Serve static files manually"""
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    file_path = os.path.join(static_dir, filename)
    
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main page or simple landing page"""
    try:
        if os.path.exists('static/crypto_screening.html'):
            return FileResponse('static/crypto_screening.html')
        else:
            # Fallback HTML if static files not available
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Tokocrypto Screener API</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .container { max-width: 800px; margin: 0 auto; }
                    .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
                    .status { color: green; font-weight: bold; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🚀 Tokocrypto Screener API</h1>
                    <p class="status">✅ API is running successfully!</p>
                    
                    <h2>📡 Available Endpoints:</h2>
                    
                    <div class="endpoint">
                        <h3>GET /screen</h3>
                        <p>Screen for bullish cryptocurrency candidates</p>
                        <p><strong>Parameters:</strong></p>
                        <ul>
                            <li>quote: Quote currency (USDT, BTC, ETH) - default: USDT</li>
                            <li>limit: Number of coins to analyze (10-100) - default: 50</li>
                            <li>min_volume: Minimum quote volume - default: 10000</li>
                        </ul>
                        <p><strong>Example:</strong> <a href="/screen?quote=USDT&limit=20">/screen?quote=USDT&limit=20</a></p>
                    </div>
                    
                    <div class="endpoint">
                        <h3>GET /health</h3>
                        <p>Health check endpoint</p>
                        <p><strong>Example:</strong> <a href="/health">/health</a></p>
                    </div>
                    
                    <div class="endpoint">
                        <h3>GET /docs</h3>
                        <p>Interactive API documentation</p>
                        <p><strong>Example:</strong> <a href="/docs">/docs</a></p>
                    </div>
                    
                    <h2>🔧 System Status:</h2>
                    <ul>
                        <li>Python Version: """ + sys.version + """</li>
                        <li>Screener Module: """ + ("✅ Available" if SCREENER_AVAILABLE else "❌ Not Available") + """</li>
                        <li>Server Time: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</li>
                    </ul>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error serving root page: {e}")
        return HTMLResponse(content=f"<h1>Error</h1><p>{str(e)}</p>", status_code=500)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "screener_available": SCREENER_AVAILABLE,
        "python_version": sys.version
    }

@app.get("/screen")
async def screen_bullish(
    request: Request,
    quote: str = Query("USDT", description="Quote currency (USDT, BTC, ETH)"),
    limit: int = Query(50, description="Number of coins to analyze", ge=10, le=100),  # Reduced max for cPanel
    min_volume: float = Query(10000, description="Minimum quote volume", ge=1000)
):
    """
    Screen for bullish cryptocurrency candidates
    Optimized for cPanel shared hosting limitations
    """
    start_time = datetime.now()
    client_ip = request.client.host
    
    logger.info(f"Screening request from {client_ip}: quote={quote}, limit={limit}")
    
    if not SCREENER_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Screener module not available",
                "message": "Please ensure all dependencies are installed",
                "fallback_available": True
            }
        )
    
    try:
        # cPanel resource limits - use smaller timeouts
        async with asyncio.timeout(300):  # 5 minute timeout max
            async with TokocryptoScreener() as screener:
                result = await screener.screen_bullish_candidates(
                    quote_currency=quote.upper(),
                    limit=min(limit, 100),  # Enforce cPanel limits
                    min_volume=min_volume
                )
                
                if 'error' in result:
                    logger.error(f"Screener error: {result['error']}")
                    raise HTTPException(status_code=500, detail=result['error'])
                
                # Add execution info
                execution_time = (datetime.now() - start_time).total_seconds()
                result['server_execution_time'] = round(execution_time, 2)
                result['server_timestamp'] = datetime.now().isoformat()
                
                logger.info(f"Screening completed in {execution_time:.2f}s, found {result.get('bullish_candidates', 0)} candidates")
                
                return JSONResponse(content=result)
                
    except asyncio.TimeoutError:
        logger.error("Screening timeout")
        return JSONResponse(
            status_code=504,
            content={
                "error": "Request timeout",
                "message": "Analysis taking too long, try reducing the limit parameter",
                "suggestion": "Use limit=20 or less for faster results"
            }
        )
    except Exception as e:
        logger.error(f"Screening error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": str(e),
                "suggestion": "Try again with smaller limit or contact support"
            }
        )

@app.get("/analyze/{symbol}")
async def analyze_symbol(
    symbol: str,
    interval: str = Query("1h", description="Kline interval (1h, 4h, 1d)"),
    limit: int = Query(50, description="Number of klines", ge=20, le=100)
):
    """Analyze a specific cryptocurrency symbol"""
    if not SCREENER_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "Screener module not available"}
        )
    
    try:
        async with TokocryptoScreener() as screener:
            klines_data = await screener.get_klines_batch([symbol.upper()], interval, limit)
            
            if symbol.upper() not in klines_data:
                raise HTTPException(status_code=404, detail="Symbol not found")
                
            klines = klines_data[symbol.upper()]
            if not klines:
                raise HTTPException(status_code=404, detail="No data available")
                
            analysis = screener.analyze_technical(klines)
            
            if 'error' in analysis:
                raise HTTPException(status_code=400, detail=analysis['error'])
                
            return JSONResponse(content={
                'symbol': symbol.upper(),
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "available_endpoints": ["/", "/screen", "/health", "/docs"],
            "requested_path": str(request.url.path)
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    logger.error(f"Internal error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "Please try again later or contact support"
        }
    )

# For cPanel Passenger WSGI compatibility
application = app

if __name__ == "__main__":
    # For local development
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
