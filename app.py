#!/usr/bin/env python3
"""
Simplified FastAPI application to avoid segmentation fault
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse

# Try to import screener
try:
    from simple_tokocrypto_screener import TokocryptoScreener
    SCREENER_AVAILABLE = True
    print("✅ Using simple TokocryptoScreener")
except ImportError:
    SCREENER_AVAILABLE = False
    print("❌ No screener available")

# Create simple FastAPI app
app = FastAPI(
    title="Crypto Screener",
    version="2.0.0"
)

@app.get("/")
async def root():
    return {
        "message": "🚀 Crypto Screener API",
        "status": "running",
        "version": "2.0.0",
        "screener_available": SCREENER_AVAILABLE
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "screener_available": SCREENER_AVAILABLE}

@app.get("/static/{filename:path}")
async def serve_static(filename: str):
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    file_path = os.path.join(static_dir, filename)
    
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/screen")
async def screen_cryptocurrencies(
    quote_currency: str = Query("USDT", description="Quote currency"),
    limit: int = Query(20, description="Number of symbols to analyze")
):
    if not SCREENER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Screener service not available")
    
    try:
        screener = TokocryptoScreener()
        result = screener.screen_bullish_candidates(quote_currency, limit)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e), "data": []}
        )
