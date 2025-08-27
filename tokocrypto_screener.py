#!/usr/bin/env python3
"""
Tokocrypto Cryptocurrency Screener - Python Version
High-performance screening with asyncio and numpy
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
import json
from datetime import datetime
import time
from typing import Dict, List, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokocryptoScreener:
    def __init__(self):
        self.binance_url = "https://api.binance.com/api/v3"
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_json(self, url: str) -> Optional[Dict]:
        """Fetch JSON data from URL with error handling"""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    async def get_all_tickers(self) -> List[Dict]:
        """Get all 24hr ticker statistics"""
        url = f"{self.binance_url}/ticker/24hr"
        data = await self.fetch_json(url)
        return data if data else []
    
    async def get_klines_batch(self, symbols: List[str], interval: str = '1h', limit: int = 50) -> Dict[str, List]:
        """Fetch klines for multiple symbols concurrently"""
        async def fetch_klines(symbol: str) -> Tuple[str, List]:
            url = f"{self.binance_url}/klines?symbol={symbol}&interval={interval}&limit={limit}"
            data = await self.fetch_json(url)
            return symbol, data if data else []
        
        # Process in batches to avoid rate limiting
        batch_size = 10
        all_results = {}
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            tasks = [fetch_klines(symbol) for symbol in batch]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, tuple) and len(result) == 2:
                    symbol, klines = result
                    all_results[symbol] = klines
                    
            # Rate limiting delay
            await asyncio.sleep(0.1)
            
        return all_results
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI using numpy for speed"""
        if len(prices) < period + 1:
            return 50.0
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
    
    def calculate_sma(self, prices: np.ndarray, period: int) -> float:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return float(np.mean(prices))
        return float(np.mean(prices[-period:]))
    
    def find_support_resistance(self, highs: np.ndarray, lows: np.ndarray, lookback: int = 20) -> Tuple[float, float]:
        """Find support and resistance levels"""
        if len(lows) < lookback:
            lookback = len(lows)
        if len(highs) < lookback:
            lookback = len(highs)
            
        support = float(np.min(lows[-lookback:]))
        resistance = float(np.max(highs[-lookback:]))
        
        return support, resistance
    
    def analyze_technical(self, klines: List) -> Dict:
        """Perform technical analysis on klines data"""
        if not klines or len(klines) < 20:
            return {'error': 'Insufficient data'}
        
        # Convert to numpy arrays for speed
        opens = np.array([float(k[1]) for k in klines])
        highs = np.array([float(k[2]) for k in klines])
        lows = np.array([float(k[3]) for k in klines])
        closes = np.array([float(k[4]) for k in klines])
        volumes = np.array([float(k[5]) for k in klines])
        
        current_price = closes[-1]
        
        # Technical indicators
        rsi = self.calculate_rsi(closes)
        sma20 = self.calculate_sma(closes, 20)
        sma50 = self.calculate_sma(closes, 50) if len(closes) >= 50 else sma20
        
        support, resistance = self.find_support_resistance(highs, lows)
        
        # Volume analysis
        avg_volume = np.mean(volumes[-20:])
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Scoring system
        score = 0
        signals = []
        
        # RSI oversold recovery (20-40 range)
        if 20 <= rsi <= 40:
            signals.append(f"RSI oversold recovery: {rsi:.1f}")
            score += 20
        elif 40 < rsi <= 60:
            signals.append(f"RSI neutral bullish: {rsi:.1f}")
            score += 10
        
        # Price near support
        if abs(current_price - support) / support < 0.05:  # Within 5% of support
            signals.append("Price near support level")
            score += 15
        
        # Volume spike
        if volume_ratio > 1.5:
            signals.append(f"Volume spike: {volume_ratio:.1f}x")
            score += 10
        
        # Price above SMA20
        if current_price > sma20:
            signals.append("Price above SMA20")
            score += 10
        
        # Bullish momentum
        price_change_5 = (closes[-1] - closes[-6]) / closes[-6] * 100 if len(closes) >= 6 else 0
        if -10 < price_change_5 < 5:  # Slight decline or flat
            signals.append("Potential reversal setup")
            score += 15
        
        # Calculate entry levels
        entry_levels = self.calculate_entry_levels(current_price, support, resistance, rsi, score)
        
        # Determine strength
        strength = 'weak'
        if score >= 50:
            strength = 'strong'
        elif score >= 30:
            strength = 'medium'
        
        return {
            'rsi': round(rsi, 2),
            'sma20': round(sma20, 8),
            'sma50': round(sma50, 8),
            'support': round(support, 8),
            'resistance': round(resistance, 8),
            'volume_ratio': round(volume_ratio, 2),
            'current_price': current_price,
            'signals': signals,
            'score': score,
            'strength': strength,
            'recommendation': 'BUY' if score >= 40 else ('WATCH' if score >= 20 else 'HOLD'),
            'entry_levels': entry_levels
        }
    
    def calculate_entry_levels(self, current_price: float, support: float, resistance: float, rsi: float, score: int) -> Dict:
        """Calculate entry and take profit levels"""
        # Entry calculation based on score
        if score >= 50:
            entry_multiplier = 0.995  # Strong signal - enter near current
        elif score >= 30:
            entry_multiplier = 0.99   # Medium signal - small pullback
        else:
            entry_multiplier = 0.985  # Weak signal - wait for pullback
        
        entry_price = current_price * entry_multiplier
        
        # Don't enter below strong support
        if entry_price < support * 0.98:
            entry_price = support * 0.995
        
        # Stop loss calculation
        stop_loss = min(entry_price * 0.97, support * 0.96)
        
        # Take profit levels
        take_profits = self.calculate_take_profit_levels(entry_price, resistance, score)
        
        # Risk/reward ratio
        risk_amount = entry_price - stop_loss
        reward_amount = take_profits['tp1']['price'] - entry_price
        risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        return {
            'entry_price': round(entry_price, 8),
            'entry_zone': {
                'min': round(entry_price * 0.995, 8),
                'max': round(entry_price * 1.005, 8)
            },
            'stop_loss': round(stop_loss, 8),
            'take_profits': take_profits,
            'risk_reward_ratio': round(risk_reward_ratio, 2),
            'max_risk_percent': round((risk_amount / entry_price) * 100, 2),
            'entry_strategy': self.get_entry_strategy(score, rsi)
        }
    
    def calculate_take_profit_levels(self, entry_price: float, resistance: float, score: int) -> Dict:
        """Calculate multiple take profit levels"""
        # Base gain based on score
        if score >= 50:
            base_gain = 0.08  # 8% for strong signals
        elif score >= 30:
            base_gain = 0.06  # 6% for medium signals
        else:
            base_gain = 0.05  # 5% for weak signals
        
        # TP1: Conservative (ensure it's above entry)
        tp1_price = max(entry_price * (1 + base_gain), resistance * 0.95)
        tp1_percent = ((tp1_price - entry_price) / entry_price) * 100
        
        # TP2: Target (ensure it's above TP1)
        tp2_price = max(tp1_price * 1.2, resistance, entry_price * (1 + base_gain * 1.5))
        tp2_percent = ((tp2_price - entry_price) / entry_price) * 100
        
        # TP3: Moon (ensure it's above TP2)
        tp3_price = max(tp2_price * 1.1, resistance * 1.1)
        tp3_percent = ((tp3_price - entry_price) / entry_price) * 100
        
        return {
            'tp1': {
                'price': round(tp1_price, 8),
                'percent': round(tp1_percent, 2),
                'label': 'Conservative (25% position)',
                'allocation': 25
            },
            'tp2': {
                'price': round(tp2_price, 8),
                'percent': round(tp2_percent, 2),
                'label': 'Target (50% position)',
                'allocation': 50
            },
            'tp3': {
                'price': round(tp3_price, 8),
                'percent': round(tp3_percent, 2),
                'label': 'Moon (25% position)',
                'allocation': 25
            }
        }
    
    def get_entry_strategy(self, score: int, rsi: float) -> str:
        """Get entry strategy description"""
        if score >= 50:
            return "Strong signal - Enter immediately on any dip"
        elif score >= 30:
            return "Medium signal - Wait for 1-2% pullback"
        else:
            return "Weak signal - Wait for support test"
    
    async def screen_bullish_candidates(self, quote_currency: str = 'USDT', limit: int = 200, min_volume: float = 10000) -> Dict:
        """Screen for bullish candidates with high performance"""
        logger.info(f"Starting screening for {limit} coins...")
        start_time = time.time()
        
        # Get all tickers
        tickers = await self.get_all_tickers()
        if not tickers:
            return {'error': 'Failed to fetch tickers'}
        
        # Filter relevant symbols
        filtered_symbols = []
        ticker_map = {}
        
        for ticker in tickers:
            symbol = ticker['symbol']
            if symbol.endswith(quote_currency):
                base_currency = symbol.replace(quote_currency, '')
                
                # Skip stablecoins and wrapped tokens
                if base_currency in ['USDT', 'USDC', 'BUSD', 'TUSD', 'WBTC', 'WETH']:
                    continue
                
                # Volume filter
                quote_volume = float(ticker['quoteVolume'])
                if quote_volume >= min_volume:
                    filtered_symbols.append(symbol)
                    ticker_map[symbol] = ticker
        
        # Limit symbols for analysis
        if len(filtered_symbols) > limit:
            filtered_symbols = filtered_symbols[:limit]
        
        logger.info(f"Analyzing {len(filtered_symbols)} symbols...")
        
        # Get klines data for all symbols concurrently
        klines_data = await self.get_klines_batch(filtered_symbols)
        
        # Analyze each symbol
        results = []
        for symbol in filtered_symbols:
            if symbol not in klines_data or not klines_data[symbol]:
                continue
                
            ticker = ticker_map[symbol]
            klines = klines_data[symbol]
            
            # Technical analysis
            analysis = self.analyze_technical(klines)
            if 'error' in analysis:
                continue
            
            # Filter bullish candidates (score >= 15)
            if analysis['score'] >= 15:
                base_currency = symbol.replace(quote_currency, '')
                result = {
                    'symbol': symbol,
                    'base_currency': base_currency,
                    'last_price': float(ticker['lastPrice']),
                    'price_change_percent': float(ticker['priceChangePercent']),
                    'volume': float(ticker['volume']),
                    'quote_volume': float(ticker['quoteVolume']),
                    'analysis': analysis
                }
                results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x['analysis']['score'], reverse=True)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Screening completed in {elapsed_time:.1f} seconds")
        
        return {
            'code': 0,
            'msg': 'success',
            'data': results,
            'timestamp': int(time.time() * 1000),
            'total_analyzed': len(filtered_symbols),
            'bullish_candidates': len(results),
            'total_coins_available': len(ticker_map),
            'quote_currency': quote_currency,
            'execution_time': round(elapsed_time, 2)
        }

async def main():
    """Main function for testing"""
    async with TokocryptoScreener() as screener:
        # Test screening
        result = await screener.screen_bullish_candidates('USDT', limit=50)
        
        if 'error' not in result:
            print(f"\n🎯 Screening Results:")
            print(f"📊 Analyzed: {result['total_analyzed']} coins")
            print(f"🚀 Bullish candidates: {result['bullish_candidates']}")
            print(f"⏱️  Execution time: {result['execution_time']} seconds")
            
            if result['data']:
                print(f"\n🏆 Top 5 Candidates:")
                for i, coin in enumerate(result['data'][:5], 1):
                    analysis = coin['analysis']
                    print(f"{i}. {coin['base_currency']} - Score: {analysis['score']} - {analysis['recommendation']}")
        else:
            print(f"Error: {result['error']}")

if __name__ == "__main__":
    asyncio.run(main())
