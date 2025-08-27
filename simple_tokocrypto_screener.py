#!/usr/bin/env python3
"""
Simple TokocryptoScreener for Python 3.6 compatibility
Minimal dependencies version
"""

import json
import time
import urllib.request
import urllib.parse
from typing import Dict, List, Any
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class TokocryptoScreener:
    def __init__(self):
        self.binance_url = 'https://api.binance.com/api/v3'
        self.tokocrypto_url = 'https://www.tokocrypto.com/open/v1'
        
    def make_request(self, url: str, timeout: int = 30) -> Dict:
        """Make HTTP request with basic error handling"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except Exception as e:
            return {'error': str(e)}
    
    def get_symbols(self) -> List[Dict]:
        """Get all available symbols from Tokocrypto"""
        url = f"{self.tokocrypto_url}/common/symbols"
        response = self.make_request(url)
        
        if 'error' in response:
            # Fallback to Binance if Tokocrypto fails
            url = f"{self.binance_url}/exchangeInfo"
            binance_response = self.make_request(url)
            if 'symbols' in binance_response:
                return binance_response['symbols']
            return []
        
        return response.get('data', {}).get('list', [])
    
    def get_ticker_data(self, symbol: str) -> Dict:
        """Get 24hr ticker data from Binance"""
        binance_symbol = symbol.replace('_', '')
        url = f"{self.binance_url}/ticker/24hr?symbol={binance_symbol}"
        
        return self.make_request(url)
    
    def get_klines(self, symbol: str, interval: str = '1h', limit: int = 20) -> List:
        """Get kline data for technical analysis"""
        binance_symbol = symbol.replace('_', '')
        url = f"{self.binance_url}/klines?symbol={binance_symbol}&interval={interval}&limit={limit}"
        
        response = self.make_request(url)
        if 'error' in response:
            return []
        
        return response if isinstance(response, list) else []
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < period:
            return 50
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return round(rsi, 2)
    
    def calculate_sma(self, prices: List[float], period: int) -> float:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return sum(prices) / len(prices) if prices else 0
        
        return sum(prices[-period:]) / period
    
    def analyze_symbol(self, symbol_info: Dict) -> Dict:
        """Analyze single symbol for bullish signals"""
        symbol = symbol_info.get('symbol', symbol_info.get('baseAsset', '') + symbol_info.get('quoteAsset', ''))
        
        # Get ticker data
        ticker = self.get_ticker_data(symbol)
        if 'error' in ticker:
            return None
        
        # Get kline data for technical analysis
        klines = self.get_klines(symbol)
        if not klines:
            return None
        
        # Extract closing prices
        closes = [float(kline[4]) for kline in klines]
        
        if len(closes) < 10:
            return None
        
        # Calculate indicators
        current_price = float(ticker['lastPrice'])
        rsi = self.calculate_rsi(closes)
        sma20 = self.calculate_sma(closes, min(20, len(closes)))
        volume_ratio = float(ticker['volume']) / max(float(ticker.get('prevVolume', ticker['volume'])), 1)
        
        # Determine support and resistance
        recent_lows = closes[-10:]
        recent_highs = closes[-10:]
        support = min(recent_lows)
        resistance = max(recent_highs)
        
        # Calculate signal strength
        score = 0
        signals = []
        
        # Calculate profit potential first
        profit_potential = ((resistance - current_price) / current_price) * 100 if resistance > current_price else 0
        
        # RSI oversold signal (good for bullish reversal)
        if rsi < 30:
            score += 35
            signals.append(f"RSI deeply oversold ({rsi:.1f}) - strong reversal potential")
        elif rsi < 35:
            score += 25
            signals.append(f"RSI oversold ({rsi:.1f}) - reversal signal")
        elif rsi < 45:
            score += 15
            signals.append(f"RSI recovering from oversold ({rsi:.1f})")
        
        # Profit potential scoring
        if profit_potential > 15:
            score += 40
            signals.append(f"Huge profit potential: +{profit_potential:.1f}% to resistance")
        elif profit_potential > 10:
            score += 30
            signals.append(f"High profit potential: +{profit_potential:.1f}% to resistance")
        elif profit_potential > 5:
            score += 20
            signals.append(f"Good profit potential: +{profit_potential:.1f}% to resistance")
        
        # Price near support (buying opportunity)
        if support > 0 and current_price <= support * 1.02:
            score += 25
            signals.append("Price near strong support - good entry zone")
        elif support > 0 and current_price <= support * 1.05:
            score += 15
            signals.append("Price approaching support level")
        
        # Price below SMA (potential reversal)
        if current_price < sma20 * 0.95:
            score += 25
            signals.append("Price significantly below SMA20 - oversold")
        elif current_price < sma20 * 0.98:
            score += 15
            signals.append("Price below SMA20 - potential bounce")
        
        # Volume analysis (momentum indicator)
        if volume_ratio > 3.0:
            score += 35
            signals.append(f"Massive volume surge ({volume_ratio:.1f}x) - strong interest")
        elif volume_ratio > 2.0:
            score += 25
            signals.append(f"Strong volume surge ({volume_ratio:.1f}x)")
        elif volume_ratio > 1.5:
            score += 20
            signals.append(f"High volume ({volume_ratio:.1f}x)")
        elif volume_ratio > 1.2:
            score += 10
            signals.append(f"Increased volume ({volume_ratio:.1f}x)")
        
        # Recent price action patterns
        if len(closes) >= 5:
            # Check for bullish reversal patterns
            if closes[-1] > closes[-2] and closes[-2] <= closes[-3]:
                score += 20
                signals.append("Bullish reversal pattern detected")
            
            # Check for higher lows (bullish trend)
            if closes[-2] > closes[-4] and closes[-1] > closes[-3]:
                score += 15
                signals.append("Higher lows pattern - uptrend forming")
            
            # Price consolidation near support
            recent_range = max(closes[-3:]) - min(closes[-3:])
            if recent_range / current_price < 0.02:  # Less than 2% range
                score += 15
                signals.append("Tight consolidation - potential breakout")
        
        # Risk/reward ratio calculation and scoring
        if support > 0:
            downside_risk = ((current_price - support) / current_price) * 100
            if downside_risk > 0 and profit_potential > 0:
                risk_reward = profit_potential / downside_risk
                if risk_reward > 4:
                    score += 25
                    signals.append(f"Excellent risk/reward: {risk_reward:.1f}:1")
                elif risk_reward > 2:
                    score += 15
                    signals.append(f"Good risk/reward: {risk_reward:.1f}:1")
        
        # Market cap bonus (prefer lower caps for higher volatility/profit potential)
        # This would need market cap data, skipping for now
        
        # Only return coins with strong signal strength
        if score < 30:  # Increased threshold for better quality
            return None
        
        # Calculate entry and take profit levels
        entry_price = current_price * 0.995  # 0.5% below current
        stop_loss = min(support * 0.97, entry_price * 0.97)  # 3% below entry or support
        take_profit_1 = min(resistance * 0.98, entry_price * 1.08)  # 8% above entry or resistance
        take_profit_2 = entry_price * 1.12  # 12% above entry
        
        # Calculate profit percentages
        tp1_percent = ((take_profit_1 - entry_price) / entry_price) * 100
        tp2_percent = ((take_profit_2 - entry_price) / entry_price) * 100
        
        # Risk/reward calculation
        risk = ((entry_price - stop_loss) / entry_price) * 100
        reward = tp1_percent
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'rsi': rsi,
            'signal_strength': min(score, 100),
            'entry_level': entry_price,
            'take_profit': take_profit_1,
            'potential_profit': tp1_percent,
            'trading_plan': f"Entry: {entry_price:.8f}, TP1: {take_profit_1:.8f} (+{tp1_percent:.1f}%), SL: {stop_loss:.8f}",
            'analysis': {
                'score': min(score, 100),
                'rsi': rsi,
                'sma20': sma20,
                'support': support,
                'resistance': resistance,
                'volume_ratio': round(volume_ratio, 2),
                'signals': signals
            }
        }
    
    def screen_bullish_candidates(self, quote_currency: str = 'USDT', limit: int = 50) -> Dict:
        """Screen for bullish candidates"""
        start_time = time.time()
        
        # Get all symbols
        symbols = self.get_symbols()
        if not symbols:
            return {
                'status': 'error',
                'message': 'Failed to fetch symbols',
                'data': []
            }
        
        # Filter by quote currency
        filtered_symbols = []
        for symbol in symbols:
            quote_asset = symbol.get('quoteAsset', '')
            if quote_currency == 'ALL' or quote_asset == quote_currency:
                filtered_symbols.append(symbol)
        
        # Limit analysis
        if limit > 0:
            filtered_symbols = filtered_symbols[:limit]
        
        # Analyze symbols using threading for better performance
        results = []
        
        def analyze_wrapper(symbol_info):
            try:
                return self.analyze_symbol(symbol_info)
            except Exception as e:
                print(f"Error analyzing {symbol_info}: {e}")
                return None
        
        # Use ThreadPoolExecutor for concurrent analysis
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(analyze_wrapper, symbol): symbol 
                for symbol in filtered_symbols
            }
            
            # Collect results
            for i, future in enumerate(as_completed(future_to_symbol)):
                try:
                    result = future.result(timeout=30)
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"Error in future: {e}")
                
                # Progress tracking
                if (i + 1) % 10 == 0:
                    print(f"Analyzed {i + 1}/{len(filtered_symbols)} symbols...")
        
        # Sort by score descending
        results.sort(key=lambda x: x['analysis']['score'], reverse=True)
        
        execution_time = time.time() - start_time
        
        return {
            'status': 'success',
            'data': results,
            'execution_time': round(execution_time, 2),
            'total_analyzed': len(filtered_symbols),
            'bullish_candidates': len(results),
            'quote_currency': quote_currency
        }

# Test function
if __name__ == "__main__":
    screener = TokocryptoScreener()
    print("Testing simple screener...")
    
    result = screener.screen_bullish_candidates('USDT', 5)
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        print(f"Found {result['bullish_candidates']} candidates in {result['execution_time']} seconds")
        
        for coin in result['data'][:3]:
            print(f"- {coin['symbol']}: Score {coin['analysis']['score']}, RSI {coin['rsi']}")
    else:
        print(f"Error: {result.get('message', 'Unknown error')}")
