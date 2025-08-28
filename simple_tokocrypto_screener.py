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
    
    def get_tokocrypto_ticker(self, symbol: str) -> Dict:
        """Get 24hr ticker data from Tokocrypto API"""
        toko_symbol = symbol.replace('_', '').upper()
        url = f"{self.tokocrypto_url}/market/ticker/24hr?symbol={toko_symbol}"
        
        return self.make_request(url)
    
    def get_tokocrypto_trades(self, symbol: str, limit: int = 100) -> List:
        """Get recent trades from Tokocrypto API"""
        toko_symbol = symbol.replace('_', '').upper()
        url = f"{self.tokocrypto_url}/market/trades?symbol={toko_symbol}&limit={limit}"
        
        response = self.make_request(url)
        if 'error' in response:
            return []
        
        return response.get('data', [])
    
    def get_tokocrypto_agg_trades(self, symbol: str, limit: int = 100) -> List:
        """Get aggregated trades from Tokocrypto API"""
        toko_symbol = symbol.replace('_', '').upper()
        url = f"{self.tokocrypto_url}/market/agg-trades?symbol={toko_symbol}&limit={limit}"
        
        response = self.make_request(url)
        if 'error' in response:
            return []
        
        return response.get('data', [])
    
    def get_tokocrypto_depth(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book depth from Tokocrypto API"""
        toko_symbol = symbol.replace('_', '').upper()
        url = f"{self.tokocrypto_url}/market/depth?symbol={toko_symbol}&limit={limit}"
        
        return self.make_request(url)
    
    def analyze_order_book(self, depth_data: Dict) -> Dict:
        """Analyze order book for support/resistance and buy/sell pressure"""
        if 'error' in depth_data or not depth_data.get('data'):
            return {'buy_pressure': 0, 'sell_pressure': 0, 'support': 0, 'resistance': 0}
        
        data = depth_data['data']
        bids = data.get('bids', [])
        asks = data.get('asks', [])
        
        # Calculate buy/sell pressure
        total_bid_volume = sum(float(bid[1]) for bid in bids[:10])  # Top 10 bids
        total_ask_volume = sum(float(ask[1]) for ask in asks[:10])  # Top 10 asks
        
        buy_pressure = total_bid_volume / (total_bid_volume + total_ask_volume) * 100 if (total_bid_volume + total_ask_volume) > 0 else 50
        sell_pressure = 100 - buy_pressure
        
        # Find support and resistance levels
        support = float(bids[0][0]) if bids else 0  # Best bid
        resistance = float(asks[0][0]) if asks else 0  # Best ask
        
        return {
            'buy_pressure': round(buy_pressure, 2),
            'sell_pressure': round(sell_pressure, 2),
            'support': support,
            'resistance': resistance,
            'bid_volume': total_bid_volume,
            'ask_volume': total_ask_volume
        }
    
    def get_klines(self, symbol: str, interval: str = '1h', limit: int = 100) -> List:
        """Get kline data for technical analysis - increased to 100 for better accuracy"""
        binance_symbol = symbol.replace('_', '')
        url = f"{self.binance_url}/klines?symbol={binance_symbol}&interval={interval}&limit={limit}"
        
        response = self.make_request(url)
        if 'error' in response:
            return []
        
        return response if isinstance(response, list) else []
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator with proper smoothing"""
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
        
        # Use Wilder's smoothing for more accurate RSI
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return round(rsi, 2)
    
    def calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD indicator"""
        if len(prices) < slow + signal:
            return {"macd": 0, "signal": 0, "histogram": 0, "trend": "neutral"}
        
        # Calculate EMAs
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        
        # Calculate MACD signal line (EMA of MACD line)
        macd_values = []
        for i in range(len(prices) - slow + 1):
            if i == 0:
                macd_values.append(macd_line)
            else:
                # Simplified for this implementation
                macd_values.append(macd_line)
        
        signal_line = macd_line * 0.9  # Simplified signal line
        histogram = macd_line - signal_line
        
        # Determine trend
        if macd_line > signal_line and histogram > 0:
            trend = "bullish"
        elif macd_line < signal_line and histogram < 0:
            trend = "bearish"
        else:
            trend = "neutral"
        
        return {
            "macd": round(macd_line, 6),
            "signal": round(signal_line, 6),
            "histogram": round(histogram, 6),
            "trend": trend
        }
    
    def calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return sum(prices) / len(prices) if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period  # Start with SMA
        
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: int = 2) -> Dict:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            avg = sum(prices) / len(prices) if prices else 0
            return {"upper": avg, "middle": avg, "lower": avg, "position": 0.5}
        
        # Calculate SMA (middle band)
        middle = sum(prices[-period:]) / period
        
        # Calculate standard deviation
        variance = sum((x - middle) ** 2 for x in prices[-period:]) / period
        std_dev_value = variance ** 0.5
        
        upper = middle + (std_dev * std_dev_value)
        lower = middle - (std_dev * std_dev_value)
        
        # Calculate position within bands (0 = lower band, 1 = upper band)
        current_price = prices[-1]
        if upper != lower:
            position = (current_price - lower) / (upper - lower)
        else:
            position = 0.5
        
        return {
            "upper": round(upper, 8),
            "middle": round(middle, 8),
            "lower": round(lower, 8),
            "position": round(position, 3)
        }
    
    def identify_support_resistance_v2(self, highs: List[float], lows: List[float], closes: List[float]) -> Dict:
        """Advanced support and resistance identification"""
        if len(highs) < 20:
            return {"support": min(lows) if lows else 0, "resistance": max(highs) if highs else 0}
        
        # Find significant highs and lows (pivot points)
        pivot_highs = []
        pivot_lows = []
        
        for i in range(2, len(highs) - 2):
            # Pivot high: higher than 2 candles before and after
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                pivot_highs.append(highs[i])
            
            # Pivot low: lower than 2 candles before and after
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                pivot_lows.append(lows[i])
        
        current_price = closes[-1]
        
        # Find resistance (nearest significant high above current price)
        resistance_levels = [h for h in pivot_highs if h > current_price]
        resistance = min(resistance_levels) if resistance_levels else max(highs)
        
        # Find support (nearest significant low below current price)
        support_levels = [l for l in pivot_lows if l < current_price]
        support = max(support_levels) if support_levels else min(lows)
        
        return {
            "support": support,
            "resistance": resistance,
            "pivot_highs": pivot_highs[-3:] if len(pivot_highs) >= 3 else pivot_highs,
            "pivot_lows": pivot_lows[-3:] if len(pivot_lows) >= 3 else pivot_lows
        }
    
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
    
    def analyze_symbol_advanced(self, symbol: str) -> Dict:
        """Advanced technical analysis with multiple indicators for maximum accuracy"""
        
        # Get comprehensive candlestick data
        klines = self.get_klines(symbol, '1h', 100)  # 100 hours of data for accuracy
        if not klines or len(klines) < 50:
            return None
        
        # Extract OHLCV data
        opens = [float(k[1]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        closes = [float(k[4]) for k in klines]
        volumes = [float(k[5]) for k in klines]
        
        current_price = closes[-1]
        
        # Calculate all technical indicators
        rsi = self.calculate_rsi(closes, 14)
        macd_data = self.calculate_macd(closes)
        bb_data = self.calculate_bollinger_bands(closes)
        sr_data = self.identify_support_resistance_v2(highs, lows, closes)
        
        # Calculate moving averages
        sma20 = self.calculate_sma(closes, 20)
        sma50 = self.calculate_sma(closes, 50)
        ema12 = self.calculate_ema(closes, 12)
        ema26 = self.calculate_ema(closes, 26)
        
        # Volume analysis
        avg_volume = sum(volumes[-20:]) / 20
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Support and resistance levels
        support = sr_data["support"]
        resistance = sr_data["resistance"]
        
        # === BEARISH TO BULLISH CRITERIA (Very Strict) ===
        
        signals = []
        score = 0
        bearish_confirmed = False
        bullish_reversal = False
        
        # 1. BEARISH CONFIRMATION (Price must be in bearish trend)
        price_below_ma = current_price < sma20 and current_price < sma50
        downtrend_confirmed = sma20 < sma50
        
        if price_below_ma and downtrend_confirmed:
            bearish_confirmed = True
            score += 20
            signals.append("✅ Bearish trend confirmed (Price below MA20 & MA50)")
        
        # 2. RSI OVERSOLD (Critical for reversal)
        if rsi <= 30:
            score += 35
            signals.append(f"🔥 RSI deeply oversold ({rsi:.1f}) - Strong reversal potential")
            bullish_reversal = True
        elif rsi <= 35:
            score += 25
            signals.append(f"⚡ RSI oversold ({rsi:.1f}) - Reversal signal")
            bullish_reversal = True
        elif rsi <= 40:
            score += 15
            signals.append(f"📈 RSI recovering from oversold ({rsi:.1f})")
        
        # 3. NEAR SUPPORT LEVEL (Critical entry zone)
        support_distance = ((current_price - support) / support * 100) if support > 0 else 100
        
        if support_distance <= 2:  # Within 2% of support
            score += 30
            signals.append(f"🎯 Price at strong support ({support_distance:.1f}% above)")
            bullish_reversal = True
        elif support_distance <= 5:  # Within 5% of support
            score += 20
            signals.append(f"📍 Price near support ({support_distance:.1f}% above)")
        
        # 4. MACD BULLISH DIVERGENCE
        if macd_data["trend"] == "bullish" and macd_data["histogram"] > 0:
            score += 25
            signals.append("🚀 MACD bullish crossover - Momentum building")
            bullish_reversal = True
        elif macd_data["histogram"] > 0:
            score += 15
            signals.append("📊 MACD histogram positive - Momentum improving")
        
        # 5. BOLLINGER BANDS POSITION
        bb_position = bb_data["position"]
        if bb_position <= 0.1:  # Near lower band
            score += 25
            signals.append("🎪 Price at Bollinger lower band - Oversold extreme")
            bullish_reversal = True
        elif bb_position <= 0.2:
            score += 15
            signals.append("📉 Price in lower Bollinger zone")
        
        # 6. VOLUME CONFIRMATION
        if volume_ratio > 2.0:
            score += 30
            signals.append(f"💥 High volume surge ({volume_ratio:.1f}x) - Strong interest")
        elif volume_ratio > 1.5:
            score += 20
            signals.append(f"📈 Increased volume ({volume_ratio:.1f}x)")
        
        # 7. CANDLESTICK PATTERNS (Last 5 candles)
        if len(closes) >= 5:
            # Bullish reversal pattern
            if closes[-1] > opens[-1] and closes[-2] <= opens[-2]:  # Green after red
                score += 20
                signals.append("🕯️ Bullish reversal candle pattern")
                bullish_reversal = True
            
            # Higher low formation
            if lows[-1] > lows[-3] and lows[-2] > lows[-4]:
                score += 15
                signals.append("📈 Higher lows forming - Uptrend starting")
        
        # === PROFIT POTENTIAL CALCULATION ===
        profit_to_resistance = ((resistance - current_price) / current_price * 100) if resistance > current_price else 0
        
        # Target profit 5-10% filter
        if 5 <= profit_to_resistance <= 10:
            score += 30
            signals.append(f"🎯 Perfect profit target: {profit_to_resistance:.1f}% to resistance")
        elif 3 <= profit_to_resistance <= 15:
            score += 20
            signals.append(f"💰 Good profit potential: {profit_to_resistance:.1f}% to resistance")
        elif profit_to_resistance > 15:
            score += 10  # Too high might be risky
            signals.append(f"⚠️ High profit potential: {profit_to_resistance:.1f}% (high risk)")
        
        # === RELAXED FILTERING FOR MORE RESULTS ===
        # Requirements: At least some bearish + oversold signals
        
        # Require at least some bearish indication OR oversold condition
        basic_requirement = (
            (price_below_ma or rsi <= 45) and  # Either below MA or moderately oversold
            (bullish_reversal or rsi <= 40 or support_distance <= 15)  # Some reversal signal
        )
        
        if not basic_requirement:
            return None
        
        # Lower minimum score for more results
        if score < 40:  # Reduced from 70 to 40
            return None
        
        # === PRECISE ENTRY & EXIT LEVELS ===
        
        # Entry: Slightly above current price for confirmation
        entry_price = current_price * 1.002  # 0.2% above current
        
        # Stop Loss: Below support with buffer
        stop_loss = support * 0.98  # 2% below support
        
        # Take Profit: Conservative target to resistance
        take_profit = min(resistance * 0.98, entry_price * 1.08)  # Max 8% gain or near resistance
        
        # Risk/Reward calculation
        risk_amount = entry_price - stop_loss
        reward_amount = take_profit - entry_price
        risk_percent = (risk_amount / entry_price) * 100
        reward_percent = (reward_amount / entry_price) * 100
        risk_reward_ratio = reward_percent / risk_percent if risk_percent > 0 else 0
        
        # Relaxed risk/reward filter - accept lower ratios for more opportunities
        if risk_reward_ratio < 1.2:  # Reduced from 2:1 to 1.2:1
            return None
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'entry_price': round(entry_price, 8),
            'take_profit': round(take_profit, 8),
            'stop_loss': round(stop_loss, 8),
            'profit_potential': round(reward_percent, 2),
            'risk_percent': round(risk_percent, 2),
            'risk_reward_ratio': round(risk_reward_ratio, 2),
            'signal_strength': min(score, 100),
            'technical_analysis': {
                'rsi': rsi,
                'macd': macd_data,
                'bollinger_bands': bb_data,
                'support': support,
                'resistance': resistance,
                'sma20': sma20,
                'sma50': sma50,
                'volume_ratio': round(volume_ratio, 2),
                'bearish_confirmed': bearish_confirmed,
                'bullish_reversal': bullish_reversal
            },
            'trading_signals': signals,
            'trading_plan': f"🎯 ENTRY: {entry_price:.8f} | 🎪 TP: {take_profit:.8f} (+{reward_percent:.1f}%) | 🛡️ SL: {stop_loss:.8f} (-{risk_percent:.1f}%) | R/R: {risk_reward_ratio:.1f}:1"
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
    
    def screen_bearish_to_bullish_advanced(self, quote_currency: str = 'USDT', limit: int = 999) -> Dict:
        """
        Advanced screener for bearish-to-bullish reversal opportunities
        Uses comprehensive technical analysis for maximum accuracy
        """
        start_time = time.time()
        
        print(f"🔍 Starting ADVANCED bearish-to-bullish screening...")
        print(f"📊 Quote Currency: {quote_currency}")
        print(f"🎯 Analysis Limit: {limit}")
        
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
                symbol_name = symbol.get('symbol', symbol.get('baseAsset', '') + quote_asset)
                filtered_symbols.append(symbol_name)
        
        # Limit analysis
        if limit > 0 and limit < 999:
            filtered_symbols = filtered_symbols[:limit]
        
        print(f"📈 Analyzing {len(filtered_symbols)} symbols with advanced technical analysis...")
        
        results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=5) as executor:  # Reduced for stability
            # Submit tasks
            future_to_symbol = {
                executor.submit(self.analyze_symbol_advanced, symbol): symbol 
                for symbol in filtered_symbols
            }
            
            # Collect results
            analyzed_count = 0
            for future in as_completed(future_to_symbol):
                try:
                    result = future.result(timeout=45)  # Increased timeout for advanced analysis
                    if result:
                        results.append(result)
                        print(f"✅ Found: {result['symbol']} (Score: {result['signal_strength']}, Profit: {result['profit_potential']}%)")
                    
                    analyzed_count += 1
                    if analyzed_count % 25 == 0:  # Progress every 25 symbols
                        print(f"🔄 Progress: {analyzed_count}/{len(filtered_symbols)} analyzed, {len(results)} candidates found...")
                        
                except Exception as e:
                    print(f"❌ Error analyzing symbol: {e}")
        
        # Sort by profit potential (highest first), then by signal strength
        results.sort(key=lambda x: (x['profit_potential'], x['signal_strength']), reverse=True)
        
        execution_time = time.time() - start_time
        
        print(f"🏁 Analysis Complete!")
        print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
        print(f"📊 Total Analyzed: {len(filtered_symbols)}")
        print(f"🎯 High-Quality Candidates Found: {len(results)}")
        
        return {
            'status': 'success',
            'analysis_type': 'advanced_bearish_to_bullish',
            'data': results,
            'execution_time': round(execution_time, 2),
            'total_analyzed': len(filtered_symbols),
            'high_quality_candidates': len(results),
            'quote_currency': quote_currency,
            'criteria': {
                'bearish_trend_or_oversold': 'Price below MA OR RSI ≤ 45',
                'reversal_signals': 'RSI ≤ 40 OR near support OR bullish patterns',
                'min_score': 40,  # Reduced for more results
                'min_risk_reward': '1.2:1',  # More realistic
                'profit_target': 'Any positive potential'
            }
        }

    def detect_dramatic_drops(self, quote_currency: str = 'USDT', limit: int = 900) -> Dict:
        """
        Deteksi coins yang turun drastis seperti gambar - menggunakan candlestick analysis
        Mencari pola: High -> Low dengan RSI drop dan volume spike
        """
        print(f"🔍 DETECTING DRAMATIC DROPS untuk {quote_currency} pairs...")
        print(f"📉 Mencari coins yang turun drastis seperti pattern di gambar...")
        start_time = time.time()
        
        # Get symbols
        all_symbols = self.get_symbols()
        if not all_symbols:
            return {'status': 'error', 'message': 'Failed to fetch symbols', 'data': []}
        
        # Filter symbols
        filtered_symbols = []
        for symbol in all_symbols:
            symbol_name = symbol.get('symbol', '').upper()
            if symbol_name.endswith(f'_{quote_currency}') or (symbol_name.endswith(quote_currency) and not symbol_name.endswith(f'_{quote_currency}')):
                if len(symbol_name) > len(quote_currency):
                    filtered_symbols.append(symbol_name)
        
        if limit and limit < len(filtered_symbols):
            filtered_symbols = filtered_symbols[:limit]
        
        print(f"📊 Analyzing {len(filtered_symbols)} symbols for dramatic drops...")
        
        results = []
        
        def detect_drop_pattern(symbol):
            try:
                # Get longer timeframe for better pattern detection
                klines = self.get_klines(symbol, '1h', 168)  # 7 days of hourly data
                if not klines or len(klines) < 50:
                    return None
                
                # Extract OHLCV data
                opens = [float(k[1]) for k in klines]
                highs = [float(k[2]) for k in klines]
                lows = [float(k[3]) for k in klines]
                closes = [float(k[4]) for k in klines]
                volumes = [float(k[5]) for k in klines]
                
                current_price = closes[-1]
                
                # Calculate RSI for multiple periods
                rsi_current = self.calculate_rsi(closes[-14:], 14) if len(closes) >= 14 else 50
                rsi_24h_ago = self.calculate_rsi(closes[-38:-24], 14) if len(closes) >= 38 else 50
                rsi_7d_ago = self.calculate_rsi(closes[-182:-168], 14) if len(closes) >= 182 else 50
                
                # Find recent high and low points
                recent_high_24h = max(highs[-24:]) if len(highs) >= 24 else current_price
                recent_low_24h = min(lows[-24:]) if len(lows) >= 24 else current_price
                recent_high_7d = max(highs[-168:]) if len(highs) >= 168 else current_price
                recent_low_7d = min(lows[-168:]) if len(lows) >= 168 else current_price
                
                # Calculate dramatic drop metrics
                drop_from_24h_high = ((recent_high_24h - current_price) / recent_high_24h) * 100 if recent_high_24h > 0 else 0
                drop_from_7d_high = ((recent_high_7d - current_price) / recent_high_7d) * 100 if recent_high_7d > 0 else 0
                
                # Volume analysis
                avg_volume_7d = sum(volumes[-168:]) / 168 if len(volumes) >= 168 else 1
                recent_volume_spike = max(volumes[-24:]) / avg_volume_7d if avg_volume_7d > 0 else 1
                
                # Pattern scoring - focus pada dramatic drops
                pattern_score = 0
                signals = []
                
                # 1. DRAMATIC DROP DETECTION (seperti gambar)
                if drop_from_24h_high >= 8:  # Drop 8%+ dalam 24h
                    pattern_score += 50
                    signals.append(f"🔴 DRAMATIC 24h DROP: -{drop_from_24h_high:.1f}% from high!")
                elif drop_from_24h_high >= 5:
                    pattern_score += 30
                    signals.append(f"📉 Significant 24h drop: -{drop_from_24h_high:.1f}%")
                
                if drop_from_7d_high >= 15:  # Drop 15%+ dalam 7 hari
                    pattern_score += 40
                    signals.append(f"🔴 MASSIVE 7d DROP: -{drop_from_7d_high:.1f}% from weekly high!")
                elif drop_from_7d_high >= 10:
                    pattern_score += 25
                    signals.append(f"📉 Big 7d drop: -{drop_from_7d_high:.1f}%")
                
                # 2. RSI PATTERN (dari tinggi ke rendah)
                rsi_drop = rsi_7d_ago - rsi_current if rsi_7d_ago > rsi_current else 0
                if rsi_drop >= 30:  # RSI turun 30+ points
                    pattern_score += 35
                    signals.append(f"📊 MASSIVE RSI drop: {rsi_7d_ago:.1f} → {rsi_current:.1f} (-{rsi_drop:.1f})")
                elif rsi_drop >= 20:
                    pattern_score += 25
                    signals.append(f"📊 Big RSI drop: {rsi_7d_ago:.1f} → {rsi_current:.1f}")
                
                if rsi_current <= 30:  # Now oversold
                    pattern_score += 25
                    signals.append(f"🔥 RSI now oversold: {rsi_current:.1f}")
                elif rsi_current <= 40:
                    pattern_score += 15
                    signals.append(f"⚡ RSI approaching oversold: {rsi_current:.1f}")
                
                # 3. VOLUME SPIKE (panic selling)
                if recent_volume_spike >= 3:
                    pattern_score += 30
                    signals.append(f"💥 VOLUME SPIKE: {recent_volume_spike:.1f}x average (panic selling!)")
                elif recent_volume_spike >= 2:
                    pattern_score += 20
                    signals.append(f"📈 High volume: {recent_volume_spike:.1f}x average")
                
                # 4. NEAR RECENT LOW (testing support)
                distance_from_low = ((current_price - recent_low_24h) / recent_low_24h) * 100 if recent_low_24h > 0 else 100
                if distance_from_low <= 2:
                    pattern_score += 25
                    signals.append(f"💪 Very close to 24h low ({distance_from_low:.1f}% above)")
                elif distance_from_low <= 5:
                    pattern_score += 15
                    signals.append(f"💪 Near 24h low ({distance_from_low:.1f}% above)")
                
                # 5. REVERSAL POTENTIAL
                potential_bounce = ((recent_high_24h - current_price) / current_price) * 100 if current_price > 0 else 0
                if potential_bounce >= 15:
                    pattern_score += 20
                    signals.append(f"🎯 HUGE bounce potential: +{potential_bounce:.1f}% to 24h high")
                elif potential_bounce >= 10:
                    pattern_score += 15
                    signals.append(f"🎯 Good bounce potential: +{potential_bounce:.1f}%")
                
                # Hanya return yang benar-benar dramatic drops
                if pattern_score < 50:  # Minimum score untuk dramatic pattern
                    return None
                
                # Get ticker data untuk info tambahan
                ticker_data = self.get_ticker_data(symbol)
                price_change_24h = float(ticker_data.get('priceChangePercent', 0)) if 'error' not in ticker_data else 0
                
                return {
                    'symbol': symbol,
                    'current_price': current_price,
                    'pattern_score': pattern_score,
                    'drop_24h_high': round(drop_from_24h_high, 2),
                    'drop_7d_high': round(drop_from_7d_high, 2),
                    'rsi_current': round(rsi_current, 1),
                    'rsi_drop': round(rsi_drop, 1),
                    'volume_spike': round(recent_volume_spike, 1),
                    'bounce_potential': round(potential_bounce, 1),
                    'distance_from_low': round(distance_from_low, 1),
                    'price_change_24h': round(price_change_24h, 1),
                    'pattern_type': 'DRAMATIC_DROP',
                    'levels': {
                        'recent_high_24h': recent_high_24h,
                        'recent_low_24h': recent_low_24h,
                        'recent_high_7d': recent_high_7d,
                        'recent_low_7d': recent_low_7d
                    },
                    'signals': signals,
                    'trading_plan': f"DRAMATIC DROP: Current {current_price:.8f} | 24h High: {recent_high_24h:.8f} | Potential: +{potential_bounce:.1f}%"
                }
                
            except Exception as e:
                print(f"❌ Error analyzing {symbol}: {str(e)}")
                return None
        
        # Process with threading
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {executor.submit(detect_drop_pattern, symbol): symbol 
                              for symbol in filtered_symbols}
            
            analyzed_count = 0
            for future in as_completed(future_to_symbol):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        print(f"🔴 FOUND DRAMATIC DROP: {result['symbol']} | Drop: -{result['drop_24h_high']:.1f}% | RSI: {result['rsi_current']:.1f} | Score: {result['pattern_score']}")
                    
                    analyzed_count += 1
                    if analyzed_count % 100 == 0:
                        print(f"🔄 Progress: {analyzed_count}/{len(filtered_symbols)} analyzed, {len(results)} dramatic drops found...")
                        
                except Exception as e:
                    print(f"❌ Error in future: {e}")
        
        # Sort by pattern score (most dramatic first)
        results.sort(key=lambda x: x['pattern_score'], reverse=True)
        
        # Return top 15 dramatic drops
        top_results = results[:15]
        
        execution_time = time.time() - start_time
        
        print(f"\n🔴 DRAMATIC DROP DETECTION COMPLETE")
        print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
        print(f"📊 Total Analyzed: {len(filtered_symbols)}")
        print(f"🔴 Dramatic Drops Found: {len(results)}")
        print(f"🏆 Top 15 Returned")
        
        if top_results:
            print(f"\n🏆 TOP DRAMATIC DROPS (LIKE YOUR CHART):")
            for i, coin in enumerate(top_results, 1):
                print(f"{i}. {coin['symbol']}: -{coin['drop_24h_high']:.1f}% | RSI {coin['rsi_current']:.1f} | Vol: {coin['volume_spike']:.1f}x | Score: {coin['pattern_score']}")
        
        return {
            'status': 'success',
            'analysis_type': 'dramatic_drop_detection',
            'data': top_results,
            'execution_time': round(execution_time, 2),
            'total_analyzed': len(filtered_symbols),
            'dramatic_drops_found': len(results),
            'top_drops_returned': len(top_results),
            'quote_currency': quote_currency,
            'criteria': {
                'pattern_focus': 'Dramatic drops like chart example (high to low crash)',
                'drop_requirements': '8%+ drop in 24h OR 15%+ drop in 7d',
                'rsi_pattern': 'RSI falling from high to low (oversold preferred)',
                'volume_requirement': 'Volume spike during drop (panic selling)',
                'scoring_system': 'Pattern-based scoring focusing on dramatic movements',
                'minimum_score': '50+ points for confirmed dramatic drops',
                'result_limit': 'Top 15 most dramatic drops',
                'sorted_by': 'Pattern score (most dramatic first)'
            }
        }

    def accurate_bullish_analysis(self, quote_currency: str = 'USDT', limit: int = 900) -> Dict:
        """
        Deteksi pola OVERSOLD REVERSAL seperti di chart:
        - Harga sudah turun drastis (minimal -15% dalam 7 hari)
        - RSI oversold (≤ 35, ideal 20-30)
        - Price sudah mendekati support level
        - Volume meningkat saat turun (kapitulasi)
        - Ready untuk bounce/reversal bullish
        - Pattern: Steep decline + oversold RSI + support test
        """
        print(f"🎯 Mencari pola OVERSOLD REVERSAL untuk {quote_currency} pairs...")
        print(f"📊 Target: Coins yang sudah turun drastis dan RSI oversold seperti gambar...")
        start_time = time.time()
        
        # Get symbols
        all_symbols = self.get_symbols()
        if not all_symbols:
            return {'status': 'error', 'message': 'Failed to fetch symbols', 'data': []}
        
        # Filter by quote currency and status
        filtered_symbols = []
        for symbol in all_symbols:
            symbol_name = symbol.get('symbol', '').upper()
            status = symbol.get('status', '')
            
            # Debug: Print first few symbols to understand the data structure
            if len(filtered_symbols) == 0 and len(all_symbols) > 0:
                print(f"🔍 Sample symbol data: {symbol}")
            
            # Check if symbol ends with the quote currency
            if symbol_name.endswith(f'_{quote_currency}'):
                # Check status - try different possible status values
                if status in ['TRADING', 'trading', 'Trading', 'ACTIVE', 'active', 'Active']:
                    filtered_symbols.append(symbol_name)
                elif not status:  # If no status field, assume trading
                    filtered_symbols.append(symbol_name)
            elif symbol_name.endswith(quote_currency) and not symbol_name.endswith(f'_{quote_currency}'):
                # Handle Binance format (BTCUSDT) if also present
                if len(symbol_name) > len(quote_currency):  # Ensure it's not just the quote currency itself
                    if status in ['TRADING', 'trading', 'Trading', 'ACTIVE', 'active', 'Active']:
                        filtered_symbols.append(symbol_name)
                    elif not status:
                        filtered_symbols.append(symbol_name)
        
        if limit and limit < len(filtered_symbols):
            filtered_symbols = filtered_symbols[:limit]
        
        print(f"📊 Found {len(filtered_symbols)} {quote_currency} pairs to analyze...")
        print(f"🔍 First 10 symbols: {filtered_symbols[:10]}")
        print(f"🎯 Limit set to: {limit}")
        print(f"📈 Will analyze: {min(len(filtered_symbols), limit) if limit else len(filtered_symbols)} symbols")
        
        if len(filtered_symbols) == 0:
            print(f"⚠️ No {quote_currency} pairs found! Available symbols format might be different.")
            # Debug: show first 10 symbols to understand format and status
            sample_symbols = []
            for i, symbol in enumerate(all_symbols[:10]):
                symbol_name = symbol.get('symbol', 'N/A')
                status = symbol.get('status', 'N/A')
                sample_symbols.append(f"{symbol_name} (status: {status})")
            
            print(f"Sample symbols from API: {sample_symbols}")
            
            # Count symbols by quote currency to help debug
            quote_counts = {}
            for symbol in all_symbols:
                symbol_name = symbol.get('symbol', '')
                if '_' in symbol_name:
                    quote = symbol_name.split('_')[-1]
                    quote_counts[quote] = quote_counts.get(quote, 0) + 1
            
            print(f"Quote currency distribution: {dict(sorted(quote_counts.items(), key=lambda x: x[1], reverse=True)[:10])}")
            
            return {
                'status': 'error',
                'message': f'No {quote_currency} trading pairs found. Check symbol format.',
                'data': [],
                'debug_info': {
                    'total_symbols_from_api': len(all_symbols),
                    'sample_symbols_with_status': sample_symbols,
                    'quote_currency_searched': quote_currency,
                    'top_quote_currencies': quote_counts
                }
            }
        
        results = []
        processed_count = 0
        
        def analyze_symbol_accurate(symbol):
            nonlocal processed_count
            try:
                # Get enhanced market data from multiple sources
                ticker_data = self.get_ticker_data(symbol)
                if 'error' in ticker_data:
                    print(f"⚠️ Failed to get ticker data for {symbol}: {ticker_data.get('error')}")
                    return None
                
                # Try to get Tokocrypto data, but don't fail if unavailable
                toko_ticker = self.get_tokocrypto_ticker(symbol)
                depth_data = self.get_tokocrypto_depth(symbol)
                recent_trades = self.get_tokocrypto_trades(symbol, 50)
                
                current_price = float(ticker_data.get('lastPrice', 0))
                if current_price <= 0:
                    print(f"⚠️ Invalid price for {symbol}: {current_price}")
                    return None
                
                # Enhanced volume analysis
                volume = float(ticker_data.get('volume', 0))
                quote_volume = float(ticker_data.get('quoteVolume', 0))
                price_change_24h = float(ticker_data.get('priceChangePercent', 0))
                
                # Volume filter - HAPUS untuk analisis semua coins
                # if quote_volume < 10000:  # DISABLED - analyze all coins
                #     return None
                
                # Analyze order book for buy/sell pressure (with fallback)
                order_book_analysis = self.analyze_order_book(depth_data)
                if not order_book_analysis or order_book_analysis.get('support', 0) == 0:
                    # Fallback order book analysis if API fails
                    order_book_analysis = {
                        'buy_pressure': 50.0,  # Neutral
                        'sell_pressure': 50.0,
                        'support': current_price * 0.95,  # 5% below current price
                        'resistance': current_price * 1.05,  # 5% above current price
                        'bid_volume': 0,
                        'ask_volume': 0
                    }
                
                # Get kline data for technical analysis
                klines = self.get_klines(symbol, '1h', 50)
                if not klines or len(klines) < 20:
                    return None
                
                closes = [float(k[4]) for k in klines]
                highs = [float(k[2]) for k in klines]
                lows = [float(k[3]) for k in klines]
                volumes = [float(k[5]) for k in klines]
                
                # Technical indicators
                rsi = self.calculate_rsi(closes)
                macd_line, signal_line = self.calculate_macd(closes)
                sma_20 = sum(closes[-20:]) / 20
                
                # === ANALISIS SEMUA COINS - HAPUS SEMUA FILTER ===
                # 1. RSI Filter - DISABLED untuk analisis semua
                # if rsi > 50:  # DISABLED
                #     return None
                
                # 2. Price decline filter - DISABLED untuk analisis semua
                price_7d_ago = closes[-min(168, len(closes))]  # 7 hari = 168 jam
                price_decline_7d = ((current_price - price_7d_ago) / price_7d_ago) * 100
                
                # if price_decline_7d > -3:  # DISABLED
                #     return None
                
                # 3. Volume sudah difilter di atas, skip filter kedua
                
                # 4. Skip support distance filter untuk sementara - terlalu ketat
                recent_low = min(lows[-24:])  # Low 24 jam terakhir
                # if current_price > recent_low * 1.15:  # Disabled temporarily
                #     return None
                
                # 5. MACD untuk konfirmasi momentum (bonus points)
                macd_improving = False
                if len(macd_line) >= 2 and len(signal_line) >= 2:
                    macd_improving = (macd_line[-1] > signal_line[-1] or 
                                    macd_line[-1] > macd_line[-2])
                
                # Enhanced support and resistance using order book + price action
                support = min(order_book_analysis['support'], recent_low)
                resistance = max(order_book_analysis['resistance'], max(highs[-10:]))
                
                # Entry dan target untuk oversold reversal
                entry_level = current_price * 1.002  # Entry sedikit di atas current (wait for confirmation)
                
                # Target berdasarkan resistance terdekat atau 15-25% profit
                target_resistance = resistance
                target_percentage = current_price * 1.20  # 20% profit target
                take_profit = min(target_resistance, target_percentage)
                
                # Calculate potential profit
                potential_profit = ((take_profit - entry_level) / entry_level) * 100
                
                # Minimum profit potential - DISABLED untuk analisis semua
                # if potential_profit < 2:  # DISABLED
                #     return None
                
                # === SCORING UNTUK SEMUA COINS - FOKUS PATTERN TURUN LALU NAIK ===
                score = 0
                
                # 1. RSI Oversold scoring (turun = bagus untuk reversal)
                if rsi <= 20:
                    score += 50  # Extremely oversold - perfect for reversal
                elif rsi <= 30:
                    score += 40  # Very oversold - excellent  
                elif rsi <= 40:
                    score += 30  # Oversold - very good
                elif rsi <= 50:
                    score += 20  # Still good for reversal
                elif rsi <= 60:
                    score += 10  # Neutral
                else:
                    score += 5   # Overbought (not ideal but still analyze)
                
                # 2. Price decline scoring (semakin turun = semakin berpotensi naik)
                if price_decline_7d <= -50:
                    score += 40  # Massive crash - huge reversal potential
                elif price_decline_7d <= -30:
                    score += 35  # Big drop - high reversal potential
                elif price_decline_7d <= -20:
                    score += 30  # Significant drop - good reversal
                elif price_decline_7d <= -10:
                    score += 25  # Moderate drop - some reversal potential
                elif price_decline_7d <= -5:
                    score += 15  # Small drop - minimal reversal
                elif price_decline_7d <= 0:
                    score += 10  # Sideways - neutral
                else:
                    score += 5   # Rising (but we still analyze for momentum)
                
                # 3. Support proximity scoring (dekat support = bagus untuk bounce)
                support_distance = ((current_price - recent_low) / recent_low) * 100 if recent_low > 0 else 100
                if support_distance <= 1:
                    score += 25  # Very close to support - perfect bounce zone
                elif support_distance <= 3:
                    score += 20  # Close to support
                elif support_distance <= 5:
                    score += 15  # Near support
                elif support_distance <= 10:
                    score += 10  # Approaching support
                else:
                    score += 5   # Away from support but still analyze
                
                # 4. Volume analysis (high volume during decline = capitulation)
                avg_volume = sum(volumes[-10:]) / 10 if len(volumes) >= 10 else 1
                current_volume = volumes[-1] if volumes else 1
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                if volume_ratio > 5:
                    score += 25  # Massive volume surge - likely capitulation
                elif volume_ratio > 3:
                    score += 20  # High volume surge
                elif volume_ratio > 2:
                    score += 15  # Increased volume
                elif volume_ratio > 1.5:
                    score += 10  # Above average volume
                else:
                    score += 5   # Normal/low volume but still analyze
                
                # 5. MACD momentum (early bullish signal)
                if macd_improving:
                    score += 20  # Momentum turning bullish
                else:
                    score += 5   # No momentum but still analyze
                
                # 6. Profit potential scoring (realistic targets)
                if potential_profit >= 25:
                    score += 20  # Excellent profit potential
                elif potential_profit >= 15:
                    score += 15  # Good profit potential
                elif potential_profit >= 10:
                    score += 12  # Decent profit
                elif potential_profit >= 5:
                    score += 8   # Small profit
                else:
                    score += 5   # Any profit still analyzed
                
                # 7. Bonus untuk pattern yang ideal (turun drastis + oversold + dekat support)
                if rsi <= 35 and price_decline_7d <= -15 and support_distance <= 5:
                    score += 30  # Perfect reversal pattern bonus
                
                # Minimum score - DISABLED untuk analisis semua
                # if score < 25:  # DISABLED
                #     return None
                
                # Calculate stop loss (below recent low)
                stop_loss = recent_low * 0.97  # 3% below recent low
                risk_reward = potential_profit / (((entry_level - stop_loss) / entry_level) * 100) if entry_level > stop_loss else 1
                
                # Generate OVERSOLD REVERSAL signals
                signals = []
                
                # RSI signals
                if rsi <= 20:
                    signals.append(f"🔥 RSI extremely oversold at {rsi:.1f} - PERFECT reversal setup!")
                elif rsi <= 25:
                    signals.append(f"🔥 RSI very oversold at {rsi:.1f} - excellent reversal potential")
                elif rsi <= 30:
                    signals.append(f"🔥 RSI oversold at {rsi:.1f} - strong reversal candidate")
                elif rsi <= 35:
                    signals.append(f"⚡ RSI at {rsi:.1f} - oversold zone, good for reversal")
                
                # Price decline signals
                if price_decline_7d <= -30:
                    signals.append(f"📉 MASSIVE 7-day drop: {price_decline_7d:.1f}% - high reversal potential!")
                elif price_decline_7d <= -20:
                    signals.append(f"📉 Big 7-day drop: {price_decline_7d:.1f}% - good reversal setup")
                elif price_decline_7d <= -15:
                    signals.append(f"📉 Significant 7-day decline: {price_decline_7d:.1f}%")
                elif price_decline_7d <= -10:
                    signals.append(f"📉 7-day decline: {price_decline_7d:.1f}% - testing support")
                
                # Support test signals
                if support_distance <= 1:
                    signals.append(f"💪 At support level - ready to bounce ({support_distance:.1f}% above low)")
                elif support_distance <= 2:
                    signals.append(f"💪 Very close to support ({support_distance:.1f}% above low)")
                elif support_distance <= 3:
                    signals.append(f"💪 Near support level ({support_distance:.1f}% above low)")
                
                # Volume signals (kapitulasi)
                if current_volume > avg_volume * 3:
                    signals.append(f"🌊 MASSIVE volume surge: {current_volume/avg_volume:.1f}x - possible capitulation!")
                elif current_volume > avg_volume * 2:
                    signals.append(f"🌊 High volume surge: {current_volume/avg_volume:.1f}x - strong selling pressure")
                elif current_volume > avg_volume * 1.5:
                    signals.append(f"📊 Above average volume: {current_volume/avg_volume:.1f}x")
                
                # MACD momentum
                if macd_improving:
                    signals.append("📈 MACD showing early bullish momentum - good timing!")
                
                # Profit potential
                if potential_profit >= 20:
                    signals.append(f"🎯 Excellent profit target: +{potential_profit:.1f}%")
                elif potential_profit >= 15:
                    signals.append(f"🎯 Great profit potential: +{potential_profit:.1f}%")
                elif potential_profit >= 10:
                    signals.append(f"🎯 Good profit target: +{potential_profit:.1f}%")
                elif potential_profit >= 8:
                    signals.append(f"🎯 Decent profit potential: +{potential_profit:.1f}%")
                
                # Risk/reward ratio
                if risk_reward >= 3:
                    signals.append(f"⚖️ Excellent risk/reward: {risk_reward:.1f}:1")
                elif risk_reward >= 2:
                    signals.append(f"⚖️ Good risk/reward: {risk_reward:.1f}:1")
                
                processed_count += 1
                if processed_count % 50 == 0:  # Progress every 50 symbols  
                    print(f"🔄 Progress: {processed_count}/{len(filtered_symbols)} analyzed, found {len(results)} candidates so far...")
                
                print(f"✅ {symbol}: Score {score} | RSI {rsi:.1f} | 7d: {price_decline_7d:.1f}% | Profit: +{potential_profit:.1f}%")
                
                return {
                    'symbol': symbol,
                    'current_price': current_price,
                    'rsi': round(rsi, 2),
                    'signal_strength': score,
                    'entry_level': round(entry_level, 8),
                    'take_profit': round(take_profit, 8),
                    'potential_profit': round(potential_profit, 1),
                    'stop_loss': round(stop_loss, 8),
                    'price_decline_7d': round(price_decline_7d, 1),
                    'support_distance': round(support_distance, 1),
                    'volume_ratio': round(current_volume / avg_volume, 2),
                    'trading_plan': f"OVERSOLD REVERSAL: Entry {entry_level:.8f}, TP {take_profit:.8f} (+{potential_profit:.1f}%), SL {stop_loss:.8f}",
                    'pattern_type': 'OVERSOLD_REVERSAL',
                    'analysis': {
                        'score': score,
                        'rsi': round(rsi, 2),
                        'price_decline_7d': round(price_decline_7d, 1),
                        'support_distance': round(support_distance, 1),
                        'sma20': round(sma_20, 8),
                        'support': round(support, 8),
                        'resistance': round(resistance, 8),
                        'recent_low': round(recent_low, 8),
                        'volume_ratio': round(current_volume / avg_volume, 2),
                        'macd_improving': macd_improving,
                        'price_change_24h': round(price_change_24h, 2),
                        'risk_reward': round(risk_reward, 1),
                        'buy_pressure': order_book_analysis['buy_pressure'],
                        'sell_pressure': order_book_analysis['sell_pressure'],
                        'signals': signals
                    }
                }
                
            except Exception as e:
                print(f"❌ Error analyzing {symbol}: {str(e)}")
                return None
        
        # Process symbols with threading for speed
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {executor.submit(analyze_symbol_accurate, symbol): symbol 
                              for symbol in filtered_symbols}
            
            for future in as_completed(future_to_symbol):
                result = future.result()
                if result:
                    results.append(result)
        
        # Sort by signal strength (highest score first) - yang paling berpotensi naik
        results.sort(key=lambda x: x['signal_strength'], reverse=True)
        
        # LIMIT HASIL HANYA TOP 10 TERBAIK
        top_results = results[:10]
        
        execution_time = time.time() - start_time
        
        print(f"\n🔥 ANALISIS SEMUA COINS COMPLETE")
        print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
        print(f"📊 Total Analyzed: {len(filtered_symbols)}")
        print(f"🎯 Total Candidates Found: {len(results)}")
        print(f"🏆 Top 10 Best Candidates Returned")
        
        if top_results:
            print(f"\n🏆 TOP 10 COINS (TURUN LALU BERPOTENSI NAIK):")
            for i, coin in enumerate(top_results, 1):
                print(f"{i}. {coin['symbol']}: Score {coin['signal_strength']} | RSI {coin['rsi']} | 7d: {coin['price_decline_7d']}% | Profit: +{coin['potential_profit']}%")
        else:
            print(f"\n❌ No candidates found after analyzing all coins")
        
        return {
            'status': 'success',
            'analysis_type': 'all_coins_reversal_analysis',
            'data': top_results,  # Hanya return top 10
            'execution_time': round(execution_time, 2),
            'total_analyzed': len(filtered_symbols),
            'total_candidates_found': len(results),
            'top_candidates_returned': len(top_results),
            'quote_currency': quote_currency,
            'criteria': {
                'pattern_type': 'ALL COINS ANALYZED - Top 10 reversal candidates',
                'filters': 'NO FILTERS - All coins analyzed regardless of RSI/decline/volume',
                'scoring_focus': 'Coins that dropped and ready to rise (oversold + decline + support)',
                'result_limit': 'Top 10 highest scoring candidates only',
                'scoring_system': 'Comprehensive 200+ point system',
                'sorted_by': 'Signal strength (reversal potential)',
                'api_sources': 'Binance (klines, 24hr ticker) + Tokocrypto (depth, trades)',
                'analysis_approach': 'Analyze ALL -> Score ALL -> Return TOP 10'
            }
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
