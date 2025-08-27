# Cryptocurrency Screener - cPanel Deployment Guide

## 📁 File Structure
```
screener/
├── app.py                      # Main FastAPI application
├── tokocrypto_screener.py      # Core screening engine
├── requirements.txt            # Python dependencies
├── passenger_wsgi.py           # cPanel WSGI entry point
└── static/
    ├── crypto_screening.html   # Web interface
    └── assets/                 # Additional assets (if needed)
```

## 🚀 cPanel Installation Steps

### 1. Upload Files to cPanel
1. **Login to cPanel** and go to **File Manager**
2. **Create a new folder** called `screener` in your domain's root directory
3. **Upload all files** from the `screener/` folder to this directory
4. **Set permissions** to 755 for the `screener` folder

### 2. Setup Python App in cPanel
1. Go to **Software** → **Python App** in cPanel
2. Click **Create Application**
3. Configure:
   - **Python version**: 3.8+ (choose highest available)
   - **Application root**: `/screener`
   - **Application URL**: `yourdomain.com/screener` or subdomain
   - **Application startup file**: `passenger_wsgi.py`
   - **Application Entry point**: `application`

### 3. Install Dependencies
1. After creating the Python app, click **Open Terminal** in cPanel Python App interface
2. Run the following commands:
```bash
pip install -r requirements.txt
```

### 4. Test Installation
1. Click **Restart** in Python App interface
2. Visit your application URL: `https://yourdomain.com/screener/static/crypto_screening.html`
3. Test the API endpoints:
   - Health check: `https://yourdomain.com/screener/`
   - API status: `https://yourdomain.com/screener/api/health`

## 🔧 Configuration Options

### Environment Variables (Optional)
Create `.env` file in screener folder:
```env
# API Configuration
BINANCE_API_URL=https://api.binance.com/api/v3
MAX_CONCURRENT_REQUESTS=50
TIMEOUT_SECONDS=30

# Logging
LOG_LEVEL=INFO
```

### Custom Domain Setup
If using subdomain (e.g., `crypto.yourdomain.com`):
1. Create subdomain in cPanel
2. Point subdomain to `/screener` folder
3. Update Python App URL accordingly

## 📊 Usage Guide

### Web Interface
1. **Access**: `https://yourdomain.com/screener/static/crypto_screening.html`
2. **Select currency**: IDR, USDT, or BTC
3. **Choose analysis limit**: 20-200 coins (or all)
4. **Click "Start Screening"** to begin analysis
5. **View results** in sortable table format

### API Endpoints
- **GET** `/` - Health check and welcome message
- **GET** `/api/health` - API health status
- **GET** `/api/screen` - Full screening analysis
  - `?quote_currency=IDR` (IDR/USDT/BTC)
  - `&limit=50` (analysis limit)
- **GET** `/api/analyze/{symbol}` - Single symbol analysis

### API Response Format
```json
{
  "status": "success",
  "data": [
    {
      "symbol": "BTCIDR",
      "current_price": 1420000000,
      "rsi": 32.5,
      "signal_strength": 7.2,
      "entry_level": 1400000000,
      "take_profit": 1540000000,
      "potential_profit": 10.5,
      "trading_plan": "Wait for RSI to drop below 30..."
    }
  ],
  "execution_time": 2.3
}
```

## 🔍 Features

### Technical Analysis
- **RSI Indicator**: Identifies oversold conditions (< 35)
- **Moving Averages**: SMA support/resistance levels
- **Price Action**: Support and resistance detection
- **Signal Strength**: Composite scoring system (1-10)

### Performance Optimizations
- **Async Processing**: Concurrent API calls using asyncio
- **Numpy Calculations**: Vectorized mathematical operations
- **Smart Caching**: Reduces redundant API calls
- **Resource Management**: Optimized for shared hosting

### Trading Features
- **Entry Points**: Calculated based on technical levels
- **Take Profit**: Conservative profit targets
- **Risk Assessment**: Signal strength indicators
- **Trading Plans**: Automated strategy suggestions

## 🛠️ Troubleshooting

### Common Issues

#### 1. Import Errors
**Error**: `ModuleNotFoundError: No module named 'fastapi'`
**Solution**: 
```bash
# In cPanel Python App terminal
pip install -r requirements.txt
pip list  # Verify installations
```

#### 2. Permission Issues
**Error**: `Permission denied`
**Solution**: Set folder permissions to 755 via File Manager

#### 3. API Timeout
**Error**: `Request timeout`
**Solution**: Reduce analysis limit or check server resources

#### 4. Memory Limits
**Error**: `Memory limit exceeded`
**Solution**: 
- Reduce concurrent requests in `tokocrypto_screener.py`
- Lower analysis limit in web interface
- Contact hosting provider for resource upgrade

### Performance Tuning

#### For Shared Hosting:
```python
# In tokocrypto_screener.py, reduce concurrent requests:
MAX_CONCURRENT_REQUESTS = 20  # Instead of 50

# In app.py, add timeout handling:
SCREENING_TIMEOUT = 60  # Seconds
```

#### Memory Optimization:
```python
# Process in smaller batches
BATCH_SIZE = 25
```

## 📈 Monitoring & Logs

### View Logs in cPanel
1. **Python App** → **View Logs**
2. **Error Logs** in cPanel main dashboard
3. **Access Logs** for request monitoring

### Health Monitoring
- API health endpoint: `/api/health`
- Returns system status and response times
- Monitor for consistent performance

## 🔄 Updates & Maintenance

### Updating the Application
1. **Upload new files** via File Manager
2. **Restart Python App** in cPanel
3. **Clear browser cache** for interface updates

### Regular Maintenance
- **Monitor API limits** from Binance
- **Check error logs** weekly
- **Update dependencies** monthly:
```bash
pip install --upgrade fastapi aiohttp numpy pandas
```

## 🔐 Security Best Practices

### Production Security
1. **API Rate Limiting**: Built into screener
2. **Input Validation**: All user inputs validated
3. **Error Handling**: Graceful error responses
4. **HTTPS Only**: Ensure SSL certificate active

### Monitoring Access
- **Access Logs**: Monitor unusual activity
- **Resource Usage**: Track CPU/memory consumption
- **API Calls**: Monitor Binance API usage

## 📞 Support

### If You Need Help
1. **Check logs** in cPanel Python App
2. **Verify dependencies** are installed
3. **Test API endpoints** directly
4. **Contact hosting support** for server issues

### Common Hosting Requirements
- **Python 3.8+**
- **500MB+ RAM** (recommended 1GB)
- **FastAPI support**
- **Outbound HTTPS** access to Binance API

---

**🎉 Congratulations!** Your crypto screener is now ready for production use on cPanel!

For questions or issues, check the application logs or contact your hosting provider.
