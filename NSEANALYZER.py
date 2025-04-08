#!/usr/bin/env python
# NSE Stock Analyzer with Supertrend, RSI, Telegram Notifications and Web Service
# Enhanced with Weekly/Monthly Analysis, HTML Tables and Debug Information

import os
import sys
import subprocess
import time
from datetime import datetime, timezone, timedelta
import json
import threading
import logging
import concurrent.futures
from functools import lru_cache
import math
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NSEANALYZER")

# Global variable to store diagnostics and errors
diagnostics = {
    "api_status": "Unknown",
    "errors": [],
    "timezone_check": None,
    "package_status": {},
    "last_error_time": None
}

# Function to install required packages - Only for local development
def install_required_packages():
    required_packages = [
        'pandas', 'numpy', 'requests', 'nsepy', 'nsetools', 
        'schedule', 'python-telegram-bot==13.7', 'tabulate', 'flask', 'pytz',
        'yfinance'  # Added as a fallback data source
    ]
    
    logger.info("Checking and installing required packages...")
    for package in required_packages:
        try:
            if '==' in package:
                module_name = package.split('==')[0]
            else:
                module_name = package
            __import__(module_name)
            logger.info(f"{package} is already installed.")
            diagnostics["package_status"][package] = "Installed"
        except ImportError:
            logger.info(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logger.info(f"{package} has been installed.")
                diagnostics["package_status"][package] = "Installed"
            except Exception as e:
                logger.error(f"Failed to install {package}: {e}")
                diagnostics["package_status"][package] = f"Failed: {str(e)}"
    
    logger.info("All required packages installation check completed.")

# Only try to install packages in development environment
if os.environ.get("ENVIRONMENT") != "production":
    install_required_packages()

# Now import the installed packages
try:
    import pandas as pd
    import numpy as np
    import requests
    import schedule
    import pytz
    from tabulate import tabulate
    from flask import Flask, jsonify, render_template_string, request

    # Try to import yfinance as a fallback
    try:
        import yfinance as yf
        diagnostics["package_status"]["yfinance"] = "Imported"
    except ImportError as e:
        logger.error(f"Failed to import yfinance: {e}")
        diagnostics["package_status"]["yfinance"] = f"Failed: {str(e)}"
    
    # Import NSE related packages
    try:
        from nsepy import get_history
        from nsetools import Nse
        diagnostics["package_status"]["nsepy"] = "Imported"
        diagnostics["package_status"]["nsetools"] = "Imported"
    except ImportError as e:
        logger.error(f"Failed to import specific package: {e}")
        diagnostics["package_status"]["NSE packages"] = f"Failed: {str(e)}"
    
    try:
        import telegram
        diagnostics["package_status"]["telegram"] = "Imported"
    except ImportError as e:
        logger.error(f"Failed to import telegram: {e}")
        diagnostics["package_status"]["telegram"] = f"Failed: {str(e)}"
    
    logger.info("All packages imported successfully.")
except Exception as e:
    logger.error(f"Failed to import required packages: {e}")
    sys.exit(1)

# Create Flask app after imports
app = Flask(__name__)
logger.info("Flask app created successfully")

# Define IST timezone
IST = pytz.timezone('Asia/Kolkata')
logger.info("Set timezone to IST (Asia/Kolkata)")

# Check if timezone is working correctly
current_ist = datetime.now(IST)
diagnostics["timezone_check"] = {
    "ist_time": current_ist.strftime("%Y-%m-%d %H:%M:%S %Z%z"),
    "utc_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z%z")
}

# Telegram Bot Configuration
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Technical Analysis Parameters
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3
RSI_PERIOD = 14
RSI_PERIOD_WEEKLY = 14
RSI_PERIOD_MONTHLY = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Stock Categories
NIFTY_LARGE_CAP_URL = "https://archives.nseindia.com/content/indices/ind_nifty100list.csv"
NIFTY_MID_CAP_URL = "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv"
NIFTY_SMALL_CAP_URL = "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv"

# Alternative Yahoo Finance symbols (for fallback)
YAHOO_INDICES = {
    "NIFTY 50": "^NSEI",
    "NIFTY BANK": "^NSEBANK",
    "SENSEX": "^BSESN"
}

# Performance settings
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 5))  # Reduced from 10 to avoid rate limiting
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 5))    # Reduced from 10 to avoid rate limiting
CACHE_TIMEOUT = int(os.environ.get("CACHE_TIMEOUT", 300))  # Cache timeout in seconds
ANALYSIS_INTERVAL = int(os.environ.get("ANALYSIS_INTERVAL", 15))  # Minutes between analyses
USE_FALLBACK = os.environ.get("USE_FALLBACK", "False").lower() == "true"  # Whether to use fallback data sources

# Global variable to store latest results
latest_results = {
    "large": [],
    "mid": [],
    "small": [],
    "weekly": [],     # For weekly analysis
    "monthly": [],    # For monthly analysis
    "indices": [],    # For major indices
    "last_update": None,
    "next_update": None,
    "status": "idle",
    "diagnostics": diagnostics  # Include diagnostics in results
}

# Global lock for accessing latest_results
results_lock = threading.Lock()

# Helper function to get current time in IST
def get_ist_time():
    return datetime.now(IST)

# Function to log errors and update diagnostics
def log_error(context, error):
    error_str = f"{context}: {str(error)}"
    error_time = get_ist_time().strftime("%Y-%m-%d %H:%M:%S")
    logger.error(error_str)
    
    # Update diagnostics with error information
    with results_lock:
        diagnostics["errors"].append({"time": error_time, "context": context, "error": str(error)})
        diagnostics["last_error_time"] = error_time
        if len(diagnostics["errors"]) > 10:  # Keep only the last 10 errors
            diagnostics["errors"] = diagnostics["errors"][-10:]
        latest_results["diagnostics"] = diagnostics

# Test NSE API connectivity
def test_nse_connectivity():
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get("https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050", headers=headers, timeout=10)
        
        if response.status_code == 200:
            with results_lock:
                diagnostics["api_status"] = "Connected"
            logger.info("NSE API connection test successful")
            return True
        else:
            with results_lock:
                diagnostics["api_status"] = f"Failed with status code: {response.status_code}"
            logger.error(f"NSE API connection test failed with status code: {response.status_code}")
            return False
    except Exception as e:
        with results_lock:
            diagnostics["api_status"] = f"Connection error: {str(e)}"
        logger.error(f"NSE API connection test failed with error: {e}")
        return False

# Function to generate HTML table from results
def generate_html_table(results, title, filter_type=None):
    if not results:
        return f"<div class='no-data'><p>No {title} data available</p></div>"
    
    # Get current timestamp in IST
    current_time = get_ist_time().strftime("%Y-%m-%d %H:%M:%S")
    
    # Filter results based on filter_type if specified
    filtered_results = results
    if filter_type:
        if filter_type == "buy":
            filtered_results = [r for r in results if r['Signal'] in ["BUY", "STRONG BUY", "OVERSOLD"]]
        elif filter_type == "sell":
            filtered_results = [r for r in results if r['Signal'] in ["SELL", "STRONG SELL", "OVERBOUGHT"]]
    
    if not filtered_results:
        return f"<div class='no-data'><p>No {title} signals matching the filter</p></div>"
    
    # Define CSS classes for signals
    signal_classes = {
        "BUY": "buy",
        "STRONG BUY": "strong-buy",
        "SELL": "sell",
        "STRONG SELL": "strong-sell",
        "OVERSOLD": "oversold",
        "OVERBOUGHT": "overbought",
        "NEUTRAL": "neutral"
    }
    
    # Generate HTML table with timestamp
    html = f"<div class='table-responsive'>"
    html += f"<div class='timestamp-info'>Last Updated: {current_time} (IST)</div>"
    html += f"<table class='stock-table'>"
    html += "<thead><tr>"
    
    # Extract all keys from first result for headers
    headers = filtered_results[0].keys()
    for header in headers:
        html += f"<th>{header}</th>"
    
    html += "</tr></thead><tbody>"
    
    for result in filtered_results:
        signal = result.get('Signal', 'NEUTRAL')
        html += f"<tr class='{signal_classes.get(signal, 'neutral')}'>"
        
        for key, value in result.items():
            if key == 'Signal':
                html += f"<td class='signal {signal_classes.get(value, 'neutral')}'>{value}</td>"
            elif isinstance(value, (int, float)):
                html += f"<td class='number'>{value}</td>"
            else:
                html += f"<td>{value}</td>"
        
        html += "</tr>"
    
    html += "</tbody></table></div>"
    return html

class StockAnalyzer:
    def __init__(self):
        try:
            self.nse = Nse()
            logger.info("NSE connection initialized")
        except Exception as e:
            log_error("NSE initialization", e)
            self.nse = None
        
        self.bot = self.setup_telegram_bot()
        self.data_cache = {}
        self.last_cache_time = {}
        self.weekly_cache = {}
        self.monthly_cache = {}
        
        # Test NSE connectivity
        self.nse_api_working = test_nse_connectivity()
    
    def setup_telegram_bot(self):
        if not TELEGRAM_TOKEN:
            logger.warning("Telegram token not set. Telegram notifications are disabled.")
            return None

        try:
            bot = telegram.Bot(token=TELEGRAM_TOKEN)
            logger.info("Telegram bot initialized successfully")
            return bot
        except Exception as e:
            log_error("Telegram bot setup", e)
            return None
    
    @lru_cache(maxsize=300)  # Cache up to 300 stock lists
    def fetch_stock_list(self, category="large"):
        try:
            if category.lower() == "large":
                url = NIFTY_LARGE_CAP_URL
            elif category.lower() == "mid":
                url = NIFTY_MID_CAP_URL
            elif category.lower() == "small":
                url = NIFTY_SMALL_CAP_URL
            else:
                raise ValueError("Invalid category. Choose from 'large', 'mid', or 'small'")
                
            # Add headers to mimic a browser request
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                logger.error(f"Failed to fetch stock list for {category} cap. Status code: {response.status_code}")
                raise Exception(f"HTTP {response.status_code}")
                
            df = pd.read_csv(url)
            symbols = df['Symbol'].tolist()
            logger.info(f"Successfully fetched {len(symbols)} stocks for {category} cap")
            
            # Limit to 20 stocks during testing/debugging
            if os.environ.get("DEBUG", "False").lower() == "true":
                symbols = symbols[:20]
                logger.info(f"Debug mode: Limited to first 20 stocks for {category} cap")
                
            return symbols
        except Exception as e:
            log_error(f"Fetch stock list ({category})", e)
            # Return a small default list for testing if real fetch fails
            if category.lower() == "large":
                return ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
            elif category.lower() == "mid":
                return ["ABCAPITAL", "ABFRL", "APLLTD", "ASTRAZEN", "ATUL"]
            else:
                return ["AHLUCONT", "AJOONI", "AKSHARCHEM", "ASTEC", "AXITA"]
    
    def get_stock_data_yf(self, symbol, days=100, timeframe="daily"):
        """Get stock data from Yahoo Finance as a fallback"""
        try:
            # Append .NS for NSE stocks
            ticker = f"{symbol}.NS"
            
            # Set period and interval based on timeframe
            if timeframe == "daily":
                period = f"{days+10}d"  # Add extra days to account for holidays
                interval = "1d"
            elif timeframe == "weekly":
                period = f"{days*2}d"
                interval = "1wk"
            elif timeframe == "monthly":
                period = f"{days*3}d"
                interval = "1mo"
            
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            
            if data.empty:
                logger.warning(f"No data returned from Yahoo Finance for {symbol}")
                return pd.DataFrame()
            
            # Rename columns to match NSE format
            data = data.rename(columns={
                'Open': 'Open', 
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            logger.info(f"Successfully fetched {timeframe} data for {symbol} from Yahoo Finance")
            return data
            
        except Exception as e:
            log_error(f"Yahoo Finance data fetch for {symbol}", e)
            return pd.DataFrame()
    
    def get_stock_data(self, symbol, days=100, timeframe="daily"):
        current_time = time.time()
        cache_key = f"{symbol}_{timeframe}"
        
        # Check if we have a cached version
        if timeframe == "daily" and symbol in self.data_cache and current_time - self.last_cache_time.get(symbol, 0) < CACHE_TIMEOUT:
            return self.data_cache[symbol]
        elif timeframe == "weekly" and cache_key in self.weekly_cache and current_time - self.last_cache_time.get(cache_key, 0) < CACHE_TIMEOUT:
            return self.weekly_cache[cache_key]
        elif timeframe == "monthly" and cache_key in self.monthly_cache and current_time - self.last_cache_time.get(cache_key, 0) < CACHE_TIMEOUT:
            return self.monthly_cache[cache_key]
        
        try:
            # Use IST timezone for start and end dates
            end_date = get_ist_time().date()
            
            # Adjust days based on timeframe to ensure enough data
            if timeframe == "weekly":
                days = days * 7  # Ensure enough daily data to create weekly data
            elif timeframe == "monthly":
                days = days * 30  # Ensure enough daily data to create monthly data
                
            start_date = end_date - pd.Timedelta(days=days)
            
            # First try NSE data if the API is working
            if self.nse_api_working and not USE_FALLBACK:
                try:
                    data = get_history(symbol=symbol, start=start_date, end=end_date)
                    if not data.empty:
                        logger.info(f"Successfully fetched {timeframe} data for {symbol} from NSE")
                        
                        # Convert to weekly or monthly data if needed
                        if timeframe == "weekly":
                            data = self.convert_to_weekly(data)
                            self.weekly_cache[cache_key] = data
                        elif timeframe == "monthly":
                            data = self.convert_to_monthly(data)
                            self.monthly_cache[cache_key] = data
                        else:
                            # Cache the daily data
                            self.data_cache[symbol] = data
                            
                        self.last_cache_time[cache_key if timeframe != "daily" else symbol] = current_time
                        return data
                    else:
                        logger.warning(f"Empty data returned from NSE for {symbol} ({timeframe})")
                except Exception as e:
                    logger.warning(f"NSE data fetch failed for {symbol}: {e}. Trying fallback source.")
            
            # Fallback to Yahoo Finance
            data = self.get_stock_data_yf(symbol, days, timeframe)
            if not data.empty:
                # Cache the data appropriately
                if timeframe == "weekly":
                    if timeframe != "weekly":
                        data = self.convert_to_weekly(data)
                    self.weekly_cache[cache_key] = data
                elif timeframe == "monthly":
                    if timeframe != "monthly":
                        data = self.convert_to_monthly(data)
                    self.monthly_cache[cache_key] = data
                else:
                    self.data_cache[symbol] = data
                    
                self.last_cache_time[cache_key if timeframe != "daily" else symbol] = current_time
                return data
            
            logger.error(f"Failed to fetch data for {symbol} ({timeframe}) from all sources")
            return pd.DataFrame()
            
        except Exception as e:
            log_error(f"Stock data fetch for {symbol} ({timeframe})", e)
            return pd.DataFrame()
    
    def convert_to_weekly(self, daily_data):
        """Convert daily data to weekly timeframe"""
        try:
            if daily_data.empty:
                return pd.DataFrame()
                
            # Ensure the index is a datetime
            if not isinstance(daily_data.index, pd.DatetimeIndex):
                daily_data.index = pd.to_datetime(daily_data.index)
            
            # Group by year and week number
            weekly_data = daily_data.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
            
            return weekly_data
        except Exception as e:
            log_error("Convert to weekly", e)
            return pd.DataFrame()
    
    def convert_to_monthly(self, daily_data):
        """Convert daily data to monthly timeframe"""
        try:
            if daily_data.empty:
                return pd.DataFrame()
                
            # Ensure the index is a datetime
            if not isinstance(daily_data.index, pd.DatetimeIndex):
                daily_data.index = pd.to_datetime(daily_data.index)
            
            # Group by year and month
            monthly_data = daily_data.resample('M').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
            
            return monthly_data
        except Exception as e:
            log_error("Convert to monthly", e)
            return pd.DataFrame()
    
    def calculate_atr(self, high, low, close, period=14):
        """Vectorized ATR calculation for better performance"""
        try:
            tr1 = high - low
            tr2 = (high - close.shift()).abs()  # Fix: Use abs() instead of direct comparison
            tr3 = (low - close.shift()).abs()   # Fix: Use abs() instead of direct comparison
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            return atr
        except Exception as e:
            log_error("ATR calculation", e)
            return pd.Series(0, index=high.index)
    
    def calculate_supertrend(self, df, period=SUPERTREND_PERIOD, multiplier=SUPERTREND_MULTIPLIER):
        """Optimized Supertrend calculation"""
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            
            # Calculate ATR
            atr = self.calculate_atr(high, low, close, period)
            
            # Calculate basic upper and lower bands
            hl2 = (high + low) / 2
            basic_upperband = hl2 + (multiplier * atr)
            basic_lowerband = hl2 - (multiplier * atr)
            
            # Initialize arrays
            final_upperband = np.zeros(len(df))
            final_lowerband = np.zeros(len(df))
            supertrend = np.zeros(len(df))
            
            # Set initial values
            for i in range(0, min(period, len(df))):
                final_upperband[i] = basic_upperband.iloc[i]
                final_lowerband[i] = basic_lowerband.iloc[i]
                supertrend[i] = basic_upperband.iloc[i]
            
            # Calculate Supertrend with optimized loop
            for i in range(period, len(df)):
                # Upper band
                if close.iloc[i-1] <= final_upperband[i-1]:
                    final_upperband[i] = min(basic_upperband.iloc[i], final_upperband[i-1])
                else:
                    final_upperband[i] = basic_upperband.iloc[i]
                
                # Lower band
                if close.iloc[i-1] >= final_lowerband[i-1]:
                    final_lowerband[i] = max(basic_lowerband.iloc[i], final_lowerband[i-1])
                else:
                    final_lowerband[i] = basic_lowerband.iloc[i]
                
                # Supertrend
                if close.iloc[i] <= final_upperband[i]:
                    supertrend[i] = final_upperband[i]
                else:
                    supertrend[i] = final_lowerband[i]
            
            # Convert back to pandas series
            df['Supertrend'] = pd.Series(supertrend, index=df.index)
            
            # Determine trend (1 for uptrend, -1 for downtrend)
            df['Supertrend_Direction'] = 0
            # Fix: Use numpy comparison to avoid ambiguous truth value of Series
            df.loc[close.values > df['Supertrend'].values, 'Supertrend_Direction'] = 1
            df.loc[close.values <= df['Supertrend'].values, 'Supertrend_Direction'] = -1
            
            return df
        except Exception as e:
            log_error("Supertrend calculation", e)
            df['Supertrend'] = 0
            df['Supertrend_Direction'] = 0
            return df
    
    def calculate_rsi(self, df, period=RSI_PERIOD):
        """Optimized RSI calculation"""
        try:
            close = df['Close']
            delta = close.diff()
            
            # Make two series: one for gains and one for losses
            gain = delta.copy()
            loss = delta.copy()
            
            # Fix: Use numpy for element-wise comparison to avoid ambiguous truth values
            gain = gain.where(delta > 0, 0)
            loss = -loss.where(delta < 0, 0)  # Make losses positive
            
            # Calculate average gain and average loss
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # Calculate RS (Relative Strength)
            # Fix: Handle division by zero
            rs = avg_gain / avg_loss.replace(0, 1e-10)  # Small value instead of zero
            
            # Calculate RSI
            rsi = 100 - (100 / (1 + rs))
            
            df['RSI'] = rsi
            return df
        except Exception as e:
            log_error("RSI calculation", e)
            df['RSI'] = 50  # Neutral RSI value
            return df
    
    def analyze_major_indices(self):
        """Analyze major indices using Yahoo Finance data"""
        results = []
        
        for index_name, yahoo_symbol in YAHOO_INDICES.items():
            try:
                # Get data from Yahoo Finance
                data = yf.download(yahoo_symbol, period="3mo", interval="1d", progress=False)
                
                if data.empty:
                    continue
                
                # Calculate indicators
                data = self.calculate_supertrend(data)
                data = self.calculate_rsi(data)
                
                if len(data) < 2:
                    continue
                
                # Get last row for analysis
                last_row = data.iloc[-1]
                prev_row = data.iloc[-2]
                
                # Determine signal
                signal = "NEUTRAL"
                supertrend_flipped = last_row['Supertrend_Direction'] != prev_row['Supertrend_Direction']
                
                # Fix: Use explicit numeric comparisons to avoid ambiguous truth values
                if (last_row['Supertrend_Direction'] == 1 and 
                    last_row['RSI'] > 50 and 
                    last_row['RSI'] < RSI_OVERBOUGHT):
                    signal = "BUY"
                    if supertrend_flipped:
                        signal = "STRONG BUY"
                elif (last_row['Supertrend_Direction'] == -1 and 
                      last_row['RSI'] < 50 and 
                      last_row['RSI'] > RSI_OVERSOLD):
                    signal = "SELL"
                    if supertrend_flipped:
                        signal = "STRONG SELL"
                elif last_row['RSI'] >= RSI_OVERBOUGHT:
                    signal = "OVERBOUGHT"
                elif last_row['RSI'] <= RSI_OVERSOLD:
                    signal = "OVERSOLD"
                
                # Format the result
                timestamp = last_row.name
                if isinstance(timestamp, pd.Timestamp):
                    date_str = timestamp.strftime('%Y-%m-%d')
                else:
                    date_str = "N/A"
                
                result = {
                    'Index': index_name,
                    'Date': date_str,
                    'Price': round(last_row['Close'], 2),
                    'Previous Close': round(prev_row['Close'], 2),
                    'Change %': round(((last_row['Close'] - prev_row['Close']) / prev_row['Close']) * 100, 2),
                    'RSI': round(last_row['RSI'], 2),
                    'Signal': signal
                }
                
                results.append(result)
                logger.info(f"Successfully analyzed {index_name}")
                
            except Exception as e:
                log_error(f"Index analysis for {index_name}", e)
        
        return results
    
    def analyze_stock(self, symbol, timeframe="daily"):
        try:
            # Get stock data for the specified timeframe
            data = self.get_stock_data(symbol, days=100 if timeframe == "daily" else 200, timeframe=timeframe)
            
            if data.empty:
                logger.warning(f"No {timeframe} data available for {symbol}")
                return None
            
            # Calculate indicators
            data = self.calculate_supertrend(data)
            
            # Use appropriate RSI period based on timeframe
            rsi_period = RSI_PERIOD
            if timeframe == "weekly":
                rsi_period = RSI_PERIOD_WEEKLY
            elif timeframe == "monthly":
                rsi_period = RSI_PERIOD_MONTHLY
                
            data = self.calculate_rsi(data, period=rsi_period)
            
            if len(data) < 2:
                logger.warning(f"Insufficient {timeframe} data points for {symbol}")
                return None
            
            # Get last row for analysis
            last_row = data.iloc[-1]
            prev_row = data.iloc[-2]
            
            # Determine signal based on Supertrend and RSI
            signal = "NEUTRAL"
            supertrend_flipped = last_row['Supertrend_Direction'] != prev_row['Supertrend_Direction']
            
            # Fix: Use explicit numeric comparisons to avoid ambiguous truth values
            if (last_row['Supertrend_Direction'] == 1 and 
                last_row['RSI'] > 50 and 
                last_row['RSI'] < RSI_OVERBOUGHT):
                signal = "BUY"
                if supertrend_flipped:
                    signal = "STRONG BUY"
            elif (last_row['Supertrend_Direction'] == -1 and 
                  last_row['RSI'] < 50 and 
                  last_row['RSI'] > RSI_OVERSOLD):
                signal = "SELL"
                if supertrend_flipped:
                    signal = "STRONG SELL"
            elif last_row['RSI'] >= RSI_OVERBOUGHT:
                signal = "OVERBOUGHT"
            elif last_row['RSI'] <= RSI_OVERSOLD:
                signal = "OVERSOLD"
            
            # Format the timestamp from the DataFrame index
            timestamp = last_row.name
            if isinstance(timestamp, pd.Timestamp):
                date_str = timestamp.strftime('%Y-%m-%d')
            else:
                date_str = "N/A"
            
            # Format the result
            result = {
                'Symbol': symbol,
                'Date': date_str,
                'Price': round(last_row['Close'], 2),
                'Previous Close': round(prev_row['Close'], 2),
                'Change %': round(((last_row['Close'] - prev_row['Close']) / prev_row['Close']) * 100, 2),
                'RSI': round(last_row['RSI'], 2),
                'Timeframe': timeframe.capitalize(),
                'Signal': signal
            }
            
            return result
        
        except Exception as e:
            log_error(f"Stock analysis ({symbol}, {timeframe})", e)
            return None
    
    def analyze_stocks_batch(self, symbols, timeframe="daily"):
        """Analyze a batch of stocks in parallel"""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks and store futures
            futures = {executor.submit(self.analyze_stock, symbol, timeframe): symbol for symbol in symbols}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        logger.info(f"Successfully analyzed {symbol} ({timeframe})")
                except Exception as e:
                    log_error(f"Stock batch analysis ({symbol}, {timeframe})", e)
        
        return results
    
    def analyze_stocks(self, category="large", timeframe="daily"):
        """Analyze all stocks in specified category"""
        try:
            # Get stock list for category
            symbols = self.fetch_stock_list(category)
            
            if not symbols:
                logger.error(f"No symbols found for {category} cap")
                return []
            
            all_results = []
            
            # Process in batches to avoid overwhelming the API
            for i in range(0, len(symbols), BATCH_SIZE):
                batch = symbols[i:i+BATCH_SIZE]
                logger.info(f"Processing batch {i//BATCH_SIZE + 1} of {math.ceil(len(symbols)/BATCH_SIZE)} for {category} cap ({timeframe})")
                
                # Analyze the batch
                batch_results = self.analyze_stocks_batch(batch, timeframe)
                all_results.extend(batch_results)
                
                # Sleep briefly to avoid rate limiting
                time.sleep(1)
            
            # Sort results by signal priority
            signal_priority = {
                "STRONG BUY": 1,
                "BUY": 2,
                "OVERSOLD": 3,
                "NEUTRAL": 4,
                "OVERBOUGHT": 5,
                "SELL": 6,
                "STRONG SELL": 7
            }
            
            all_results.sort(key=lambda x: signal_priority.get(x['Signal'], 999))
            
            return all_results
        
        except Exception as e:
            log_error(f"Stock analysis for {category} cap ({timeframe})", e)
            return []
    
    def send_telegram_notification(self, message):
        """Send notification to Telegram"""
        if not self.bot or not TELEGRAM_CHAT_ID:
            logger.warning("Telegram bot or chat ID not set. Notification not sent.")
            return False
        
        try:
            self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode="HTML")
            logger.info("Telegram notification sent successfully")
            return True
        except Exception as e:
            log_error("Telegram notification", e)
            return False
    
    def format_results_message(self, results, category, timeframe="daily"):
        """Format results for Telegram message"""
        if not results:
            return f"No {timeframe} signals for {category} cap stocks"
            
        # Group by signal
        signals = {}
        for result in results:
            signal = result['Signal']
            if signal not in signals:
                signals[signal] = []
            signals[signal].append(result)
        
        # Format message
        message = f"<b>{timeframe.capitalize()} Analysis for {category.capitalize()} Cap Stocks</b>\n"
        message += f"<i>Generated at {get_ist_time().strftime('%Y-%m-%d %H:%M:%S')} IST</i>\n\n"
        
        # Order signals by priority
        signal_order = ["STRONG BUY", "BUY", "OVERSOLD", "NEUTRAL", "OVERBOUGHT", "SELL", "STRONG SELL"]
        
        for signal in signal_order:
            if signal in signals and signals[signal]:
                message += f"<b>{signal} ({len(signals[signal])}):</b>\n"
                
                # List top stocks for each signal (max 5)
                for i, result in enumerate(signals[signal][:5]):
                    message += f"• {result['Symbol']}: ₹{result['Price']} (RSI: {result['RSI']})\n"
                
                if len(signals[signal]) > 5:
                    message += f"<i>...and {len(signals[signal]) - 5} more</i>\n"
                
                message += "\n"
        
        return message
    
    def run_analysis(self):
        """Run the complete analysis process"""
        try:
            logger.info("Starting stock analysis...")
            
            # Update status
            with results_lock:
                latest_results["status"] = "running"
            
            # Analyze major indices
            logger.info("Analyzing major indices...")
            indices_results = self.analyze_major_indices()
            
            # Analyze daily data for all categories
            logger.info("Analyzing daily data for large cap stocks...")
            large_cap_results = self.analyze_stocks("large", "daily")
            
            logger.info("Analyzing daily data for mid cap stocks...")
            mid_cap_results = self.analyze_stocks("mid", "daily")
            
            logger.info("Analyzing daily data for small cap stocks...")
            small_cap_results = self.analyze_stocks("small", "daily")
            
            # Analyze weekly data for all categories combined
            logger.info("Analyzing weekly data...")
            # Combine top stocks from each category for weekly analysis to reduce load
            top_stocks = []
            for results in [large_cap_results, mid_cap_results, small_cap_results]:
                # Add stocks with strong signals
                for result in results:
                    if result['Signal'] in ["STRONG BUY", "STRONG SELL", "BUY", "SELL", "OVERSOLD", "OVERBOUGHT"]:
                        top_stocks.append(result['Symbol'])
            
            # Remove duplicates and limit to 50 stocks for weekly analysis
            top_stocks = list(set(top_stocks))[:50]
            weekly_results = self.analyze_stocks_batch(top_stocks, "weekly")
            
            # Analyze monthly data (just top 20 stocks to reduce load)
            logger.info("Analyzing monthly data...")
            monthly_stocks = top_stocks[:20]
            monthly_results = self.analyze_stocks_batch(monthly_stocks, "monthly")
            
            # Send notifications if enabled
            if self.bot and TELEGRAM_CHAT_ID:
                # Send summary notifications for each timeframe
                logger.info("Sending Telegram notifications...")
                
                # Daily summary
                daily_message = "<b>🔔 DAILY ANALYSIS SUMMARY</b>\n\n"
                
                # Add indices summary
                daily_message += "<b>📊 INDICES:</b>\n"
                for idx in indices_results:
                    trend = "🔴" if idx['Change %'] < 0 else "🟢"
                    daily_message += f"{trend} {idx['Index']}: {idx['Price']} ({idx['Change %']}%), {idx['Signal']}\n"
                daily_message += "\n"
                
                # Add top signals from each category
                for category, results in [("Large Cap", large_cap_results), 
                                         ("Mid Cap", mid_cap_results), 
                                         ("Small Cap", small_cap_results)]:
                    strong_signals = [r for r in results if r['Signal'] in ["STRONG BUY", "STRONG SELL"]][:3]
                    if strong_signals:
                        daily_message += f"<b>{category} Strong Signals:</b>\n"
                        for r in strong_signals:
                            signal_emoji = "🟢" if r['Signal'] == "STRONG BUY" else "🔴"
                            daily_message += f"{signal_emoji} {r['Symbol']}: ₹{r['Price']} ({r['Signal']})\n"
                        daily_message += "\n"
                
                self.send_telegram_notification(daily_message)
                
                # Weekly and Monthly summaries (optional, based on configuration)
                if os.environ.get("SEND_WEEKLY_SUMMARY", "True").lower() == "true":
                    weekly_message = self.format_results_message(weekly_results, "all", "weekly")
                    self.send_telegram_notification(weekly_message)
                
                if os.environ.get("SEND_MONTHLY_SUMMARY", "True").lower() == "true":
                    monthly_message = self.format_results_message(monthly_results, "all", "monthly")
                    self.send_telegram_notification(monthly_message)
            
            # Update global results
            current_time = get_ist_time()
            next_update = current_time + timedelta(minutes=ANALYSIS_INTERVAL)
            
            with results_lock:
                latest_results["large"] = large_cap_results
                latest_results["mid"] = mid_cap_results
                latest_results["small"] = small_cap_results
                latest_results["weekly"] = weekly_results
                latest_results["monthly"] = monthly_results
                latest_results["indices"] = indices_results
                latest_results["last_update"] = current_time.strftime("%Y-%m-%d %H:%M:%S")
                latest_results["next_update"] = next_update.strftime("%Y-%m-%d %H:%M:%S")
                latest_results["status"] = "idle"
            
            logger.info("Analysis completed successfully")
            return True
        
        except Exception as e:
            log_error("Complete analysis run", e)
            
            # Update status to error
            with results_lock:
                latest_results["status"] = "error"
            
            return False

# Web interface functions
@app.route('/')
def index():
    """Main web interface for stock analysis results"""
    current_time = get_ist_time().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get results with filter type if specified
    filter_type = request.args.get('filter', None)
    
    # Create HTML template with some styling
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NSE Stock Analyzer</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f4f7f9;
                color: #333;
            }
            .header {
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .tab-container {
                display: flex;
                margin-bottom: 20px;
                background: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .tab {
                padding: 15px 20px;
                cursor: pointer;
                transition: background-color 0.3s;
                border-bottom: 3px solid transparent;
                font-weight: bold;
            }
            .tab.active {
                background-color: #f4f7f9;
                border-bottom: 3px solid #3498db;
                color: #3498db;
            }
            .tab:hover:not(.active) {
                background-color: #f9f9f9;
            }
            .tab-content {
                display: none;
                background: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .tab-content.active {
                display: block;
            }
            .filter-container {
                margin-bottom: 20px;
                display: flex;
                justify-content: flex-start;
                gap: 10px;
            }
            .filter-btn {
                padding: 8px 15px;
                background-color: #f4f7f9;
                border: 1px solid #ddd;
                border-radius: 5px;
                cursor: pointer;
                transition: all 0.3s;
                font-size: 14px;
            }
            .filter-btn.active {
                background-color: #3498db;
                color: white;
                border-color: #3498db;
            }
            .filter-btn:hover:not(.active) {
                background-color: #e9ecef;
            }
            .stock-table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            .stock-table th, .stock-table td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            .stock-table th {
                background-color: #f4f7f9;
                font-weight: bold;
                position: sticky;
                top: 0;
                z-index: 10;
            }
            .stock-table tr:hover {
                background-color: #f9f9f9;
            }
            .buy, .strong-buy, .oversold {
                background-color: rgba(46, 204, 113, 0.1);
            }
            .sell, .strong-sell, .overbought {
                background-color: rgba(231, 76, 60, 0.1);
            }
            .signal {
                font-weight: bold;
                padding: 5px 10px;
                border-radius: 3px;
                text-align: center;
            }
            .signal.buy, .signal.strong-buy {
                background-color: #2ecc71;
                color: white;
            }
            .signal.sell, .signal.strong-sell {
                background-color: #e74c3c;
                color: white;
            }
            .signal.oversold {
                background-color: #3498db;
                color: white;
            }
            .signal.overbought {
                background-color: #f39c12;
                color: white;
            }
            .signal.neutral {
                background-color: #95a5a6;
                color: white;
            }
            .number {
                text-align: right;
            }
            .status-bar {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px 20px;
                background-color: #ecf0f1;
                border-radius: 5px;
                margin-bottom: 20px;
                font-size: 14px;
            }
            .status-badge {
                padding: 5px 10px;
                border-radius: 3px;
                font-weight: bold;
            }
            .status-idle {
                background-color: #2ecc71;
                color: white;
            }
            .status-running {
                background-color: #f39c12;
                color: white;
            }
            .status-error {
                background-color: #e74c3c;
                color: white;
            }
            .timestamp-info {
                font-style: italic;
                color: #7f8c8d;
                margin-bottom: 10px;
                font-size: 14px;
            }
            .no-data {
                padding: 20px;
                text-align: center;
                color: #7f8c8d;
                background-color: #f9f9f9;
                border-radius: 5px;
                margin: 20px 0;
            }
            .table-responsive {
                overflow-x: auto;
            }
            .footer {
                margin-top: 30px;
                text-align: center;
                color: #7f8c8d;
                font-size: 14px;
                padding: 20px;
            }
            .error-info {
                background-color: #ffecec;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
                border: 1px solid #f5c6cb;
            }
            .error-title {
                font-weight: bold;
                color: #721c24;
                margin-bottom: 10px;
            }
            .error-list {
                font-family: monospace;
                white-space: pre-wrap;
                background-color: #f8f9fa;
                padding: 10px;
                border-radius: 3px;
                max-height: 200px;
                overflow-y: auto;
                font-size: 12px;
            }
            .debug-section {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-top: 20px;
                border: 1px solid #dee2e6;
            }
            .debug-title {
                font-weight: bold;
                margin-bottom: 10px;
                color: #495057;
            }
            @media (max-width: 768px) {
                .tab-container {
                    flex-wrap: wrap;
                }
                .tab {
                    flex-grow: 1;
                    text-align: center;
                    padding: 10px;
                }
                .stock-table th, .stock-table td {
                    padding: 8px;
                    font-size: 14px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>NSE Stock Analyzer</h1>
                <p>Technical Analysis with Supertrend and RSI indicators</p>
            </div>
            
            <div class="status-bar">
                <div>
                    <span>Status: </span>
                    <span class="status-badge status-{status}">{status}</span>
                </div>
                <div>Last update: {last_update}</div>
                <div>Next update: {next_update}</div>
            </div>
            
            <div class="filter-container">
                <a href="/?filter=all" class="filter-btn {all_active}">All Signals</a>
                <a href="/?filter=buy" class="filter-btn {buy_active}">Buy/Oversold</a>
                <a href="/?filter=sell" class="filter-btn {sell_active}">Sell/Overbought</a>
            </div>
            
            <div class="tab-container">
                <div class="tab active" data-tab="indices">Indices</div>
                <div class="tab" data-tab="large">Large Cap</div>
                <div class="tab" data-tab="mid">Mid Cap</div>
                <div class="tab" data-tab="small">Small Cap</div>
                <div class="tab" data-tab="weekly">Weekly</div>
                <div class="tab" data-tab="monthly">Monthly</div>
                <div class="tab" data-tab="debug">Diagnostics</div>
            </div>
            
            <div id="indices" class="tab-content active">
                {indices_content}
            </div>
            
            <div id="large" class="tab-content">
                {large_content}
            </div>
            
            <div id="mid" class="tab-content">
                {mid_content}
            </div>
            
            <div id="small" class="tab-content">
                {small_content}
            </div>
            
            <div id="weekly" class="tab-content">
                {weekly_content}
            </div>
            
            <div id="monthly" class="tab-content">
                {monthly_content}
            </div>
            
            <div id="debug" class="tab-content">
                <h2>System Diagnostics</h2>
                
                <div class="debug-section">
                    <div class="debug-title">API Status</div>
                    <p>NSE API Status: {api_status}</p>
                    <p>Current Time (IST): {current_time}</p>
                    <p>Current Time (UTC): {utc_time}</p>
                </div>
                
                {error_section}
                
                <div class="debug-section">
                    <div class="debug-title">Package Status</div>
                    <ul>
                        {package_status}
                    </ul>
                </div>
            </div>
            
            <div class="footer">
                <p>NSE Stock Analyzer - Technical Analysis Tool</p>
                <p>Trading based solely on technical indicators involves risk - please do your own research.</p>
            </div>
        </div>
        
        <script>
            // Tab switching functionality
            document.querySelectorAll('.tab').forEach(tab => {
                tab.addEventListener('click', () => {
                    // Remove active class from all tabs and contents
                    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                    
                    // Add active class to clicked tab
                    tab.classList.add('active');
                    
                    // Show corresponding content
                    const tabId = tab.getAttribute('data-tab');
                    document.getElementById(tabId).classList.add('active');
                });
            });
        </script>
    </body>
    </html>
    """
    
    # Format current filter status
    filter_status = {
        'all_active': 'active' if filter_type is None or filter_type == 'all' else '',
        'buy_active': 'active' if filter_type == 'buy' else '',
        'sell_active': 'active' if filter_type == 'sell' else ''
    }
    
    # Generate HTML content for each tab
    indices_content = generate_html_table(latest_results["indices"], "Indices", filter_type)
    large_content = generate_html_table(latest_results["large"], "Large Cap Stocks", filter_type)
    mid_content = generate_html_table(latest_results["mid"], "Mid Cap Stocks", filter_type)
    small_content = generate_html_table(latest_results["small"], "Small Cap Stocks", filter_type)
    weekly_content = generate_html_table(latest_results["weekly"], "Weekly Analysis", filter_type)
    monthly_content = generate_html_table(latest_results["monthly"], "Monthly Analysis", filter_type)
    
    # Generate error section for debug tab
    error_section = ""
    if diagnostics["errors"]:
        error_section = """
        <div class="error-info">
            <div class="error-title">Recent Errors</div>
            <div class="error-list">
        """
        for error in diagnostics["errors"]:
            error_section += f"{error['time']} [{error['context']}]: {error['error']}\n"
        error_section += """
            </div>
        </div>
        """
    
    # Generate package status list
    package_status_html = ""
    for package, status in diagnostics["package_status"].items():
        package_status_html += f"<li><strong>{package}:</strong> {status}</li>"
    
    # Substitute all values in template
    html = html.format(
        status=latest_results["status"],
        last_update=latest_results["last_update"] or "Not available",
        next_update=latest_results["next_update"] or "Not scheduled",
        indices_content=indices_content,
        large_content=large_content,
        mid_content=mid_content,
        small_content=small_content,
        weekly_content=weekly_content,
        monthly_content=monthly_content,
        api_status=diagnostics["api_status"],
        current_time=current_time,
        utc_time=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z"),
        error_section=error_section,
        package_status=package_status_html,
        **filter_status
    )
    
    return html

@app.route('/api/results')
def api_results():
    """API endpoint to get analysis results"""
    # Return the latest results as JSON
    with results_lock:
        # Remove diagnostics info from API results
        api_results = latest_results.copy()
        api_results.pop("diagnostics", None)
        return jsonify(api_results)

@app.route('/api/refresh', methods=['POST'])
def api_refresh():
    """API endpoint to trigger a refresh of the analysis"""
    # Check if analysis is already running
    if latest_results["status"] == "running":
        return jsonify({"success": False, "message": "Analysis is already running"})
    
    # Start analysis in a separate thread
    threading.Thread(target=analyzer.run_analysis).start()
    
    return jsonify({"success": True, "message": "Analysis refresh triggered"})

@app.route('/api/diagnostics')
def api_diagnostics():
    """API endpoint to get system diagnostics"""
    return jsonify(diagnostics)

def run_scheduler():
    """Run the scheduler for periodic analysis"""
    logger.info(f"Setting up scheduler to run analysis every {ANALYSIS_INTERVAL} minutes")
    
    # Function to run analysis and reschedule
    def run_scheduled_analysis():
        try:
            logger.info("Running scheduled analysis...")
            analyzer.run_analysis()
        except Exception as e:
            log_error("Scheduled analysis", e)
        
        # Reschedule the next run
        threading.Timer(ANALYSIS_INTERVAL * 60, run_scheduled_analysis).start()
    
    # Run the first analysis immediately
    threading.Timer(1, run_scheduled_analysis).start()

# Initialize the analyzer
analyzer = StockAnalyzer()

# Check initialization status
if __name__ == "__main__":
    try:
        # Run initial analysis
        logger.info("Running initial analysis on startup")
        analyzer.run_analysis()
        
        # Start scheduler in a separate thread
        threading.Thread(target=run_scheduler, daemon=True).start()
        
        # Run Flask app
        port = int(os.environ.get("PORT", 5000))
        logger.info(f"Starting web server on port {port}")
        app.run(host='0.0.0.0', port=port, debug=os.environ.get("DEBUG", "False").lower() == "true")
    except Exception as e:
        logger.critical(f"Failed to start application: {e}")
        traceback.print_exc()
