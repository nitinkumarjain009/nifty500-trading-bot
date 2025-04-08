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
            tr2 = abs(high - close.shift())  # Fix: Use abs() directly
            tr3 = abs(low - close.shift())   # Fix: Use abs() directly
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            return atr
        except Exception as e:
            log_error("ATR calculation", e)
            return pd.Series(0, index=high.index)
    
    def calculate_supertrend(self, df, period=SUPERTREND_PERIOD, multiplier=SUPERTREND_MULTIPLIER):
    """Fixed Supertrend calculation"""
    try:
        if df.empty or len(df) < period:
            logger.warning("DataFrame is empty or has insufficient data for Supertrend calculation")
            df['Supertrend'] = 0
            df['Supertrend_Direction'] = 0
            return df
            
        # Make a copy to avoid modifying the original DataFrame
        df = df.copy()
        
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # Calculate ATR
        atr = self.calculate_atr(high, low, close, period)
        
        # Calculate basic upper and lower bands
        hl2 = (high + low) / 2
        basic_upperband = hl2 + (multiplier * atr)
        basic_lowerband = hl2 - (multiplier * atr)
        
        # Initialize Series with NaN values
        final_upperband = pd.Series(index=df.index, dtype=float)
        final_lowerband = pd.Series(index=df.index, dtype=float)
        supertrend = pd.Series(index=df.index, dtype=float)
        
        # Fill initial values
        for i in range(period):
            final_upperband.iloc[i] = basic_upperband.iloc[i]
            final_lowerband.iloc[i] = basic_lowerband.iloc[i]
            supertrend.iloc[i] = basic_upperband.iloc[i]  # Default to upper band for initial values
        
        # Calculate the rest of the values iteratively
        for i in range(period, len(df)):
            # Upper band
            if basic_upperband.iloc[i] < final_upperband.iloc[i-1] or close.iloc[i-1] > final_upperband.iloc[i-1]:
                final_upperband.iloc[i] = basic_upperband.iloc[i]
            else:
                final_upperband.iloc[i] = final_upperband.iloc[i-1]
            
            # Lower band
            if basic_lowerband.iloc[i] > final_lowerband.iloc[i-1] or close.iloc[i-1] < final_lowerband.iloc[i-1]:
                final_lowerband.iloc[i] = basic_lowerband.iloc[i]
            else:
                final_lowerband.iloc[i] = final_lowerband.iloc[i-1]
            
            # Supertrend value - simplified logic
            if supertrend.iloc[i-1] == final_upperband.iloc[i-1]:
                if close.iloc[i] <= final_upperband.iloc[i]:
                    supertrend.iloc[i] = final_upperband.iloc[i]
                else:
                    supertrend.iloc[i] = final_lowerband.iloc[i]
            elif supertrend.iloc[i-1] == final_lowerband.iloc[i-1]:
                if close.iloc[i] >= final_lowerband.iloc[i]:
                    supertrend.iloc[i] = final_lowerband.iloc[i]
                else:
                    supertrend.iloc[i] = final_upperband.iloc[i]
            else:
                # Fallback logic (should rarely happen with proper initialization)
                supertrend.iloc[i] = final_lowerband.iloc[i] if close.iloc[i] > supertrend.iloc[i-1] else final_upperband.iloc[i]
        
        # Add Supertrend to DataFrame
        df['Supertrend'] = supertrend
        
        # Calculate trend direction (1 for uptrend, -1 for downtrend)
        df['Supertrend_Direction'] = np.where(close > supertrend, 1, -1)
        
        return df
        
    except Exception as e:
        log_error("Supertrend calculation", e)
        df['Supertrend'] = 0
        df['Supertrend_Direction'] = 0
        return df
    
    def calculate_rsi(self, df, period=RSI_PERIOD):
        """Fixed RSI calculation"""
        try:
            if df.empty or len(df) < period + 1:
                logger.warning("DataFrame is empty or has insufficient data for RSI calculation")
                df['RSI'] = 50
                return df
                
            delta = df['Close'].diff()
            
            # Create gain and loss series
            gain = delta.copy()
            loss = delta.copy()
            
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = -loss  # Make losses positive
            
            # First average gain and loss
            avg_gain = gain.rolling(window=period).mean().fillna(0)
            avg_loss = loss.rolling(window=period).mean().fillna(0)
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss.replace(0, 1e-10)  # Avoid division by zero
            rsi = 100.0 - (100.0 / (1.0 + rs))
            
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
                supertrend_direction = last_row['Supertrend_Direction']
                prev_supertrend_direction = prev_row['Supertrend_Direction']
                supertrend_flipped = supertrend_direction != prev_supertrend_direction
                rsi_value = last_row['RSI']
                
                # Determine signal based on fixed conditions
                if supertrend_direction == 1:
                    if rsi_value > 50 and rsi_value < RSI_OVERBOUGHT:
                        signal = "BUY"
                        if supertrend_flipped:
                            signal = "STRONG BUY"
                    elif rsi_value >= RSI_OVERBOUGHT:
                        signal = "OVERBOUGHT"
                elif supertrend_direction == -1:
                    if rsi_value < 50 and rsi_value > RSI_OVERSOLD:
                        signal = "SELL"
                        if supertrend_flipped:
                            signal = "STRONG SELL"
                    elif rsi_value <= RSI_OVERSOLD:
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
            if timeframe == "weekly":
                data = self.calculate_rsi(data, period=RSI_PERIOD_WEEKLY)
            elif timeframe == "monthly":
                data = self.calculate_rsi(data, period=RSI_PERIOD_MONTHLY)
            else:
                data = self.calculate_rsi(data, period=RSI_PERIOD)
            
            if len(data) < 2:
                logger.warning(f"Insufficient data points for {symbol} ({timeframe})")
                return None
            
            # Get last row for analysis
            last_row = data.iloc[-1]
            prev_row = data.iloc[-2]
            
            # Determine signal
            signal = "NEUTRAL"
            supertrend_direction = last_row['Supertrend_Direction']
            prev_supertrend_direction = prev_row['Supertrend_Direction']
            supertrend_flipped = supertrend_direction != prev_supertrend_direction
            rsi_value = last_row['RSI']
            
            # Determine signal based on combined indicators
            if supertrend_direction == 1:
                if rsi_value > 50 and rsi_value < RSI_OVERBOUGHT:
                    signal = "BUY"
                    if supertrend_flipped:
                        signal = "STRONG BUY"
                elif rsi_value >= RSI_OVERBOUGHT:
                    signal = "OVERBOUGHT"
            elif supertrend_direction == -1:
                if rsi_value < 50 and rsi_value > RSI_OVERSOLD:
                    signal = "SELL"
                    if supertrend_flipped:
                        signal = "STRONG SELL"
                elif rsi_value <= RSI_OVERSOLD:
                    signal = "OVERSOLD"
            
            # Format the result
            timestamp = last_row.name
            if isinstance(timestamp, pd.Timestamp):
                date_str = timestamp.strftime('%Y-%m-%d')
            else:
                date_str = "N/A"
            
            # Get stock name if available (for better readability)
            stock_name = symbol
            try:
                if self.nse:
                    stock_info = self.nse.get_quote(symbol)
                    if stock_info and 'companyName' in stock_info:
                        stock_name = stock_info['companyName']
            except:
                pass  # Use symbol if name fetch fails
            
            result = {
                'Symbol': symbol,
                'Name': stock_name,
                'Timeframe': timeframe.capitalize(),
                'Date': date_str,
                'Price': round(last_row['Close'], 2),
                'Previous Close': round(prev_row['Close'], 2),
                'Change %': round(((last_row['Close'] - prev_row['Close']) / prev_row['Close']) * 100, 2),
                'RSI': round(rsi_value, 2),
                'Signal': signal
            }
            
            logger.info(f"Successfully analyzed {symbol} ({timeframe}): {signal}")
            return result
            
        except Exception as e:
            log_error(f"Stock analysis for {symbol} ({timeframe})", e)
            return None
    
    def analyze_stocks_batch(self, symbols, timeframe="daily"):
        results = []
        for symbol in symbols:
            try:
                result = self.analyze_stock(symbol, timeframe)
                if result:
                    results.append(result)
            except Exception as e:
                log_error(f"Batch analysis for {symbol}", e)
        return results
    
    def analyze_stocks_parallel(self, symbols, timeframe="daily"):
        # Split symbols into batches
        batches = [symbols[i:i + BATCH_SIZE] for i in range(0, len(symbols), BATCH_SIZE)]
        all_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit batch jobs
            future_to_batch = {
                executor.submit(self.analyze_stocks_batch, batch, timeframe): batch 
                for batch in batches
            }
            
            # Process completed jobs
            for future in concurrent.futures.as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    logger.info(f"Completed batch analysis of {len(batch)} stocks")
                except Exception as e:
                    log_error(f"Parallel analysis batch", e)
        
        return all_results
    
    def send_telegram_notification(self, message):
        if not self.bot or not TELEGRAM_CHAT_ID:
            logger.info("Telegram notification skipped (bot or chat ID not configured)")
            return False
        
        try:
            self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode="HTML")
            logger.info("Telegram notification sent successfully")
            return True
        except Exception as e:
            log_error("Telegram notification", e)
            return False
    
    def generate_notification_message(self, results, category):
        if not results:
            return None
        
        # Filter for strong signals only
        strong_signals = [r for r in results if r['Signal'] in ["STRONG BUY", "STRONG SELL"]]
        
        if not strong_signals:
            return None
        
        # Group by signal type
        buy_signals = [r for r in strong_signals if r['Signal'] == "STRONG BUY"]
        sell_signals = [r for r in strong_signals if r['Signal'] == "STRONG SELL"]
        
        message = f"<b>🔔 {category.upper()} CAP STOCK SIGNALS</b>\n\n"
        
        if buy_signals:
            message += "<b>🟢 STRONG BUY Signals:</b>\n"
            for signal in buy_signals[:5]:  # Limit to 5 signals
                message += f"• {signal['Symbol']} - ₹{signal['Price']} (RSI: {signal['RSI']})\n"
            if len(buy_signals) > 5:
                message += f"...and {len(buy_signals) - 5} more\n"
        
        if sell_signals:
            message += "\n<b>🔴 STRONG SELL Signals:</b>\n"
            for signal in sell_signals[:5]:  # Limit to 5 signals
                message += f"• {signal['Symbol']} - ₹{signal['Price']} (RSI: {signal['RSI']})\n"
            if len(sell_signals) > 5:
                message += f"...and {len(sell_signals) - 5} more\n"
        
        message += f"\n<i>Generated on {get_ist_time().strftime('%Y-%m-%d %H:%M:%S')} (IST)</i>"
        return message
    
    def run_analysis(self):
        try:
            logger.info("Starting stock analysis...")
            with results_lock:
                latest_results["status"] = "running"
                latest_results["last_update"] = get_ist_time().strftime("%Y-%m-%d %H:%M:%S")
                latest_results["next_update"] = (get_ist_time() + timedelta(minutes=ANALYSIS_INTERVAL)).strftime("%Y-%m-%d %H:%M:%S")
            
            # Analyze major indices first (always do this)
            indices_results = self.analyze_major_indices()
            
            # Get stock lists
            large_cap_stocks = self.fetch_stock_list("large")
            mid_cap_stocks = self.fetch_stock_list("mid")
            small_cap_stocks = self.fetch_stock_list("small")
            
            # Analyze stocks in parallel
            large_cap_results = self.analyze_stocks_parallel(large_cap_stocks)
            mid_cap_results = self.analyze_stocks_parallel(mid_cap_stocks)
            small_cap_results = self.analyze_stocks_parallel(small_cap_stocks)
            
            # Also perform weekly and monthly analyses (using a subset for efficiency)
            weekly_stocks = large_cap_stocks[:20] + mid_cap_stocks[:10] + small_cap_stocks[:10]
            monthly_stocks = large_cap_stocks[:10] + mid_cap_stocks[:5] + small_cap_stocks[:5]
            
            weekly_results = self.analyze_stocks_parallel(weekly_stocks, "weekly")
            monthly_results = self.analyze_stocks_parallel(monthly_stocks, "monthly")
            
            # Update global results
            with results_lock:
                latest_results["indices"] = indices_results
                latest_results["large"] = large_cap_results
                latest_results["mid"] = mid_cap_results
                latest_results["small"] = small_cap_results
                latest_results["weekly"] = weekly_results
                latest_results["monthly"] = monthly_results
                latest_results["last_update"] = get_ist_time().strftime("%Y-%m-%d %H:%M:%S")
                latest_results["next_update"] = (get_ist_time() + timedelta(minutes=ANALYSIS_INTERVAL)).strftime("%Y-%m-%d %H:%M:%S")
                latest_results["status"] = "idle"
            
            logger.info("Stock analysis completed successfully")
            
            # Send telegram notifications for strong signals
            for category, results in [("Large", large_cap_results), 
                                     ("Mid", mid_cap_results), 
                                     ("Small", small_cap_results)]:
                notification_message = self.generate_notification_message(results, category)
                if notification_message:
                    self.send_telegram_notification(notification_message)
            
            return True
        except Exception as e:
            log_error("Run analysis", e)
            with results_lock:
                latest_results["status"] = "error"
            return False
    
    def scheduled_analysis(self):
        """Run analysis on schedule"""
        try:
            self.run_analysis()
        except Exception as e:
            log_error("Scheduled analysis", e)

# Flask routes for the web service
@app.route('/')
def index():
    """Main page with analysis results"""
    # Define basic HTML/CSS template with tabs
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NSE Stock Analyzer</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            header {
                background-color: #2c3e50;
                color: white;
                padding: 1rem;
                text-align: center;
                margin-bottom: 20px;
            }
            h1 {
                margin: 0;
            }
            .status-bar {
                background-color: #34495e;
                color: white;
                padding: 10px;
                border-radius: 4px;
                margin-bottom: 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .status-indicator {
                padding: 5px 10px;
                border-radius: 20px;
                font-weight: bold;
            }
            .status-idle {
                background-color: #27ae60;
            }
            .status-running {
                background-color: #f39c12;
                animation: pulse 1.5s infinite;
            }
            .status-error {
                background-color: #e74c3c;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.7; }
                100% { opacity: 1; }
            }
            .tab {
                overflow: hidden;
                background-color: #f1f1f1;
                border-radius: 5px 5px 0 0;
            }
            .tab button {
                background-color: inherit;
                float: left;
                border: none;
                outline: none;
                cursor: pointer;
                padding: 14px 16px;
                transition: 0.3s;
                font-size: 17px;
            }
            .tab button:hover {
                background-color: #ddd;
            }
            .tab button.active {
                background-color: #34495e;
                color: white;
            }
            .tabcontent {
                display: none;
                padding: 20px;
                border: 1px solid #ccc;
                border-top: none;
                border-radius: 0 0 5px 5px;
                background-color: white;
                margin-bottom: 20px;
            }
            .tabcontent.active {
                display: block;
            }
            .filter-bar {
                margin-bottom: 15px;
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 4px;
            }
            .filter-bar button {
                margin-right: 5px;
                padding: 5px 10px;
                border: none;
                border-radius: 3px;
                cursor: pointer;
                background-color: #3498db;
                color: white;
            }
            .filter-bar button.active {
                background-color: #2980b9;
            }
            .stock-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
                font-size: 14px;
            }
            .stock-table th {
                background-color: #34495e;
                color: white;
                padding: 10px;
                text-align: left;
            }
            .stock-table td {
                padding: 8px 10px;
                border-bottom: 1px solid #ddd;
            }
            .stock-table tr:hover {
                background-color: #f9f9f9;
            }
            .stock-table tr.buy {
                background-color: rgba(46, 204, 113, 0.1);
            }
            .stock-table tr.strong-buy {
                background-color: rgba(46, 204, 113, 0.3);
            }
            .stock-table tr.sell {
                background-color: rgba(231, 76, 60, 0.1);
            }
            .stock-table tr.strong-sell {
                background-color: rgba(231, 76, 60, 0.3);
            }
            .stock-table tr.oversold {
                background-color: rgba(52, 152, 219, 0.2);
            }
            .stock-table tr.overbought {
                background-color: rgba(155, 89, 182, 0.2);
            }
            .signal {
                font-weight: bold;
                padding: 3px 6px;
                border-radius: 3px;
            }
            .signal.buy {
                background-color: rgba(46, 204, 113, 0.2);
                color: #27ae60;
            }
            .signal.strong-buy {
                background-color: rgba(46, 204, 113, 0.4);
                color: #27ae60;
            }
            .signal.sell {
                background-color: rgba(231, 76, 60, 0.2);
                color: #c0392b;
            }
            .signal.strong-sell {
                background-color: rgba(231, 76, 60, 0.4);
                color: #c0392b;
            }
            .signal.oversold {
                background-color: rgba(52, 152, 219, 0.2);
                color: #2980b9;
            }
            .signal.overbought {
                background-color: rgba(155, 89, 182, 0.2);
                color: #8e44ad;
            }
            .signal.neutral {
                background-color: rgba(149, 165, 166, 0.2);
                color: #7f8c8d;
            }
            .timestamp-info {
                font-size: 12px;
                color: #7f8c8d;
                text-align: right;
                margin-bottom: 5px;
            }
            .number {
                text-align: right;
            }
            .diagnostics-section {
                margin-top: 20px;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 5px;
                border: 1px solid #e9ecef;
            }
            .diagnostics-title {
                font-size: 18px;
                margin-bottom: 10px;
                color: #495057;
            }
            .diagnostic-item {
                margin-bottom: 5px;
            }
            .diagnostic-label {
                font-weight: bold;
                color: #495057;
            }
            .error-list {
                max-height: 200px;
                overflow-y: auto;
                margin-top: 10px;
                border: 1px solid #e9ecef;
                padding: 10px;
                background-color: #fff;
            }
            .error-item {
                margin-bottom: 8px;
                padding-bottom: 8px;
                border-bottom: 1px solid #e9ecef;
            }
            .error-time {
                color: #6c757d;
                font-size: 12px;
            }
            .error-context {
                font-weight: bold;
                color: #212529;
            }
            .error-message {
                color: #dc3545;
            }
            .no-data {
                text-align: center;
                padding: 20px;
                color: #6c757d;
                font-style: italic;
            }
            @media (max-width: 768px) {
                .tab button {
                    padding: 10px 8px;
                    font-size: 14px;
                }
                .stock-table {
                    font-size: 12px;
                }
                .stock-table th, .stock-table td {
                    padding: 6px;
                }
            }
        </style>
    </head>
    <body>
        <header>
            <h1>NSE Stock Analyzer</h1>
            <p>Technical Analysis with Supertrend and RSI</p>
        </header>
        
        <div class="container">
            <div class="status-bar">
                <div>
                    <strong>Last Update:</strong> {{last_update}}
                    <span style="margin-left: 15px;"><strong>Next Update:</strong> {{next_update}}</span>
                </div>
                <div>
                    <span class="status-indicator status-{{status}}">{{status|upper}}</span>
                </div>
            </div>
            
            <div class="tab">
                <button class="tablinks active" onclick="openTab(event, 'tab-indices')">Market Indices</button>
                <button class="tablinks" onclick="openTab(event, 'tab-large')">Large Cap</button>
                <button class="tablinks" onclick="openTab(event, 'tab-mid')">Mid Cap</button>
                <button class="tablinks" onclick="openTab(event, 'tab-small')">Small Cap</button>
                <button class="tablinks" onclick="openTab(event, 'tab-weekly')">Weekly Analysis</button>
                <button class="tablinks" onclick="openTab(event, 'tab-monthly')">Monthly Analysis</button>
                <button class="tablinks" onclick="openTab(event, 'tab-diagnostics')">Diagnostics</button>
            </div>
            
            <div id="tab-indices" class="tabcontent active">
                <h2>Market Indices</h2>
                {{indices_table}}
            </div>
            
            <div id="tab-large" class="tabcontent">
                <h2>Large Cap Stocks</h2>
                <div class="filter-bar">
                    <button class="filter-btn active" data-filter="all">All</button>
                    <button class="filter-btn" data-filter="buy">Buy Signals</button>
                    <button class="filter-btn" data-filter="sell">Sell Signals</button>
                </div>
                <div class="all-content">{{large_cap_table}}</div>
                <div class="buy-content" style="display:none;">{{large_cap_buy_table}}</div>
                <div class="sell-content" style="display:none;">{{large_cap_sell_table}}</div>
            </div>
            
            <div id="tab-mid" class="tabcontent">
                <h2>Mid Cap Stocks</h2>
                <div class="filter-bar">
                    <button class="filter-btn active" data-filter="all">All</button>
                    <button class="filter-btn" data-filter="buy">Buy Signals</button>
                    <button class="filter-btn" data-filter="sell">Sell Signals</button>
                </div>
                <div class="all-content">{{mid_cap_table}}</div>
                <div class="buy-content" style="display:none;">{{mid_cap_buy_table}}</div>
                <div class="sell-content" style="display:none;">{{mid_cap_sell_table}}</div>
            </div>
            
            <div id="tab-small" class="tabcontent">
                <h2>Small Cap Stocks</h2>
                <div class="filter-bar">
                    <button class="filter-btn active" data-filter="all">All</button>
                    <button class="filter-btn" data-filter="buy">Buy Signals</button>
                    <button class="filter-btn" data-filter="sell">Sell Signals</button>
                </div>
                <div class="all-content">{{small_cap_table}}</div>
                <div class="buy-content" style="display:none;">{{small_cap_buy_table}}</div>
                <div class="sell-content" style="display:none;">{{small_cap_sell_table}}</div>
            </div>
            
            <div id="tab-weekly" class="tabcontent">
                <h2>Weekly Analysis</h2>
                <div class="filter-bar">
                    <button class="filter-btn active" data-filter="all">All</button>
                    <button class="filter-btn" data-filter="buy">Buy Signals</button>
                    <button class="filter-btn" data-filter="sell">Sell Signals</button>
                </div>
                <div class="all-content">{{weekly_table}}</div>
                <div class="buy-content" style="display:none;">{{weekly_buy_table}}</div>
                <div class="sell-content" style="display:none;">{{weekly_sell_table}}</div>
            </div>
            
            <div id="tab-monthly" class="tabcontent">
                <h2>Monthly Analysis</h2>
                <div class="filter-bar">
                    <button class="filter-btn active" data-filter="all">All</button>
                    <button class="filter-btn" data-filter="buy">Buy Signals</button>
                    <button class="filter-btn" data-filter="sell">Sell Signals</button>
                </div>
                <div class="all-content">{{monthly_table}}</div>
                <div class="buy-content" style="display:none;">{{monthly_buy_table}}</div>
                <div class="sell-content" style="display:none;">{{monthly_sell_table}}</div>
            </div>
            
            <div id="tab-diagnostics" class="tabcontent">
                <h2>System Diagnostics</h2>
                <div class="diagnostics-section">
                    <div class="diagnostics-title">System Status</div>
                    <div class="diagnostic-item">
                        <span class="diagnostic-label">NSE API Status:</span> {{api_status}}
                    </div>
                    <div class="diagnostic-item">
                        <span class="diagnostic-label">Current Time (IST):</span> {{current_time}}
                    </div>
                    <div class="diagnostic-item">
                        <span class="diagnostic-label">Timezone Check:</span> {{timezone_check}}
                    </div>
                    
                    <div class="diagnostics-title" style="margin-top: 15px;">Package Status</div>
                    {{package_status}}
                    
                    <div class="diagnostics-title" style="margin-top: 15px;">Recent Errors</div>
                    <div class="error-list">
                        {{error_list}}
                    </div>
                </div>
            </div>
        </div>
        
        <script>
        function openTab(evt, tabName) {
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].className = tabcontent[i].className.replace(" active", "");
            }
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }
            document.getElementById(tabName).className += " active";
            evt.currentTarget.className += " active";
        }
        
        // Filter buttons functionality
        document.addEventListener('DOMContentLoaded', function() {
            var filterBtns = document.querySelectorAll('.filter-btn');
            
            filterBtns.forEach(function(btn) {
                btn.addEventListener('click', function() {
                    var filter = this.getAttribute('data-filter');
                    var tabContent = this.closest('.tabcontent');
                    
                    // Update button active state
                    tabContent.querySelectorAll('.filter-btn').forEach(function(filterBtn) {
                        filterBtn.classList.remove('active');
                    });
                    this.classList.add('active');
                    
                    // Show/hide content based on filter
                    tabContent.querySelector('.all-content').style.display = (filter === 'all') ? 'block' : 'none';
                    tabContent.querySelector('.buy-content').style.display = (filter === 'buy') ? 'block' : 'none';
                    tabContent.querySelector('.sell-content').style.display = (filter === 'sell') ? 'block' : 'none';
                });
            });
            
            // Auto-refresh page every 5 minutes
            setTimeout(function() {
                location.reload();
            }, 300000);
        });
        </script>
    </body>
    </html>
    """
    
    # Get current data with lock
    with results_lock:
        results_copy = latest_results.copy()
        diagnostics_copy = diagnostics.copy()
    
    # Generate HTML tables for each category
    indices_table = generate_html_table(results_copy.get("indices", []), "Market Indices")
    
    large_cap_table = generate_html_table(results_copy.get("large", []), "Large Cap Stocks")
    large_cap_buy_table = generate_html_table(results_copy.get("large", []), "Large Cap Stocks", "buy")
    large_cap_sell_table = generate_html_table(results_copy.get("large", []), "Large Cap Stocks", "sell")
    
    mid_cap_table = generate_html_table(results_copy.get("mid", []), "Mid Cap Stocks")
    mid_cap_buy_table = generate_html_table(results_copy.get("mid", []), "Mid Cap Stocks", "buy")
    mid_cap_sell_table = generate_html_table(results_copy.get("mid", []), "Mid Cap Stocks", "sell")
    
    small_cap_table = generate_html_table(results_copy.get("small", []), "Small Cap Stocks")
    small_cap_buy_table = generate_html_table(results_copy.get("small", []), "Small Cap Stocks", "buy")
    small_cap_sell_table = generate_html_table(results_copy.get("small", []), "Small Cap Stocks", "sell")
    
    weekly_table = generate_html_table(results_copy.get("weekly", []), "Weekly Analysis")
    weekly_buy_table = generate_html_table(results_copy.get("weekly", []), "Weekly Analysis", "buy")
    weekly_sell_table = generate_html_table(results_copy.get("weekly", []), "Weekly Analysis", "sell")
    
    monthly_table = generate_html_table(results_copy.get("monthly", []), "Monthly Analysis")
    monthly_buy_table = generate_html_table(results_copy.get("monthly", []), "Monthly Analysis", "buy")
    monthly_sell_table = generate_html_table(results_copy.get("monthly", []), "Monthly Analysis", "sell")
    
    # Generate diagnostics information
    package_status_html = "<div>"
    for package, status in diagnostics_copy.get("package_status", {}).items():
        status_color = "green" if "Failed" not in status else "red"
        package_status_html += f"<div class='diagnostic-item'><span class='diagnostic-label'>{package}:</span> <span style='color:{status_color}'>{status}</span></div>"
    package_status_html += "</div>"
    
    error_list_html = ""
    if diagnostics_copy.get("errors"):
        for error in diagnostics_copy["errors"]:
            error_list_html += f"""
            <div class='error-item'>
                <div class='error-time'>{error['time']}</div>
                <div class='error-context'>{error['context']}</div>
                <div class='error-message'>{error['error']}</div>
            </div>
            """
    else:
        error_list_html = "<div class='no-data'>No errors recorded</div>"
    
    # Replace placeholders in the template
    html = html_template.replace("{{indices_table}}", indices_table)
    html = html.replace("{{large_cap_table}}", large_cap_table)
    html = html.replace("{{large_cap_buy_table}}", large_cap_buy_table)
    html = html.replace("{{large_cap_sell_table}}", large_cap_sell_table)
    html = html.replace("{{mid_cap_table}}", mid_cap_table)
    html = html.replace("{{mid_cap_buy_table}}", mid_cap_buy_table)
    html = html.replace("{{mid_cap_sell_table}}", mid_cap_sell_table)
    html = html.replace("{{small_cap_table}}", small_cap_table)
    html = html.replace("{{small_cap_buy_table}}", small_cap_buy_table)
    html = html.replace("{{small_cap_sell_table}}", small_cap_sell_table)
    html = html.replace("{{weekly_table}}", weekly_table)
    html = html.replace("{{weekly_buy_table}}", weekly_buy_table)
    html = html.replace("{{weekly_sell_table}}", weekly_sell_table)
    html = html.replace("{{monthly_table}}", monthly_table)
    html = html.replace("{{monthly_buy_table}}", monthly_buy_table)
    html = html.replace("{{monthly_sell_table}}", monthly_sell_table)
    
    html = html.replace("{{last_update}}", results_copy.get("last_update", "Never"))
    html = html.replace("{{next_update}}", results_copy.get("next_update", "Unknown"))
    html = html.replace("{{status}}", results_copy.get("status", "idle"))
    
    html = html.replace("{{api_status}}", diagnostics_copy.get("api_status", "Unknown"))
    html = html.replace("{{current_time}}", get_ist_time().strftime("%Y-%m-%d %H:%M:%S"))
    html = html.replace("{{timezone_check}}", diagnostics_copy.get("timezone_check", {}).get("ist_time", "Unknown"))
    html = html.replace("{{package_status}}", package_status_html)
    html = html.replace("{{error_list}}", error_list_html)
    
    return html

@app.route('/api/status')
def api_status():
    """Return status and summary as JSON"""
    with results_lock:
        results_copy = latest_results.copy()
    
    # Create summary of signals
    summary = {
        "indices": {},
        "large": {"BUY": 0, "STRONG BUY": 0, "SELL": 0, "STRONG SELL": 0, "OVERSOLD": 0, "OVERBOUGHT": 0, "NEUTRAL": 0},
        "mid": {"BUY": 0, "STRONG BUY": 0, "SELL": 0, "STRONG SELL": 0, "OVERSOLD": 0, "OVERBOUGHT": 0, "NEUTRAL": 0},
        "small": {"BUY": 0, "STRONG BUY": 0, "SELL": 0, "STRONG SELL": 0, "OVERSOLD": 0, "OVERBOUGHT": 0, "NEUTRAL": 0},
        "weekly": {"BUY": 0, "STRONG BUY": 0, "SELL": 0, "STRONG SELL": 0, "OVERSOLD": 0, "OVERBOUGHT": 0, "NEUTRAL": 0},
        "monthly": {"BUY": 0, "STRONG BUY": 0, "SELL": 0, "STRONG SELL": 0, "OVERSOLD": 0, "OVERBOUGHT": 0, "NEUTRAL": 0}
    }
    
    # Count indices signals
    for idx in results_copy.get("indices", []):
        index_name = idx.get("Index")
        signal = idx.get("Signal", "NEUTRAL")
        if index_name not in summary["indices"]:
            summary["indices"][index_name] = signal
    
    # Count stock signals for each category
    for category in ["large", "mid", "small", "weekly", "monthly"]:
        for stock in results_copy.get(category, []):
            signal = stock.get("Signal", "NEUTRAL")
            summary[category][signal] += 1
    
    response = {
        "status": results_copy.get("status", "idle"),
        "last_update": results_copy.get("last_update"),
        "next_update": results_copy.get("next_update"),
        "summary": summary
    }
    
    return jsonify(response)

@app.route('/api/data/<category>')
def api_data(category):
    """Return data for a specific category as JSON"""
    if category not in ["indices", "large", "mid", "small", "weekly", "monthly"]:
        return jsonify({"error": "Invalid category"}), 400
    
    with results_lock:
        results_copy = latest_results.copy()
    
    return jsonify({
        "category": category,
        "data": results_copy.get(category, []),
        "last_update": results_copy.get("last_update")
    })

@app.route('/api/signals/<signal_type>')
def api_signals(signal_type):
    """Return stocks with specific signal type"""
    if signal_type not in ["buy", "sell", "strong_buy", "strong_sell", "oversold", "overbought"]:
        return jsonify({"error": "Invalid signal type"}), 400
    
    with results_lock:
        results_copy = latest_results.copy()
    
    # Map signal_type to actual signal values
    signal_map = {
        "buy": ["BUY", "STRONG BUY"],
        "strong_buy": ["STRONG BUY"],
        "sell": ["SELL", "STRONG SELL"],
        "strong_sell": ["STRONG SELL"],
        "oversold": ["OVERSOLD"],
        "overbought": ["OVERBOUGHT"]
    }
    
    signals = signal_map.get(signal_type)
    matching_stocks = []
    
    # Collect matching stocks from all categories
    for category in ["large", "mid", "small", "weekly", "monthly"]:
        for stock in results_copy.get(category, []):
            if stock.get("Signal") in signals:
                stock_with_category = stock.copy()
                stock_with_category["Category"] = category
                matching_stocks.append(stock_with_category)
    
    return jsonify({
        "signal_type": signal_type,
        "count": len(matching_stocks),
        "stocks": matching_stocks,
        "last_update": results_copy.get("last_update")
    })

@app.route('/force-update')
def force_update():
    """Force an analysis update"""
    # Get secret key from request (simple security)
    secret = request.args.get('key', '')
    if secret != os.environ.get("UPDATE_SECRET", "update123"):
        return jsonify({"error": "Unauthorized"}), 401
    
    # Start update in background thread to avoid blocking
    update_thread = threading.Thread(target=analyzer.run_analysis)
    update_thread.daemon = True
    update_thread.start()
    
    return jsonify({
        "status": "update_started",
        "message": "Stock analysis update has been initiated"
    })

@app.route('/diagnostics')
def diagnostics_page():
    """Return diagnostics information"""
    with results_lock:
        diagnostics_copy = diagnostics.copy()
    
    return jsonify(diagnostics_copy)

def setup_scheduler():
    """Set up the schedule for recurring analysis"""
    # Define market hours in IST
    market_open_hour = 9  # 9:00 AM IST
    market_close_hour = 16  # 4:00 PM IST
    
    # Schedule analysis during market hours on weekdays
    def job():
        now = get_ist_time()
        # Check if market hours (9:00 AM to 4:00 PM IST) and weekday (0-4, Mon-Fri)
        if (market_open_hour <= now.hour < market_close_hour) and (0 <= now.weekday() <= 4):
            logger.info("Scheduled analysis triggered during market hours")
            analyzer.scheduled_analysis()
        else:
            logger.info("Skipping scheduled analysis (outside market hours)")
    
    # Schedule job every ANALYSIS_INTERVAL minutes
    schedule.every(ANALYSIS_INTERVAL).minutes.do(job)
    logger.info(f"Scheduled analysis job every {ANALYSIS_INTERVAL} minutes")
    
    # Run scheduler in a separate thread
    def run_scheduler():
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                log_error("Scheduler", e)
                time.sleep(60)  # Wait a minute before retrying
    
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()
    logger.info("Scheduler thread started")

def main():
    """Main function to start the application"""
    try:
        global analyzer
        
        # Create analyzer
        analyzer = StockAnalyzer()
        
        # Run initial analysis
        analyzer.run_analysis()
        
        # Setup scheduler
        setup_scheduler()
        
        # Start Flask app
        port = int(os.environ.get("PORT", 5000))
        debug = os.environ.get("DEBUG", "False").lower() == "true"
        
        logger.info(f"Starting Flask app on port {port} (Debug: {debug})")
        app.run(host='0.0.0.0', port=port, debug=debug, use_reloader=False)
        
    except Exception as e:
        log_error("Application startup", e)
        logger.critical(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
