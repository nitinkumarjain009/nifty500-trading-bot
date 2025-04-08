#!/usr/bin/env python
# NSE Stock Analyzer with Supertrend, RSI, Telegram Notifications and Web Services
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
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
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
            df.loc[close > df['Supertrend'], 'Supertrend_Direction'] = 1
            df.loc[close <= df['Supertrend'], 'Supertrend_Direction'] = -1
            
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
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = abs(loss)
            
            # Calculate average gain and average loss
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # Calculate RS (Relative Strength)
            rs = avg_gain / avg_loss.replace(0, 0.00001)  # Avoid division by zero
            
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
                
                if last_row['Supertrend_Direction'] == 1 and last_row['RSI'] > 50 and last_row['RSI'] < RSI_OVERBOUGHT:
                    signal = "BUY"
                    if supertrend_flipped:
                        signal = "STRONG BUY"
                elif last_row['Supertrend_Direction'] == -1 and last_row['RSI'] < 50 and last_row['RSI'] > RSI_OVERSOLD:
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
            
            if last_row['Supertrend_Direction'] == 1 and last_row['RSI'] > 50 and last_row['RSI'] < RSI_OVERBOUGHT:
                signal = "BUY"
                if supertrend_flipped:
                    signal = "STRONG BUY"
            elif last_row['Supertrend_Direction'] == -1 and last_row['RSI'] < 50 and last_row['RSI'] > RSI_OVERSOLD:
                signal = "SELL"
                if supertrend_flipped:
                    signal = "STRONG SELL"
            elif last_row['RSI'] >= RSI_OVERBOUGHT:
                signal = "OVERBOUGHT"
            elif last_row['RSI'] <= RSI_OVERSOLD:
                signal = "OVERSOLD"
            
           # Get current price and format the result
            timestamp = last_row.name
            if isinstance(timestamp, pd.Timestamp):
                date_str = timestamp.strftime('%Y-%m-%d')
            else:
                date_str = "N/A"
            
            # Format the result as a dictionary
            result = {
                'Symbol': symbol,
                'Date': date_str,
                'Timeframe': timeframe.capitalize(),
                'Price': round(last_row['Close'], 2),
                'Previous Close': round(prev_row['Close'], 2),
                'Change %': round(((last_row['Close'] - prev_row['Close']) / prev_row['Close']) * 100, 2),
                'RSI': round(last_row['RSI'], 2),
                'Supertrend': round(last_row['Supertrend'], 2),
                'Signal': signal
            }
            
            logger.info(f"Successfully analyzed {symbol} ({timeframe}): {signal}")
            return result
            
        except Exception as e:
            log_error(f"Stock analysis for {symbol} ({timeframe})", e)
            return None
    
    def analyze_stocks_batch(self, symbols, timeframe="daily"):
        """Analyze a batch of stocks for better performance"""
        results = []
        
        for symbol in symbols:
            try:
                result = self.analyze_stock(symbol, timeframe)
                if result:
                    results.append(result)
            except Exception as e:
                log_error(f"Batch analysis for {symbol}", e)
        
        return results
    
    def analyze_stocks_in_parallel(self, symbols, timeframe="daily"):
        """Analyze stocks in parallel using thread pool"""
        results = []
        
        # Split symbols into batches
        batches = [symbols[i:i + BATCH_SIZE] for i in range(0, len(symbols), BATCH_SIZE)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit batch tasks
            future_to_batch = {
                executor.submit(self.analyze_stocks_batch, batch, timeframe): batch 
                for batch in batches
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    symbols_str = ", ".join(batch[:5]) + "..." if len(batch) > 5 else ", ".join(batch)
                    log_error(f"Parallel analysis for batch {symbols_str}", e)
        
        return results
    
    def send_telegram_notification(self, message):
        """Send notification to Telegram"""
        if not self.bot or not TELEGRAM_CHAT_ID:
            logger.debug("Telegram notification skipped: Bot or chat ID not configured")
            return False
        
        try:
            self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message,
                parse_mode=telegram.ParseMode.HTML
            )
            logger.info("Telegram notification sent successfully")
            return True
        except Exception as e:
            log_error("Telegram notification", e)
            return False
    
    def format_results_for_notification(self, results, category):
        """Format analysis results for Telegram notification"""
        if not results:
            return f"No signals found for {category} cap stocks"
        
        # Sort results by signal priority
        signal_priority = {
            "STRONG BUY": 1,
            "STRONG SELL": 2,
            "BUY": 3,
            "SELL": 4,
            "OVERSOLD": 5,
            "OVERBOUGHT": 6,
            "NEUTRAL": 7
        }
        
        sorted_results = sorted(results, key=lambda x: signal_priority.get(x['Signal'], 999))
        
        # Format message
        message = f"<b>{category.upper()} CAP STOCK SIGNALS</b>\n\n"
        count = 0
        
        for result in sorted_results:
            signal = result['Signal']
            
            # Skip neutral signals to keep message focused
            if signal == "NEUTRAL":
                continue
                
            # Limit to 20 stocks per message to avoid hitting Telegram limits
            if count >= 20:
                message += f"\n... and {len(sorted_results) - 20} more stocks with signals"
                break
                
            symbol = result['Symbol']
            price = result['Price']
            rsi = result['RSI']
            change = result['Change %']
            
            # Format each entry with signal emoji
            emoji = "🟢" if signal in ["BUY", "STRONG BUY", "OVERSOLD"] else "🔴"
            message += f"{emoji} <b>{symbol}</b>: {signal} at ₹{price} (RSI: {rsi}, Chg: {change}%)\n"
            count += 1
        
        if count == 0:
            message += "No significant signals found."
            
        # Add timestamp
        message += f"\n\nUpdated: {get_ist_time().strftime('%Y-%m-%d %H:%M:%S')} (IST)"
        
        return message
    
    def run_complete_analysis(self):
        """Run a complete analysis of all stock categories"""
        with results_lock:
            latest_results["status"] = "analyzing"
        
        try:
            logger.info("Starting complete stock analysis...")
            
            # Analyze major indices first
            indices_results = self.analyze_major_indices()
            with results_lock:
                latest_results["indices"] = indices_results
            
            # Get stock lists for each category
            large_cap_stocks = self.fetch_stock_list("large")
            mid_cap_stocks = self.fetch_stock_list("mid")
            small_cap_stocks = self.fetch_stock_list("small")
            
            # Analyze each category in parallel
            logger.info(f"Analyzing {len(large_cap_stocks)} large cap stocks...")
            large_cap_results = self.analyze_stocks_in_parallel(large_cap_stocks)
            
            logger.info(f"Analyzing {len(mid_cap_stocks)} mid cap stocks...")
            mid_cap_results = self.analyze_stocks_in_parallel(mid_cap_stocks)
            
            logger.info(f"Analyzing {len(small_cap_stocks)} small cap stocks...")
            small_cap_results = self.analyze_stocks_in_parallel(small_cap_stocks)
            
            # Weekly timeframe analysis (use a subset of large and mid cap)
            weekly_stocks = large_cap_stocks[:30] + mid_cap_stocks[:20]
            logger.info(f"Analyzing {len(weekly_stocks)} stocks on weekly timeframe...")
            weekly_results = self.analyze_stocks_in_parallel(weekly_stocks, timeframe="weekly")
            
            # Monthly timeframe analysis (use top large cap only)
            monthly_stocks = large_cap_stocks[:20]
            logger.info(f"Analyzing {len(monthly_stocks)} stocks on monthly timeframe...")
            monthly_results = self.analyze_stocks_in_parallel(monthly_stocks, timeframe="monthly")
            
            # Update global results
            with results_lock:
                latest_results["large"] = large_cap_results
                latest_results["mid"] = mid_cap_results
                latest_results["small"] = small_cap_results
                latest_results["weekly"] = weekly_results
                latest_results["monthly"] = monthly_results
                latest_results["last_update"] = get_ist_time().strftime("%Y-%m-%d %H:%M:%S")
                
                # Calculate next update time
                next_update = get_ist_time() + timedelta(minutes=ANALYSIS_INTERVAL)
                latest_results["next_update"] = next_update.strftime("%Y-%m-%d %H:%M:%S")
                latest_results["status"] = "idle"
            
            # Send Telegram notifications for important signals only
            if self.bot:
                # Filter for strong signals only
                strong_large_signals = [r for r in large_cap_results if r['Signal'] in ["STRONG BUY", "STRONG SELL"]]
                strong_mid_signals = [r for r in mid_cap_results if r['Signal'] in ["STRONG BUY", "STRONG SELL"]]
                
                if strong_large_signals:
                    message = self.format_results_for_notification(strong_large_signals, "LARGE")
                    self.send_telegram_notification(message)
                
                if strong_mid_signals:
                    message = self.format_results_for_notification(strong_mid_signals, "MID")
                    self.send_telegram_notification(message)
                
                # Weekly signals for longer term trends
                strong_weekly = [r for r in weekly_results if r['Signal'] in ["STRONG BUY", "STRONG SELL"]]
                if strong_weekly:
                    message = self.format_results_for_notification(strong_weekly, "WEEKLY")
                    self.send_telegram_notification(message)
            
            logger.info("Complete analysis finished successfully")
            return True
            
        except Exception as e:
            with results_lock:
                latest_results["status"] = "error"
            log_error("Complete analysis", e)
            return False

# Flask routes for the web service

@app.route('/')
def index():
    """Render the main dashboard page"""
    with results_lock:
        current_results = latest_results.copy()
    
    # Generate HTML content
    html_content = """
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
                background-color: #f5f7fa;
            }
            .container {
                width: 95%;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            header {
                background-color: #2c3e50;
                color: white;
                padding: 20px 0;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1, h2, h3 {
                margin-top: 0;
            }
            h1 {
                text-align: center;
                font-size: 2.5rem;
            }
            h2 {
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
                margin-top: 30px;
            }
            .card {
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
                overflow: hidden;
            }
            .card-header {
                background: #3498db;
                color: white;
                padding: 15px 20px;
                font-size: 1.2rem;
                font-weight: bold;
            }
            .card-body {
                padding: 20px;
            }
            .tabs {
                display: flex;
                border-bottom: 1px solid #ddd;
                margin-bottom: 15px;
            }
            .tab {
                padding: 10px 20px;
                cursor: pointer;
                border: 1px solid transparent;
                border-bottom: none;
                background: #f1f1f1;
                margin-right: 5px;
                border-radius: 5px 5px 0 0;
            }
            .tab.active {
                background: white;
                border-color: #ddd;
                border-bottom: 1px solid white;
                margin-bottom: -1px;
                font-weight: bold;
                color: #3498db;
            }
            .tab-content {
                display: none;
            }
            .tab-content.active {
                display: block;
            }
            .timestamp-info {
                text-align: right;
                font-size: 0.9rem;
                color: #666;
                margin: 10px 0;
            }
            .stock-table {
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
                font-size: 0.9rem;
            }
            .stock-table th {
                background-color: #f1f3f5;
                padding: 12px 15px;
                text-align: left;
                font-weight: bold;
                color: #2c3e50;
                border-bottom: 2px solid #ddd;
            }
            .stock-table td {
                padding: 10px 15px;
                border-bottom: 1px solid #ddd;
            }
            .stock-table tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .stock-table tr:hover {
                background-color: #e9f5fe;
            }
            .buy {
                background-color: rgba(72, 199, 116, 0.1);
            }
            .strong-buy {
                background-color: rgba(72, 199, 116, 0.3);
                font-weight: bold;
            }
            .sell {
                background-color: rgba(234, 84, 85, 0.1);
            }
            .strong-sell {
                background-color: rgba(234, 84, 85, 0.3);
                font-weight: bold;
            }
            .oversold {
                background-color: rgba(72, 199, 116, 0.2);
            }
            .overbought {
                background-color: rgba(234, 84, 85, 0.2);
            }
            .neutral {
                background-color: rgba(255, 255, 255, 0.7);
            }
            .signal {
                font-weight: bold;
            }
            .signal.buy, .signal.strong-buy, .signal.oversold {
                color: #28a745;
            }
            .signal.sell, .signal.strong-sell, .signal.overbought {
                color: #dc3545;
            }
            .signal.neutral {
                color: #6c757d;
            }
            .filter-buttons {
                margin: 15px 0;
            }
            .filter-btn {
                padding: 8px 15px;
                margin-right: 10px;
                background-color: #e9ecef;
                border: 1px solid #ced4da;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.9rem;
            }
            .filter-btn.active {
                background-color: #3498db;
                color: white;
                border-color: #3498db;
            }
            .table-responsive {
                overflow-x: auto;
            }
            .status-bar {
                background-color: #fff;
                border-radius: 4px;
                padding: 10px 15px;
                margin-bottom: 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
            }
            .status-active {
                background-color: #28a745;
            }
            .status-idle {
                background-color: #17a2b8;
            }
            .status-error {
                background-color: #dc3545;
            }
            .reload-btn {
                padding: 8px 15px;
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.9rem;
            }
            .reload-btn:hover {
                background-color: #2980b9;
            }
            .footer {
                text-align: center;
                margin-top: 40px;
                padding: 20px;
                color: #6c757d;
                font-size: 0.9rem;
                border-top: 1px solid #ddd;
            }
            .no-data {
                padding: 20px;
                text-align: center;
                color: #6c757d;
            }
            @media (max-width: 768px) {
                .container {
                    width: 100%;
                    padding: 10px;
                }
                .card-header, .card-body {
                    padding: 10px;
                }
                .tab {
                    padding: 8px 12px;
                    font-size: 0.9rem;
                }
                .stock-table {
                    font-size: 0.8rem;
                }
                .stock-table th, .stock-table td {
                    padding: 8px 10px;
                }
            }
        </style>
    </head>
    <body>
        <header>
            <div class="container">
                <h1>NSE Stock Analyzer</h1>
            </div>
        </header>
        
        <div class="container">
            <div class="status-bar">
                <div>
                    <span class="status-indicator status-{{ current_results['status'] }}"></span>
                    Status: {{ current_results['status'].capitalize() }}
                    {% if current_results['last_update'] %}
                    | Last update: {{ current_results['last_update'] }}
                    {% endif %}
                    {% if current_results['next_update'] %}
                    | Next update: {{ current_results['next_update'] }}
                    {% endif %}
                </div>
                <button class="reload-btn" onclick="location.reload()">Refresh Data</button>
            </div>
            
            <!-- Major Indices Section -->
            <div class="card">
                <div class="card-header">Major Indices</div>
                <div class="card-body">
                    {{ generate_html_table(current_results['indices'], 'Major Indices') }}
                </div>
            </div>
            
            <!-- Daily Analysis Section -->
            <div class="card">
                <div class="card-header">Daily Analysis</div>
                <div class="card-body">
                    <div class="tabs">
                        <div class="tab active" onclick="changeTab(event, 'large-cap')">Large Cap</div>
                        <div class="tab" onclick="changeTab(event, 'mid-cap')">Mid Cap</div>
                        <div class="tab" onclick="changeTab(event, 'small-cap')">Small Cap</div>
                    </div>
                    
                    <div class="filter-buttons">
                        <button class="filter-btn active" onclick="filterTable(event, 'all')">All Signals</button>
                        <button class="filter-btn" onclick="filterTable(event, 'buy')">Buy Signals</button>
                        <button class="filter-btn" onclick="filterTable(event, 'sell')">Sell Signals</button>
                    </div>
                    
                    <div id="large-cap" class="tab-content active">
                        {{ generate_html_table(current_results['large'], 'Large Cap') }}
                    </div>
                    
                    <div id="mid-cap" class="tab-content">
                        {{ generate_html_table(current_results['mid'], 'Mid Cap') }}
                    </div>
                    
                    <div id="small-cap" class="tab-content">
                        {{ generate_html_table(current_results['small'], 'Small Cap') }}
                    </div>
                </div>
            </div>
            
            <!-- Weekly & Monthly Analysis Section -->
            <div class="card">
                <div class="card-header">Longer Term Analysis</div>
                <div class="card-body">
                    <div class="tabs">
                        <div class="tab active" onclick="changeTab(event, 'weekly-analysis')">Weekly Analysis</div>
                        <div class="tab" onclick="changeTab(event, 'monthly-analysis')">Monthly Analysis</div>
                    </div>
                    
                    <div class="filter-buttons">
                        <button class="filter-btn active" onclick="filterTable(event, 'all')">All Signals</button>
                        <button class="filter-btn" onclick="filterTable(event, 'buy')">Buy Signals</button>
                        <button class="filter-btn" onclick="filterTable(event, 'sell')">Sell Signals</button>
                    </div>
                    
                    <div id="weekly-analysis" class="tab-content active">
                        {{ generate_html_table(current_results['weekly'], 'Weekly Analysis') }}
                    </div>
                    
                    <div id="monthly-analysis" class="tab-content">
                        {{ generate_html_table(current_results['monthly'], 'Monthly Analysis') }}
                    </div>
                </div>
            </div>
            
            <!-- Diagnostics Section -->
            <div class="card">
                <div class="card-header">System Diagnostics</div>
                <div class="card-body">
                    <h3>API Status: {{ current_results['diagnostics']['api_status'] }}</h3>
                    
                    <h3>Timezone Check:</h3>
                    <pre>{{ current_results['diagnostics']['timezone_check'] | tojson(indent=2) }}</pre>
                    
                    <h3>Package Status:</h3>
                    <ul>
                        {% for package, status in current_results['diagnostics']['package_status'].items() %}
                        <li>{{ package }}: {{ status }}</li>
                        {% endfor %}
                    </ul>
                    
                    <h3>Recent Errors:</h3>
                    {% if current_results['diagnostics']['errors'] %}
                    <table class="stock-table">
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Context</th>
                                <th>Error</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for error in current_results['diagnostics']['errors'] %}
                            <tr>
                                <td>{{ error['time'] }}</td>
                                <td>{{ error['context'] }}</td>
                                <td>{{ error['error'] }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                    {% else %}
                    <p>No errors recorded.</p>
                    {% endif %}
                </div>
            </div>
            
            <div class="footer">
                <p>NSE Stock Analyzer &copy; 2025 | Last Updated: {{ current_results['last_update'] or 'Never' }}</p>
            </div>
        </div>
        
        <script>
            function changeTab(event, tabId) {
                // Hide all tab contents
                const tabContents = event.currentTarget.closest('.card-body').querySelectorAll('.tab-content');
                tabContents.forEach(tab => tab.classList.remove('active'));
                
                // Show the selected tab content
                document.getElementById(tabId).classList.add('active');
                
                // Update tab buttons
                const tabs = event.currentTarget.closest('.tabs').querySelectorAll('.tab');
                tabs.forEach(tab => tab.classList.remove('active'));
                event.currentTarget.classList.add('active');
            }
            
            function filterTable(event, filterType) {
                // Update button styles
                const buttons = event.currentTarget.closest('.filter-buttons').querySelectorAll('.filter-btn');
                buttons.forEach(btn => btn.classList.remove('active'));
                event.currentTarget.classList.add('active');
                
                // Get the active tab content
                const activeTabContent = event.currentTarget.closest('.card-body').querySelector('.tab-content.active');
                
                // Filter table rows based on signal
                const rows = activeTabContent.querySelectorAll('table tbody tr');
                rows.forEach(row => {
                    const signal = row.querySelector('.signal').textContent;
                    
                    if (filterType === 'all') {
                        row.style.display = '';
                    } else if (filterType === 'buy' && ['BUY', 'STRONG BUY', 'OVERSOLD'].includes(signal)) {
                        row.style.display = '';
                    } else if (filterType === 'sell' && ['SELL', 'STRONG SELL', 'OVERBOUGHT'].includes(signal)) {
                        row.style.display = '';
                    } else {
                        row.style.display = 'none';
                    }
                });
            }
        </script>
    </body>
    </html>
    """
    
    # Render the template with the current results
    return render_template_string(
        html_content,
        current_results=current_results,
        generate_html_table=generate_html_table,
        tojson=json.dumps
    )

@app.route('/api/data')
def api_data():
    """API endpoint to get the latest analysis data"""
    with results_lock:
        current_results = latest_results.copy()
    
    # Remove sensitive information from diagnostics
    if "diagnostics" in current_results:
        sanitized_diagnostics = current_results["diagnostics"].copy()
        # Remove any potentially sensitive information
        for key in ["errors"]:
            if key in sanitized_diagnostics:
                # Keep only the last 5 errors and sanitize them
                sanitized_diagnostics[key] = sanitized_diagnostics[key][-5:] if sanitized_diagnostics[key] else []
        
        current_results["diagnostics"] = sanitized_diagnostics
    
    return jsonify(current_results)

@app.route('/api/stocks/<category>')
def api_stocks(category):
    """API endpoint to get stocks from a specific category"""
    if category not in ["large", "mid", "small", "weekly", "monthly", "indices"]:
        return jsonify({"error": "Invalid category"}), 400
    
    with results_lock:
        category_data = latest_results.get(category, [])
    
    # Get filter query parameter
    filter_type = request.args.get('filter')
    
    if filter_type:
        if filter_type == "buy":
            category_data = [r for r in category_data if r['Signal'] in ["BUY", "STRONG BUY", "OVERSOLD"]]
        elif filter_type == "sell":
            category_data = [r for r in category_data if r['Signal'] in ["SELL", "STRONG SELL", "OVERBOUGHT"]]
    
    return jsonify(category_data)

@app.route('/force-update')
def force_update():
    """Force a new analysis run"""
    def run_analysis_thread():
        analyzer = StockAnalyzer()
        analyzer.run_complete_analysis()
    
    # Only start if not already analyzing
    with results_lock:
        if latest_results["status"] != "analyzing":
            latest_results["status"] = "analyzing"
            threading.Thread(target=run_analysis_thread).start()
            return jsonify({"status": "Analysis started"})
        else:
            return jsonify({"status": "Analysis already in progress"})

def start_scheduler():
    """Start the scheduler for periodic analysis"""
    analyzer = StockAnalyzer()
    
    def scheduled_analysis():
        logger.info("Running scheduled analysis...")
        
        # Only run if not currently analyzing
        with results_lock:
            if latest_results["status"] != "analyzing":
                analyzer.run_complete_analysis()
    
    # Schedule the analysis to run every ANALYSIS_INTERVAL minutes
    schedule.every(ANALYSIS_INTERVAL).minutes.do(scheduled_analysis)
    
    # Also run at specific times during market hours (9:30 AM, 11:30 AM, 1:30 PM, 3:30 PM IST)
    market_hours = ["09:30", "11:30", "13:30", "15:30"]
    for hour in market_hours:
        schedule.every().day.at(hour).do(scheduled_analysis)
    
    # Run initial analysis
    threading.Thread(target=analyzer.run_complete_analysis).start()
    
    # Run the scheduler in a separate thread
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(30)  # Check every 30 seconds
    
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    logger.info("Scheduler started successfully")

if __name__ == "__main__":
    # Start the scheduler
    start_scheduler()
    
    # Get port from environment variable or use default 5000
    port = int(os.environ.get("PORT", 5000))
    
    # Start the Flask app
    app.run(host="0.0.0.0", port=port, debug=False)
