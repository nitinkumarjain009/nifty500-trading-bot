#!/usr/bin/env python
# NSE Stock Analyzer with Supertrend, RSI, Telegram Notifications and Web Service
# Enhanced with Weekly/Monthly Analysis and HTML Tables

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

# Function to install required packages - Only for local development
def install_required_packages():
    required_packages = [
        'pandas', 'numpy', 'requests', 'nsepy', 'nsetools', 
        'schedule', 'python-telegram-bot==13.7', 'tabulate', 'flask', 'pytz'
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
        except ImportError:
            logger.info(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logger.info(f"{package} has been installed.")
            except Exception as e:
                logger.error(f"Failed to install {package}: {e}")
    
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
    
    # Import NSE related packages
    try:
        from nsepy import get_history
        from nsetools import Nse
        import telegram
    except ImportError as e:
        logger.error(f"Failed to import specific package: {e}")
        # Still allow the app to start and provide appropriate error messages later
    
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

# Telegram Bot Configuration
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")

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

# Performance settings
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 10))  # Number of parallel workers
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 10))  # Process stocks in batches to avoid rate limiting
CACHE_TIMEOUT = int(os.environ.get("CACHE_TIMEOUT", 300))  # Cache timeout in seconds
ANALYSIS_INTERVAL = int(os.environ.get("ANALYSIS_INTERVAL", 15))  # Minutes between analyses

# Global variable to store latest results
latest_results = {
    "large": [],
    "mid": [],
    "small": [],
    "weekly": [],     # For weekly analysis
    "monthly": [],    # For monthly analysis
    "last_update": None,
    "next_update": None,
    "status": "idle"
}

# Global lock for accessing latest_results
results_lock = threading.Lock()

# Helper function to get current time in IST
def get_ist_time():
    return datetime.now(IST)

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
            logger.error(f"Failed to initialize NSE connection: {e}")
            self.nse = None
        
        self.bot = self.setup_telegram_bot()
        self.data_cache = {}
        self.last_cache_time = {}
        self.weekly_cache = {}
        self.monthly_cache = {}
    
    def setup_telegram_bot(self):
        try:
            bot = telegram.Bot(token=TELEGRAM_TOKEN)
            logger.info("Telegram bot initialized successfully")
            return bot
        except Exception as e:
            logger.error(f"Error setting up Telegram bot: {e}")
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
                
            df = pd.read_csv(url)
            symbols = df['Symbol'].tolist()
            logger.info(f"Successfully fetched {len(symbols)} stocks for {category} cap")
            return symbols
        except Exception as e:
            logger.error(f"Error fetching stock list for {category} cap: {e}")
            # Return a small default list for testing if real fetch fails
            if category.lower() == "large":
                return ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
            elif category.lower() == "mid":
                return ["ABCAPITAL", "ABFRL", "APLLTD", "ASTRAZEN", "ATUL"]
            else:
                return ["AHLUCONT", "AJOONI", "AKSHARCHEM", "ASTEC", "AXITA"]
    
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
            
            # Try to get data with retry logic
            retries = 3
            for attempt in range(retries):
                try:
                    data = get_history(symbol=symbol, start=start_date, end=end_date)
                    if not data.empty:
                        logger.info(f"Successfully fetched {timeframe} data for {symbol}")
                        
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
                        logger.warning(f"Empty data returned for {symbol} ({timeframe}), attempt {attempt+1}/{retries}")
                except Exception as inner_e:
                    logger.warning(f"Attempt {attempt+1}/{retries} failed for {symbol} ({timeframe}): {inner_e}")
                    time.sleep(1)  # Reduced wait time before retry
            
            logger.error(f"Failed to fetch {timeframe} data for {symbol} after {retries} attempts")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error in get_stock_data for {symbol} ({timeframe}): {e}")
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
            logger.error(f"Error converting to weekly data: {e}")
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
            logger.error(f"Error converting to monthly data: {e}")
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
            logger.error(f"Error calculating ATR: {e}")
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
            for i in range(0, period):
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
            logger.error(f"Error calculating Supertrend: {e}")
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
            logger.error(f"Error calculating RSI: {e}")
            df['RSI'] = 50  # Neutral RSI value
            return df
    
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
            
            # Get current price and other data
            current_price = last_row['Close']
            supertrend_value = last_row['Supertrend']
            rsi_value = last_row['RSI']
            
            # Get timestamp for the analysis
            timestamp = last_row.name
            if isinstance(timestamp, pd.Timestamp):
                date_str = timestamp.strftime('%Y-%m-%d')
            else:
                date_str = "N/A"
            
            result = {
                'Symbol': symbol,
                'Date': date_str,
                'Price': round(current_price, 2),
                'Supertrend': round(supertrend_value, 2),
                'RSI': round(rsi_value, 2),
                'Signal': signal,
                'Timeframe': timeframe.capitalize()
            }
            
            return result
        except Exception as e:
            logger.error(f"Error analyzing {symbol} ({timeframe}): {e}")
            return None
    
    def analyze_batch(self, symbols, timeframe="daily"):
        """Analyze a batch of symbols"""
        results = []
        for symbol in symbols:
            try:
                result = self.analyze_stock(symbol, timeframe)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {symbol} ({timeframe}): {e}")
        return results
    
    def analyze_category(self, category, timeframe="daily"):
        """Analyze a category using parallel processing"""
        stocks = self.fetch_stock_list(category)
        results = []
        
        logger.info(f"Analyzing {len(stocks)} stocks from {category} cap category ({timeframe} timeframe)")
        
        # Process in batches with parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            # Create batches to avoid overwhelming the API
            for i in range(0, len(stocks), BATCH_SIZE):
                batch = stocks[i:i+BATCH_SIZE]
                futures.append(executor.submit(self.analyze_batch, batch, timeframe))
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                batch_results = future.result()
                if batch_results:
                    results.extend(batch_results)
        
        logger.info(f"Completed analysis for {category} cap with {len(results)} results ({timeframe} timeframe)")
        return results
    
    def format_results_table(self, results):
        if not results:
            return "No results found."
        
        # Filter for interesting signals
        interesting_results = [r for r in results if r['Signal'] in ["BUY", "STRONG BUY", "SELL", "STRONG SELL", "OVERSOLD", "OVERBOUGHT"]]
        
        if not interesting_results:
            return "No actionable signals found."
        
        # Organize by signal type
        buy_signals = [r for r in interesting_results if r['Signal'] in ["BUY", "STRONG BUY"]]
        sell_signals = [r for r in interesting_results if r['Signal'] in ["SELL", "STRONG SELL"]]
        oversold = [r for r in interesting_results if r['Signal'] == "OVERSOLD"]
        overbought = [r for r in interesting_results if r['Signal'] == "OVERBOUGHT"]
        
        # Create tables
        tables = []
        
        if buy_signals:
            tables.append("🟢 BUY RECOMMENDATIONS 🟢")
            tables.append(tabulate(buy_signals, headers="keys", tablefmt="pipe"))
        
        if sell_signals:
            tables.append("\n🔴 SELL RECOMMENDATIONS 🔴")
            tables.append(tabulate(sell_signals, headers="keys", tablefmt="pipe"))
        
        if oversold:
            tables.append("\n🟡 OVERSOLD (Potential Buy) 🟡")
            tables.append(tabulate(oversold, headers="keys", tablefmt="pipe"))
        
        if overbought:
            tables.append("\n🟡 OVERBOUGHT (Potential Sell) 🟡")
            tables.append(tabulate(overbought, headers="keys", tablefmt="pipe"))
        
        return "\n".join(tables)
    
    def send_telegram_message(self, message):
        if not self.bot:
            logger.error("Telegram bot not initialized")
            return False
        
        try:
            # Split messages if too long
            max_length = 4096
            if len(message) <= max_length:
                self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode="Markdown")
            else:
                parts = [message[i:i+max_length] for i in range(0, len(message), max_length)]
                for part in parts:
                    self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=part, parse_mode="Markdown")
            
            logger.info("Message sent to Telegram successfully")
            return True
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def run_analysis(self):
        global latest_results  # Proper declaration of using the global variable
        
        try:
            with results_lock:
                latest_results["status"] = "running"
            
            # Use IST time for timestamps
            current_time = get_ist_time().strftime("%Y-%m-%d %H:%M:%S")
            logger.info("Starting market analysis at " + current_time)
            
            # Calculate next update time in IST
            next_update_time = get_ist_time() + timedelta(minutes=ANALYSIS_INTERVAL)
            next_update = next_update_time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Analyze each category
            categories = ["large", "mid", "small"]
            timeframes = ["daily", "weekly", "monthly"]
            all_results = {}
            
            # Analyze daily timeframe for each category
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(categories)) as executor:
                daily_futures = {executor.submit(self.analyze_category, category): category for category in categories}
                for future in concurrent.futures.as_completed(daily_futures):
                    category = daily_futures[future]
                    try:
                        results = future.result()
                        all_results[category] = results
                        
                        # Update global results dictionary with thread safety
                        with results_lock:
                            latest_results[category] = results
                    except Exception as e:
                        logger.error(f"Error analyzing {category} category (daily): {e}")
            
            # Analyze weekly data (combine all categories)
            logger.info("Starting weekly timeframe analysis")
            weekly_results = []
            for category in categories:
                try:
                    category_stocks = self.fetch_stock_list(category)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        futures = []
                        for i in range(0, len(category_stocks), BATCH_SIZE):
                            batch = category_stocks[i:i+BATCH_SIZE]
                            futures.append(executor.submit(self.analyze_batch, batch, "weekly"))
                        
                        for future in concurrent.futures.as_completed(futures):
                            batch_results = future.result()
                            if batch_results:
                                weekly_results.extend(batch_results)
                except Exception as e:
                    logger.error(f"Error analyzing {category} category weekly data: {e}")
            
            # Update weekly results in global dictionary
            with results_lock:
                latest_results["weekly"] = weekly_results
            
            # Analyze monthly data (combine all categories)
            logger.info("Starting monthly timeframe analysis")
            monthly_results = []
            for category in categories:
                try:
                    category_stocks = self.fetch_stock_list(category)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        futures = []
                        for i in range(0, len(category_stocks), BATCH_SIZE):
                            batch = category_stocks[i:i+BATCH_SIZE]
                            futures.append(executor.submit(self.analyze_batch, batch, "monthly"))
                        
                        for future in concurrent.futures.as_completed(futures):
                            batch_results = future.result()
                            if batch_results:
                                monthly_results.extend(batch_results)
                except Exception as e:
                    logger.error(f"Error analyzing {category} category monthly data: {e}")
            
            # Update monthly results in global dictionary
            with results_lock:
                latest_results["monthly"] = monthly_results
                latest_results["last_update"] = current_time
                latest_results["next_update"] = next_update
                latest_results["status"] = "idle"
            
            # Format and send reports for each category and timeframe
            # Daily reports by category
            for category, results in all_results.items():
                message = f"📊 *NSE {category.upper()} CAP ANALYSIS (DAILY)* 📊\n"
                message += f"*Date & Time (IST):* {current_time}\n"
                message += f"*Next Update (IST):* {next_update}\n\n"
                message += self.format_results_table(results)
                
                self.send_telegram_message(message)
                
                # Save to file
                with open(f"{category}_cap_analysis.txt", "w") as f:
                    f.write(message)
                
                logger.info(f"Analysis for {category} cap completed and saved")
            
            # Weekly report
            if weekly_results:
                message = f"📊 *NSE WEEKLY TIMEFRAME ANALYSIS* 📊\n"
                message += f"*Date & Time (IST):* {current_time}\n"
                message += f"*Next Update (IST):* {next_update}\n\n"
                message += self.format_results_table(weekly_results)
                
                self.send_telegram_message(message)
                
                # Save to file
                with open("
