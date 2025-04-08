#!/usr/bin/env python
# NSE Stock Analyzer with Supertrend, RSI, Telegram Notifications and Web Service

import os
import sys
import subprocess
import time
from datetime import datetime
import json
import threading
import logging
import concurrent.futures
from functools import lru_cache

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
        'schedule', 'python-telegram-bot==13.7', 'tabulate', 'flask'
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
    from tabulate import tabulate
    from flask import Flask, jsonify
    
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

# Telegram Bot Configuration
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")

# Technical Analysis Parameters
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Stock Categories
NIFTY_LARGE_CAP_URL = "https://archives.nseindia.com/content/indices/ind_nifty100list.csv"
NIFTY_MID_CAP_URL = "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv"
NIFTY_SMALL_CAP_URL = "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv"

# Performance settings
MAX_WORKERS = os.environ.get("MAX_WORKERS", 10)  # Number of parallel workers
BATCH_SIZE = 10  # Process stocks in batches to avoid rate limiting
CACHE_TIMEOUT = 300  # Cache timeout in seconds

# Global variable to store latest results
latest_results = {
    "large": [],
    "mid": [],
    "small": [],
    "last_update": None
}

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
    
    def get_stock_data(self, symbol, days=100):
        current_time = time.time()
        
        # Check if we have a cached version
        if symbol in self.data_cache and current_time - self.last_cache_time.get(symbol, 0) < CACHE_TIMEOUT:
            return self.data_cache[symbol]
        
        try:
            end_date = datetime.now().date()
            start_date = end_date - pd.Timedelta(days=days)
            
            # Try to get data with retry logic
            retries = 3
            for attempt in range(retries):
                try:
                    data = get_history(symbol=symbol, start=start_date, end=end_date)
                    if not data.empty:
                        logger.info(f"Successfully fetched data for {symbol}")
                        # Cache the result
                        self.data_cache[symbol] = data
                        self.last_cache_time[symbol] = current_time
                        return data
                    else:
                        logger.warning(f"Empty data returned for {symbol}, attempt {attempt+1}/{retries}")
                except Exception as inner_e:
                    logger.warning(f"Attempt {attempt+1}/{retries} failed for {symbol}: {inner_e}")
                    time.sleep(1)  # Reduced wait time before retry
            
            logger.error(f"Failed to fetch data for {symbol} after {retries} attempts")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error in get_stock_data for {symbol}: {e}")
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
    
    def analyze_stock(self, symbol):
        try:
            # Get stock data
            data = self.get_stock_data(symbol)
            
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return None
            
            # Calculate indicators
            data = self.calculate_supertrend(data)
            data = self.calculate_rsi(data)
            
            if len(data) < 2:
                logger.warning(f"Insufficient data points for {symbol}")
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
            
            result = {
                'Symbol': symbol,
                'Price': round(current_price, 2),
                'Supertrend': round(supertrend_value, 2),
                'RSI': round(rsi_value, 2),
                'Signal': signal
            }
            
            return result
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def analyze_batch(self, symbols):
        """Analyze a batch of symbols"""
        results = []
        for symbol in symbols:
            try:
                result = self.analyze_stock(symbol)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        return results
    
    def analyze_category(self, category):
        """Analyze a category using parallel processing"""
        stocks = self.fetch_stock_list(category)
        results = []
        
        logger.info(f"Analyzing {len(stocks)} stocks from {category} cap category")
        
        # Remove testing limit to process all stocks
        # stocks = stocks[:5]  # This line is commented out
        
        # Process in batches with parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(MAX_WORKERS)) as executor:
            futures = []
            # Create batches to avoid overwhelming the API
            for i in range(0, len(stocks), BATCH_SIZE):
                batch = stocks[i:i+BATCH_SIZE]
                futures.append(executor.submit(self.analyze_batch, batch))
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                batch_results = future.result()
                if batch_results:
                    results.extend(batch_results)
        
        logger.info(f"Completed analysis for {category} cap with {len(results)} results")
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
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info("Starting market analysis at " + current_time)
            
            # Analyze each category
            categories = ["large", "mid", "small"]
            all_results = {}
            
            # Use ThreadPoolExecutor to analyze categories in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(categories)) as executor:
                futures = {executor.submit(self.analyze_category, category): category for category in categories}
                for future in concurrent.futures.as_completed(futures):
                    category = futures[future]
                    try:
                        results = future.result()
                        all_results[category] = results
                        latest_results[category] = results
                    except Exception as e:
                        logger.error(f"Error analyzing {category} category: {e}")
            
            latest_results["last_update"] = current_time
            
            # Format and send reports
            for category, results in all_results.items():
                message = f"📊 *NSE {category.upper()} CAP ANALYSIS* 📊\n"
                message += f"*Date & Time:* {current_time}\n\n"
                message += self.format_results_table(results)
                
                self.send_telegram_message(message)
                
                # Save to file
                with open(f"{category}_cap_analysis.txt", "w") as f:
                    f.write(message)
                
                logger.info(f"Analysis for {category} cap completed and saved")
            
            logger.info("Market analysis completed")
            return all_results
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            return None

# Flask routes
@app.route('/')
def index():
    last_update_value = latest_results.get("last_update") or "Not yet run"
    
    return """
    <html>
        <head>
            <title>NSE Stock Analyzer</title>
            <style>
                body {font-family: Arial, sans-serif; margin: 20px;}
                h1 {color: #333;}
                .container {max-width: 800px; margin: 0 auto;}
                .links {margin: 20px 0;}
                .links a {display: inline-block; margin-right: 15px; padding: 10px; 
                         background-color: #0066cc; color: white; text-decoration: none; 
                         border-radius: 5px;}
                .status {margin: 20px 0; padding: 15px; background-color: #f0f0f0; 
                        border-radius: 5px;}
                .performance {margin: 20px 0; background-color: #e8f5e9; padding: 15px;
                             border-radius: 5px;}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>NSE Stock Analyzer</h1>
                <div class="status">
                    <p><strong>Status:</strong> Running</p>
                    <p><strong>Last Update:</strong> """ + last_update_value + """</p>
                </div>
                <div class="performance">
                    <p><strong>Performance:</strong> Optimized for speed with parallel processing</p>
                    <p><strong>Workers:</strong> """ + str(MAX_WORKERS) + """</p>
                </div>
                <div class="links">
                    <a href="/api/analyze">Run Analysis Now</a>
                    <a href="/api/results">View Latest Results</a>
                </div>
                <div>
                    <h2>About</h2>
                    <p>This service analyzes NSE stocks using Supertrend and RSI indicators, 
                    identifies potential buy/sell opportunities, and sends alerts to Telegram.</p>
                </div>
            </div>
        </body>
    </html>
    """

@app.route('/api/analyze')
def trigger_analysis():
    # Run analysis in a separate thread to avoid blocking the response
    threading.Thread(target=lambda: StockAnalyzer().run_analysis()).start()
    return jsonify({
        "status": "Analysis triggered", 
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": "Analysis is running in background. Check results in a few minutes."
    })

@app.route('/api/results')
def get_results():
    return jsonify(latest_results)

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"})

# Environment configuration endpoint
@app.route('/api/config')
def get_config():
    return jsonify({
        "MAX_WORKERS": MAX_WORKERS,
        "BATCH_SIZE": BATCH_SIZE,
        "CACHE_TIMEOUT": CACHE_TIMEOUT
    })

# Background thread for scheduled tasks
def run_scheduler():
    analyzer = StockAnalyzer()
    
    # Run once at startup
    analyzer.run_analysis()
    
    # Schedule runs - every 30 minutes during market hours
    for hour in range(9, 16):
        for minute in [0, 30]:
            if (hour == 9 and minute < 30) or (hour == 15 and minute > 15):
                continue
            schedule.every().monday.at(f"{hour:02}:{minute:02}").do(analyzer.run_analysis)
            schedule.every().tuesday.at(f"{hour:02}:{minute:02}").do(analyzer.run_analysis)
            schedule.every().wednesday.at(f"{hour:02}:{minute:02}").do(analyzer.run_analysis)
            schedule.every().thursday.at(f"{hour:02}:{minute:02}").do(analyzer.run_analysis)
            schedule.every().friday.at(f"{hour:02}:{minute:02}").do(analyzer.run_analysis)
    
    logger.info("Scheduler has been set up. The script will run during market hours.")
    
    # Keep the scheduler running
    while True:
        schedule.run_pending()
        time.sleep(60)

def main():
    try:
        # Start the scheduler in a background thread
        scheduler_thread = threading.Thread(target=run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        
        # Start the Flask app
        port = int(os.environ.get("PORT", 8080))
        app.run(host="0.0.0.0", port=port)
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
