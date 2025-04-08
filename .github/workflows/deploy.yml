#!/usr/bin/env python
# NSE Stock Analyzer with Supertrend, RSI, Telegram Notifications and Web Service

import subprocess
import sys
import os
import time
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import schedule
import logging
import threading
from flask import Flask, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Function to install required packages
def install_required_packages():
    required_packages = [
        'pandas', 'numpy', 'requests', 'nsepy', 'nsetools', 
        'schedule', 'python-telegram-bot', 'tabulate', 'flask'
    ]
    
    logger.info("Checking and installing required packages...")
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"{package} is already installed.")
        except ImportError:
            logger.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logger.info(f"{package} has been installed.")

# Install required packages
install_required_packages()

# Now import the packages that were installed
from nsepy import get_history
from nsetools import Nse
from tabulate import tabulate
import telegram

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

# Create Flask app
app = Flask(__name__)

# Global variable to store latest results
latest_results = {
    "large": [],
    "mid": [],
    "small": [],
    "last_update": None
}

class StockAnalyzer:
    def __init__(self):
        self.nse = Nse()
        self.bot = self.setup_telegram_bot()
    
    def setup_telegram_bot(self):
        try:
            bot = telegram.Bot(token=TELEGRAM_TOKEN)
            logger.info("Telegram bot initialized successfully")
            return bot
        except Exception as e:
            logger.error(f"Error setting up Telegram bot: {e}")
            return None
    
    def fetch_stock_list(self, category="large"):
        try:
            if category.lower() == "large":
                df = pd.read_csv(NIFTY_LARGE_CAP_URL)
                return df['Symbol'].tolist()
            elif category.lower() == "mid":
                df = pd.read_csv(NIFTY_MID_CAP_URL)
                return df['Symbol'].tolist()
            elif category.lower() == "small":
                df = pd.read_csv(NIFTY_SMALL_CAP_URL)
                return df['Symbol'].tolist()
            else:
                raise ValueError("Invalid category. Choose from 'large', 'mid', or 'small'")
        except Exception as e:
            logger.error(f"Error fetching stock list for {category} cap: {e}")
            return []
    
    def get_stock_data(self, symbol, days=100):
        try:
            end_date = datetime.now().date()
            data = get_history(symbol=symbol, start=end_date - pd.Timedelta(days=days), end=end_date)
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_atr(self, high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
    
    def calculate_supertrend(self, df, period=SUPERTREND_PERIOD, multiplier=SUPERTREND_MULTIPLIER):
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # Calculate ATR
        atr = self.calculate_atr(high, low, close, period)
        
        # Calculate basic upper and lower bands
        hl2 = (high + low) / 2
        basic_upperband = hl2 + (multiplier * atr)
        basic_lowerband = hl2 - (multiplier * atr)
        
        # Initialize final upper and lower bands
        final_upperband = basic_upperband.copy()
        final_lowerband = basic_lowerband.copy()
        
        # Initialize Supertrend
        supertrend = pd.Series(0, index=df.index)
        
        # Calculate final upper and lower bands and supertrend
        for i in range(period, len(df)):
            if close[i] > final_upperband[i-1]:
                final_upperband[i] = basic_upperband[i]
            else:
                final_upperband[i] = min(final_upperband[i-1], basic_upperband[i])
                
            if close[i] < final_lowerband[i-1]:
                final_lowerband[i] = basic_lowerband[i]
            else:
                final_lowerband[i] = max(final_lowerband[i-1], basic_lowerband[i])
            
            if close[i] <= final_upperband[i]:
                supertrend[i] = final_upperband[i]
            else:
                supertrend[i] = final_lowerband[i]
        
        # Determine trend (1 for uptrend, -1 for downtrend)
        df['Supertrend'] = supertrend
        df['Supertrend_Direction'] = 0
        df.loc[close > supertrend, 'Supertrend_Direction'] = 1
        df.loc[close <= supertrend, 'Supertrend_Direction'] = -1
        
        return df
    
    def calculate_rsi(self, df, period=RSI_PERIOD):
        close = df['Close']
        delta = close.diff()
        
        # Make two series: one for gains and one for losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and average loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        df['RSI'] = rsi
        return df
    
    def analyze_stock(self, symbol):
        try:
            # Get stock data
            data = self.get_stock_data(symbol)
            
            if data.empty:
                return None
            
            # Calculate indicators
            data = self.calculate_supertrend(data)
            data = self.calculate_rsi(data)
            
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
            
            return {
                'Symbol': symbol,
                'Price': round(current_price, 2),
                'Supertrend': round(supertrend_value, 2),
                'RSI': round(rsi_value, 2),
                'Signal': signal
            }
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def analyze_category(self, category):
        stocks = self.fetch_stock_list(category)
        results = []
        
        logger.info(f"Analyzing {len(stocks)} stocks from {category} cap category")
        
        for symbol in stocks:
            try:
                result = self.analyze_stock(symbol)
                if result:
                    results.append(result)
                    logger.info(f"Analyzed {symbol}: Signal = {result['Signal']}")
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
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
            
            # Check if market is open (Monday to Friday, 9:15 AM to 3:30 PM IST)
            now = datetime.now()
            is_weekend = now.weekday() >= 5  # 5: Saturday, 6: Sunday
            market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
            is_market_hours = market_start <= now <= market_end
            
            if is_weekend:
                logger.info("Market closed (Weekend). Skipping analysis.")
                return
            
            if not is_market_hours:
                logger.info("Outside market hours. Running analysis anyway for demonstration.")
                # Note: For production, you may want to uncomment the following line
                # return
            
            # Analyze each category
            categories = ["large", "mid", "small"]
            all_results = {}
            
            for category in categories:
                results = self.analyze_category(category)
                all_results[category] = results
                
                # Update global results - proper assignment to the global variable
                latest_results[category] = results
            
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

# Background thread for scheduled tasks
def run_scheduler():
    analyzer = StockAnalyzer()
    
    # Run once at startup
    analyzer.run_analysis()
    
    # Schedule runs
    # During market hours on weekdays (9:30 AM to 3:15 PM), run every 30 minutes
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

# Flask routes
@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>NSE Stock Analyzer</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                .container { max-width: 800px; margin: 0 auto; }
                .links { margin: 20px 0; }
                .links a { display: inline-block; margin-right: 15px; padding: 10px; 
                           background-color: #0066cc; color: white; text-decoration: none; 
                           border-radius: 5px; }
                .status { margin: 20px 0; padding: 15px; background-color: #f0f0f0; 
                         border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>NSE Stock Analyzer</h1>
                <div class="status">
                    <p><strong>Status:</strong> Running</p>
                    <p><strong>Last Update:</strong> {last_update}</p>
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
    """.format(last_update=latest_results["last_update"] or "Not yet run")

@app.route('/api/analyze')
def trigger_analysis():
    analyzer = StockAnalyzer()
    analyzer.run_analysis()
    return jsonify({"status": "Analysis triggered", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

@app.route('/api/results')
def get_results():
    return jsonify(latest_results)

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"})

def main():
    # Start the scheduler in a background thread
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()
    
    # Start the Flask app
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
