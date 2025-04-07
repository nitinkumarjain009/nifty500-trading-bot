# main.py
import pandas as pd
import numpy as np
import yfinance as yf
import time
import schedule
import requests
import logging
from datetime import datetime, timedelta
import os
import json
from typing import List, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '8017759392:AAEwM-W-y83lLXTjlPl8sC_aBmizuIrFXnU')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '711856868')
WHATSAPP_NUMBER = os.environ.get('WHATSAPP_NUMBER', '+918376906697')
WHATSAPP_API_URL = os.environ.get('WHATSAPP_API_URL', 'https://api.whatsapp.com/send')  # Example URL, you'll need a proper WhatsApp Business API

# Load Nifty 500 stocks
def load_nifty500_stocks() -> List[str]:
    try:
        # You might need to update this with a proper source for Nifty 500 stocks
        # This is a placeholder - in production, fetch from NSE website or use a local file
        url = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
        df = pd.read_csv(url)
        # Convert to Yahoo Finance compatible symbols (NSE: prefix)
        symbols = [f"{symbol}.NS" for symbol in df['Symbol'].tolist()]
        logger.info(f"Loaded {len(symbols)} Nifty 500 stocks")
        return symbols
    except Exception as e:
        logger.error(f"Error loading Nifty 500 stocks: {e}")
        # Return some sample stocks as fallback
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

# Calculate SuperTrend indicator
def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: int = 3) -> pd.DataFrame:
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Calculate ATR
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift()))
    tr3 = pd.DataFrame(abs(low - close.shift()))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    # Calculate SuperTrend
    upper_band = (high + low) / 2 + multiplier * atr
    lower_band = (high + low) / 2 - multiplier * atr
    
    super_trend = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)
    
    for i in range(period, len(df)):
        if close.iloc[i] > upper_band.iloc[i-1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
            
            if direction.iloc[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i-1]:
                lower_band.iloc[i] = lower_band.iloc[i-1]
            if direction.iloc[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i-1]:
                upper_band.iloc[i] = upper_band.iloc[i-1]
        
        if direction.iloc[i] == 1:
            super_trend.iloc[i] = lower_band.iloc[i]
        else:
            super_trend.iloc[i] = upper_band.iloc[i]
    
    df['SuperTrend'] = super_trend
    df['SuperTrend_Direction'] = direction
    
    return df

# Calculate Chandelier Exit indicator
def calculate_chandelier_exit(df: pd.DataFrame, period: int = 22, multiplier: int = 3) -> pd.DataFrame:
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Calculate ATR
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift()))
    tr3 = pd.DataFrame(abs(low - close.shift()))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    # Calculate highest high and lowest low for the period
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    # Calculate long exit and short exit
    long_exit = highest_high - multiplier * atr
    short_exit = lowest_low + multiplier * atr
    
    df['ChandelierLongExit'] = long_exit
    df['ChandelierShortExit'] = short_exit
    
    # Determine Chandelier Exit signals
    df['ChandelierExitLong'] = close > long_exit
    df['ChandelierExitShort'] = close < short_exit
    
    return df

# Get stock data
def get_stock_data(symbol: str, interval: str = '30m', period: str = '7d') -> Optional[pd.DataFrame]:
    try:
        data = yf.download(symbol, interval=interval, period=period)
        if data.empty:
            logger.warning(f"No data found for {symbol}")
            return None
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None

# Generate trading signals
def generate_signals(symbol: str) -> Optional[Dict]:
    try:
        # Get stock data
        df = get_stock_data(symbol)
        if df is None or len(df) < 25:
            return None
        
        # Calculate indicators
        df = calculate_supertrend(df)
        df = calculate_chandelier_exit(df)
        
        # Check for signals in the latest candle
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        signal = None
        reason = []
        
        # SuperTrend signal
        if previous['SuperTrend_Direction'] == -1 and latest['SuperTrend_Direction'] == 1:
            signal = "BUY"
            reason.append("SuperTrend turned bullish")
        elif previous['SuperTrend_Direction'] == 1 and latest['SuperTrend_Direction'] == -1:
            signal = "SELL"
            reason.append("SuperTrend turned bearish")
            
        # Chandelier Exit signal
        if previous['ChandelierExitLong'] == False and latest['ChandelierExitLong'] == True:
            if signal != "SELL":  # Avoid conflicting signals
                signal = "BUY"
                reason.append("Chandelier Exit Long signal")
        elif previous['ChandelierExitShort'] == False and latest['ChandelierExitShort'] == True:
            if signal != "BUY":  # Avoid conflicting signals
                signal = "SELL"
                reason.append("Chandelier Exit Short signal")
        
        if signal:
            return {
                'symbol': symbol,
                'signal': signal,
                'price': latest['Close'],
                'time': df.index[-1].strftime('%Y-%m-%d %H:%M'),
                'reason': ', '.join(reason)
            }
        return None
    except Exception as e:
        logger.error(f"Error generating signals for {symbol}: {e}")
        return None

# Send Telegram message
def send_telegram_message(message: str) -> bool:
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'Markdown'
        }
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            logger.info("Telegram message sent successfully")
            return True
        else:
            logger.error(f"Failed to send Telegram message: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error sending Telegram message: {e}")
        return False

# Send WhatsApp message (This is a placeholder - you'll need to integrate with a proper WhatsApp API)
def send_whatsapp_message(message: str) -> bool:
    try:
        # This is a placeholder for WhatsApp Business API integration
        # In a real implementation, you would use a proper WhatsApp Business API
        logger.info(f"WhatsApp message would be sent to {WHATSAPP_NUMBER}: {message}")
        
        # Example code for when you have a proper API integration:
        # url = "https://your-whatsapp-api-endpoint.com/send"
        # payload = {
        #     'phone': WHATSAPP_NUMBER,
        #     'message': message
        # }
        # response = requests.post(url, json=payload, headers={'Authorization': 'your-auth-token'})
        # return response.status_code == 200
        
        return True
    except Exception as e:
        logger.error(f"Error sending WhatsApp message: {e}")
        return False

# Format signal message
def format_signal_message(signal: Dict) -> str:
    emoji = "🟢" if signal['signal'] == "BUY" else "🔴"
    return f"{emoji} *{signal['signal']} SIGNAL*\n\n" \
           f"*Stock:* {signal['symbol'].replace('.NS', '')}\n" \
           f"*Price:* ₹{signal['price']:.2f}\n" \
           f"*Time:* {signal['time']}\n" \
           f"*Reason:* {signal['reason']}\n\n" \
           f"*Indicators:* SuperTrend & Chandelier Exit (30m)"

# Process all stocks for signals
def process_stocks():
    logger.info("Starting stock analysis...")
    stocks = load_nifty500_stocks()
    signals_found = False
    
    for symbol in stocks:
        signal = generate_signals(symbol)
        if signal:
            signals_found = True
            message = format_signal_message(signal)
            logger.info(f"Signal found for {symbol}: {signal['signal']}")
            
            # Send notifications
            send_telegram_message(message)
            send_whatsapp_message(message.replace('*', ''))  # Remove markdown for WhatsApp
            
            # Rate limiting to avoid API throttling
            time.sleep(1)
    
    if not signals_found:
        logger.info("No signals found in this scan")

# Main function to schedule jobs
def main():
    logger.info("Starting Nifty 500 Stock Trading Signal Bot")
    
    # Define market hours (Indian Standard Time)
    # Market hours: 9:15 AM to 3:30 PM, Monday to Friday
    
    # Run immediately once at startup
    process_stocks()
    
    # Schedule to run every 30 minutes during market hours
    # Adjust timings according to your server's timezone
    for hour in range(9, 16):  # 9 AM to 3 PM
        for minute in [15, 45]:  # Run at xx:15 and xx:45
            if hour == 15 and minute == 45:
                continue  # Skip 3:45 PM as market closes at 3:30 PM
            
            schedule_time = f"{hour:02d}:{minute:02d}"
            schedule.every().monday.at(schedule_time).do(process_stocks)
            schedule.every().tuesday.at(schedule_time).do(process_stocks)
            schedule.every().wednesday.at(schedule_time).do(process_stocks)
            schedule.every().thursday.at(schedule_time).do(process_stocks)
            schedule.every().friday.at(schedule_time).do(process_stocks)
    
    logger.info("Bot scheduled successfully")
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()
