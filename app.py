# app.py
import os
import pandas as pd
import numpy as np
from numpy import nan as NaN
import yfinance as yf
import pytz
import datetime
import requests
import pandas_ta as ta
from flask import Flask, render_template, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Constants
TELEGRAM_BOT_TOKEN = "8017759392:AAEwM-W-y83lLXTjlPl8sC_aBmizuIrFXnU"
TELEGRAM_CHAT_ID = "-1001234567890"  # Replace with your actual chat ID
IST = pytz.timezone('Asia/Kolkata')

# Dictionary to store the previous analysis results
previous_signals = {}

# NSE Stock Categories
LARGE_CAP_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS"
]

MID_CAP_STOCKS = [
    "BERGEPAINT.NS", "GODREJCP.NS", "HAVELLS.NS", "MPHASIS.NS", "OFSS.NS",
    "COFORGE.NS", "LTTS.NS", "TATACOMM.NS", "AUROPHARMA.NS", "TRENT.NS",
    "ABCAPITAL.NS", "TVSMOTOR.NS", "MFSL.NS", "PERSISTENT.NS", "CUMMINSIND.NS"
]

SMALL_CAP_STOCKS = [
    "CAMS.NS", "POLYMED.NS", "METROPOLIS.NS", "FINEORG.NS", "GRINDWELL.NS",
    "VGUARD.NS", "AJANTPHARM.NS", "CESC.NS", "REDINGTON.NS", "JKLAKSHMI.NS",
    "ESCORTS.NS", "ZYDUSWELL.NS", "CEATLTD.NS", "IRB.NS", "BALRAMCHIN.NS"
]

ALL_STOCKS = {"large_cap": LARGE_CAP_STOCKS, "mid_cap": MID_CAP_STOCKS, "small_cap": SMALL_CAP_STOCKS}

# Analysis results storage
last_analysis_results = {}
last_analysis_time = None

def get_stock_data(symbol, period="1mo"):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        if data.empty:
            return None
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_rsi(data):
    """Calculate RSI indicator using pandas_ta"""
    if data is None or len(data) < 14:
        return None
    
    # Calculate RSI using pandas_ta
    data['RSI'] = ta.rsi(data['Close'], length=14)
    
    return data

def generate_signals(data, symbol):
    """Generate buy/sell signals based on RSI levels"""
    if data is None or len(data) < 2:
        return {"symbol": symbol, "signal": "No Data", "price": 0, "rsi": 0, "change_pct": 0}
    
    # Get the latest data point
    latest = data.iloc[-1]
    prev_day = data.iloc[-2]
    
    current_price = latest['Close']
    rsi = latest['RSI']
    
    # Calculate daily change percentage
    daily_change = ((current_price - prev_day['Close']) / prev_day['Close']) * 100
    
    # Generate signals based on RSI
    signal = "Hold"
    
    # RSI-based signals
    if rsi <= 30:
        signal = "Strong Buy"
    elif rsi > 30 and rsi <= 40:
        signal = "Buy"
    elif rsi >= 70:
        signal = "Strong Sell"
    elif rsi >= 60 and rsi < 70:
        signal = "Sell"
    
    return {
        "symbol": symbol,
        "signal": signal,
        "price": round(current_price, 2),
        "rsi": round(rsi, 2),
        "change_pct": round(daily_change, 2)
    }

def analyze_stocks():
    """Analyze all stocks and generate signals"""
    results = {"large_cap": [], "mid_cap": [], "small_cap": []}
    new_signals = []
    
    for category, stocks in ALL_STOCKS.items():
        for symbol in stocks:
            data = get_stock_data(symbol)
            if data is not None:
                data = calculate_rsi(data)
                if data is not None:
                    signal = generate_signals(data, symbol)
                    results[category].append(signal)
                    
                    # Check if we should send a Telegram alert (for Strong Buy/Sell only)
                    if symbol in previous_signals:
                        prev_signal = previous_signals[symbol]
                        curr_signal = signal["signal"]
                        
                        if prev_signal != curr_signal and ("Strong" in curr_signal):
                            new_signals.append(signal)
                    
                    # Update previous signal
                    previous_signals[symbol] = signal["signal"]
    
    # Store results for later access
    global last_analysis_results, last_analysis_time
    last_analysis_results = results
    last_analysis_time = datetime.datetime.now(IST)
    
    # Send combined Telegram alert at the end of the day
    if new_signals and is_after_market_hours():
        send_after_hours_summary(results, new_signals)
    
    return results

def is_after_market_hours():
    """Check if current time is after market hours (after 3:30 PM IST)"""
    current_time = datetime.datetime.now(IST)
    market_close_time = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
    return current_time.weekday() < 5 and current_time >= market_close_time

def get_top_picks(results):
    """Get top buy and sell recommendations"""
    all_stocks = []
    for category in results.values():
        all_stocks.extend(category)
    
    # Filter out stocks with no data
    valid_stocks = [stock for stock in all_stocks if stock["signal"] != "No Data"]
    
    # Sort for buy and sell recommendations
    buy_picks = [s for s in valid_stocks if "Buy" in s["signal"]]
    sell_picks = [s for s in valid_stocks if "Sell" in s["signal"]]
    
    # Sort by RSI (ascending for buy, descending for sell)
    buy_picks = sorted(buy_picks, key=lambda x: x["rsi"])[:5]  # Lowest RSI first for buy
    sell_picks = sorted(sell_picks, key=lambda x: -x["rsi"])[:5]  # Highest RSI first for sell
    
    return buy_picks, sell_picks

def send_after_hours_summary(results, new_signals):
    """Send after-hours summary to Telegram"""
    buy_picks, sell_picks = get_top_picks(results)
    
    # Construct message
    message = f"📊 *NSE STOCK ANALYSIS - AFTER HOURS SUMMARY* 📊\n"
    message += f"Date: {datetime.datetime.now(IST).strftime('%Y-%m-%d')}\n\n"
    
    # Top buy recommendations
    message += "*🟢 TOP BUY RECOMMENDATIONS 🟢*\n"
    for i, pick in enumerate(buy_picks, 1):
        message += f"{i}. {pick['symbol'].replace('.NS', '')}: ₹{pick['price']} (RSI: {pick['rsi']}) - {pick['signal']}\n"
    
    message += "\n*🔴 TOP SELL RECOMMENDATIONS 🔴*\n"
    for i, pick in enumerate(sell_picks, 1):
        message += f"{i}. {pick['symbol'].replace('.NS', '')}: ₹{pick['price']} (RSI: {pick['rsi']}) - {pick['signal']}\n"
    
    # New signals
    if new_signals:
        message += "\n*🔔 NEW SIGNALS TODAY 🔔*\n"
        for signal in new_signals:
            message += f"• {signal['symbol'].replace('.NS', '')}: {signal['signal']} at ₹{signal['price']} (RSI: {signal['rsi']})\n"
    
    # Add link to web dashboard
    message += "\n*🌐 View Detailed Analysis:*\n"
    message += "https://nifty500-trading-bot.onrender.com/\n"
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, json=payload)
        print(f"Telegram summary sent: {response.text}")
    except Exception as e:
        print(f"Error sending Telegram summary: {e}")

def generate_chart(symbol):
    """Generate stock chart with RSI and volume"""
    data = get_stock_data(symbol, period="3mo")
    if data is None:
        return None
    
    data = calculate_rsi(data)
    if data is None:
        return None
    
    # Create figure with subplots
    fig = Figure(figsize=(12, 8))
    
    # Price plot
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(data.index, data['Close'], 'b-', label='Close Price')
    
    # Add 20-day moving average
    data['MA20'] = data['Close'].rolling(window=20).mean()
    ax1.plot(data.index, data['MA20'], 'r--', label='20-day MA')
    
    # Add 50-day moving average
    data['MA50'] = data['Close'].rolling(window=50).mean()
    ax1.plot(data.index, data['MA50'], 'g--', label='50-day MA')
    
    ax1.set_title(f'{symbol} Price Chart')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    ax1.legend()
    
    # RSI plot
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(data.index, data['RSI'], 'g-')
    ax2.axhline(y=70, color='r', linestyle='-')
    ax2.axhline(y=30, color='g', linestyle='-')
    ax2.set_title('RSI (14)')
    ax2.set_ylabel('RSI')
    ax2.grid(True)
    
    # Volume plot
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.bar(data.index, data['Volume'])
    ax3.set_title('Volume')
    ax3.set_ylabel('Volume')
    ax3.grid(True)
    
    fig.tight_layout()
    
    # Convert plot to PNG image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode PNG image to base64 string
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return img_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze')
def analyze():
    # Use cached results if available and less than 1 hour old
    global last_analysis_results, last_analysis_time
    
    current_time = datetime.datetime.now(IST)
    force_refresh = request.args.get('force', 'false').lower() == 'true'
    
    # If no analysis yet or last analysis is more than 1 hour old or force refresh
    if not last_analysis_time or (current_time - last_analysis_time).total_seconds() > 3600 or force_refresh:
        results = analyze_stocks()
    else:
        results = last_analysis_results
        
    return jsonify({
        "results": results,
        "last_updated": last_analysis_time.strftime("%Y-%m-%d %H:%M:%S") if last_analysis_time else None
    })

@app.route('/chart/<symbol>')
def chart(symbol):
    img_str = generate_chart(symbol)
    if img_str:
        return jsonify({"image": img_str})
    return jsonify({"error": "Failed to generate chart"})

@app.route('/after-hours-analysis')
def after_hours_analysis():
    """API endpoint for after-hours analysis"""
    if not is_after_market_hours():
        return jsonify({"error": "This API is only available after market hours (3:30 PM IST onwards)"})
    
    results = analyze_stocks()
    buy_picks, sell_picks = get_top_picks(results)
    
    return jsonify({
        "date": datetime.datetime.now(IST).strftime("%Y-%m-%d"),
        "top_buys": buy_picks,
        "top_sells": sell_picks,
        "market_status": "closed",
        "dashboard_url": "https://nifty500-trading-bot.onrender.com/"
    })

def scheduled_after_hours_analysis():
    """Perform scheduled after-hours analysis"""
    current_time = datetime.datetime.now(IST)
    
    # Run analysis after market closes (at 4:00 PM IST) on weekdays
    if current_time.weekday() < 5 and current_time.hour == 16 and current_time.minute == 0:
        print(f"Running after-hours analysis at {current_time}")
        analyze_stocks()

# Set up the scheduler
scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
scheduler.add_job(scheduled_after_hours_analysis, 'cron', hour=16, minute=0)

# Create templates directory if it doesn't exist
os.makedirs('templates', exist_ok=True)

# Create index.html template
with open('templates/index.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>NSE Stock Analyzer - After Hours Edition</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
        }
        .strong-buy { color: darkgreen; font-weight: bold; }
        .buy { color: green; font-weight: bold; }
        .strong-sell { color: darkred; font-weight: bold; }
        .sell { color: red; font-weight: bold; }
        .hold { color: gray; }
        .chart-container {
            margin-top: 30px;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .stock-tabs {
            margin-bottom: 20px;
        }
        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(0, 0, 0, 0.3);
            border-radius: 50%;
            border-top-color: #000;
            animation: spin 1s ease-in-out infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .timestamp {
            font-size: 12px;
            color: #666;
            margin-bottom: 10px;
        }
        .card {
            margin-bottom: 20px;
            box-shadow: 0 0 5px rgba(0,0,0,0.1);
        }
        .header-panel {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 0 5px rgba(0,0,0,0.1);
        }
        .positive-change { color: green; }
        .negative-change { color: red; }
        .tab-content {
            background-color: white;
            padding: 20px;
            border-radius: 0 0 8px 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .summary-section {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .after-hours-banner {
            background-color: #343a40;
            color: white;
            padding: 10px 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
        }
        .after-hours-banner a {
            color: #17a2b8;
            text-decoration: none;
        }
        .after-hours-banner a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header-panel">
            <div class="row">
                <div class="col-md-8">
                    <h1>NSE Stock Analyzer</h1>
                    <h4>After Hours Analysis</h4>
                    <p class="timestamp" id="timestamp">Loading...</p>
                </div>
                <div class="col-md-4 text-end">
                    <button class="btn btn-primary" onclick="loadData(true)">Refresh Data</button>
                </div>
            </div>
        </div>
        
        <!-- After Hours Banner -->
        <div class="after-hours-banner" id="after-hours-banner" style="display: none;">
            <h4><i class="bi bi-moon-stars"></i> After Hours Mode Active</h4>
            <p>Market is closed. Analysis is based on today's closing prices.</p>
            <p>Visit <a href="https://nifty500-trading-bot.onrender.com/" target="_blank">https://nifty500-trading-bot.onrender.com/</a> for more detailed analysis.</p>
        </div>
        
        <!-- Top Picks Summary -->
        <div class="summary-section">
            <h3>Today's Top Picks</h3>
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header bg-success text-white">
                            <h5 class="mb-0">Top Buy Recommendations</h5>
                        </div>
                        <div class="card-body">
                            <div id="top-buys-loading">Loading... <div class="loading-spinner"></div></div>
                            <table class="table table-hover" id="top-buys-table">
                                <thead>
                                    <tr>
                                        <th>Symbol</th>
                                        <th>Price</th>
                                        <th>RSI</th>
                                        <th>Signal</th>
                                    </tr>
                                </thead>
                                <tbody></tbody>
                            </table>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header bg-danger text-white">
                            <h5 class="mb-0">Top Sell Recommendations</h5>
                        </div>
                        <div class="card-body">
                            <div id="top-sells-loading">Loading... <div class="loading-spinner"></div></div>
                            <table class="table table-hover" id="top-sells-table">
                                <thead>
                                    <tr>
                                        <th>Symbol</th>
                                        <th>Price</th>
                                        <th>RSI</th>
                                        <th>Signal</th>
                                    </tr>
                                </thead>
                                <tbody></tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- All Stocks by Category -->
        <div class="stock-tabs">
            <ul class="nav nav-tabs" id="stockTabs" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active" id="large-tab" data-bs-toggle="tab" data-bs-target="#large" type="button" role="tab">Large Cap</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="mid-tab" data-bs-toggle="tab" data-bs-target="#mid" type="button" role="tab">Mid Cap</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="small-tab" data-bs-toggle="tab" data-bs-target="#small" type="button" role="tab">Small Cap</button>
                </li>
            </ul>
            <div class="tab-content" id="stockTabContent">
                <div class="tab-pane fade show active" id="large" role="tabpanel">
                    <div id="large-loading" class="mt-3">Loading... <div class="loading-spinner"></div></div>
                    <table class="table table-hover" id="large-cap-table">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Price</th>
                                <th>Change %</th>
                                <th>RSI</th>
                                <th>Signal</th>
                                <th>Action</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
                <div class="tab-pane fade" id="mid" role="tabpanel">
                    <div id="mid-loading" class="mt-3">Loading... <div class="loading-spinner"></div></div>
                    <table class="table table-hover" id="mid-cap-table">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Price</th>
                                <th>Change %</th>
                                <th>RSI</th>
                                <th>Signal</th>
                                <th>Action</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
                <div class="tab-pane fade" id="small" role="tabpanel">
                    <div id="small-loading" class="mt-3">Loading... <div class="loading-spinner"></div></div>
                    <table class="table table-hover" id="small-cap-table">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Price</th>
                                <th>Change %</th>
                                <th>RSI</th>
                                <th>Signal</th>
                                <th>Action</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <!-- Stock Chart -->
        <div class="chart-container">
            <h3>Stock Chart Analysis</h3>
            <p>Select a stock from the table to view detailed analysis</p>
            <div id="chart-loading" style="display: none;">Loading chart... <div class="loading-spinner"></div></div>
            <div id="chart"></div>
        </div>
        
        <!-- After Hours Analysis Section -->
        <div class="summary-section" id="after-hours-section" style="display: none;">
            <h3>After Hours Trading Strategy</h3>
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header bg-dark text-white">
                            <h5 class="mb-0">Next Day Trading Plan</h5>
                        </div>
                        <div class="card-body">
                            <h6>Buy Strategy:</h6>
                            <p>Consider accumulating positions in stocks with "Strong Buy" signals, especially when RSI is below 30.</p>
                            <ul id="strong-buy-list">
                                <li>Loading strong buy recommendations...</li>
                            </ul>
                            
                            <h6>Sell Strategy:</h6>
                            <p>Consider booking profits on stocks with "Strong Sell" signals, especially when RSI is above 70.</p>
                            <ul id="strong-sell-list">
                                <li>Loading strong sell recommendations...</li>
                            </ul>
                            
                            <div class="alert alert-info mt-3">
                                <strong>Note:</strong> This is an algorithmic analysis based on RSI indicators. Always combine with your own research and risk management strategy.
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            loadData();
            checkMarketHours();
            
            // Refresh data every hour
            setInterval(loadData, 60 * 60 * 1000);
            setInterval(checkMarketHours, 5 * 60 * 1000);
        });
        
        function checkMarketHours() {
            const now = new Date();
            const hour = now.getHours();
            const minute = now.getMinutes();
            const day = now.getDay();
            
            // Check if it's after market hours (after 3:30 PM IST) on weekdays (0 = Sunday, 6 = Saturday)
            const isAfterHours = (day > 0 && day < 6) && ((hour > 15) || (hour === 15 && minute >= 30));
            
            // Show/hide after hours sections
            document.getElementById('after-hours-banner').style.display = isAfterHours ? 'block' : 'none';
            document.getElementById('after-hours-section').style.display = isAfterHours ? 'block' : 'none';
            
            if (isAfterHours) {
                updateAfterHoursStrategies();
            }
        }
        
        function updateAfterHoursStrategies() {
            // Get strong buy and strong sell stocks
            const allTables = ['large-cap-table', 'mid-cap-table', 'small-cap-table'];
            const strongBuys = [];
            const strongSells = [];
            
            allTables.forEach(tableId => {
                const rows = document.querySelectorAll(`#${tableId} tbody tr`);
                rows.forEach(row => {
                    const cells = row.querySelectorAll('td');
                    if (cells.length >= 5) {
                        const symbol = cells[0].textContent;
                        const price = cells[1].textContent;
                        const rsi = cells[3].textContent;
                        const signal = cells[4].textContent;
                        
                        if (signal === 'Strong Buy') {
                            strongBuys.push({ symbol, price, rsi });
                        } else if (signal === 'Strong Sell') {
                            strongSells.push({ symbol, price, rsi });
                        }
                    }
                });
            });
            
            // Update the lists
            const buyList = document.getElementById('strong-buy-list');
            const sellList = document.getElementById('strong-sell-list');
            
            buyList.innerHTML = '';
            sellList.innerHTML = '';
            
            if (strongBuys.length > 0) {
                strongBuys.forEach(stock => {
                    const li = document.createElement('li');
                    li.innerHTML = `<strong>${stock.symbol}</strong>: ${stock.price} (RSI: ${stock.rsi})`;
                    buyList.appendChild(li);
                });
            } else {
                buyList.innerHTML = '<li>No strong buy signals today</li>';
            }
            
            if (strongSells.length > 0) {
                strongSells.forEach(stock => {
                    const li = document.createElement('li');
                    li.innerHTML = `<strong>${stock.symbol}</strong>: ${stock.price} (RSI: ${stock.rsi})`;
                    sellList.appendChild(li);
                });
            } else {
                sellList.innerHTML = '<li>No strong sell signals today</li>';
            }
        }
        
        function loadData(forceRefresh = false) {
            document.getElementById('large-loading').style.display = 'block';
            document.getElementById('mid-loading').style.display = 'block';
            document.getElementById('small-loading').style.display = 'block';
            document.getElementById('top-buys-loading').style.display = 'block';
            document.getElementById('top-sells-loading').style.display = 'block';
            
            const url = forceRefresh ? '/analyze?force=true' : '/analyze';
            
            fetch(url)
                .then(response => response.json())
                .then(data => {
                    const results = data.results;
                    
                    updateTable('large-cap-table', results.large_cap);
                    updateTable('mid-cap-table', results.mid_cap);
                    updateTable('small-cap-table', results.small_cap);
                    
                    updateTopPicks(results);
                    
                    document.getElementById('timestamp').innerHTML = `Last updated: ${data.last_updated}`;
                    document.getElementById('large-loading').style.display = 'none';
                    document.getElementById('mid-loading').style.display = 'none';
                    document.getElementById('small-loading').style.display = 'none';
                    document.getElementById('top-buys-loading').style.display = 'none';
                    document.getElementById('top-sells-loading').style.display = 'none';
                })
                .catch(error => {
                    console.error('Error fetching stock data:', error);
                    document.getElementById('large-loading').style.display = 'none';
                    document.getElementById('mid-loading').style.display = 'none';
                    document.getElementById('small-loading').style.display = 'none';
                    document.getElementById('top-buys-loading').style.display = 'none';
                    document.getElementById('top-sells-loading').style.display = 'none';
                    alert('Error loading stock data. Please try again later.');
                });
        }
        
        function updateTable(tableId, data) {
            const tableBody = document.querySelector(`#${tableId} tbody`);
            tableBody.innerHTML = '';
            
            data.forEach(stock => {
                const row = document.createElement('tr');
                
                // Symbol column
                const symbolCell = document.createElement('td');
                symbolCell.textContent = stock.symbol.replace('.NS', '');
                row.appendChild(symbolCell);
                
                // Price column
                const priceCell = document.createElement('td');
                priceCell.textContent = stock.price;
                row.appendChild(priceCell);
                
                // Change percentage column
                const changeCell = document.createElement('td');
                if (stock.change_pct > 0) {
                    changeCell.textContent = `+${stock.change_pct}%`;
                    changeCell.className = 'positive-change';
                } else if (stock.change_pct < 0) {
                    changeCell.textContent = `${stock.change_pct}%`;
                    changeCell.className = 'negative-change';
                } else {
                    changeCell.textContent = '0.00%';
                }
                row.appendChild(changeCell);
                
                // RSI column
                const rsiCell = document.createElement('td');
                rsiCell.textContent = stock.rsi;
                row.appendChild(rsiCell);
                
                // Signal column
                const signalCell = document.createElement('td');
                signalCell.textContent = stock.signal;
                
                // Apply color class based on signal
                if (stock.signal === 'Strong Buy') {
                    signalCell.className = 'strong-buy';
                } else if (stock.signal === 'Buy') {
                    signalCell.className = 'buy';
                } else if (stock.signal === 'Strong Sell') {
                    signalCell.className = 'strong-sell';
                } else if (stock.signal === 'Sell') {
                    signalCell.className = 'sell';
                } else {
                    signalCell.className = 'hold';
                }
                
                row.appendChild(signalCell);
                
                // Action column - View Chart button
                const actionCell = document.createElement('td');
                if (stock.signal !== 'No Data') {
                    const chartButton = document.createElement('button');
                    chartButton.textContent = 'View Chart';
                    chartButton.className = 'btn btn-sm btn-outline-primary';
                    chartButton.onclick = function() {
                        loadChart(stock.symbol);
                    };
                    actionCell.appendChild(chartButton);
                } else {
                    actionCell.textContent = 'No Data';
                }
                row.appendChild(actionCell);
                
                tableBody.appendChild(row);
            });
        }
        
        function updateTopPicks(results) {
            // Combine all stocks
            let allStocks = [];
            
            for (const category in results) {
                allStocks = allStocks.concat(results[category]);
            }
            
            // Filter and sort for buy recommendations
            const buyStocks = allStocks.filter(stock => 
                stock.signal === 'Strong Buy' || stock.signal === 'Buy'
            ).sort((a, b) => a.rsi - b.rsi); // Sort by RSI ascending (lower is better for buying)
            
            // Filter and sort for sell recommendations
            const sellStocks = allStocks.filter(stock => 
                stock.signal === 'Strong Sell' || stock.signal === 'Sell'
            ).sort((a, b) => b.rsi - a.rsi); // Sort by RSI descending (higher is better for selling)
            
            // Take top 5 from each
            const topBuys = buyStocks.slice(0, 5);
            const topSells = sellStocks.slice(0, 5);
            
            // Update top buys table
            const buyTableBody = document.querySelector('#top-buys-table tbody');
            buyTableBody.innerHTML = '';
            
            if (topBuys.length > 0) {
                topBuys.forEach(stock => {
                    const row = document.createElement('tr');
                    
                    // Symbol
                    const symbolCell = document.createElement('td');
                    symbolCell.textContent = stock.symbol.replace('.NS', '');
                    row.appendChild(symbolCell);
                    
                    // Price
                    const priceCell = document.createElement('td');
                    priceCell.textContent = stock.price;
                    row.appendChild(priceCell);
                    
                    // RSI
                    const rsiCell = document.createElement('td');
                    rsiCell.textContent = stock.rsi;
                    row.appendChild(rsiCell);
                    
                    // Signal
                    const signalCell = document.createElement('td');
                    signalCell.textContent = stock.signal;
                    signalCell.className = stock.signal === 'Strong Buy' ? 'strong-buy' : 'buy';
                    row.appendChild(signalCell);
                    
                    buyTableBody.appendChild(row);
                });
            } else {
                const row = document.createElement('tr');
                const cell = document.createElement('td');
                cell.colSpan = 4;
                cell.textContent = 'No buy signals available';
                cell.style.textAlign = 'center';
                row.appendChild(cell);
                buyTableBody.appendChild(row);
            }
            
            // Update top sells table
            const sellTableBody = document.querySelector('#top-sells-table tbody');
            sellTableBody.innerHTML = '';
            
            if (topSells.length > 0) {
                topSells.forEach(stock => {
                    const row = document.createElement('tr');
                    
                    // Symbol
                    const symbolCell = document.createElement('td');
                    symbolCell.textContent = stock.symbol.replace('.NS', '');
                    row.appendChild(symbolCell);
                    
                    // Price
                    const priceCell = document.createElement('td');
                    priceCell.textContent = stock.price;
                    row.appendChild(priceCell);
                    
                    // RSI
                    const rsiCell = document.createElement('td');
                    rsiCell.textContent = stock.rsi;
                    row.appendChild(rsiCell);
                    
                    // Signal
                    const signalCell = document.createElement('td');
                    signalCell.textContent = stock.signal;
                    signalCell.className = stock.signal === 'Strong Sell' ? 'strong-sell' : 'sell';
                    row.appendChild(signalCell);
                    
                    sellTableBody.appendChild(row);
                });
            } else {
                const row = document.createElement('tr');
                const cell = document.createElement('td');
                cell.colSpan = 4;
                cell.textContent = 'No sell signals available';
                cell.style.textAlign = 'center';
                row.appendChild(cell);
                sellTableBody.appendChild(row);
            }
        }
        
        function loadChart(symbol) {
            document.getElementById('chart-loading').style.display = 'block';
            document.getElementById('chart').innerHTML = '';
            
            fetch(`/chart/${symbol}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('chart').innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                    } else {
                        const img = document.createElement('img');
                        img.src = `data:image/png;base64,${data.image}`;
                        img.className = 'img-fluid';
                        img.alt = `${symbol} Chart`;
                        
                        document.getElementById('chart').innerHTML = '';
                        document.getElementById('chart').appendChild(img);
                    }
                    document.getElementById('chart-loading').style.display = 'none';
                })
                .catch(error => {
                    console.error('Error loading chart:', error);
                    document.getElementById('chart').innerHTML = '<div class="alert alert-danger">Failed to load chart. Please try again.</div>';
                    document.getElementById('chart-loading').style.display = 'none';
                });
        }
    </script>
</body>
</html>
''')

# Start the scheduler when the app starts
@app.before_first_request
def start_scheduler():
    scheduler.start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
