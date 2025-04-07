import subprocess
import sys

def install_required_packages():
    required_packages = ['pandas', 'nsepy', 'nsetools', 'requests']  # Added requests for Telegram API
    for package in required_packages:
        try:
            __import__(package)
            print(f"{package} is already installed.")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} installed successfully.")

print("Checking and installing required packages...")
install_required_packages()

# Now import the packages
import os
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from nsetools import Nse
import nsepy
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import requests  # Added for Telegram API

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('trading_bot')

# List of Nifty 50 stocks with proper NSE symbols
STOCKS = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", 
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BPCL", "BHARTIARTL", 
    "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY", 
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", 
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC", 
    "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LT", 
    "M&M", "MARUTI", "NTPC", "NESTLEIND", "ONGC", 
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SUNPHARMA", 
    "TCS", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TECHM", 
    "TITAN", "UPL", "ULTRACEMCO", "WIPRO", "YESBANK"
]


# New Telegram notification class
class TelegramNotifier:
    def __init__(self, bot_token, chat_id):
        """Initialize Telegram notifier with bot token and chat ID."""
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
    def send_message(self, message, parse_mode="HTML"):
        """Send a message to the Telegram chat."""
        try:
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            response = requests.post(self.api_url, data=payload)
            
            if response.status_code == 200:
                logger.info(f"Message sent to Telegram successfully")
                return True
            else:
                logger.error(f"Failed to send Telegram message: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def send_trading_signals(self, opportunities):
        """Format and send trading signals to Telegram."""
        # Create message with buy recommendations
        message = "<b>📈 NSE Trading Bot - Signals Update</b>\n\n"
        message += f"<b>🟢 BUY RECOMMENDATIONS:</b>\n"
        
        if opportunities['buy']:
            for rec in opportunities['buy']:
                message += f"• <b>{rec['symbol']}</b>: ₹{rec['current_price']:.2f} → ₹{rec['target_price']:.2f} (+{rec['potential_gain_pct']:.2f}%)\n"
                message += f"  RSI: {rec['rsi']:.2f}, Change: {rec['price_change']:.2f}%\n"
        else:
            message += "No buy recommendations at this time.\n"
        
        message += "\n<b>🔴 SELL RECOMMENDATIONS:</b>\n"
        
        # Add sell recommendations
        if opportunities['sell']:
            for rec in opportunities['sell']:
                message += f"• <b>{rec['symbol']}</b>: ₹{rec['current_price']:.2f} → ₹{rec['target_price']:.2f} (-{rec['potential_loss_pct']:.2f}%)\n"
                message += f"  RSI: {rec['rsi']:.2f}, Change: {rec['price_change']:.2f}%\n"
        else:
            message += "No sell recommendations at this time.\n"
        
        # Add timestamp
        message += f"\n<i>Update time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        
        # Send the message
        return self.send_message(message)


class NSEDataSource:
    def __init__(self):
        """Initialize NSE API client."""
        self.nse = Nse()
        self.rate_limit_delay = 1  # Add small delay between requests to avoid overwhelming the NSE servers
    
    def get_daily_data(self, symbol, days=100):
        """Get daily stock data from NSE."""
        logger.info(f"Requesting daily data for {symbol}")
        try:
            # Calculate the start date (going back approximately 'days' trading days)
            end_date = datetime.now().date()
            # Approximately 5 trading days per week, so go back days*7/5 calendar days
            start_date = end_date - timedelta(days=int(days * 7/5) + 10)  # Add buffer for holidays
            
            # Get data using nsepy
            df = nsepy.get_history(symbol=symbol, start=start_date, end=end_date)
            
            if df.empty:
                logger.error(f"No data found for {symbol}")
                return None
                
            # Rename columns to standardize with our expected format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Keep only necessary columns
            columns_to_keep = ['open', 'high', 'low', 'close', 'volume']
            df = df[[col for col in columns_to_keep if col in df.columns]]
            
            logger.info(f"Successfully retrieved {len(df)} datapoints for {symbol}")
            
            # Add small delay to avoid overloading the server
            time.sleep(self.rate_limit_delay)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            time.sleep(self.rate_limit_delay)
            return None
    
    def get_current_price(self, symbol):
        """Get current price data from NSE."""
        logger.info(f"Requesting current price for {symbol}")
        try:
            # Get quote data
            quote = self.nse.get_quote(symbol)
            
            if not quote:
                logger.error(f"No quote data found for {symbol}")
                return None
            
            return quote
            
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            time.sleep(self.rate_limit_delay)
            return None
    
    def get_latest_data(self, symbol):
        """Get latest available data point from NSE."""
        logger.info(f"Requesting current data for {symbol}")
        try:
            # Get quote data
            quote = self.nse.get_quote(symbol)
            
            if not quote:
                logger.error(f"No quote data found for {symbol}")
                return None
            
            # Create a single-row DataFrame with the latest data
            data = {
                'open': [quote['open']],
                'high': [quote['dayHigh']],
                'low': [quote['dayLow']],
                'close': [quote['lastPrice']],
                'volume': [quote['totalTradedVolume']]
            }
            
            df = pd.DataFrame(data, index=[datetime.now()])
            
            # Add small delay to avoid overloading the server
            time.sleep(self.rate_limit_delay)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            time.sleep(self.rate_limit_delay)
            return None

    def search_symbol(self, pattern):
        """Search for a stock symbol."""
        logger.info(f"Searching for symbol with pattern: {pattern}")
        try:
            matches = self.nse.get_stock_codes(cached=False)
            
            # Filter matches based on pattern
            filtered_matches = {code: name for code, name in matches.items() 
                             if pattern.lower() in code.lower() or pattern.lower() in name.lower()}
            
            if not filtered_matches:
                logger.info(f"No matches found for pattern: {pattern}")
                return []
                
            logger.info(f"Found {len(filtered_matches)} matches for {pattern}")
            
            # Format the results
            results = [{"symbol": code, "name": name} for code, name in filtered_matches.items()]
            
            # Add small delay to avoid overloading the server
            time.sleep(self.rate_limit_delay)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching for pattern {pattern}: {e}")
            time.sleep(self.rate_limit_delay)
            return None


class StockDataManager:
    def __init__(self, cache_dir="cache"):
        """Initialize Stock data manager."""
        self.data_source = NSEDataSource()
        self.cache_dir = cache_dir
        self.stocks = STOCKS
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def _get_cache_path(self, symbol, data_type):
        """Get cache file path for a symbol and data type."""
        # Clean symbol for filename (remove any special characters)
        clean_symbol = symbol.replace("-", "_").replace("/", "_")
        return os.path.join(self.cache_dir, f"{clean_symbol}_{data_type}.csv")
    
    def _is_cache_fresh(self, filepath, max_age_hours=12):
        """Check if cache file is fresh."""
        if not os.path.exists(filepath):
            return False
        
        file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        age = datetime.now() - file_time
        return age.total_seconds() < max_age_hours * 3600
    
    def get_daily_data(self, symbol, force_refresh=False):
        """Get daily data for a symbol, using cache if available."""
        cache_path = self._get_cache_path(symbol, "daily")
        
        # Check if we have fresh cache
        if not force_refresh and self._is_cache_fresh(cache_path):
            logger.info(f"Using cached daily data for {symbol}")
            try:
                return pd.read_csv(cache_path, index_col=0, parse_dates=True)
            except Exception as e:
                logger.error(f"Error reading cache file for {symbol}: {e}")
                # Continue to fetch from API if cache read fails
        
        # Get data from API
        df = self.data_source.get_daily_data(symbol)
            
        # Save cache if we got data
        if df is not None and not df.empty:
            try:
                df.to_csv(cache_path)
                logger.info(f"Saved {symbol} data to cache")
            except Exception as e:
                logger.error(f"Error saving cache for {symbol}: {e}")
            
        return df
    
    def get_latest_data(self, symbol, force_refresh=False):
        """Get latest data for a symbol, using cache if available."""
        cache_path = self._get_cache_path(symbol, "latest")
        
        # For latest data, we want more frequent refreshes
        # Check if we have fresh cache (max 15 minutes for latest data)
        if not force_refresh and self._is_cache_fresh(cache_path, max_age_hours=0.25):
            logger.info(f"Using cached latest data for {symbol}")
            try:
                return pd.read_csv(cache_path, index_col=0, parse_dates=True)
            except Exception as e:
                logger.error(f"Error reading cache file for {symbol} latest: {e}")
                # Continue to fetch from API if cache read fails
        
        # Get data from API
        df = self.data_source.get_latest_data(symbol)
            
        # Save cache if we got data
        if df is not None and not df.empty:
            try:
                df.to_csv(cache_path)
                logger.info(f"Saved {symbol} latest data to cache")
            except Exception as e:
                logger.error(f"Error saving cache for {symbol} latest: {e}")
            
        return df
    
    def get_current_price(self, symbol):
        """Get current price for a symbol directly from NSE."""
        quote = self.data_source.get_current_price(symbol)
        if quote and 'lastPrice' in quote:
            return quote['lastPrice']
        return None
    
    def fetch_all_daily_data(self, max_workers=4):
        """Fetch daily data for all stocks."""
        logger.info(f"Fetching daily data for all {len(self.stocks)} stocks with {max_workers} workers")
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.get_daily_data, symbol): symbol 
                for symbol in self.stocks
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        results[symbol] = data
                        logger.info(f"Successfully processed {symbol}")
                    else:
                        logger.warning(f"Failed to fetch data for {symbol}")
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
        
        logger.info(f"Successfully fetched data for {len(results)} out of {len(self.stocks)} stocks")
        return results
    
    def calculate_performance_metrics(self, dataframes_dict, days=30):
        """Calculate performance metrics for each stock."""
        metrics = []
        
        for symbol, df in dataframes_dict.items():
            if df is None or len(df) < days + 1:
                logger.warning(f"Insufficient data for {symbol}, skipping metrics calculation")
                continue
                
            try:
                # Calculate daily returns
                df['daily_return'] = df['close'].pct_change()
                
                # Get recent data
                recent_df = df.iloc[-days-1:]
                
                # Make sure we have enough data
                if len(recent_df) < days:
                    logger.warning(f"Insufficient recent data for {symbol}, skipping metrics calculation")
                    continue
                
                # Calculate metrics
                start_price = recent_df['close'].iloc[0]
                end_price = recent_df['close'].iloc[-1]
                price_change_pct = ((end_price - start_price) / start_price) * 100
                
                avg_daily_return = recent_df['daily_return'].mean() * 100
                volatility = recent_df['daily_return'].std() * 100 * (252 ** 0.5)  # Annualized
                
                # Calculate average volume
                avg_volume = recent_df['volume'].mean()
                
                # Calculate RSI (14-day) with error handling
                try:
                    delta = df['close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                    
                    # Avoid division by zero
                    rs = gain / loss.replace(0, float('nan'))
                    rs = rs.fillna(0)  # Replace NaN with 0
                    rsi = 100 - (100 / (1 + rs))
                    current_rsi = rsi.iloc[-1]
                    
                    # Handle potential NaN in RSI
                    if pd.isna(current_rsi):
                        current_rsi = 50  # Default to neutral RSI if calculation fails
                except Exception as e:
                    logger.error(f"Error calculating RSI for {symbol}: {e}")
                    current_rsi = 50  # Default value
                
                # Get current price (latest) from NSE directly
                current_price = self.get_current_price(symbol)
                if current_price is None:
                    current_price = end_price  # Fall back to historical data if live price not available
                
                metrics.append({
                    'symbol': symbol,
                    'start_price': start_price,
                    'end_price': end_price,
                    'current_price': current_price,
                    'price_change_pct': price_change_pct,
                    'avg_daily_return': avg_daily_return,
                    'volatility': volatility,
                    'avg_volume': avg_volume,
                    'rsi': current_rsi
                })
                
            except Exception as e:
                logger.error(f"Error calculating metrics for {symbol}: {e}")
        
        if not metrics:
            logger.warning("No metrics calculated for any stock")
            return pd.DataFrame(columns=['symbol', 'start_price', 'end_price', 'current_price', 'price_change_pct', 
                                         'avg_daily_return', 'volatility', 'avg_volume', 'rsi'])
        
        return pd.DataFrame(metrics)
    
    def calculate_target_price(self, current_price, rsi, price_change_pct, is_buy):
        """Calculate target price based on RSI, current price and recent movement."""
        if is_buy:
            # For buy recommendations (oversold territory)
            rsi_factor = (30 - rsi) / 30 if rsi < 30 else 0.05  # RSI influence factor
            decline_factor = min(abs(price_change_pct), 20) / 100 if price_change_pct < 0 else 0.05
            upside_potential = max(0.05, min(0.15, rsi_factor + decline_factor))
            target_price = current_price * (1 + upside_potential)
        else:
            # For sell recommendations (overbought territory)
            rsi_factor = (rsi - 70) / 30 if rsi > 70 else 0.05  # RSI influence factor
            rise_factor = min(abs(price_change_pct), 20) / 100 if price_change_pct > 0 else 0.05
            downside_potential = max(0.05, min(0.15, rsi_factor + rise_factor))
            target_price = current_price * (1 - downside_potential)
        
        return round(target_price, 2)
    
    def identify_trading_opportunities(self, metrics_df):
        """Identify trading opportunities based on metrics."""
        buy_recommendations = []
        sell_recommendations = []
        
        if metrics_df.empty:
            logger.warning("No metrics data available to identify trading opportunities")
            return {'buy': [], 'sell': []}
        
        # Basic strategy:
        # Buy: Oversold (RSI < 30) + Negative price change
        # Sell: Overbought (RSI > 70) + Positive price change
        
        for _, row in metrics_df.iterrows():
            symbol = row['symbol']
            current_price = row['current_price']
            
            # Buy conditions
            if row['rsi'] < 30 and row['price_change_pct'] < 0:
                target_price = self.calculate_target_price(
                    current_price, row['rsi'], row['price_change_pct'], is_buy=True
                )
                
                buy_recommendations.append({
                    'symbol': symbol,
                    'current_price': current_price,
                    'target_price': target_price,
                    'potential_gain_pct': round(((target_price - current_price) / current_price) * 100, 2),
                    'rsi': row['rsi'],
                    'price_change': row['price_change_pct'],
                    'reason': 'Oversold (RSI < 30) with recent price decline'
                })
            
            # Sell conditions
            elif row['rsi'] > 70 and row['price_change_pct'] > 0:
                target_price = self.calculate_target_price(
                    current_price, row['rsi'], row['price_change_pct'], is_buy=False
                )
                
                sell_recommendations.append({
                    'symbol': symbol,
                    'current_price': current_price,
                    'target_price': target_price,
                    'potential_loss_pct': round(((current_price - target_price) / current_price) * 100, 2),
                    'rsi': row['rsi'],
                    'price_change': row['price_change_pct'],
                    'reason': 'Overbought (RSI > 70) with recent price increase'
                })
        
        return {
            'buy': buy_recommendations,
            'sell': sell_recommendations
        }
    
    def export_opportunities_to_json(self, opportunities, filepath="trading_signals.json"):
        """Export trading opportunities to JSON file."""
        # Format decimal values for JSON serialization
        for rec_type in ['buy', 'sell']:
            for rec in opportunities[rec_type]:
                rec['current_price'] = float(rec['current_price'])
                rec['target_price'] = float(rec['target_price'])
                rec['rsi'] = float(rec['rsi'])
                rec['price_change'] = float(rec['price_change'])
                if 'potential_gain_pct' in rec:
                    rec['potential_gain_pct'] = float(rec['potential_gain_pct'])
                if 'potential_loss_pct' in rec:
                    rec['potential_loss_pct'] = float(rec['potential_loss_pct'])
        
        with open(filepath, 'w') as f:
            json.dump(opportunities, f, indent=4)
        logger.info(f"Trading signals exported to {filepath}")
    
    def run_analysis(self):
        """Run full analysis pipeline with Telegram notifications."""
        # 1. Fetch data for all stocks
        all_data = self.fetch_all_daily_data(max_workers=4)
        
        # Check if we got any data
        if not all_data:
            logger.error("Failed to fetch data for any stocks. Check connectivity to NSE servers.")
            return {'buy': [], 'sell': []}
        
        # 2. Calculate performance metrics
        metrics = self.calculate_performance_metrics(all_data)
        
        # 3. Identify trading opportunities
        opportunities = self.identify_trading_opportunities(metrics)
        
        # 4. Export to JSON
        self.export_opportunities_to_json(opportunities)
        
        # 5. Send Telegram notification
        try:
            # Your Telegram bot token and chat ID from the parameters
            bot_token = "8017759392:AAEwM-W-y83lLXTjlPl8sC_aBmizuIrFXnU"
            chat_id = "711856868"
            
            # Create notifier and send message
            telegram_notifier = TelegramNotifier(bot_token, chat_id)
            telegram_notifier.send_trading_signals(opportunities)
            logger.info("Trading signals sent to Telegram")
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
        
        # 6. Print summary
        buy_count = len(opportunities['buy'])
        sell_count = len(opportunities['sell'])
        logger.info(f"Analysis complete. Found {buy_count} buy and {sell_count} sell opportunities.")
        
        return opportunities


# Define the HTTP Handler for our web server
class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Main page
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Basic HTML response showing the bot is running
            response = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>NSE Trading Bot</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #2c3e50; }
                    .status { padding: 10px; background-color: #e8f5e9; border-radius: 5px; }
                    .info { color: #1b5e20; }
                </style>
            </head>
            <body>
                <h1>NSE Trading Bot</h1>
                <div class="status">
                    <p class="info">✅ Trading bot is running and analyzing NSE stocks.</p>
                </div>
                <p>The bot is analyzing Nifty 50 stocks and generating trading signals.</p>
                <p>Last updated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
                <p><a href="/signals">View latest trading signals</a></p>
            </body>
            </html>
            """
            self.wfile.write(response.encode())
            
        # Signals endpoint
        elif self.path == '/signals':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Try to read the latest trading signals
            try:
                with open('trading_signals.json', 'r') as f:
                    signals = json.load(f)
                    
                # Create HTML for the signals
                signals_html = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Trading Signals</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        h1, h2 { color: #2c3e50; }
                        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                        th { background-color: #f2f2f2; }
                        tr:nth-child(even) { background-color: #f9f9f9; }
                        .buy { color: green; }
                        .sell { color: red; }
                        .no-data { font-style: italic; color: #777; }
                    </style>
                </head>
                <body>
                    <h1>Trading Signals</h1>
                    <p>Last updated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
                    
                    <h2 class="buy">Buy Recommendations</h2>
                """
                
                # Add buy recommendations table
                if signals['buy']:
                    signals_html += """
                    <table>
                        <tr>
                            <th>Symbol</th>
                            <th>Current Price (₹)</th>
                            <th>Target Price (₹)</th>
                            <th>Potential Gain (%)</th>
                            <th>RSI</th>
                            <th>Recent Change (%)</th>
                            <th>Reason</th>
                        </tr>
                    """
                    
                    for rec in signals['buy']:
                        signals_html += f"""
                        <tr>
                            <td>{rec['symbol']}</td>
                            <td>{rec['current_price']}</td>
                            <td>{rec['target_price']}</td>
                            <td class="buy">+{rec['potential_gain_pct']}%</td>
                            <td>{rec['rsi']:.2f}</td>
                            <td>{rec['price_change']:.2f}%</td>
                            <td>{rec['reason']}</td>
                        </tr>
                        """
                    
                    signals_html += "</table>"
                else:
                    signals_html += '<p class="no-data">No buy recommendations at this time.</p>'
                
                # Add sell recommendations table
                signals_html += '<h2 class="sell">Sell Recommendations</h2>'
                
                if signals['sell']:
                    signals_html += """
                    <table>
                        <tr>
                            <th>Symbol</th>
                            <th>Current Price (₹)</th>
                            <th>Target Price (₹)</th>
                            <th>Potential Loss (%)</th>
                            <th>RSI</th>
                            <th>Recent Change (%)</th>
                            <th>Reason</th>
                        </tr>
                    """
                    
                    for rec in signals['sell']:
                        signals_html += f"""
                        <tr>
                            <td>{rec['symbol']}</td>
                            <td>{rec['current_price']}</td>
                            <td>{rec['target_price']}</td>
                            <td class="sell">-{rec['potential_loss_pct']}%</td>
                            <td>{rec['rsi']:.2f}</td>
                            <td>{rec['price_change']:.2f}%</td>
                            <td>{rec['reason']}</td>
                        </tr>
                        """
                    
                    signals_html += "</table>"
                else:
                    signals_html += '<p class="no-data">No sell recommendations at this time.</p>'
                
                # Complete the HTML
                signals_html += """
                    <p><a href="/">Back to home</a></p>
                </body>
                </html>
                """
                
                self.wfile.write(signals_html.encode())
                
            except FileNotFoundError:
                self.wfile.write(b"No trading signals available yet. Please run analysis first.")
            except Exception as e:
                self.wfile.write(f"Error displaying signals: {str(e)}".encode())
                
        # Health check endpoint
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy"}).encode())
            
        # Handle 404 for other paths
# Define the HTTP Handler for our web server
class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Main page
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Basic HTML response showing the bot is running
            response = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>NSE Trading Bot</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #2c3e50; }
                    .status { padding: 10px; background-color: #e8f5e9; border-radius: 5px; }
                    .info { color: #1b5e20; }
                </style>
            </head>
            <body>
                <h1>NSE Trading Bot</h1>
                <div class="status">
                    <p class="info">✅ Trading bot is running and analyzing NSE stocks.</p>
                </div>
                <p>The bot is analyzing Nifty 50 stocks and generating trading signals.</p>
                <p>Last updated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
                <p><a href="/signals">View latest trading signals</a></p>
            </body>
            </html>
            """
            self.wfile.write(response.encode())
            
        # Signals endpoint
        elif self.path == '/signals':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Try to read the latest trading signals
            try:
                with open('trading_signals.json', 'r') as f:
                    signals = json.load(f)
                    
                # Create HTML for the signals
                signals_html = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Trading Signals</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        h1, h2 { color: #2c3e50; }
                        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                        th { background-color: #f2f2f2; }
                        tr:nth-child(even) { background-color: #f9f9f9; }
                        .buy { color: green; }
                        .sell { color: red; }
                        .no-data { font-style: italic; color: #777; }
                    </style>
                </head>
                <body>
                    <h1>Trading Signals</h1>
                    <p>Last updated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
                    
                    <h2 class="buy">Buy Recommendations</h2>
                """
                
                # Add buy recommendations table
                if signals['buy']:
                    signals_html += """
                    <table>
                        <tr>
                            <th>Symbol</th>
                            <th>Current Price (₹)</th>
                            <th>Target Price (₹)</th>
                            <th>Potential Gain (%)</th>
                            <th>RSI</th>
                            <th>Recent Change (%)</th>
                            <th>Reason</th>
                        </tr>
                    """
                    
                    for rec in signals['buy']:
                        signals_html += f"""
                        <tr>
                            <td>{rec['symbol']}</td>
                            <td>{rec['current_price']}</td>
                            <td>{rec['target_price']}</td>
                            <td class="buy">+{rec['potential_gain_pct']}%</td>
                            <td>{rec['rsi']:.2f}</td>
                            <td>{rec['price_change']:.2f}%</td>
                            <td>{rec['reason']}</td>
                        </tr>
                        """
                    
                    signals_html += "</table>"
                else:
                    signals_html += '<p class="no-data">No buy recommendations at this time.</p>'
                
                # Add sell recommendations table
                signals_html += '<h2 class="sell">Sell Recommendations</h2>'
                
                if signals['sell']:
                    signals_html += """
                    <table>
                        <tr>
                            <th>Symbol</th>
                            <th>Current Price (₹)</th>
                            <th>Target Price (₹)</th>
                            <th>Potential Loss (%)</th>
                            <th>RSI</th>
                            <th>Recent Change (%)</th>
                            <th>Reason</th>
                        </tr>
                    """
                    
                    for rec in signals['sell']:
                        signals_html += f"""
                        <tr>
                            <td>{rec['symbol']}</td>
                            <td>{rec['current_price']}</td>
                            <td>{rec['target_price']}</td>
                            <td class="sell">-{rec['potential_loss_pct']}%</td>
                            <td>{rec['rsi']:.2f}</td>
                            <td>{rec['price_change']:.2f}%</td>
                            <td>{rec['reason']}</td>
                        </tr>
                        """
                    
                    signals_html += "</table>"
                else:
                    signals_html += '<p class="no-data">No sell recommendations at this time.</p>'
                
                # Complete the HTML
                signals_html += """
                    <p><a href="/">Back to home</a></p>
                </body>
                </html>
                """
                
                self.wfile.write(signals_html.encode())
                
            except FileNotFoundError:
                self.wfile.write(b"No trading signals available yet. Please run analysis first.")
            except Exception as e:
                self.wfile.write(f"Error displaying signals: {str(e)}".encode())
                
        # Health check endpoint
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy"}).encode())
            
        # Handle 404 for other paths
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"404 Not Found")

    def log_message(self, format, *args):
        """Override logging to use our logger instead of stdout."""
        logger.info("%s - - [%s] %s" % (
            self.address_string(),
            self.log_date_time_string(),
            format % args
        ))


class TradingBotApp:
    def __init__(self, server_port=8080, analysis_interval=3600):
        """Initialize Trading Bot application.
        
        Args:
            server_port: Port to run the web server on
            analysis_interval: Interval (in seconds) between analyses
        """
        self.server_port = server_port
        self.analysis_interval = analysis_interval
        self.data_manager = StockDataManager(cache_dir="cache")
        self.stop_event = threading.Event()
        self.http_server = None
        
    def start_http_server(self):
        """Start the HTTP server in a separate thread."""
        try:
            server_address = ('', self.server_port)
            self.http_server = HTTPServer(server_address, SimpleHandler)
            logger.info(f"Starting HTTP server on port {self.server_port}")
            
            # Run the server in a separate thread
            server_thread = threading.Thread(target=self.http_server.serve_forever)
            server_thread.daemon = True  # So it will close when the main thread exits
            server_thread.start()
            logger.info("HTTP server started successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to start HTTP server: {e}")
            return False
    
    def analysis_loop(self):
        """Run the analysis loop at regular intervals."""
        logger.info(f"Starting analysis loop with interval of {self.analysis_interval} seconds")
        
        while not self.stop_event.is_set():
            try:
                logger.info("Running stock analysis...")
                self.data_manager.run_analysis()
                logger.info(f"Analysis complete. Waiting {self.analysis_interval} seconds until next run...")
                
                # Wait for the next interval or until stop is requested
                self.stop_event.wait(self.analysis_interval)
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                # Wait a bit before retrying to avoid tight error loops
                time.sleep(60)
    
    def start(self):
        """Start the trading bot and its components."""
        logger.info("Starting NSE Trading Bot...")
        
        # Start the HTTP server
        if not self.start_http_server():
            logger.error("Failed to start HTTP server, exiting.")
            return False
        
        # Start the analysis thread
        analysis_thread = threading.Thread(target=self.analysis_loop)
        analysis_thread.daemon = True
        analysis_thread.start()
        
        logger.info("NSE Trading Bot started successfully")
        return True
    
    def stop(self):
        """Stop the trading bot and its components."""
        logger.info("Stopping NSE Trading Bot...")
        
        # Signal threads to stop
        self.stop_event.set()
        
        # Stop the HTTP server
        if self.http_server:
            self.http_server.shutdown()
        
        logger.info("NSE Trading Bot stopped")


def main():
    """Main function to run the trading bot."""
    try:
        # Parse command line arguments (could be extended)
        import argparse
        parser = argparse.ArgumentParser(description='NSE Trading Bot')
        parser.add_argument('--port', type=int, default=8080, help='Web server port')
        parser.add_argument('--interval', type=int, default=3600, help='Analysis interval in seconds')
        args = parser.parse_args()
        
        # Initialize and start the bot
        bot = TradingBotApp(server_port=args.port, analysis_interval=args.interval)
        if not bot.start():
            sys.exit(1)
        
        # Keep the main thread running
        logger.info("Bot is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        if 'bot' in locals():
            bot.stop()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
