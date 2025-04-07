# Check if required packages are installed, if not install them
import subprocess
import sys

def install_required_packages():
    required_packages = ['pandas', 'nsepy', 'nsetools']
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


class NSEDataSource:
    def __init__(self):
        """Initialize NSE API client."""
        self.nse = Nse()
        self.rate_limit_delay = 1  # Add small delay between requests to avoid overwhelming the NSE servers
    
    def get_daily_data(self, symbol, days=100):
        """Get daily stock data from NSE.
        
        Args:
            symbol (str): Stock symbol
            days (int): Number of trading days of data to fetch
            
        Returns:
            pandas.DataFrame: Stock data with columns open, high, low, close, volume
        """
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
        """Get current price data from NSE.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Current price data including lastPrice
        """
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
        """Get latest available data point from NSE.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            pandas.DataFrame: Latest stock data
        """
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
        """Search for a stock symbol.
        
        Args:
            pattern (str): Pattern to search for
            
        Returns:
            list: List of matching symbols
        """
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
        """Initialize Stock data manager.
        
        Args:
            cache_dir (str, optional): Directory to cache data. Defaults to "cache".
        """
        self.data_source = NSEDataSource()
        self.cache_dir = cache_dir
        self.stocks = STOCKS
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def _get_cache_path(self, symbol, data_type):
        """Get cache file path for a symbol and data type.
        
        Args:
            symbol (str): Stock symbol
            data_type (str): Type of data (daily, intraday, etc.)
            
        Returns:
            str: Cache file path
        """
        # Clean symbol for filename (remove any special characters)
        clean_symbol = symbol.replace("-", "_").replace("/", "_")
        return os.path.join(self.cache_dir, f"{clean_symbol}_{data_type}.csv")
    
    def _is_cache_fresh(self, filepath, max_age_hours=12):
        """Check if cache file is fresh.
        
        Args:
            filepath (str): Path to cache file
            max_age_hours (int, optional): Maximum age in hours. Defaults to 12.
            
        Returns:
            bool: True if cache is fresh, False otherwise
        """
        if not os.path.exists(filepath):
            return False
        
        file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        age = datetime.now() - file_time
        return age.total_seconds() < max_age_hours * 3600
    
    def get_daily_data(self, symbol, force_refresh=False):
        """Get daily data for a symbol, using cache if available.
        
        Args:
            symbol (str): Stock symbol
            force_refresh (bool, optional): Force refresh from API. Defaults to False.
            
        Returns:
            pandas.DataFrame: Daily stock data
        """
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
        """Get latest data for a symbol, using cache if available.
        
        Args:
            symbol (str): Stock symbol
            force_refresh (bool, optional): Force refresh from API. Defaults to False.
            
        Returns:
            pandas.DataFrame: Latest stock data
        """
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
        """Get current price for a symbol directly from NSE.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            float: Current price or None if not available
        """
        quote = self.data_source.get_current_price(symbol)
        if quote and 'lastPrice' in quote:
            return quote['lastPrice']
        return None
    
    def fetch_all_daily_data(self, max_workers=4):
        """Fetch daily data for all stocks.
        
        Args:
            max_workers (int, optional): Maximum number of worker threads. Defaults to 4.
            
        Returns:
            dict: Dictionary mapping symbols to DataFrames
        """
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
        """Calculate performance metrics for each stock.
        
        Args:
            dataframes_dict (dict): Dictionary mapping symbols to DataFrames
            days (int, optional): Number of days to calculate metrics for. Defaults to 30.
            
        Returns:
            pandas.DataFrame: DataFrame with performance metrics
        """
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
        """Calculate target price based on RSI, current price and recent movement.
        
        Args:
            current_price (float): Current stock price
            rsi (float): Relative Strength Index
            price_change_pct (float): Recent price change percentage
            is_buy (bool): True if calculating buy target, False for sell target
            
        Returns:
            float: Target price
        """
        if is_buy:
            # For buy recommendations (oversold territory)
            # Target is a reversion to mean - stronger the oversold, higher the expected bounce
            # Formula: The more oversold (lower RSI), the higher potential rebound
            rsi_factor = (30 - rsi) / 30 if rsi < 30 else 0.05  # RSI influence factor
            
            # Recent decline factor (the bigger the decline, the stronger the potential rebound)
            decline_factor = min(abs(price_change_pct), 20) / 100 if price_change_pct < 0 else 0.05
            
            # Combined upside potential (between 5% and 15%)
            upside_potential = max(0.05, min(0.15, rsi_factor + decline_factor))
            
            target_price = current_price * (1 + upside_potential)
            
        else:
            # For sell recommendations (overbought territory)
            # Target is a reversion to mean - stronger the overbought, higher the expected correction
            # Formula: The more overbought (higher RSI), the deeper potential correction
            rsi_factor = (rsi - 70) / 30 if rsi > 70 else 0.05  # RSI influence factor
            
            # Recent rise factor (the bigger the rise, the stronger the potential correction)
            rise_factor = min(abs(price_change_pct), 20) / 100 if price_change_pct > 0 else 0.05
            
            # Combined downside potential (between 5% and 15%)
            downside_potential = max(0.05, min(0.15, rsi_factor + rise_factor))
            
            target_price = current_price * (1 - downside_potential)
        
        return round(target_price, 2)
    
    def identify_trading_opportunities(self, metrics_df):
        """Identify trading opportunities based on metrics.
        
        Args:
            metrics_df (pandas.DataFrame): DataFrame with performance metrics
            
        Returns:
            dict: Dictionary with buy and sell recommendations
        """
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
        """Export trading opportunities to JSON file.
        
        Args:
            opportunities (dict): Dictionary with buy and sell recommendations
            filepath (str, optional): Output file path. Defaults to "trading_signals.json".
        """
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
        """Run full analysis pipeline.
        
        Returns:
            dict: Dictionary with buy and sell recommendations
        """
        # 1. Fetch data for all stocks
        all_data = self.fetch_all_daily_data(max_workers=4)  # We can use more workers with NSE-Python
        
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
        
        # 5. Print summary
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
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"404 Not Found")


def start_web_server():
    """Start a simple web server to keep the application alive on Render."""
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 8080))
    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, SimpleHandler)
    logger.info(f"Starting web server on port {port}")
    httpd.serve_forever()


def main():
    """Main function to run both analysis and web server."""
    # Initialize data manager
    data_manager = StockDataManager()
    
    # Run initial analysis
    opportunities = data_manager.run_analysis()
    
    # Print opportunities
    print("\nBuy Recommendations:")
    for rec in opportunities['buy']:
        print(f"{rec['symbol']}: Current = ₹{rec['current_price']:.2f}, Target = ₹{rec['target_price']:.2f} " +
              f"(+{rec['potential_gain_pct']:.2f}%), RSI = {rec['rsi']:.2f}, Change = {rec['price_change']:.2f}%")
    
    print("\nSell Recommendations:")
    for rec in opportunities['sell']:
        print(f"{rec['symbol']}: Current = ₹{rec['current_price']:.2f}, Target = ₹{rec['target_price']:.2f} " +
              f"(-{rec['potential_loss_pct']:.2f}%), RSI = {rec['rsi']:.2f}, Change = {rec['price_change']:.2f}%")
    
    # Start the web server in a separate thread
    import threading
    server_thread = threading.Thread(target=start_web_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Set up scheduled analysis runs (every 6 hours)
    import sched
    import time
    
    scheduler = sched.scheduler(time.time, time.sleep)
    
    def scheduled_analysis():
        """Run analysis on schedule and reschedule next run."""
        logger.info("Running scheduled analysis...")
        try:
            data_manager.run_analysis()
            logger.info("Scheduled analysis complete")
        except Exception as e:
            logger.error(f"Error in scheduled analysis: {e}")
        
        # Schedule next run (6 hours later)
        scheduler.enter(6 * 60 * 60, 1, scheduled_analysis)
    
    # Schedule first run (1 hour after startup)
    scheduler.enter(60 * 60, 1, scheduled_analysis)
    
    try:
        # Run the scheduler in the main thread
        scheduler.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error in main thread: {e}")


if __name__ == "__main__":
    main()
