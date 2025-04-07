import os
import pandas as pd
import requests
import time
import logging
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('trading_bot')

# List of Nifty 50 stocks with proper NSE/BSE symbols and added NSIT (Insight Enterprises Inc.)
STOCKS = [
     "NSIT", "NSE:ADANIENT", "NSE:ADANIPORTS", "NSE:APOLLOHOSP", "NSE:ASIANPAINT", "NSE:AXISBANK", 
    "NSE:BAJAJ-AUTO", "NSE:BAJFINANCE", "NSE:BAJAJFINSV", "NSE:BPCL", "NSE:BHARTIARTL", 
    "NSE:BRITANNIA", "NSE:CIPLA", "NSE:COALINDIA", "NSE:DIVISLAB", "NSE:DRREDDY", 
    "NSE:EICHERMOT", "NSE:GRASIM", "NSE:HCLTECH", "NSE:HDFCBANK", "NSE:HDFCLIFE", 
    "NSE:HEROMOTOCO", "NSE:HINDALCO", "NSE:HINDUNILVR", "NSE:ICICIBANK", "NSE:ITC", 
    "NSE:INDUSINDBK", "NSE:INFY", "NSE:JSWSTEEL", "NSE:KOTAKBANK", "NSE:LT", 
    "NSE:M&M", "NSE:MARUTI", "NSE:NTPC", "NSE:NESTLEIND", "NSE:ONGC", 
    "NSE:POWERGRID", "NSE:RELIANCE", "NSE:SBILIFE", "NSE:SBIN", "NSE:SUNPHARMA", 
    "NSE:TCS", "NSE:TATACONSUM", "NSE:TATAMOTORS", "NSE:TATASTEEL", "NSE:TECHM", 
    "NSE:TITAN", "NSE:UPL", "NSE:ULTRACEMCO", "NSE:WIPRO", "NSE:YESBANK",
     # Adding Insight Enterprises Inc.
]

# Hardcoded Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = "UW3VK4H7EYAEXZ6K"

class AlphaVantageDataSource:
    def __init__(self, api_key=None):
        """Initialize Alpha Vantage API client."""
        # Use hardcoded key as primary, fallback to parameter or env var
        self.api_key = api_key or ALPHA_VANTAGE_API_KEY or os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required.")
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 15  # Alpha Vantage free tier has a limit of 5 API calls per minute
    
    def get_daily_data(self, symbol, outputsize="compact"):
        """Get daily stock data from Alpha Vantage.
        
        Args:
            symbol (str): Stock symbol. NSE/BSE stocks already have proper prefixes
            outputsize (str): 'compact' returns the latest 100 datapoints, 'full' returns up to 20 years of data
            
        Returns:
            pandas.DataFrame: Stock data with columns open, high, low, close, volume
        """
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": outputsize,
            "apikey": self.api_key,
            "datatype": "json"
        }
        
        logger.info(f"Requesting daily data for {symbol}")
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Check for error messages
            if "Error Message" in data:
                logger.error(f"API returned error for {symbol}: {data['Error Message']}")
                # Try with BSE prefix if NSE fails
                if symbol.startswith("NSE:"):
                    bse_symbol = symbol.replace("NSE:", "BSE:")
                    logger.info(f"Retrying with BSE symbol: {bse_symbol}")
                    time.sleep(self.rate_limit_delay)  # Wait before retry
                    return self.get_daily_data(bse_symbol, outputsize)
                # Try without any prefix as a last resort
                elif symbol.startswith("BSE:"):
                    plain_symbol = symbol.replace("BSE:", "")
                    logger.info(f"Retrying with plain symbol: {plain_symbol}")
                    time.sleep(self.rate_limit_delay)  # Wait before retry
                    return self.get_daily_data(plain_symbol, outputsize)
                return None
                
            # Check for Note key which indicates API limit reached
            if "Note" in data:
                logger.warning(f"API limit warning: {data['Note']}")
                logger.info(f"Waiting {self.rate_limit_delay * 3} seconds due to API limit")
                time.sleep(self.rate_limit_delay * 3)  # Wait longer when limit is hit
                return self.get_daily_data(symbol, outputsize)  # Retry the same request
                
            # Alpha Vantage returns data in a nested dictionary format
            time_series_data = data.get("Time Series (Daily)")
            if not time_series_data:
                logger.error(f"No time series data found in response for {symbol}")
                logger.debug(f"Response content: {data}")
                # Try with alternative symbol format
                if symbol.startswith("NSE:"):
                    bse_symbol = symbol.replace("NSE:", "BSE:")
                    logger.info(f"Retrying with BSE symbol: {bse_symbol}")
                    time.sleep(self.rate_limit_delay)  # Wait before retry
                    return self.get_daily_data(bse_symbol, outputsize)
                # Try without any prefix as a last resort
                elif symbol.startswith("BSE:"):
                    plain_symbol = symbol.replace("BSE:", "")
                    logger.info(f"Retrying with plain symbol: {plain_symbol}")
                    time.sleep(self.rate_limit_delay)  # Wait before retry
                    return self.get_daily_data(plain_symbol, outputsize)
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series_data, orient='index')
            
            # Check if columns need renaming
            if all(col.startswith(('1. ', '2. ', '3. ', '4. ', '5. ')) for col in df.columns):
                df.columns = [col.split(". ")[1] for col in df.columns]
            
            df.index = pd.to_datetime(df.index)
            df = df.sort_index(ascending=True)
            
            # Convert columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
                
            logger.info(f"Successfully retrieved {len(df)} datapoints for {symbol}")
            return df
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for {symbol}: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {symbol}: {e}")
        except ValueError as e:
            logger.error(f"JSON parsing error for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error for {symbol}: {e}")
            
        logger.info(f"Waiting {self.rate_limit_delay} seconds before next request due to rate limiting")
        time.sleep(self.rate_limit_delay)
        return None
    
    def get_intraday_data(self, symbol, interval="15min", outputsize="compact"):
        """Get intraday stock data from Alpha Vantage.
        
        Args:
            symbol (str): Stock symbol. NSE/BSE stocks already have proper prefixes
            interval (str): Time interval between data points. Options: 1min, 5min, 15min, 30min, 60min
            outputsize (str): 'compact' returns the latest 100 datapoints, 'full' returns up to 30 days of data
            
        Returns:
            pandas.DataFrame: Stock data with columns open, high, low, close, volume
        """
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key,
            "datatype": "json"
        }
        
        logger.info(f"Requesting {interval} intraday data for {symbol}")
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Check for error messages
            if "Error Message" in data:
                logger.error(f"API returned error for {symbol}: {data['Error Message']}")
                # Try with BSE prefix if NSE fails
                if symbol.startswith("NSE:"):
                    bse_symbol = symbol.replace("NSE:", "BSE:")
                    logger.info(f"Retrying with BSE symbol: {bse_symbol}")
                    time.sleep(self.rate_limit_delay)  # Wait before retry
                    return self.get_intraday_data(bse_symbol, interval, outputsize)
                # Try without any prefix as a last resort
                elif symbol.startswith("BSE:"):
                    plain_symbol = symbol.replace("BSE:", "")
                    logger.info(f"Retrying with plain symbol: {plain_symbol}")
                    time.sleep(self.rate_limit_delay)  # Wait before retry
                    return self.get_intraday_data(plain_symbol, interval, outputsize)
                return None
                
            # Check for Note key which indicates API limit reached
            if "Note" in data:
                logger.warning(f"API limit warning: {data['Note']}")
                logger.info(f"Waiting {self.rate_limit_delay * 3} seconds due to API limit")
                time.sleep(self.rate_limit_delay * 3)  # Wait longer when limit is hit
                return self.get_intraday_data(symbol, interval, outputsize)  # Retry
                
            # Alpha Vantage returns data in a nested dictionary format
            time_series_key = f"Time Series ({interval})"
            time_series_data = data.get(time_series_key)
            if not time_series_data:
                logger.error(f"No time series data found in response for key: {time_series_key} for {symbol}")
                logger.debug(f"Response content: {data}")
                # Try with alternative symbol format
                if symbol.startswith("NSE:"):
                    bse_symbol = symbol.replace("NSE:", "BSE:")
                    logger.info(f"Retrying with BSE symbol: {bse_symbol}")
                    time.sleep(self.rate_limit_delay)  # Wait before retry
                    return self.get_intraday_data(bse_symbol, interval, outputsize)
                # Try without any prefix as a last resort
                elif symbol.startswith("BSE:"):
                    plain_symbol = symbol.replace("BSE:", "")
                    logger.info(f"Retrying with plain symbol: {plain_symbol}")
                    time.sleep(self.rate_limit_delay)  # Wait before retry
                    return self.get_intraday_data(plain_symbol, interval, outputsize)
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series_data, orient='index')
            
            # Check if columns need renaming
            if all(col.startswith(('1. ', '2. ', '3. ', '4. ', '5. ')) for col in df.columns):
                df.columns = [col.split(". ")[1] for col in df.columns]
                
            df.index = pd.to_datetime(df.index)
            df = df.sort_index(ascending=True)
            
            # Convert columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
                
            logger.info(f"Successfully retrieved {len(df)} datapoints for {symbol}")
            return df
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for {symbol}: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {symbol}: {e}")
        except ValueError as e:
            logger.error(f"JSON parsing error for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error for {symbol}: {e}")
            
        logger.info(f"Waiting {self.rate_limit_delay} seconds before next request due to rate limiting")
        time.sleep(self.rate_limit_delay)
        return None

    def search_symbol(self, keywords):
        """Search for a stock symbol.
        
        Args:
            keywords (str): Keywords to search for
            
        Returns:
            list: List of matching symbols and their details
        """
        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": keywords,
            "apikey": self.api_key
        }
        
        logger.info(f"Searching for symbol with keywords: {keywords}")
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Check for error messages
            if "Error Message" in data:
                logger.error(f"API returned error: {data['Error Message']}")
                return None
                
            # Check for API limit reached
            if "Note" in data:
                logger.warning(f"API limit warning: {data['Note']}")
                time.sleep(self.rate_limit_delay * 3)
                return self.search_symbol(keywords)  # Retry
                
            results = data.get("bestMatches", [])
            if not results:
                logger.info(f"No matches found for keywords: {keywords}")
                return []
                
            logger.info(f"Found {len(results)} matches for {keywords}")
            return results
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
        except ValueError as e:
            logger.error(f"JSON parsing error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            
        logger.info(f"Waiting {self.rate_limit_delay} seconds before next request due to rate limiting")
        time.sleep(self.rate_limit_delay)
        return None


class StockDataManager:
    def __init__(self, api_key=None, cache_dir="cache"):
        """Initialize Stock data manager.
        
        Args:
            api_key (str, optional): Alpha Vantage API key. Defaults to None.
            cache_dir (str, optional): Directory to cache data. Defaults to "cache".
        """
        self.data_source = AlphaVantageDataSource(api_key)
        self.cache_dir = cache_dir
        self.stocks = STOCKS
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def _get_cache_path(self, symbol, data_type):
        """Get cache file path for a symbol and data type.
        
        Args:
            symbol (str): Stock symbol
            data_type (str): Type of data (daily, intraday_15min, etc.)
            
        Returns:
            str: Cache file path
        """
        # Clean symbol for filename (remove any special characters)
        clean_symbol = symbol.replace(":", "_").replace("/", "_")
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
    
    def get_intraday_data(self, symbol, interval="15min", force_refresh=False):
        """Get intraday data for a symbol, using cache if available.
        
        Args:
            symbol (str): Stock symbol
            interval (str, optional): Time interval. Defaults to "15min".
            force_refresh (bool, optional): Force refresh from API. Defaults to False.
            
        Returns:
            pandas.DataFrame: Intraday stock data
        """
        cache_path = self._get_cache_path(symbol, f"intraday_{interval}")
        
        # For intraday data, we want more frequent refreshes
        # Check if we have fresh cache (max 1 hour for intraday)
        if not force_refresh and self._is_cache_fresh(cache_path, max_age_hours=1):
            logger.info(f"Using cached {interval} intraday data for {symbol}")
            try:
                return pd.read_csv(cache_path, index_col=0, parse_dates=True)
            except Exception as e:
                logger.error(f"Error reading cache file for {symbol} intraday: {e}")
                # Continue to fetch from API if cache read fails
        
        # Get data from API
        df = self.data_source.get_intraday_data(symbol, interval=interval)
            
        # Save cache if we got data
        if df is not None and not df.empty:
            try:
                df.to_csv(cache_path)
                logger.info(f"Saved {symbol} intraday data to cache")
            except Exception as e:
                logger.error(f"Error saving cache for {symbol} intraday: {e}")
            
        return df
    
    def fetch_all_daily_data(self, max_workers=1):
        """Fetch daily data for all stocks.
        
        Args:
            max_workers (int, optional): Maximum number of worker threads. Defaults to 1.
            
        Returns:
            dict: Dictionary mapping symbols to DataFrames
        """
        logger.info(f"Fetching daily data for all {len(self.stocks)} stocks with {max_workers} workers")
        results = {}
        
        # Limit max_workers to respect API rate limits (free tier is very limited)
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
                
                metrics.append({
                    'symbol': symbol,
                    'start_price': start_price,
                    'end_price': end_price,
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
            return pd.DataFrame(columns=['symbol', 'start_price', 'end_price', 'price_change_pct', 
                                         'avg_daily_return', 'volatility', 'avg_volume', 'rsi'])
        
        return pd.DataFrame(metrics)
    
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
            
            # Buy conditions
            if row['rsi'] < 30 and row['price_change_pct'] < 0:
                buy_recommendations.append({
                    'symbol': symbol,
                    'price': row['end_price'],
                    'rsi': row['rsi'],
                    'price_change': row['price_change_pct'],
                    'reason': 'Oversold (RSI < 30) with recent price decline'
                })
            
            # Sell conditions
            elif row['rsi'] > 70 and row['price_change_pct'] > 0:
                sell_recommendations.append({
                    'symbol': symbol,
                    'price': row['end_price'],
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
                rec['price'] = float(rec['price'])
                rec['rsi'] = float(rec['rsi'])
                rec['price_change'] = float(rec['price_change'])
        
        with open(filepath, 'w') as f:
            json.dump(opportunities, f, indent=4)
        logger.info(f"Trading signals exported to {filepath}")
    
    def run_analysis(self):
        """Run full analysis pipeline.
        
        Returns:
            dict: Dictionary with buy and sell recommendations
        """
        # 1. Fetch data for all stocks
        all_data = self.fetch_all_daily_data(max_workers=1)  # Use single worker for free tier
        
        # Check if we got any data
        if not all_data:
            logger.error("Failed to fetch data for any stocks. Check API key and connectivity.")
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


def main():
    """Example usage of the StockDataManager class."""
    # Initialize data manager with API key
    data_manager = StockDataManager(ALPHA_VANTAGE_API_KEY)
    
    # Run analysis
    opportunities = data_manager.run_analysis()
    
    # Print opportunities
    print("\nBuy Recommendations:")
    for rec in opportunities['buy']:
        # Format price display based on stock market (₹ for NSE/BSE stocks, $ for US stocks)
        currency_symbol = "₹" if "NSE:" in rec['symbol'] or "BSE:" in rec['symbol'] else "$"
        print(f"{rec['symbol']}: Price = {currency_symbol}{rec['price']:.2f}, RSI = {rec['rsi']:.2f}, Change = {rec['price_change']:.2f}%")
    
    print("\nSell Recommendations:")
    for rec in opportunities['sell']:
        # Format price display based on stock market (₹ for NSE/BSE stocks, $ for US stocks)
        currency_symbol = "₹" if "NSE:" in rec['symbol'] or "BSE:" in rec['symbol'] else "$"
        print(f"{rec['symbol']}: Price = {currency_symbol}{rec['price']:.2f}, RSI = {rec['rsi']:.2f}, Change = {rec['price_change']:.2f}%")


if __name__ == "__main__":
    main()
