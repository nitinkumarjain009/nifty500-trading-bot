@app.route('/')
def index():
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
