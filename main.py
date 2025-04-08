# main.py
from app import app
import logging

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()
        ]
    )
    
    # Start the app
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        logging.error(f"Application error: {e}")
