import webview
import threading
import time
from app import app
import socket

def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('localhost', 0))
    port = s.getsockname()[1]
    s.close()
    return port

def run_flask(port):
    app.run(host='127.0.0.1', port=port, debug=False, threaded=True)

if __name__ == '__main__':
    port = 5040 # Use fixed port for now, or dynamic if preferred
    
    # Start Flask in a background thread
    t = threading.Thread(target=run_flask, args=(port,))
    t.daemon = True
    t.start()
    
    # Wait for Flask to start
    time.sleep(1)
    
    # Create pywebview window
    webview.create_window('Sattmal', f'http://127.0.0.1:{port}', width=1280, height=800)
    webview.start()
