import base64
import os
import socket
import threading
import time

import webview

from app import app


class DesktopApi:
    def __init__(self):
        self.window = None

    def save_file(self, filename, base64_content):
        if not self.window:
            return {"success": False, "error": "Desktop window is not ready."}

        safe_filename = os.path.basename(filename or "document")
        destination = self.window.create_file_dialog(
            webview.SAVE_DIALOG,
            save_filename=safe_filename,
        )
        if not destination:
            return {"success": False, "cancelled": True}

        if isinstance(destination, (list, tuple)):
            destination = destination[0]

        try:
            with open(destination, "wb") as f:
                f.write(base64.b64decode(base64_content))
        except Exception as exc:
            return {"success": False, "error": str(exc)}

        return {"success": True, "path": destination}

def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('localhost', 0))
    port = s.getsockname()[1]
    s.close()
    return port

def run_flask(port):
    app.run(host='127.0.0.1', port=port, debug=False, threaded=True)

if __name__ == '__main__':
    port = get_free_port()
    api = DesktopApi()
    
    # Start Flask in a background thread
    t = threading.Thread(target=run_flask, args=(port,))
    t.daemon = True
    t.start()
    
    # Wait for Flask to start
    time.sleep(1)
    
    # Create pywebview window
    window = webview.create_window('Sattmal', f'http://127.0.0.1:{port}', width=1280, height=800, js_api=api)
    api.window = window
    webview.start()
