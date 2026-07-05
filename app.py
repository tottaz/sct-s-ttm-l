from flask import Flask
from routes import signature_bp, get_flask_secret_key, get_version_info
from datetime import datetime

app = Flask(__name__)
app.secret_key = get_flask_secret_key()
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

# Register blueprint
app.register_blueprint(signature_bp)


@app.context_processor
def inject_version_info():
    version_info = get_version_info()
    return {
        "app_version": version_info,
        "app_version_label": f"v{version_info.get('version', '0.0.0')} ({version_info.get('build', 'dev')})",
    }


@app.template_filter('todatetime')
def todatetime_filter(value):
    return datetime.fromisoformat(value)


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5040)
