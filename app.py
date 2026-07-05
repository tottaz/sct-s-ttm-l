from flask import Flask
from routes import signature_bp, get_flask_secret_key
from datetime import datetime

app = Flask(__name__)
app.secret_key = get_flask_secret_key()
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

# Register blueprint
app.register_blueprint(signature_bp)


@app.template_filter('todatetime')
def todatetime_filter(value):
    return datetime.fromisoformat(value)


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5040)
