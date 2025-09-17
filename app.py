from flask import Flask
from routes import signature_bp
from datetime import datetime

app = Flask(__name__)
app.secret_key = "super-secret-key"

# Register blueprint
app.register_blueprint(signature_bp)


@app.template_filter('todatetime')
def todatetime_filter(value):
    return datetime.fromisoformat(value)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5040)
