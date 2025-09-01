from flask import Flask
from routes import signature_bp

app = Flask(__name__)
app.secret_key = "super-secret-key"

# Register blueprint
app.register_blueprint(signature_bp)

if __name__ == "__main__":
    app.run(debug=True)
