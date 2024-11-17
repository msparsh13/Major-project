from flask import Flask
from routes import setup_routes
import os

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join("static", "audio"), exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # Max file size: 10 MB

# Register routes
setup_routes(app)

if __name__ == "__main__":
    app.run(debug=True)
