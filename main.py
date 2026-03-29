import os

from dotenv import load_dotenv
from flask import Flask, render_template

load_dotenv()

HOST = os.getenv("HOST")
PORT = int(os.getenv("PORT"))

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=True)
