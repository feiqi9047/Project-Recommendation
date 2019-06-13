from flask import Flask, render_template
app = Flask(__name__)

@app.route("/")
def index():
    return "<h2>First Homepage</h2>"

@app.route("/profile/<name>")
def profile(name):
    return "my name is <b>%s</b>" % name

if __name__ == "__main__":
    app.run(debug=True)
