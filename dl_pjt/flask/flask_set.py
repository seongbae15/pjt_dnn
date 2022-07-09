from flask import Flask
from flask import render_template
from flask import request

app = Flask(__name__)


@app.route("/")
@app.route("/FacialKeypoints")
def FacialKeypointsPrediction():
    picture_number = request.args.get("picture")
    
    return
