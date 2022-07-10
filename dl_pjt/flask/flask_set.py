from flask import Flask
from flask import render_template
from flask import request
from tensorflow.keras.models import load_model
from model import Facial_Kepoints_Detect


# Set model
web_model = Facial_Kepoints_Detect(
    input_size=[96, 96, 1], output_size=2, init_conv_filters=6,
)
# str = model_file name
model_file_name = ""
web_model.set_loaded_model(model_file_name)


# set Flask
app = Flask(__name__)


@app.route("/")
@app.route("/FacialKeypoints")
def FacialKeypointsPrediction():
    # picture_number == test file index
    picture_number = request.args.get("picture")

    # test file image preprocessing

    # disp output

    return render_template("facial_keypoints.html")


app.run(host="0.0.0.0", port=5000)
