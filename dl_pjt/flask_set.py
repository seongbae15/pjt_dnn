from flask import Flask
from flask import render_template
from flask import request
from tensorflow.keras.models import load_model
from model import Facial_Kepoints_Detect


# Set model
web_model = Facial_Kepoints_Detect(
    input_size=[96, 96, 1], output_size=2, init_conv_filters=6,
)
model_file_name = "029-2.0174-0.9949.hdf5"
web_model.set_loaded_model(model_file_name)


# Set Flask
app = Flask(__name__)


@app.route("/")
@app.route("/FacialKeypoints")
def FacialKeypoints():
    # picture_number == test file index
    picture_number = request.args.get("picture_number")

    if picture_number == None:
        return render_template("FacialKeypoints.html", Output="")
    else:

        return render_template("FacialKeypoints.html", Output=picture_number)


app.run(host="0.0.0.0", port=5000)
