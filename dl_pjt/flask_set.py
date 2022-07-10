from flask import Flask
from flask import render_template
from flask import request
from tensorflow.keras.models import load_model
from model import Facial_Kepoints_Detect
from utils import get_test_dataset, normalize_image
import matplotlib.pyplot as plt

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
    picture_number = request.args.get("picture_number")
    if picture_number == None:
        return render_template("FacialKeypoints.html", Output="")
    elif (int(picture_number) >= 1) and (int(picture_number) <= 1783):
        ith_image_data = get_test_dataset()
        ith_image_data = normalize_image(ith_image_data)
        result = web_model.get_model().predict(ith_image_data)
        picture_idx = int(picture_number) - 1
        plt.figure()
        plt.gca().invert_yaxis()
        plt.imshow(ith_image_data[picture_idx], cmap="gray")
        plt.plot(result[picture_idx][0], result[picture_idx][1], "rx")
        plt.savefig("static/img/result.png")
        return render_template("FacialKeypoints.html", Output=result[0])
    else:
        return render_template("FacialKeypoints.html", Output="")


app.run(host="0.0.0.0", port=5000)
