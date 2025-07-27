from flask import Blueprint, render_template

main = Blueprint('main', __name__)

@main.route("/")
def home():
    return render_template("home.html")

@main.route("/face-detection")
def face_detection():
    return render_template("face_detection.html")

@main.route("/face-eda")
def face_eda():
    return render_template("face_eda.html")

@main.route("/image-manipulation")
def image_manipulation():
    return render_template("image_manipulation.html")

@main.route("/image-eda")
def image_eda():
    return render_template("image_eda.html")

@main.route("/signature-verification")
def signature_verification():
    return render_template("signature_verification.html")

@main.route("/signature-eda")
def signature_eda():
    return render_template("signature_eda.html")