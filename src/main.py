import os
import traceback

from flask import Flask, request, make_response, jsonify

import detector

app = Flask(__name__)

IMG_FOLDER = '/app/images'


@app.route("/odlc", methods=["POST"])
def process_image():
    """
    Queue image POST request
    """

    # If any info is missing, throw an error
    try:
        req = request.get_json()
        assert "img_name" in req, "field 'img_name' is missing"

        # Process image
        img_path = f"{IMG_FOLDER}/{req['img_name']}"
        offsetX, offsetY = detector.detect(img_path)
        os.remove(img_path)

    except Exception:
        traceback.print_exc()
        return make_response("Badly formed image update", 400)

    return jsonify([offsetX, offsetY])
