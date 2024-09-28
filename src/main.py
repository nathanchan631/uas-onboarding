import os
import json
import redis
import traceback

from flask import Flask, request, make_response, jsonify

import detector


app = Flask(__name__)

r = redis.Redis(host='redis', port=6379, db=0)
r.set('detector/curr_detection', json.dumps({}))

IMG_FOLDER = '/app/images'

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route("/odlc", methods=["GET"])
def get_detection():
    curr_detection = r.get('detector/curr_detection')
    print(f"{curr_detection=}")

    return jsonify(curr_detection)

@app.route("/odlc", methods=["POST"])
def queue_image_for_odlc():
    """
    Queue image POST request
    """

    # If any info is missing, throw an error
    try:
        telemetry = request.get_json()
        assert "img_name" in telemetry, "field 'img_name' is missing"
        assert "altitude" in telemetry, "field 'altitude' is missing"
        assert "latitude" in telemetry, "field 'latitude' is missing"
        assert "longitude" in telemetry, "field 'longitude' is missing"

        # Process image
        img_path = f"{IMG_FOLDER}/{telemetry.pop('img_name')}"
        detector.process_queued_image(img_path, telemetry)
        os.remove(img_path)

    except Exception as exc:
        traceback.print_exc()
        return make_response("Badly formed image update", 400)

    return make_response("Success", 200)
