from flask import Flask, request
from flask_restful import Resource, Api
from vision import detect_contour, four_point_transform, black_and_white, detect_words
from rect import rectify
import numpy as np
import cv2
import base64

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
api = Api(app)


def format_point(p):
    return {'x': p[0], 'y': p[1]}


class DetectContour(Resource):

    def post(self):
        image = from_base64(request.data)
        contour = detect_contour(image)
        warped = four_point_transform(image, rectify(contour))
        cv2.imshow("Warped", warped)
        cv2.waitKey(0)

        detect_words(warped)
        cv2.waitKey(0)
        return {
            "success": "true" if len(contour) == 4 else "false",
            "points": np.array([format_point(p[0]) for p in contour]).tolist()
        }


api.add_resource(DetectContour, '/api/contour')


def to_base64(img):
    _, buf = cv2.imencode(".jpg", img)

    return base64.b64encode(buf)


def from_base64(buf):
    buf_decode = base64.b64decode(buf)

    buf_arr = np.fromstring(buf_decode, dtype=np.uint8)

    return cv2.imdecode(buf_arr, cv2.IMREAD_UNCHANGED)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


