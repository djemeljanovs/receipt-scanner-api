from flask import Flask, request
from flask_restful import Resource, Api
from vision import detect_contour, four_point_transform, black_and_white
from text import detect_words, detect_all_words
from rect import rectify
from imutils import grab_contours
import numpy as np
import cv2
import base64
from pytesseract import image_to_string

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
api = Api(app)


def format_point(p):
    return {'x': p[0], 'y': p[1]}


def format_contour(c):
    return [
        {"x": int(c[0][0][0]), "y": int(c[0][0][1])},
        {"x": int(c[1][0][0]), "y": int(c[1][0][1])},
        {"x": int(c[2][0][0]), "y": int(c[2][0][1])},
        {"x": int(c[3][0][0]), "y": int(c[3][0][1])}
    ]


class DetectContour(Resource):

    def post(self):
        image = from_base64(request.data)
        contour = detect_contour(image)
        return {
            "success": "true" if len(contour) == 4 else "false",
            "points": format_contour(contour) if len(contour) == 4 else []
        }


class TransformPerspective(Resource):

    def post(self):
        image = from_base64(request.data.image)

        contour = request.data.contour if request.data.contour else detect_contour(image)
        warped = four_point_transform(image, rectify(contour))
        return to_base64(warped)


class ReadReceipt(Resource):


    def post(self):
        image = from_base64(request.data)
        return detect_all_words(image)



ENCODING = 'utf-8'


class AutoReadReceipt(Resource):


    def post(self):
        image = from_base64(request.data)
        contour = detect_contour(image)
        warped = four_point_transform(image, rectify(contour))
        text = detect_all_words(warped)
        return {
            "success": "true" if len(contour) == 4 else "false",
            "text": text,
            "points": format_contour(contour) if len(contour) == 4 else [],
            "warped": to_base64(warped)
        }


api.add_resource(DetectContour, '/api/contour')
api.add_resource(TransformPerspective, '/api/transform')
api.add_resource(ReadReceipt, '/api/read')
api.add_resource(AutoReadReceipt, '/api/autoread')


def to_base64(img):
    _, buf = cv2.imencode(".jpg", img)

    return base64.b64encode(buf).decode(ENCODING)


def from_base64(buf):
    buf_decode = base64.b64decode(buf)

    buf_arr = np.fromstring(buf_decode, dtype=np.uint8)

    return cv2.imdecode(buf_arr, cv2.IMREAD_UNCHANGED)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


