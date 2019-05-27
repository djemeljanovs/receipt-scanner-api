from flask import Flask, request
from flask_restful import Resource, Api
import numpy as np
import cv2
import imutils
import base64

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
api = Api(app)


class DetectContour(Resource):
    def post(self):
        image = from_base64(request.data)
        if image is not None:
            width = image.shape[0]
            height = image.shape[1]
            blured = cv2.GaussianBlur(image, (5, 5), 0)
            edged = cv2.Canny(blured, 75, 200)

            contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

            # loop over the contours
            for contour in contours:
                # approximate the contour
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(cv2.convexHull(contour), 0.01 * perimeter, True)

                # if our approximated contour has four points, then we
                # can assume that we have found the receipt
                if len(approx) == 4:
                    points = np.array([format_point(p[0]) for p in approx.tolist()])
                    return {
                        'success': 'true',
                        'points': points.tolist()
                    }

            return {
                'success': 'true',
                'points':  [
                    format_point([0, 0]),
                    format_point([0, height]),
                    format_point([width, height]),
                    format_point([0, height])
                ]
            }

        return {'success': 'false'}


def format_point(p):
    return {'x': p[0], 'y': p[1]}


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


