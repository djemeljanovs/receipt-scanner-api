import numpy as np
import cv2
import imutils
import pytesseract

from skimage.filters import threshold_local


def auto_canny(image, sigma=0.33):
    v = np.median(image)

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def detect_contour(image):
    if image is not None:
        width = image.shape[0]
        height = image.shape[1]
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        edged = auto_canny(blurred)
        #cv2.imshow("Canny", edged)
        #cv2.waitKey(0)
        contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        result = image.copy()
        #cv2.drawContours(result, contours, -1, (0, 255, 0), 3)
        #cv2.imshow("Contours", result)
        #cv2.waitKey(0)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        # loop over the contours
        for contour in contours:
            # approximate the contour
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(cv2.convexHull(contour), 0.01 * perimeter, True)

            # if our approximated contour has four points, then we
            # can assume that we have found the receipt
            if len(approx) == 4:
                return approx

        return np.array([
                [0, 0],
                [0, height],
                [width, height],
                [width, 0]
            ])

    return []


def four_point_transform(image, rect):
    # obtain a consistent order of the points and unpack them
    # individually
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def black_and_white(image):
    bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = threshold_local(bw, 11, offset=10, method="gaussian")
    bw = (bw > T).astype("uint8") * 255
    return bw


def detect_words(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # --- choosing the right kernel
    # --- kernel size of 3 rows (to join dots above letters 'i' and 'j')
    # --- and 10 columns to join neighboring letters in words and neighboring words
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
    #cv2.imshow('dilation', dilation)

    # ---Finding contours ---
    _, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    im2 = image.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = image[y:y+h, x:x+w]
        #cv2.imshow("ROI", roi)
        #cv2.waitKey(0)
        cv2.dilate(roi, (5, 5), roi)
        text = pytesseract.image_to_string(roi, config='--psm 7')
        print("{}\n".format(text))

    cv2.imshow('final', im2)
    cv2.waitKey(0)
