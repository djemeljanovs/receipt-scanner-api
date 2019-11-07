vcap = cv2.VideoCapture("rtsp://admin:GANmobilly20@213.100.182.170:1001/Streaming/Channels/1")
while(1):
    ret, frame = vcap.read()
    if ret:
        height, width, layers = frame.shape
        new_h = int(height / 2)
        new_w = int(width / 2)
        img = cv2.resize(frame, (new_w, new_h))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grey scale
        edged = cv2.Canny(gray, 30, 200)  # Perform Edge detection
        cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

        screenCnt = None
        # loop over our contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(cv2.convexHull(c), 0.01 * peri, True)
            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            if len(approx) == 4:
                screenCnt = approx
                break

        if screenCnt is not None:
            cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
            # Masking the part other than the number plate
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
            new_image = cv2.bitwise_and(img, img, mask=mask)
            # Now crop
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))
            Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]
            # Read the number plate
            text = image_to_string(Cropped, config='--psm 11')
            print("Detected Number is:", text)
        cv2.imshow('VIDEO', img)
        cv2.waitKey(1)
    else:
        break