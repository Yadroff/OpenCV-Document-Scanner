import cv2
import os


def segmentation(image):
    # Load image, grayscale, Gaussian blur, Otsu's threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create rectangular structuring element and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=7)

    # Find contours and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    return cnts


def write_contours(image, cnts, out_dir):
    orig_img = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i, cnt in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.imwrite(os.path.join(out_dir, f'contour_{i}.jpg'), orig_img[y:y + h, x:x + w])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image
