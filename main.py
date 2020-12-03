from ocr import *

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import difflib


class Line:
    def __init__(self, a, b, c):
        # ax + by + c = 0
        self.a = a
        self.b = b
        self.c = c

    @staticmethod
    def from_2_points(x1, y1, x2, y2):
        a, b, c = y2-y1, x1-x2, y1*x2-x1*y2
        if a < 0 or (a == 0 > b) or (a == b == 0 > c):
            a = -a
            b = -b
            c = -c

        _gcd = abs(math.gcd(math.gcd(a, b), c))
        if _gcd > 0:
            a //= _gcd
            b //= _gcd
            c //= _gcd

        return Line(a, b, c)

    def get_intersect(self, other):
        if other.a * self.b == other.b * self.a:
            return [] if other.a * self.c != other.c * self.a else [(0, -self.c), None]

        a1, b1, c1 = self.a, self.b, self.c
        a2, b2, c2 = other.a, other.b, other.c
        return [((b1*c2-b2*c1)/(a1*b2-a2*b1), (c1*a2-c2*a1)/(a1*b2-a2*b1))]

    def __contains__(self, point):
        x, y = point
        return self.a * x + self.b * y + self.c == 0

    def __repr__(self):
        return "({}, {}, {})".format(self.a, self.b, self.c)


def line_from(x1, y1, x2, y2):
    # ax + by + c == 0
    a, b, c = y2-y1, x1-x2, y1*x2-x1*y2
    if a < 0 or (a == 0 > b) or (a == b == 0 > c):
        a = -a
        b = -b
        c = -c

    _gcd = abs(math.gcd(math.gcd(a, b), c))
    if _gcd > 0:
        a //= _gcd
        b //= _gcd
        c //= _gcd

    return (a, b, c)


class ImgProcessing:
    window_name = "test"

    @staticmethod
    def remove_noise(img):
        from scipy import ndimage
        im_blur = ndimage.gaussian_filter(img, 4)
        # im_blur = cv2.GaussianBlur(self.gray, (0, 0), 5, 5)
        return im_blur

    @staticmethod
    def canny(img, blur_ksize=5, min_threshold=100, max_threshold=200):
        """
        blur_ksize: Gaussian kernel size
        """
        # blurred = cv2.GaussianBlur(self.gray, (blur_ksize, blur_ksize), 0)
        img_canny = cv2.Canny(img, min_threshold, max_threshold)
        return img_canny

    @staticmethod
    def hough(img, canny_min_threshold, canny_max_threshold):
        edges = ImgProcessing.canny(img, 5, canny_min_threshold, canny_max_threshold)

        indices = np.where(edges != [0])
        canny_points = list(zip(indices[0], indices[1]))

        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        hough_lines = []

        if lines is not None:
            lines = lines.reshape((-1, 2))

            for r, theta in lines:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*r
                y0 = b*r
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                # cv2.line draws a line in img from the point(x1,y1) to (x2,y2). 
                # (0,0,255) denotes the colour of the line to be drawn. In this case, it is red.
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                hough_lines.append(Line.from_2_points(x1, y1, x2, y2))

            cv2.imwrite("test.jpg", img)
        else:
            print("There's no line to be detected.")

        return sorted(hough_lines, key=lambda t: (t.a, t.b, t.c))

    @staticmethod
    def cut_image(img_id):
        img = cv2.imread(get_image_filename(img_id))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img_id > 200:
            img = ImgProcessing.remove_noise(gray)
            hough_lines = ImgProcessing.hough(img, 10, 15)
        else:
            img = gray
            hough_lines = ImgProcessing.hough(img, 50, 200)

        # for l in hough_lines: print(l)

        start_y, start_x = 0, 0
        end_y, end_x = img.shape[:2]

        intersections = []
        for l1 in hough_lines:
            for l2 in hough_lines:
                i = l1.get_intersect(l2)
                if len(i) == 1:
                    intersections += i

        # print(sorted(intersections))

        return img[start_y:end_y, start_x:end_x]

    def cv2_show(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow(ImgProcessing.window_name, gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def plt_show(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plt.figure()
        plt.imshow(gray, cmap='gray')
        plt.show()


def compare_output(libname, img_id):
    expected_filename = 'expected_output/expected{0:0>3}.txt'.format(img_id)
    output_filename = 'output/output_{1}_{0:0>3}.txt'.format(img_id, libname)

    file1 = open(expected_filename, 'r', encoding='utf8')
    file2 = open(output_filename, 'r', encoding='utf8')

    file1_content = file1.read()
    file2_content = file2.read()

    s = difflib.SequenceMatcher(None, file1_content, file2_content)
    return s.ratio()


def summary(max_img_id):
    overall = {"pytesseract": [], "pyocr": [], "tesserocr": []}
    for img_id in range(1, max_img_id+1):
        if get_image_filename(img_id) is not None:
            for libname in ["pytesseract", "pyocr", "tesserocr"]:
                ratio = compare_output(libname, img_id)
                print(img_id, libname, round(ratio*100, 2))
                overall[libname].append(ratio)

    print("OVERALL")
    for libname in ["pytesseract", "pyocr", "tesserocr"]:
        s = sum(overall[libname])
        n = len(overall[libname])
        print(libname, round(100*s/n, 2))


if __name__ == '__main__':
    # print(compare_output())
    imgproc = ImgProcessing()
    # ImgProcessing.cut_image(205)
    # imgproc.cv2_show()
    # imgproc.plt_show(cv2.imread("test.jpg"))
    summary()
