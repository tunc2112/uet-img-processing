from PIL import Image
import cv2
import pytesseract
import tesserocr
from pyocr import pyocr
from pyocr import builders
import sys
import os


def get_image_filename(img_id):
    filename = "img_src/src{0:0>3}".format(img_id)
    for ext in [".png", ".jpg", ".jpeg"]:
        if os.path.exists(os.path.join(os.getcwd(), filename + ext)):
            return filename + ext


def write_output(libname, img_id, text):
    filename = "./output/output_{1}_{0:0>3}.txt".format(img_id, libname)
    with open(filename, "w") as f:
        f.write(text)


def ocr_pytesseract(img_id):
    img_filename = './' + get_image_filename(img_id)
    img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    original = pytesseract.image_to_string(gray, config='')
    write_output(img, original)


def ocr_pyocr(img_id):
    img_filename = './' + get_image_filename(img_id)
    img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)

    tool = tools[0]
    langs = tool.get_available_languages()
    lang = langs[0]
    txt = tool.image_to_string(
        Image.open(img_filename),
        lang=lang,
        builder=pyocr.tesseract.builders.TextBuilder()
    )
    write_output(img_id, txt)


def ocr_tesseract(img_id):
    img_filename = './' + get_image_filename(img_id)
    txt = tesserocr.image_to_text(Image.open(img_filename))
    write_output(img_id, txt)
