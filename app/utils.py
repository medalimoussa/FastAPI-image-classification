import base64
from PIL import Image
from io import BytesIO
import numpy as np


def decode_base64_image(image_str):
    image_data = base64.b64decode(image_str)
    return Image.open(BytesIO(image_data)).convert('RGB')

