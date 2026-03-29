from transformers import pipeline
from PIL import Image

class ImageAnalyzer:

    def __init__(self):
        self.model = pipeline(
            "image-classification",
            model="google/vit-base-patch16-224"
        )

    def analyze(self, image_path):

        image = Image.open(image_path)

        result = self.model(image)

        label = result[0]["label"]
        score = result[0]["score"]

        return label, score