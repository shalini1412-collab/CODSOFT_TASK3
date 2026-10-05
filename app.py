from flask import Flask, render_template, request
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

print("Loading image captioning model...")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

print("Model loaded successfully!")


@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    image_path = None

    if request.method == "POST":
        if "image" not in request.files:
            return render_template(
                "index.html",
                error="Please select an image."
            )

        image = request.files["image"]

        if image.filename == "":
            return render_template(
                "index.html",
                error="Please select an image."
            )

        filename = secure_filename(image.filename)
        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        image.save(image_path)

        # Open image using Pillow
        img = Image.open(image_path).convert("RGB")

        # Process image
        inputs = processor(images=img, return_tensors="pt")

        # Generate caption
        output = model.generate(**inputs, max_new_tokens=50)

        caption = processor.decode(
            output[0],
            skip_special_tokens=True
        )

    return render_template(
        "index.html",
        caption=caption,
        image_path=image_path
    )


if __name__ == "__main__":
    app.run(debug=True)