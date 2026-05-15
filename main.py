from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove
from io import BytesIO
from PIL import Image, ImageEnhance
import cv2
import numpy as np

app = FastAPI()

# Enable CORS so the frontend can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# High-resolution passport dimensions (2x original resolution)
PASSPORT_WIDTH = 826
PASSPORT_HEIGHT = 1062


@app.get("/")
def home():
    return {"message": "AI Passport Photo Generator API is running"}


def enhance_image(image: Image.Image) -> Image.Image:
    """
    Improve brightness, contrast, and sharpness.
    """
    # Brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.05)

    # Contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.15)

    # Sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.35)

    return image


def smart_crop_with_face(image_np: np.ndarray) -> np.ndarray:
    """
    Detect the largest face and crop around the head and shoulders.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(80, 80)
    )

    # If no face detected, return original image
    if len(faces) == 0:
        return image_np

    # Select the largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    img_h, img_w = image_np.shape[:2]

    # Expand crop around face
    top = max(y - int(0.7 * h), 0)
    bottom = min(y + int(2.8 * h), img_h)
    left = max(x - int(1.0 * w), 0)
    right = min(x + int(2.0 * w), img_w)

    cropped = image_np[top:bottom, left:right]

    return cropped


@app.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
    # Read uploaded image
    input_bytes = await file.read()

    # Remove background
    output_bytes = remove(input_bytes)

    # Open transparent image
    transparent_img = Image.open(BytesIO(output_bytes)).convert("RGBA")

    # Create white background
    white_bg = Image.new("RGBA", transparent_img.size, (255, 255, 255, 255))
    white_bg.paste(transparent_img, (0, 0), transparent_img)

    # Convert to RGB
    rgb_img = white_bg.convert("RGB")

    # Convert to NumPy
    img_np = np.array(rgb_img)

    # Smart face crop
    img_np = smart_crop_with_face(img_np)

    # Convert back to Pillow
    final_img = Image.fromarray(img_np)

    # Enhance image quality
    final_img = enhance_image(final_img)

    # Resize to high-resolution passport dimensions
    final_img = final_img.resize(
        (PASSPORT_WIDTH, PASSPORT_HEIGHT),
        Image.LANCZOS
    )

    # Save to memory
    buffer = BytesIO()
    final_img.save(
        buffer,
        format="JPEG",
        quality=98,
        optimize=True
    )
    buffer.seek(0)

    # Return final image
    return StreamingResponse(
        buffer,
        media_type="image/jpeg",
        headers={
            "Content-Disposition": 'attachment; filename="passport_photo.jpg"'
        },
    )