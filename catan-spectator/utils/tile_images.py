from catan.board import Terrain
from PIL import Image

IMAGE_DIR = "./tile_images"
IMAGE_SIZE = (86, 100)

def gen_tile_images():
    # Dict mapping terrain name to PIL image
    images = {}

    for terrain in Terrain:
        image_name = f"{IMAGE_DIR}/{terrain.value}.png"
        pilImage = Image.open(image_name)
        pilImage = pilImage.resize(IMAGE_SIZE, Image.ANTIALIAS) # Resize to fit on board appropriately
        images[terrain.value] = pilImage
    
    return images
