from PIL import Image
import math
import os
import cv2

async def crop_image(path:str, resolution: int):
    try:
        # Open the original image
        image = Image.open(path)

        # Define the size of the smaller images
        size = resolution
        # 1024,768,512

        # Calculate the number of rows and columns of the grid
        rows = math.ceil(image.height / size)
        cols = math.ceil(image.width / size)

        # Iterate over the rows and columns and crop the image
        for r in range(rows):
            for c in range(cols):
                left = c * size
                top = r * size
                right = min(left + size, image.width)
                bottom = min(top + size, image.height)
                box = (left, top, right, bottom)
                cropped_image = image.crop(box)
                cropped_image.save(f"cropped_{r}_{c}.png")
        
        return "image success"
    except Exception as e:
        return f"An error occurred while processing the image: {str(e)}"

def create_grid(image,coordinate):
    frame = cv2.imread(image)
    results = coordinate.pandas().xyxy[0].to_dict(orient="records")
    for result in results:
        x1 = int(result['xmin'])
        y1 = int(result['ymin'])
        x2 = int(result['xmax'])
        y2 = int(result['ymax'])
        # Do whatever you want
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), 2)
    return frame

def split_image(image_path: str, crop_size: int, output_path : str):
    """
    Splits a large image into multiple pieces of the same size and saves them to disk.

    Args:
    - image_path (str): the path to the large image file
    - crop_size (int): the desired size of each cropped image (width and height)
    - output_path (str) : the path to output will be generated

    Returns:
    - A list of paths to the cropped image files.
    """
    
    # Open the original image
    image = Image.open(image_path)

    # Calculate the number of rows and columns of the grid
    rows = math.ceil(image.height / crop_size)
    cols = math.ceil(image.width / crop_size)

    cropped_path = f"{output_path}/cropped_{crop_size}"
    if not os.path.exists(cropped_path):
       os.makedirs(cropped_path)

    cropped_image_path =""
    
    # Iterate over the rows and columns and crop the image
    for r in range(rows):
        for c in range(cols):
            left = c * crop_size
            top = r * crop_size
            right = min(left + crop_size, image.width)
            bottom = min(top + crop_size, image.height)
            box = (left, top, right, bottom)

            # Crop the image
            cropped_image = image.crop(box)

            # Fill the rest of the image with black pixels if it doesn't have the exact same size as crop_size
            if cropped_image.size != (crop_size, crop_size):
                new_image = Image.new('RGB', (crop_size, crop_size), color = 'black')
                new_image.paste(cropped_image)
                cropped_image = new_image

            # Save the cropped image to disk
            cropped_image_path = f"{cropped_path}/cropped_{r}_{c}.jpg"
            cropped_image.save(cropped_image_path)
    return cropped_path, rows, cols

def combine_images(input_path: str, rows: int, cols: int, output_path: str):
    """
    Combines multiple images into one large image and saves it to disk.

    Args:
    - input_path (str): the path to the folder containing the cropped images
    - rows (int): the number of rows in the grid of cropped images
    - cols (int): the number of columns in the grid of cropped images
    - output_path (str): the path to the output image file

    Returns:
    - None
    """
    crop_size = int(input_path.split("_")[1])
    
    # Calculate the size of the output image
    width = cols * crop_size
    height = rows * crop_size

    # Create a new image to combine the cropped images
    new_image = Image.new('RGB', (width, height))

    # Iterate over the rows and columns and paste the cropped images into the new image
    for r in range(rows):
        for c in range(cols):
            # Open the cropped image
            cropped_image_path = os.path.join(input_path, f"cropped_{r}_{c}.jpg")
            cropped_image = Image.open(cropped_image_path)

            # Calculate the position of the cropped image in the new image
            left = c * crop_size
            top = r * crop_size
            right = left + crop_size
            bottom = top + crop_size
            box = (left, top, right, bottom)

            # Paste the cropped image into the new image
            new_image.paste(cropped_image, box)

    # Save the new image to disk
    new_image.save(f"{output_path}/result.jpg")
    return output_path