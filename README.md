# Neural Style Transfer (NST)

This project implements Neural Style Transfer using a pre-trained VGG19 model. It combines the content of one image with the style of another to generate a new image.

## Directory Structure


## Setup

1. Clone the repository.
2. Install the dependencies:

3. Place your content and style images in the `data/content/` and `data/style/` directories respectively.

4. Run the style transfer:

5. The output image will be saved in the `data/output/` directory.

## Notes

- You can adjust the weights for content and style in `src/nst.py` to fine-tune the style transfer result.
- A GPU is highly recommended for faster processing.
