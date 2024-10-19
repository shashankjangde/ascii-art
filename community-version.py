import argparse
import base64
import os
import platform
import random
import sys
from typing import List

import imageio.v3 as iio
import numpy as np
import streamlit as st
from PIL import (Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont,
                 ImageOps)

# ASCII patterns and color themes
ASCII_PATTERNS = {
    "basic": ["@", "#", "S", "%", "?", "*", "+", "-", "/", ";", ":", ",", "."],
    "complex": ["▓", "▒", "░", "█", "▄", "▀", "▌", "▐", "▆", "▇", "▅", "▃", "▂"],
    "emoji": ["😁", "😎", "🤔", "😱", "🤩", "😏", "😴", "😬", "😵", "😃"],
}

COLOR_THEMES = {
    "neon": [(57, 255, 20), (255, 20, 147), (0, 255, 255)],
    "pastel": [(255, 179, 186), (255, 223, 186), (186, 255, 201), (186, 225, 255)],
    "grayscale": [(i, i, i) for i in range(0, 255, 25)],
}


# Function to apply filters to the image
def apply_image_filters(
    image: Image.Image, brightness: float, contrast: float, blur: bool, sharpen: bool
) -> Image.Image:
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)

    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)

    if blur:
        image = image.filter(ImageFilter.BLUR)

    if sharpen:
        image = image.filter(ImageFilter.SHARPEN)

    return image


# Function to create contours
def create_contours(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.FIND_EDGES)


# Function to flip image
def flip_image(
    image: Image.Image, flip_horizontal: bool, flip_vertical: bool
) -> Image.Image:
    if flip_horizontal:
        image = ImageOps.mirror(image)  # Flip horizontally
    if flip_vertical:
        image = ImageOps.flip(image)  # Flip vertically
    return image


# Function to dynamically adjust aspect ratio based on ASCII pattern
def get_aspect_ratio(pattern: str) -> float:
    if pattern == "basic":
        return 0.55
    elif pattern == "complex":
        return 0.65
    elif pattern == "emoji":
        return 1.0
    return 0.55


# Function to resize the image dynamically based on the aspect ratio of the selected pattern
def resize_image(image: Image.Image, width: int, pattern: str) -> Image.Image:
    aspect_ratio = get_aspect_ratio(pattern)
    new_height = int((image.height / image.width) * width * aspect_ratio)
    return image.resize((width, new_height))


# resizes the image based on the original image's aspect ratios
def preserve_resize_image(image: Image.Image, width: int) -> Image.Image:
    original_aspect_ratio = image.height / image.width
    new_height = int(width * original_aspect_ratio)
    return image.resize((width, new_height))


# Function to map pixels to ASCII characters
def map_pixels_to_ascii(image: Image.Image, pattern: list) -> str:
    grayscale_image = image.convert("L")
    pixels = np.array(grayscale_image)
    ascii_chars = np.vectorize(
        lambda pixel: pattern[min(pixel // (256 // len(pattern)), len(pattern) - 1)]
    )(pixels)
    ascii_image = "\n".join(["".join(row) for row in ascii_chars])
    return ascii_image


# Function to create colorized ASCII art in HTML format
def create_colorized_ascii_html(image: Image.Image, pattern: list, theme: str) -> str:
    image = resize_image(image, 80, "basic")
    pixels = np.array(image)

    ascii_image_html = """
    <div style='font-family: monospace; white-space: pre;'>
    """

    color_palette = COLOR_THEMES.get(theme, COLOR_THEMES["grayscale"])

    for row in pixels:
        for pixel in row:
            ascii_char = pattern[int(np.mean(pixel) / 255 * (len(pattern) - 1))]
            color = random.choice(color_palette)
            ascii_image_html += f"<span style='color:rgb({color[0]},{color[1]},{color[2]})'>{ascii_char}</span>"
        ascii_image_html += "<br>"

    ascii_image_html += "</div>"
    return ascii_image_html


# function to get default monospace font from system
def get_monospace_font():
    # Determine the font path based on the operating system
    os_name = platform.system()
    if os_name == "Windows":
        font_path = "C:\\Windows\\Fonts\\cour.ttf"  # Courier New font
    elif os_name == "Darwin":  # macOS
        font_path = "/System/Library/Fonts/Menlo.ttc"  # Menlo font
    else:  # Assume Linux
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"  # DejaVu Sans Mono font
    # Load the font with a specified size
    font_size = 12
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # Fallback to default font if the specified font is not found
        print(
            "Could not load the specified monospaced font. Falling back to default font."
        )
        font = ImageFont.load_default()
    return font


def create_colored_ascii_art_image(
    ascii_text: str, font: ImageFont.ImageFont
, color_theme) -> Image.Image:
    # Split the ASCII text into lines
    lines = ascii_text.splitlines()

    # Calculate the width and height of the image based on the font size
    max_line_length = max(len(line) for line in lines)
    char_width = int(font.getlength("A"))
    char_height = int(font.getlength("A"))
    image_width = char_width * max_line_length + 20  # Add some padding
    image_height = char_height * len(lines) + 20  # Add some padding

    # Create a new image with a white background
    image = Image.new("RGB", (image_width, image_height), color="white")

    # Initialize the drawing context
    draw = ImageDraw.Draw(image)
    #colors = [
    #    (255, 180, 128),
    #    (255, 200, 150),
    #    (255, 170, 130),
    #    (255, 160, 200),
    #    (255, 180, 220),
    #    (128, 255, 140),
    #    (255, 140, 200),
    #    (255, 130, 180),
    #    (255, 210, 140),
    #    (200, 255, 130),
    #]
    colors = COLOR_THEMES[color_theme]

    # Start drawing text line by line, keeping alignment
    y_position = 10  # Start drawing at 10 pixels from the top
    for line in lines:
        x_position = 10  # Start drawing each line at 10 pixels from the left
        for char in line:
            # Generate a random color
            # Draw the character with the random color
            random_color = random.choice(colors)
            draw.text((x_position, y_position), char, fill=random_color, font=font)
            x_position += char_width  # Move to the next character position
        y_position += char_height  # Move to the next line

    return image


def create_gif(image_paths: List, output, animation_speed):
    images = []
    for i in image_paths:
        images.append(iio.imread(i))
    iio.imwrite(output, images, duration=animation_speed, loop=0)


def delete_temp_files(image_paths: List):
    for i in image_paths:
        os.remove(i)


def gif_loader(output):
    file_ = open(f"{output}", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    return data_url


# Streamlit app for the ASCII art generator
def run_streamlit_app():
    st.title("🌟 Customizable ASCII Art Generator")

    # Sidebar for options and settings
    st.sidebar.title("Settings")
    pattern_type = st.sidebar.selectbox(
        "Choose ASCII Pattern", options=["basic", "complex", "emoji"]
    )
    colorize = st.sidebar.checkbox("Enable Colorized ASCII Art")
    color_theme = st.sidebar.selectbox(
        "Choose Color Theme", options=list(COLOR_THEMES.keys())
    )
    if colorize and pattern_type!="emoji":
        animate = st.sidebar.checkbox("Animate Ascii-Art")
        animation_speed = st.sidebar.slider("Animation Speed in ms", 100, 800, step=100)
    else:
        animate = 0


    width = st.sidebar.slider("Set ASCII Art Width", 50, 150, 100)

    # New Flip Image Feature
    flip_horizontal = st.sidebar.checkbox("Flip Image Horizontally")
    flip_vertical = st.sidebar.checkbox("Flip Image Vertically")

    # Image filters
    brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0)
    contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0)
    apply_blur = st.sidebar.checkbox("Apply Blur")
    apply_sharpen = st.sidebar.checkbox("Apply Sharpen")

    # New Contour Feature
    apply_contours = st.sidebar.checkbox("Apply Contours")

    # Upload image
    uploaded_file = st.file_uploader(
        "Upload an image (JPEG/PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)

        # Apply filters to the image
        image = apply_image_filters(
            image, brightness, contrast, apply_blur, apply_sharpen
        )

        # Apply contour effect if selected
        if apply_contours:
            image = create_contours(image)

        # Flip the image if requested
        image = flip_image(image, flip_horizontal, flip_vertical)

        # Display the original processed image
        st.image(image, caption="Processed Image", use_column_width=True)

        if colorize and animate:
            image_resized = preserve_resize_image(image, width)
        else:
            # Resize the image based on the pattern type's aspect ratio
            image_resized = resize_image(image, width, pattern_type)

        # Generate ASCII art
        ascii_pattern = ASCII_PATTERNS[pattern_type]
        if colorize:
            if animate:
                ascii_text = map_pixels_to_ascii(image_resized, ascii_pattern)
                font = get_monospace_font()
                image_paths = []
                for i in range(5):
                    image = create_colored_ascii_art_image(ascii_text, font, color_theme)
                    path = f"image{i}.png"
                    image.save(path)
                    image_paths.append(path)
                output_path = "temp/animated.gif"
                create_gif(image_paths, output_path, animation_speed)
                delete_temp_files(image_paths)
                st.subheader("Animated ASCII Art Preview:")
                st.markdown(
                    f'<img src="data:image/gif;base64,{gif_loader(output_path)}" alt="output gif">',
                    unsafe_allow_html=True,
                )

            else:
                st.subheader("Colorized ASCII Art Preview:")
                ascii_html = create_colorized_ascii_html(
                    image_resized, ascii_pattern, color_theme
                )
                st.markdown(ascii_html, unsafe_allow_html=True)
        else:
            st.subheader("Grayscale ASCII Art Preview:")
            ascii_art = map_pixels_to_ascii(image_resized, ascii_pattern)
            st.text(ascii_art)

        # Download options
        if colorize:
            if animate:
                with open(output_path, "rb") as download_file:
                    data = download_file.read()
                    st.download_button(
                        "Download ASCII Art as GIF",
                        data,
                        file_name="ascii_art.gif",
                        mime="image/gif",
                    )

            else:
                st.download_button(
                    "Download ASCII Art as HTML",
                    ascii_html,
                    file_name="ascii_art.html",
                    mime="text/html",
                )
        else:
            st.download_button(
                "Download ASCII Art as Text",
                ascii_art,
                file_name="ascii_art.txt",
                mime="text/plain",
            )

    # Instructions for the user
    st.markdown(
        """
        - 🎨 Use the **Settings** panel to customize your ASCII art with patterns, colors, and image filters.
        - 📤 Upload an image in JPEG or PNG format to start generating your ASCII art.
        - 💾 Download your creation as a **text file**, **GIF** or **HTML** for colorized output.
    """
    )


# Check if the file path is valid
def is_valid_image_path(file_path: str) -> bool:
    if not os.path.exists(file_path):
        return False
    if not os.path.isfile(file_path):
        return False
    return True


# Command Line Interface (CLI) Function
def run_cli(
    input_image: str,
    output: str,
    pattern_type: str,
    width: int,
    brightness: float,
    contrast: float,
    blur: bool,
    sharpen: bool,
    colorize: bool,
    theme: str,
    apply_contours: bool,
    flip_horizontal: bool,
    flip_vertical: bool,
    animate: bool,
    animation_speed: int,
):
    image = Image.open(input_image)

    # Apply filters
    image = apply_image_filters(image, brightness, contrast, blur, sharpen)

    # Apply contour effect if selected
    if apply_contours:
        image = create_contours(image)

    # Flip the image if requested
    image = flip_image(image, flip_horizontal, flip_vertical)

    if colorize and animate:
        image_resized = preserve_resize_image(image, width)
    else:
        # Resize image
        image_resized = resize_image(image, width, pattern_type)

    # Generate ASCII art
    ascii_pattern = ASCII_PATTERNS[pattern_type]

    if colorize:
        if pattern_type!="emoji" and animate:
            # Animation for colorized ASCII art
            ascii_text = map_pixels_to_ascii(image_resized, ascii_pattern)
            font = get_monospace_font()
            image_paths = []
            for i in range(5):  # Create multiple frames for the GIF
                image_frame = create_colored_ascii_art_image(ascii_text, font)
                frame_path = f"frame_{i}.png"
                image_frame.save(frame_path)
                image_paths.append(frame_path)
            # Create a GIF
            create_gif(image_paths, output, animation_speed)
            delete_temp_files(image_paths)

    elif colorize:
        # Create colorized ASCII art in HTML format
        ascii_html = create_colorized_ascii_html(image_resized, ascii_pattern, theme)
        with open(output, "w", encoding="utf-8") as file:
            file.write(ascii_html)

    else:
        # Generate grayscale ASCII art
        ascii_art = map_pixels_to_ascii(image_resized, ascii_pattern)
        with open(output, "w", encoding="utf-8") as file:
            file.write(ascii_art)

    print(f"ASCII art saved to {output}")


# Main function for CLI execution
if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="Generate ASCII art from an image."
        )
        parser.add_argument("input_image", help="Path to the input image file.")
        parser.add_argument(
            "-o", "--output", default="output.txt", help="Output file name."
        )
        parser.add_argument(
            "-p",
            "--pattern",
            choices=ASCII_PATTERNS.keys(),
            default="basic",
            help="ASCII pattern.",
        )
        parser.add_argument(
            "-w", "--width", type=int, default=100, help="Width of ASCII art."
        )
        parser.add_argument(
            "-b", "--brightness", type=float, default=1.0, help="Brightness factor."
        )
        parser.add_argument(
            "-c", "--contrast", type=float, default=1.0, help="Contrast factor."
        )
        parser.add_argument("--blur", action="store_true", help="Apply blur effect.")
        parser.add_argument(
            "--sharpen", action="store_true", help="Apply sharpen effect."
        )
        parser.add_argument(
            "--colorize", action="store_true", help="Enable colorized ASCII art."
        )
        parser.add_argument(
            "-t",
            "--theme",
            choices=COLOR_THEMES.keys(),
            default="grayscale",
            help="Color theme.",
        )
        parser.add_argument(
            "--contours", action="store_true", help="Apply contour effect to the image."
        )
        parser.add_argument(
            "--flip_horizontal", action="store_true", help="Flip image horizontally."
        )
        parser.add_argument(
            "--flip_vertical", action="store_true", help="Flip image vertically."
        )
        parser.add_argument(
            "--animate", action="store_true", help="Animate ASCII art as a GIF."
        )
        parser.add_argument(
            "--animation_speed",
            type=int,
            default=300,
            help="Animation speed in milliseconds.",
        )

        args = parser.parse_args()
        run_cli(
            args.input_image,
            args.output,
            args.pattern,
            args.width,
            args.brightness,
            args.contrast,
            args.blur,
            args.sharpen,
            args.colorize,
            args.theme,
            args.contours,
            args.flip_horizontal,
            args.flip_vertical,
            args.animate,
            args.animation_speed,
        )
    else:
        run_streamlit_app()
