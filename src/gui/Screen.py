from win32api import GetSystemMetrics


def get_screen_resolution():
    return GetSystemMetrics(0), GetSystemMetrics(1)  # width, height


def get_max_size(original_size, bounding_box, tiles=(1, 1), upscaling=False):
    """
    Get the maximum size to which to scale an image so it fits into the given bounding box while maintaining the
    aspect ratio.
    :param original_size: The original size of the image that is being resized (width, height)
    :param bounding_box: The bounding box in which to fit the scaled image (width, height)
    :param tiles: The number of tiles of the image that need to fit into the bounding box (rows, columns)
                  Default is (1, 1), i.e. single image with no tiling.
    :param upscaling: Whether or not the output size is alowed to be larger than the original size. Default: False
    :return: The recommended size of the image (width, height) so it will fit in the bounding box
    """
    total_width = original_size[0] * tiles[1]
    total_height = original_size[1] * tiles[0]
    scale_h = bounding_box[0] / total_width
    scale_v = bounding_box[1] / total_height

    scale = min(scale_h, scale_v)

    if upscaling is False:
        scale = min(1.0, scale)

    scaled_width = int(original_size[0] * scale)
    scaled_height = int(original_size[1] * scale)

    return scaled_width, scaled_height


def get_max_size_on_screen(image_size, tiles=(1, 1), margin=(0, 0), upscaling=False):
    """
    Get the size to which to scale an image so it will fit on the primary screen with the given margin,
    maintaining ascpect ratio.
    :param image_size: The original size of the image (width, height)
    :param tiles: The number of tiles of the image that need to fit onto the screen (rows, columns)
                  Default is (1, 1), i.e. single image with no tiling.
    :param margin: A margin by which the output size is smaller than the screen size (width_margin, height_margin).
                   (0, 0) for fullscreen
    :param upscaling: Whether or not the output size is alowed to be larger than the original size. Default: False
    :return: The recommended size of the image (width, height) so it will fit on the primary screen
    """
    screen = get_screen_resolution()
    bounding_box = (screen[0] - margin[0], screen[1] - margin[1])  # bb is screen size minus margins
    return get_max_size(image_size, bounding_box, tiles, upscaling)
