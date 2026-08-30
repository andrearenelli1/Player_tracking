"""
Tile-grid utilities for running inference on crops of a frame instead of
the whole frame, to recover effective resolution on wide shots where the
ball becomes too small after the network's mandatory resize (e.g. out13:
a ball ~55px wide on a 3840x2160 frame becomes ~7px after resizing the
full frame to 512x288 -- below what HRNet can reliably detect).
"""


def compute_tile_grid(frame_w, frame_h, n=2, overlap_frac=0.1):
    """
    Builds an n x n grid of overlapping tiles covering the full frame.

    Using the SAME n for both rows and columns (rather than independent
    n_cols/n_rows) is what keeps each tile's aspect ratio equal to the
    frame's: our frames are 3840x2160 (16:9), which matches the network's
    512x288 input aspect ratio, so dividing both width and height by the
    same n preserves that 16:9 ratio per tile too -- no distortion beyond
    what the full-frame path already has.

    `overlap_frac` inflates each tile beyond the non-overlapping n x n
    split (e.g. 0.1 = 10% wider/taller than a plain 1/n split) so a ball
    sitting exactly on a grid line still falls well inside at least one
    tile instead of being split across two. Pick it comfortably larger
    than the expected ball size in pixels.

    Returns a list of (x0, y0, x1, y1) integer pixel boxes, row-major
    (top-left tile first).
    """
    if n <= 1:
        return [(0, 0, frame_w, frame_h)]

    base_w = frame_w / n
    base_h = frame_h / n
    tile_w = min(frame_w, base_w * (1 + overlap_frac))
    tile_h = min(frame_h, base_h * (1 + overlap_frac))

    xs = [round(i * (frame_w - tile_w) / (n - 1)) for i in range(n)]
    ys = [round(i * (frame_h - tile_h) / (n - 1)) for i in range(n)]

    tiles = []
    for y0 in ys:
        for x0 in xs:
            x1 = min(frame_w, int(round(x0 + tile_w)))
            y1 = min(frame_h, int(round(y0 + tile_h)))
            tiles.append((int(x0), int(y0), x1, y1))
    return tiles


def clip_rect_to_tile(rect, tile_box):
    """
    Converts a rectangle (x0,y0,x1,y1) in full-frame pixel coordinates
    into tile-local coordinates, clipped to the tile's bounds. Returns
    None if the rectangle doesn't overlap the tile at all. Used to remap
    --mask-rect regions (always specified in full-frame coordinates) onto
    each individual tile.
    """
    rx0, ry0, rx1, ry1 = rect
    tx0, ty0, tx1, ty1 = tile_box

    ix0, iy0 = max(rx0, tx0), max(ry0, ty0)
    ix1, iy1 = min(rx1, tx1), min(ry1, ty1)
    if ix0 >= ix1 or iy0 >= iy1:
        return None

    return (ix0 - tx0, iy0 - ty0, ix1 - tx0, iy1 - ty0)
