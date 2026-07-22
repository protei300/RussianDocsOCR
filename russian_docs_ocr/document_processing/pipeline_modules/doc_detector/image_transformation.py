import numpy as np
import cv2



def iou(bbox1: np.ndarray, bbox2: np.ndarray):
    """Compute intersection over union between two bboxes.

    Args:
        bbox1 (ndarray): First bounding box
        bbox2 (ndarray): Second bounding box

    Returns:
        IoU ratio value
    """
    area1 = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
    area2 = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])

    #intersection
    x1 = np.maximum(bbox1[..., 0], bbox2[..., 0])
    x2 = np.minimum(bbox1[..., 2], bbox2[..., 2])
    y1 = np.maximum(bbox1[..., 1], bbox2[..., 1])
    y2 = np.minimum(bbox1[..., 3], bbox2[..., 3])

    intersection = np.maximum(0, (x2-x1)) * np.maximum(0, (y2-y1))

    ratio = intersection / (area1 + area2 - intersection)

    return ratio

def xywh2xyxy(x):
    """Convert bboxes from (x,y,w,h) to (x1,y1,x2,y2) format.

    Args:
        x (ndarray): Bounding boxes in x,y,w,h format

    Returns:
        Converted bboxes in x1,y1,x2,y2 format
    """
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def sort_coordinates(list_of_xy_coords):
    """Sorts coordinates clockwise from centroid.

    Args:
        coords (list): List of (x, y) coordinate tuples

    Returns:
        Sorted coordinate list
    """
    cx, cy = list_of_xy_coords.mean(0)
    x, y = list_of_xy_coords.T
    angles = np.arctan2(x-cx, y-cy)
    indices = np.argsort(angles)
    return list_of_xy_coords[indices]

def get_angles(list_of_coords):
    """Calculates angle (degrees) between each coordinate and the next one.

    Args:
        coords (list): List of (x, y) tuples

    Returns:
        np.ndarray: Angles in degrees between adjacent coordinate pairs.
    """
    x, y = list_of_coords.T
    x_shifted, y_shifted = np.roll(list_of_coords,-1, axis=0).T
    return np.arctan2(x_shifted-x, y_shifted - y)/np.pi*180


def perp( a ) :
    """Calculates perpendicular vector.

    Args:
        a (list): Input 2D vector

    Returns:
        Perpendicular 2D vector
    """
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def seg_intersect(a1,a2, b1,b2) :
    """Finds intersection point of two lines.

    Args:
        a1, a2 (list): Endpoints of first line
        b1, b2 (list): Endpoints of second line

    Returns:
        Intersection point
    """
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1

def get_len(list_of_coords):
    """Approximates quadrangle from coordinates.

    Args:
        coords (list): List of (x, y) tuples

    Returns:
        Approx quadrangle coordinates
    """

    n_coord = len(list_of_coords)

    # searching 4 segments with max len
    list_of_coords_shifted = np.roll(list_of_coords,-1, axis=0)
    dist = np.linalg.norm(list_of_coords-list_of_coords_shifted, axis=1, ord=2)
    args_dist = np.sort(np.argsort(dist)[:-5:-1]) #searching 4 biggest segments


    # searching inconsistency in vectors order. like 0->2, where dist between em more then 1
    args_inconsistency = np.roll(args_dist,-1, axis=0) - args_dist
    args_inconsistency[-1] += n_coord


    # getting 4 points of 2 segments
    p2_ix = (args_dist[np.argwhere(args_inconsistency>1)[0]][0] + 1) % n_coord
    p1_ix = args_dist[np.argwhere(args_inconsistency>1)[0]][0]
    p3_ix = args_dist[(np.argwhere(args_inconsistency>1)[0] + 1) % len(args_dist)][0]
    p4_ix = (args_dist[(np.argwhere(args_inconsistency>1)[0] + 1) % len(args_dist)][0] + 1) % n_coord
    p = list_of_coords[[p1_ix, p2_ix, p3_ix, p4_ix]]

    # searching crossing point
    cross_p = seg_intersect(*p).astype(int)

    # adding point to list of coords
    list_of_coords = np.insert(list_of_coords, 0, cross_p, axis=0)
    list_of_coords = sort_coordinates(list_of_coords)

    # searching approx 4 points from new list of points
    peri = cv2.arcLength(list_of_coords, True)
    approx = cv2.approxPolyDP(list_of_coords, 0.05 * peri, True)
    approx = approx.reshape(approx.shape[0], -1)
    approx = sort_coordinates(approx)

    return approx


def order_points(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left.

    Uses the coordinate sum/difference method which is robust to rotation:
    the top-left has the smallest x+y, the bottom-right the largest; the
    top-right has the smallest y-x, the bottom-left the largest.

    Args:
        pts: array-like of 4 (x, y) points.

    Returns:
        np.ndarray (4, 2) float32 ordered [TL, TR, BR, BL].
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]       # TL: min(x + y)
    rect[2] = pts[np.argmax(s)]       # BR: max(x + y)
    d = np.diff(pts, axis=1)[:, 0]    # y - x
    rect[1] = pts[np.argmin(d)]       # TR: min(y - x)
    rect[3] = pts[np.argmax(d)]       # BL: max(y - x)
    return rect


def extract_quad(contour):
    """Extract a 4-point quadrilateral from a segmentation contour.

    Tries convex hull + adaptive polygon approximation; falls back to the
    minimum-area rotated rectangle (which always yields 4 corners).

    Args:
        contour: array-like (N, 2) contour points.

    Returns:
        np.ndarray (4, 2) float32, or None if the contour is degenerate.
    """
    cnt = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    if len(cnt) < 4:
        return None
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    for frac in (0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15):
        approx = cv2.approxPolyDP(hull, frac * peri, True).reshape(-1, 2)
        if len(approx) == 4:
            return approx.astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32)


def four_point_transform(img: np.ndarray, quad: np.ndarray):
    """Warp a quadrilateral to an axis-aligned rectangle.

    The output size is derived from the real side lengths of the quad, so the
    document aspect ratio is preserved (no stretching of tilted documents).

    Args:
        img: Source image.
        quad: 4 points (any order — they are canonically reordered).

    Returns:
        Warped image, or None if the target rectangle is degenerate.
    """
    rect = order_points(quad)
    tl, tr, br, bl = rect
    width = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    height = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    if width < 2 or height < 2:
        return None
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_LINEAR)


def fix_perspective(img: np.ndarray, segments: np.ndarray, stack: str = 'auto'):
    """Fix perspective of a document image using segmentation contours.

    Each segment is rectified independently with a robust four-point
    transform. When two documents are present (e.g. a passport spread) they
    are merged according to ``stack``:
      - 'auto'       : direction chosen from the pages' actual layout —
                       side-by-side -> horizontal, stacked -> vertical (default)
      - 'horizontal' : force left-to-right, common height, np.hstack
      - 'vertical'   : force top-to-bottom, common width, np.vstack

    Args:
        img: Input document image (H, W, 3).
        segments: List of contours (each (N, 2)) from the segmentation model.
        stack: Multi-document merge direction ('auto', 'horizontal', 'vertical').

    Returns:
        warped: Rectified image (or the original if no valid quad is found).
        cnt_img: Original image with the detected quadrilaterals drawn.
    """
    quads, warps = [], []

    for cnt in segments:
        quad = extract_quad(cnt)
        if quad is None:
            continue
        rect = order_points(quad)
        # clip corners to image bounds
        rect[:, 0] = np.clip(rect[:, 0], 0, img.shape[1])
        rect[:, 1] = np.clip(rect[:, 1], 0, img.shape[0])
        warped = four_point_transform(img, rect)
        if warped is None:
            continue
        quads.append(rect)
        warps.append(warped)

    cnt_img = img.copy()
    for q in quads:
        cnt_img = cv2.polylines(cnt_img, [q.astype(np.int32)], True, (255, 0, 0), 4)

    if not warps:
        return img, cnt_img
    if len(warps) == 1:
        return warps[0], cnt_img

    # multiple documents (two pages of a spread): decide merge direction.
    # 'auto' picks it from the pages' actual layout — if their centroids are
    # further apart horizontally than vertically they sit side by side
    # (stitch left->right), otherwise one above the other (stitch top->bottom).
    if stack == 'auto':
        c = [q.mean(axis=0) for q in quads]
        dx = abs(c[0][0] - c[1][0])
        dy = abs(c[0][1] - c[1][1])
        direction = 'horizontal' if dx >= dy else 'vertical'
    else:
        direction = stack

    if direction == 'horizontal':
        # side by side, left-to-right by leftmost x, common height
        order = np.argsort([q[:, 0].min() for q in quads])
        warps = [warps[i] for i in order]
        common_h = min(w.shape[0] for w in warps)
        warps = [
            cv2.resize(w, (max(1, int(round(w.shape[1] * common_h / w.shape[0]))), common_h),
                       interpolation=cv2.INTER_LINEAR)
            for w in warps
        ]
        return np.hstack(warps), cnt_img

    # vertical: top-to-bottom by topmost y, common width
    order = np.argsort([q[:, 1].min() for q in quads])
    warps = [warps[i] for i in order]
    common_w = min(w.shape[1] for w in warps)
    warps = [
        cv2.resize(w, (common_w, max(1, int(round(w.shape[0] * common_w / w.shape[1])))),
                   interpolation=cv2.INTER_LINEAR)
        for w in warps
    ]
    return np.vstack(warps), cnt_img



