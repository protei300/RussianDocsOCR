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
    """Calculates angle between adjacent coordinates.

    Args:
        coords (list): List of (x, y) tuples

    Prints:
        Angles between adjacent coordinate pairs
    """
    x, y = list_of_coords.T
    x_shifted, y_shifted = np.roll(list_of_coords,-1, axis=0).T
    print(np.arctan2(x_shifted-x, y_shifted - y)/np.pi*180)


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


def fix_perspective(img: np.ndarray, segments: np.ndarray, ):
    """Fix perspective of doc image using segments.

    Args:
        img: Input document image
        segments: Document segment coordinates

    Returns:
        warped: Rectified document image
        border_img: Image with segment borders drawn
    """
    cnts, imgs = [], []

    for cnt in segments:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
        approx = approx.reshape(approx.shape[0], -1)
        approx = sort_coordinates(approx)
        # print(approx)




        if approx.shape[0] != 4:
            approx = get_len(approx)

        if approx.shape[0] !=4: #if we fail to find 4 points then just skip
            continue

        approx[:, 0] = np.where(approx[:, 0] > img.shape[1], img.shape[1], approx[:, 0])
        approx[:, 1] = np.where(approx[:, 1] > img.shape[0], img.shape[0], approx[:, 1])

        top_left = [0, 0]
        bottom_right = approx.max(axis=0) - approx.min(axis=0)
        top_right = [top_left[0], bottom_right[1]]
        bottom_left = [bottom_right[0], top_left[0]]

        ideal_marks = np.vstack([top_left,
                                 top_right,
                                 bottom_right,
                                 bottom_left], )


        M = cv2.getPerspectiveTransform(approx, ideal_marks.astype(np.float32))
        imgs.append(cv2.warpPerspective(img.copy(), M, bottom_right.astype(np.int32), flags=cv2.INTER_LINEAR))
        cnts.append(approx)

    cnt_img = img.copy()
    for cnt in cnts:
        cnt_img = cv2.polylines(cnt_img, [cnt.astype(np.int32)], True, (255, 0, 0), 4)

    if 1 < len(cnts) :
        imgs[1] = cv2.resize(imgs[1], imgs[0].shape[:2][::-1], cv2.INTER_LINEAR)
        if cnts[0][0, 1] > cnts[1][0, 1]:
            warpimg = np.vstack(imgs[::-1])
        else:
            warpimg = np.vstack(imgs)
    elif len(imgs)>0:
        warpimg = imgs[0]
    else:
        warpimg = img


    return warpimg, cnt_img



