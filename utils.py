import cv2
import numpy as np

from celluloid import Camera
from matplotlib import pyplot as plt

def project_point(image, point, model_view_matrix, projection_matrix, transformation_mat=None) -> tuple:
    """
    This function is project 3d Points in world space to image plane

    :param image: Image to project points to, do not modified used only for image size retrieval
    :param point: 3D point that we want to project
    :param model_view_matrix: This is inverse matrix of camera position in 3D space to tell where are we looking from
    :param projection_matrix: Projection matrix for camera that we used for image render
    :param transformation_mat: Matrix of transformation
    :return: Projected X and Y coordinate
    """
    height, width = image.shape[:2]

    pm = projection_matrix @ model_view_matrix
    if transformation_mat is not None:
        pm = pm @ transformation_mat

    x, y, z, w = pm @ point
    px, py = (x / w, -y / w)  # this point is normalized between -1 and 1
    return (px / 2 + 0.5) * width, (py / 2 + 0.5) * height  # denormalized values


def project_points(image, points, model_view_matrix, projection_matrix, transformation_mat=None) -> list:
    return [project_point(image, point, model_view_matrix, projection_matrix, transformation_mat)
            for point in points]


lasers = np.float32((
    ((-0.145, -0.1, -0.01), (-0.145, -0.3, -0.01)),
    ((-0.145, -0.1,  0.01), (-0.145, -0.3,  0.01)),
    (( 0.145, -0.1, -0.01), ( 0.145, -0.3, -0.01)),
    (( 0.145, -0.1,  0.01), ( 0.145, -0.3,  0.01))
))

bounding_box_points_in_3d = np.float32(
    [(-0.156, -0.182, -0.032),
     ( 0.156, -0.182, -0.032),
     (-0.156,  0.163, -0.032),
     ( 0.156,  0.163, -0.032),
     (-0.156, -0.182,  0.034),
     ( 0.156, -0.182,  0.034),
     (-0.156,  0.163,  0.034),
     ( 0.156,  0.163,  0.034)])

K = np.array([
    [497.7778,   0.0000, 256.0000],
    [0.0000, 746.6667, 256.0000],
    [0.0000,   0.0000,   1.0000]
    ], dtype=np.float32)

def laser_generator(origin, direction, rvect, tvec, max_len = 0.2, step_mul=0.2):
    o, d = origin.copy(), direction.copy()
    step = (direction - origin) * step_mul
    d = o.copy()
    yield np.array([o, d]), rvect, tvec
    while True:
        if np.linalg.norm(d - o) > max_len:
            o += step
        d += step
        yield np.array([o, d]), rvect, tvec

def rays_shooting_probability(rays_amount, period=25, variance=5):
    i = 0
    while True:
        if i % period == 0:
            next_shoot = np.random.randint(i, i+variance, rays_amount)
        shoot = i == next_shoot
        yield shoot
        i += 1

def produce_animation(frames, predicted_points_projections):
    """
    This function is produce animation of shooting xwing lasers based on predicted points projections
    :param frames: Images of flying xwing
    :param predicted_points_projections: predicted bounding box points projections
    :return: Animation that could be saved or displayed with IPython.display.HTML in notebook
    """
    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    plt.axis('off')
    camera = Camera(fig)

    laser_generators = []
    rays_prob_gen = rays_shooting_probability(lasers.shape[0])

    for fnum, (frame, pred_points) in enumerate(zip(frames, predicted_points_projections)):
        img = frame.copy()

        success, rotation_vector, translation_vector = cv2.solvePnP(bounding_box_points_in_3d,
                                                                    pred_points, K, distCoeffs=None,
                                                                    flags=cv2.SOLVEPNP_SQPNP)

        for l, prob in zip(lasers, next(rays_prob_gen)):
            if prob:
                laser_generators.append(laser_generator(l[0], l[1], rotation_vector, translation_vector))

        for lg in laser_generators:
            lpts, rvec, tvec = next(lg)
            points = cv2.projectPoints(lpts, rvec, tvec, K, distCoeffs=None)[0]
            if points.max() > 1000:
                continue
            p1 = points[0][0].astype(int)
            p2 = points[1][0].astype(int)
            img = cv2.line(img, tuple(p1), tuple(p2), (0, 255, 0), 1, cv2.LINE_AA)
        plt.imshow(img)
        camera.snap()

    return camera.animate(interval=33)
