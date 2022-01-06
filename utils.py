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
