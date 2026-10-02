import torch


def build_detection_targets(
    boxes,
    labels,
    image_size=256,
    stride=8,
):
    """
    Convierte cajas [x1, y1, x2, y2] en objetivos
    para una cuadrícula anchor-free.

    Parámetros
    ----------
    boxes : Tensor [N, 4]
        Cajas en coordenadas de la imagen.

    labels : Tensor [N]
        Clase de cada caja.

    image_size : int
        Tamaño de la imagen cuadrada.

    stride : int
        Salto espacial entre la imagen y el feature map.

    Retorna
    -------
    objectness : Tensor [1, H, W]

    bbox : Tensor [4, H, W]

    classes : Tensor [H, W]
        Clase asignada a cada celda.
        -1 significa que la celda no contiene objeto.
    """

    grid_size = image_size // stride

    objectness = torch.zeros(
        1,
        grid_size,
        grid_size,
        dtype=torch.float32,
    )

    bbox = torch.zeros(
        4,
        grid_size,
        grid_size,
        dtype=torch.float32,
    )

    classes = torch.full(
        (grid_size, grid_size),
        -1,
        dtype=torch.long,
    )

    for box, label in zip(boxes, labels):

        x1, y1, x2, y2 = box

        # Centro de la caja
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Celda responsable
        grid_x = int(cx / stride)
        grid_y = int(cy / stride)

        # Protección contra límites
        grid_x = min(max(grid_x, 0), grid_size - 1)
        grid_y = min(max(grid_y, 0), grid_size - 1)

        # Distancias desde el centro hasta los bordes
        left = cx - x1
        top = cy - y1
        right = x2 - cx
        bottom = y2 - cy

        # Asignar target
        objectness[0, grid_y, grid_x] = 1.0

        bbox[0, grid_y, grid_x] = left
        bbox[1, grid_y, grid_x] = top
        bbox[2, grid_y, grid_x] = right
        bbox[3, grid_y, grid_x] = bottom

        classes[grid_y, grid_x] = label

    return objectness, bbox, classes