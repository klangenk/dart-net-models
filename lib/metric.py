

def my_accuracy(y_pred, y_true, thresh: float = 0.5, sigmoid: bool = True):
    n, c = y_true.shape
    DARTS = y_pred.shape[1] // 27
    y_pred = y_pred.view(n, DARTS, -1).argmax(dim=-1).view(n, -1) % 63
    y_true = y_true.view(n, DARTS, -1).argmax(dim=-1).view(n, -1) % 63
    return (y_pred == y_true).float().mean()

def my_accuracy2(y_pred, y_true, thresh: float = 0.5, sigmoid: bool = True):
    n, c = y_true.shape
    DARTS = y_pred.shape[1] // 27
    #y_true = y_true.repeat(1, 2)
    y_true_slice = y_true.view(n * DARTS, -1)[:, :20].argmax(dim=-1)
    y_true_ring = y_true.view(n * DARTS, -1)[:, 20:].argmax(dim=-1)
    y_pred_slice = y_pred.view(n * DARTS, -1)[:, :20].argmax(dim=-1)
    y_pred_ring = y_pred.view(n * DARTS, -1)[:, 20:].argmax(dim=-1)
    mask = y_true_ring < 3
    y_true_slice = y_true_slice[mask]
    y_pred_slice = y_pred_slice[mask]
    result = (y_pred_ring == y_true_ring).float().mean() * len(y_true_ring)
    if len(y_true_slice) > 0:
        result += (y_pred_slice == y_true_slice).float().mean() * len(y_true_slice)
    return result / (len(y_true_slice) + len(y_true_ring))


def my_accuracy3(y_pred, y_true, thresh: float = 0.5, sigmoid: bool = True):
    n, c = y_true.shape
    DARTS = y_pred.shape[1] // 27
    #y_true = y_true.repeat(1, 2)
    y_true_slice = y_true.view(n * DARTS, -1)[:, :20].argmax(dim=-1)
    y_true_ring = y_true.view(n * DARTS, -1)[:, 20:].argmax(dim=-1)
    y_pred_slice = y_pred.view(n * DARTS, -1)[:, :20].argmax(dim=-1)
    y_pred_ring = y_pred.view(n * DARTS, -1)[:, 20:].argmax(dim=-1)
    mask = y_true_ring < 3
    y_true_slice = y_true_slice[mask]
    y_pred_slice = y_pred_slice[mask]
    y_true_ring_1 = y_true_ring[~mask]
    y_pred_ring_1 = y_pred_ring[~mask]
    y_true_ring_2 = y_true_ring[mask]
    y_pred_ring_2 = y_pred_ring[mask]
    result = 0
    if len(y_true_slice) > 0:
        result += ((y_pred_slice == y_true_slice) & (y_pred_ring_2 == y_true_ring_2)).float().mean() * len(y_true_slice)
    if len(y_true_ring_1) > 0:
        result += (y_pred_ring_1 == y_true_ring_1).float().mean() * len(y_true_ring_1)
    return result / (len(y_true_slice) + len(y_true_ring_1))

def my_accuracy4(y_pred, y_true, thresh: float = 0.5, sigmoid: bool = True):
    n, c = y_true.shape
    DARTS = y_pred.shape[1] // 27
    y_true_slice = y_true.view(n * 3, -1)[:, :20].argmax(dim=-1)
    y_true_ring = y_true.view(n * 3, -1)[:, 20:].argmax(dim=-1)
    y_pred_slice = y_pred.view(n * 3, -1)[:, :20].argmax(dim=-1)
    y_pred_ring = y_pred.view(n * 3, -1)[:, 20:].argmax(dim=-1)
    mask = y_true_ring < 3
    y_true_slice = y_true_slice[mask]
    y_pred_slice = y_pred_slice[mask]
    y_true_ring_1 = y_true_ring[~mask]
    y_pred_ring_1 = y_pred_ring[~mask]
    y_true_ring_2 = y_true_ring[mask]
    y_pred_ring_2 = y_pred_ring[mask]
    result = 0
    if len(y_true_slice) > 0:
        result += ((y_pred_slice == y_true_slice) & (y_pred_ring_2 == y_true_ring_2)).float().mean() * len(y_true_slice)
    if len(y_true_ring_1) > 0:
        result += (y_pred_ring_1 == y_true_ring_1).float().mean() * len(y_true_ring_1)
    return result / (len(y_true_slice) + len(y_true_ring_1))


def my_accuracy5(y_pred, y_class, y_pos, thresh: float = 0.5, sigmoid: bool = True):
    n, c = y_class.shape
    DARTS = y_pred.shape[1] // 27
    #y_class = y_class.repeat(1, 2)
    y_class_slice = y_class.view(n * DARTS, -1)[:, :20].argmax(dim=-1)
    y_class_ring = y_class.view(n * DARTS, -1)[:, 20:].argmax(dim=-1)
    y_pred_slice = y_pred.view(n * DARTS, -1)[:, :20].argmax(dim=-1)
    y_pred_ring = y_pred.view(n * DARTS, -1)[:, 20:].argmax(dim=-1)
    mask = y_class_ring < 3
    y_class_slice = y_class_slice[mask]
    y_pred_slice = y_pred_slice[mask]
    y_class_ring_1 = y_class_ring[~mask]
    y_pred_ring_1 = y_pred_ring[~mask]
    y_class_ring_2 = y_class_ring[mask]
    y_pred_ring_2 = y_pred_ring[mask]
    result = 0
    if len(y_class_slice) > 0:
        result += ((y_pred_slice == y_class_slice) & (y_pred_ring_2 == y_class_ring_2)).float().mean() * len(y_class_slice)
    if len(y_class_ring_1) > 0:
        result += (y_pred_ring_1 == y_class_ring_1).float().mean() * len(y_class_ring_1)
    return result / (len(y_class_slice) + len(y_class_ring_1))


def double_acc(y_pred, y_true, thresh: float = 0.5, sigmoid: bool = True):
    n, c = y_true.shape
    y_pred = y_pred[:, :81]
    y_true_slice = y_true.reshape(n * 3, -1)[:, :20].argmax(dim=-1)
    y_true_ring = y_true.reshape(n * 3, -1)[:, 20:].argmax(dim=-1)
    y_pred_slice = y_pred.reshape(n * 3, -1)[:, :20].argmax(dim=-1)
    y_pred_ring = y_pred.reshape(n * 3, -1)[:, 20:].argmax(dim=-1)
    mask = y_true_ring < 3
    y_true_slice = y_true_slice[mask]
    y_pred_slice = y_pred_slice[mask]
    y_true_ring_1 = y_true_ring[~mask]
    y_pred_ring_1 = y_pred_ring[~mask]
    y_true_ring_2 = y_true_ring[mask]
    y_pred_ring_2 = y_pred_ring[mask]
    result = 0
    if len(y_true_slice) > 0:
        result += ((y_pred_slice == y_true_slice) & (y_pred_ring_2 == y_true_ring_2)).float().mean() * len(y_true_slice)
    if len(y_true_ring_1) > 0:
        result += (y_pred_ring_1 == y_true_ring_1).float().mean() * len(y_true_ring_1)
    return result / (len(y_true_slice) + len(y_true_ring_1))


def single_acc(y_pred, y_true, thresh: float = 0.5, sigmoid: bool = True):
    n, c = y_true.shape
    y_pred = y_pred[:, 81:]
    y_true = y_true.repeat(1, 2)
    y_true_slice = y_true.reshape(n * 6, -1)[:, :20].argmax(dim=-1)
    y_true_ring = y_true.reshape(n * 6, -1)[:, 20:].argmax(dim=-1)
    y_pred_slice = y_pred.reshape(n * 6, -1)[:, :20].argmax(dim=-1)
    y_pred_ring = y_pred.reshape(n * 6, -1)[:, 20:].argmax(dim=-1)
    mask = y_true_ring < 3
    y_true_slice = y_true_slice[mask]
    y_pred_slice = y_pred_slice[mask]
    y_true_ring_1 = y_true_ring[~mask]
    y_pred_ring_1 = y_pred_ring[~mask]
    y_true_ring_2 = y_true_ring[mask]
    y_pred_ring_2 = y_pred_ring[mask]
    result = 0
    if len(y_true_slice) > 0:
        result += ((y_pred_slice == y_true_slice) & (y_pred_ring_2 == y_true_ring_2)).float().mean() * len(y_true_slice)
    if len(y_true_ring_1) > 0:
        result += (y_pred_ring_1 == y_true_ring_1).float().mean() * len(y_true_ring_1)
    return result / (len(y_true_slice) + len(y_true_ring_1))

def total_acc(y_pred, y_true, thresh: float = 0.5, sigmoid: bool = True):
    n, c = y_true.shape
    DARTS = y_pred.shape[1] // 27
    y_true = y_true.repeat(1, DARTS // 3)
    y_true_slice = y_true.view(n * DARTS, -1)[:, :20].argmax(dim=-1)
    y_true_ring = y_true.view(n * DARTS, -1)[:, 20:].argmax(dim=-1)
    y_pred_slice = y_pred.view(n * DARTS, -1)[:, :20].argmax(dim=-1)
    y_pred_ring = y_pred.view(n * DARTS, -1)[:, 20:].argmax(dim=-1)
    mask = y_true_ring < 3
    y_true_slice = y_true_slice[mask]
    y_pred_slice = y_pred_slice[mask]
    y_true_ring_1 = y_true_ring[~mask]
    y_pred_ring_1 = y_pred_ring[~mask]
    y_true_ring_2 = y_true_ring[mask]
    y_pred_ring_2 = y_pred_ring[mask]
    result = 0
    if len(y_true_slice) > 0:
        result += ((y_pred_slice == y_true_slice) & (y_pred_ring_2 == y_true_ring_2)).float().mean() * len(y_true_slice)
    if len(y_true_ring_1) > 0:
        result += (y_pred_ring_1 == y_true_ring_1).float().mean() * len(y_true_ring_1)
    return result / (len(y_true_slice) + len(y_true_ring_1))