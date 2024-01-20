from enum import Enum


class Cropping(Enum):
    CENTER = 1
    CAUSAL = 2


def cropping_to_function(cropping: Cropping):
    if cropping == Cropping.CENTER:
        return center_crop_and_f
    if cropping == Cropping.CAUSAL:
        return causal_crop_and_f
    raise ValueError(f"What crop? {cropping}")


def center_crop_and_f(conv, x_res, f):
    seq_len = x_res.shape[-2]
    final_seq_len = conv.shape[-2]
    res_chop = (seq_len - final_seq_len) // 2
    return f(conv, x_res[:, res_chop:-res_chop, :])


def causal_crop_and_f(conv, x_res, f):
    final_seq_len = conv.shape[-2]
    return f(conv, x_res[:, -final_seq_len:, :])


def center_crop_and_add(conv, x_res):
    return center_crop_and_f(conv, x_res, lambda x, y: x + y)


def causal_crop_and_add(conv, x_res):
    return causal_crop_and_f(conv, x_res, lambda x, y: x + y)
