import torch
import numpy as np


def is_tensor_or_array_or_value(obj):
    if isinstance(obj, torch.Tensor):
        return True
    if isinstance(obj, np.ndarray):
        return True
    if np.isscalar(obj):
        return True

    if isinstance(obj, bool) or isinstance(obj, str) or isinstance(obj, type(None)):
        raise NotImplementedError(f"not applicable for \'{obj}\'")

    return False


def flatten_structure(obj, tensor_list=None):
    if tensor_list is None:
        tensor_list = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            flatten_structure(value, tensor_list)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for item in obj:
            flatten_structure(item, tensor_list)
    elif is_tensor_or_array_or_value(obj):
        tensor_list.append(obj)

    return tensor_list


def restore_structure_recursive(original, tensor_list):
    if isinstance(original, dict):
        new_dict = {}
        for key, value in original.items():
            new_dict[key], tensor_list = restore_structure_recursive(value, tensor_list)
        return new_dict, tensor_list

    elif isinstance(original, list) or isinstance(original, tuple):
        new_list = []
        for item in original:
            restored_item, tensor_list = restore_structure_recursive(item, tensor_list)
            new_list.append(restored_item)
        if isinstance(original, tuple):
            new_list = tuple(new_list)
        return new_list, tensor_list

    elif is_tensor_or_array_or_value(original):
        return tensor_list.pop(0), tensor_list


def restore_structure(original, tensor_list):
    return restore_structure_recursive(original, tensor_list)[0]

if __name__ == "__main__":

    def test_obj(nest_obj):
        val_list = flatten_structure(nest_obj)
        val_list = [v * 2 for v in val_list]
        print(restore_structure(nest_obj, val_list))


    test_obj((2,3))

    test_obj((2,[3]))

    test_obj([2,3])

    test_obj({"a": 3.,
              "B": {'c': (1., 2.),
                    "d": [5., np.array([1, 3, 5, 7])], },
              "e": [3., 0, 1.]})
