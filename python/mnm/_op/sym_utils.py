from numbers import Number

import numpy as np

from mnm._core.ndarray import Symbol, ndarray
from mnm._core.value import TensorValue, Value


def to_any(a):
    if isinstance(a, Symbol):
        return a._Symbol__expr  # pylint: disable=protected-access

    if a is None:
        return None

    if isinstance(a, (Number, str)):
        return a

    return to_tensor(a)


def to_tensor(a):
    if a is None:
        return None

    if isinstance(a, Symbol):
        return a._Symbol__expr  # pylint: disable=protected-access

    if isinstance(a, ndarray):
        return a._ndarray__handle._expr  # pylint: disable=protected-access

    if not isinstance(a, np.ndarray):
        a = np.array(a)

    return Value.as_const_expr(TensorValue.from_numpy(a))


def to_int_tuple(a):
    if isinstance(a, Symbol):
        return a._Symbol__expr  # pylint: disable=protected-access

    if isinstance(a, np.ndarray):
        a = a.tolist()

    if isinstance(a, Number):
        if int(a) != a:
            raise ValueError("Cannot convert to List[int]")

        return int(a)

    if not isinstance(a, (tuple, list)):
        raise ValueError("Cannot convert to List[int]")
    result = []

    for item in a:
        if isinstance(item, Number) and int(item) == item:
            result.append(int(item))
        else:
            raise ValueError("Cannot convert to List[int]")

    return result


def to_optional_int_tuple(a):
    return None if a is None else to_int_tuple(a)


def to_int(a):
    if isinstance(a, Symbol):
        return a._Symbol__expr  # pylint: disable=protected-access

    if isinstance(a, np.ndarray) and a.size == 1 and a.ndim <= 1:
        a = a.item()

    if isinstance(a, Number) and int(a) == a:
        return int(a)
    raise ValueError("Cannot convert to int")


def to_double(a):
    if isinstance(a, Symbol):
        return a._Symbol__expr  # pylint: disable=protected-access

    if isinstance(a, np.ndarray) and a.size == 1 and a.ndim <= 1:
        a = a.item()

    if isinstance(a, Number) and float(a) == a:
        return float(a)
    raise ValueError("Cannot convert to double")


def to_bool(a):
    if isinstance(a, Symbol):
        return a._Symbol__expr  # pylint: disable=protected-access

    if isinstance(a, np.ndarray) and a.size == 1 and a.ndim <= 1:
        a = a.item()

    if isinstance(a, Number) and bool(a) == a:
        return bool(a)
    raise ValueError("Cannot convert to bool")


def to_string(a):
    if isinstance(a, Symbol):
        return a._Symbol__expr  # pylint: disable=protected-access

    if isinstance(a, str):
        return a
    raise ValueError("Cannot convert to str")