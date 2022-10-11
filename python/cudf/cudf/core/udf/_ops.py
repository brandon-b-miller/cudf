# Copyright (c) 2021-2022, NVIDIA CORPORATION.

import math
import operator

arith_ops = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.mod,
    operator.pow,
    operator.iadd,
    operator.iand,
    operator.ifloordiv,
    operator.mod,
    operator.imul,
    operator.ior,
    operator.ipow,
    operator.isub,
    operator.itruediv,
    operator.ixor,
]

# only distinct for separate testing
shift_ops = [
    operator.lshift,
    operator.rshift,
    operator.ilshift,
    operator.irshift,
]

bitwise_ops = [operator.and_, operator.or_, operator.xor]

unary_ops = [
    math.acos,
    math.acosh,
    math.asin,
    math.asinh,
    math.atan,
    math.atanh,
    math.ceil,
    math.cos,
    math.degrees,
    math.erf,
    math.erfc,
    math.exp,
    math.expm1,
    math.fabs,
    math.floor,
    math.gamma,
    math.lgamma,
    math.log,
    math.log10,
    math.log1p,
    math.log2,
    math.radians,
    math.sin,
    math.sinh,
    math.sqrt,
    math.tan,
    math.tanh,
    operator.pos,
    operator.neg,
    operator.not_,
    operator.invert,
    operator.truth,
]

comparison_ops = [
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
]
