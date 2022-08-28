# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-function-docstring, undefined-loop-variable, unused-argument, invalid-name
"""Compute definition and schedules for data transform operators"""
import numpy as np

from raf._tvm_op.nn import schedule_generic
from .._lib import generic_func
from .._lib import register_compute
from .._lib import strategy
from .._lib import tvm as _tvm  # pylint: disable=unused-import
from .._lib import _reg

_topi = _tvm.topi  # pylint: disable=no-member

def _get_max_threads(batch_size):
    target = _tvm.target.Target.current()
    max_threads = _tvm.target.Target.current(allow_none=False).max_num_threads
    if "vulkan" in str(target) and not isinstance(batch_size, _tvm.tir.IntImm):
        # SPIR-V does not support dynamic thread group size
        return max_threads
    return _tvm.tir.min(batch_size, max_threads)

def _calc_tensor_shape_ir(num_elements, data, output):
    ib = _tvm.tir.ir_builder.create()
    num_elements_ptr = ib.buffer_ptr(num_elements)
    if len(data.shape) == 2:
        n = _tvm.topi.cast(data.shape[1], dtype="int64")
    else:
        n = _tvm.tir.const(len(data.shape), "int64")    
    output_ptr = ib.buffer_ptr(output)
    max_threads = _get_max_threads(1)
    with ib.new_scope():
        nthread_tx = max_threads
        nthread_bx = _tvm.topi.utils.ceil_div(1, max_threads)
        tx = _tvm.te.thread_axis("threadIdx.x")
        bx = _tvm.te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        tid = bx * max_threads + tx
        with ib.if_scope(tid == 0):
            output_ptr[0] = num_elements_ptr[0]
            output_ptr[1] = n
    return ib.get()

def _calc_tensor_shape_scalar_ir(num_elements, data, output):
    ib = _tvm.tir.ir_builder.create()
    num_elements_ptr = ib.buffer_ptr(num_elements)
    #if len(data.shape) == 2:
    #    n = _tvm.topi.cast(data.shape[1], dtype="int64")
    #else:
    #    n = _tvm.tir.const(len(data.shape), "int64")    
    output_ptr = ib.buffer_ptr(output)
    max_threads = _get_max_threads(1)
    with ib.new_scope():
        nthread_tx = max_threads
        nthread_bx = _tvm.topi.utils.ceil_div(1, max_threads)
        tx = _tvm.te.thread_axis("threadIdx.x")
        bx = _tvm.te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        tid = bx * max_threads + tx
        with ib.if_scope(tid == 0):
            output_ptr[0] = num_elements_ptr[0]
            #output_ptr[1] = n
    return ib.get()

def _calc_tensor_shape_like_ir(data, output):
    ib = _tvm.tir.ir_builder.create()    
    ndim = _tvm.tir.const(len(data.shape), "int64")    
    output_ptr = ib.buffer_ptr(output)
    max_threads = _get_max_threads(1)
    with ib.new_scope():
        nthread_tx = max_threads
        nthread_bx = _tvm.topi.utils.ceil_div(1, max_threads)
        tx = _tvm.te.thread_axis("threadIdx.x")
        bx = _tvm.te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        tid = bx * max_threads + tx
        with ib.if_scope(tid == 0):
            output_ptr[0] = _tvm.topi.cast(data.shape[0], dtype="int64")
            output_ptr[1] = ndim
    return ib.get()    

def _calc_tensor_shape_like_scalar_ir(data, output):
    ib = _tvm.tir.ir_builder.create()    
    #ndim = _tvm.tir.const(len(data.shape), "int64")    
    #import pdb; pdb.set_trace()
    output_ptr = ib.buffer_ptr(output)
    max_threads = _get_max_threads(1)
    with ib.new_scope():
        nthread_tx = max_threads
        nthread_bx = _tvm.topi.utils.ceil_div(1, max_threads)
        tx = _tvm.te.thread_axis("threadIdx.x")
        bx = _tvm.te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        tid = bx * max_threads + tx
        with ib.if_scope(tid == 0):
            output_ptr[0] = _tvm.topi.cast(data.shape[0], dtype="int64")
            #output_ptr[1] = ndim
    return ib.get()      

def _calc_tensor_shape(num_elements, data, scalar=False):
    """
    num_elements is a tensor.
    """
    num_elemenets_buf = _tvm.tir.decl_buffer(num_elements.shape, num_elements.dtype, "num_elements_buf", data_alignment=8)
    data_buf = _tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    if scalar:
        output_buf = _tvm.tir.decl_buffer((1,), "int64", "output_buf", data_alignment=8)
        f_compute = lambda ins, outs: _calc_tensor_shape_scalar_ir(*ins, *outs)
    else:
        output_buf = _tvm.tir.decl_buffer((2,), "int64", "output_buf", data_alignment=8)
        f_compute = lambda ins, outs: _calc_tensor_shape_ir(*ins, *outs)
    return _tvm.te.extern(
        [(1,)] if scalar else [(2,)],
        [num_elements, data],
        #lambda ins, outs: _calc_tensor_shape_ir(ins[0], ins[1], outs[0]),
        f_compute,
        #lambda ins, outs: _calc_tensor_shape_scalar_ir(ins[0], ins[1], outs[0]) if scalar else lambda ins, outs: _calc_tensor_shape_ir(ins[0], ins[1], outs[0])
        dtype=["int64"],
        in_buffers=[num_elemenets_buf, data_buf],
        out_buffers=[output_buf],
        name="_calc_tensor_shape",
        tag="_calc_tensor_shape_gpu",
    )

def _calc_tensor_shape_like(data, scalar=False):
    """
    num_elements is a number.
    """    
    data_buf = _tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    if scalar:
        output_buf = _tvm.tir.decl_buffer((1,), "int64", "output_buf", data_alignment=8)
        f_compute = lambda ins, outs: _calc_tensor_shape_like_scalar_ir(*ins, *outs)
    else:
        output_buf = _tvm.tir.decl_buffer((2,), "int64", "output_buf", data_alignment=8)
        f_compute = lambda ins, outs: _calc_tensor_shape_like_ir(*ins, *outs)
    #import pdb; pdb.set_trace()
    return _tvm.te.extern(
        [(1,)] if scalar else [(2,)],
        [data],
        #lambda ins, outs: _calc_tensor_shape_like_ir(ins[0], outs[0]),
        f_compute,
        #lambda ins, outs: _calc_tensor_shape_like_scalar_ir(*ins, *outs) if scalar else lambda ins, outs: _calc_tensor_shape_like_ir(ins[0], outs[0]),
        dtype=["int64"],
        in_buffers=[data_buf],
        out_buffers=[output_buf],
        name="_calc_tensor_shape_like",
        tag="_calc_tensor_shape_like_gpu",
    )    

def _calc_num_unique_ir(data, output):
    ib = _tvm.tir.ir_builder.create()
    data_ptr = ib.buffer_ptr(data)
    output_ptr = ib.buffer_ptr(output)
    max_threads = _get_max_threads(1)
    with ib.new_scope():
        nthread_tx = max_threads
        nthread_bx = _tvm.topi.utils.ceil_div(1, max_threads)
        tx = _tvm.te.thread_axis("threadIdx.x")
        bx = _tvm.te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        tid = bx * max_threads + tx
        with ib.if_scope(tid == 0):
            output_ptr[tid] = data_ptr[data.shape[0] - 1] + 1
    return ib.get()

def _calc_num_unique(inc_scan):
    inc_scan_buf = _tvm.tir.decl_buffer(inc_scan.shape, inc_scan.dtype, "inc_scan_buf", data_alignment=8)
    output_buf = _tvm.tir.decl_buffer((1,), "int64", "output_buf", data_alignment=8)
    return _tvm.te.extern(
        [(1,)],
        [inc_scan],
        lambda ins, outs: _calc_num_unique_ir(ins[0], outs[0]),
        dtype=["int64"],
        in_buffers=[inc_scan_buf],
        out_buffers=[output_buf],
        name="_calc_num_unique",
        tag="_calc_num_unique_gpu",
    )

def _calc_unique_ir(
    data, argsorted_indices, inc_scan, unique_elements, inverse_indices, counts
):
    ib = _tvm.tir.ir_builder.create()
    data_ptr = ib.buffer_ptr(data)
    argsorted_indices_ptr = ib.buffer_ptr(argsorted_indices)
    inc_scan_ptr = ib.buffer_ptr(inc_scan)
    unique_elements_ptr = ib.buffer_ptr(unique_elements)
    inverse_indices_ptr = ib.buffer_ptr(inverse_indices)

    if isinstance(counts, _tvm.tir.Buffer):
        counts_ptr = ib.buffer_ptr(counts)
        # use indices_ptr as a tmp buffer to store tids with inc_scan[tid] != inc_scan[tid-1]
        unique_seq_indices_ptr = ib.buffer_ptr(inverse_indices)

    n_indices = _tvm.topi.cast(data.shape[0], dtype="int64")
    #n = data.shape[1]
    max_threads = _get_max_threads(n_indices)

    # if need to return counts
    if isinstance(counts, _tvm.tir.Buffer):
        num_unique = inc_scan_ptr[inc_scan.shape[0] - 1] + 1
        with ib.new_scope():
            nthread_tx = max_threads
            nthread_bx = _tvm.topi.utils.ceil_div(n_indices, max_threads)
            tx = _tvm.te.thread_axis("threadIdx.x")
            bx = _tvm.te.thread_axis("blockIdx.x")
            ib.scope_attr(tx, "thread_extent", nthread_tx)
            ib.scope_attr(bx, "thread_extent", nthread_bx)
            tid = bx * max_threads + tx
            with ib.if_scope(tid < n_indices):
                with ib.if_scope(tid == 0):
                    unique_seq_indices_ptr[num_unique - 1] = n_indices
                with ib.else_scope():
                    with ib.if_scope(inc_scan_ptr[tid] != inc_scan_ptr[tid - 1]):
                        unique_seq_indices_ptr[inc_scan_ptr[tid] - 1] = tid
        with ib.new_scope():
            nthread_tx = max_threads
            nthread_bx = _tvm.topi.utils.ceil_div(n_indices, max_threads)
            tx = _tvm.te.thread_axis("threadIdx.x")
            bx = _tvm.te.thread_axis("blockIdx.x")
            ib.scope_attr(tx, "thread_extent", nthread_tx)
            ib.scope_attr(bx, "thread_extent", nthread_bx)
            tid = bx * max_threads + tx
            with ib.if_scope(tid < num_unique):
                unique_idx = tid
                with ib.if_scope(tid == 0):
                    counts_ptr[unique_idx] = unique_seq_indices_ptr[tid]
                with ib.else_scope():
                    counts_ptr[unique_idx] = (
                        unique_seq_indices_ptr[tid] - unique_seq_indices_ptr[tid - 1]
                    )

    # calculate unique elements and inverse indices
    with ib.new_scope():
        nthread_tx = max_threads
        nthread_bx = _tvm.topi.utils.ceil_div(n_indices, max_threads)
        tx = _tvm.te.thread_axis("threadIdx.x")
        bx = _tvm.te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        tid = bx * max_threads + tx
        with ib.if_scope(tid < n_indices):
            data_idx = argsorted_indices_ptr[tid]
            unique_idx = inc_scan_ptr[tid]
            inverse_indices_ptr[data_idx] = unique_idx
    
    n_indices = data.shape[0] # 4
    n = data.shape[1] # 3
    max_threads = _get_max_threads(n_indices * n)
    with ib.new_scope():
        nthread_tx = max_threads
        nthread_bx = _tvm.topi.utils.ceil_div(n_indices * n, max_threads)
        tx = _tvm.te.thread_axis("threadIdx.x")
        bx = _tvm.te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        tid = bx * max_threads + tx
        with ib.if_scope(tid < n_indices * n):
            first_idx = _tvm.te.floordiv(tid, n)            
            data_idx = argsorted_indices_ptr[first_idx]            
            unique_idx = inc_scan_ptr[first_idx]
            with ib.if_scope(first_idx == 0):
                unique_elements_ptr[unique_idx * n + tid % n] = data_ptr[data_idx * n + tid % n]
            with ib.else_scope():
                with ib.if_scope(inc_scan_ptr[first_idx] != inc_scan_ptr[first_idx - 1]):
                    unique_elements_ptr[unique_idx * n + tid % n] = data_ptr[data_idx * n + tid % n]
    return ib.get()

def sort_dim_zero_thrust(data):
    """Performs sorting along the dimension 0 of the array.
    """
    #print(indices_buf.shape)
    out = _tvm.te.extern(
        [(data.shape[0],)],
        [data],
        lambda ins, outs: _tvm.tir.call_packed(
            "tvm.contrib.thrust.sort_dim_zero", ins[0], outs[0]
        ),    
        #in_buffers=[value_buf],
        #out_buffers=[indices_buf],
        name="sort_dim_zero",
        tag="sort_dim_zero_gpu",
    )
    #import pdb; pdb.set_trace()
    return out

@register_compute("raf.op.tvm.upper_bound.unique_dim", level=15)
#@register_compute("raf.op.tvm.unique_dim", level=15)
def unique_dim_compute(attrs, inputs, output_type):
    """
    This function follows the implementation from Pytorch
    https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/Unique.cu
    NOTE: We set dim to 0 for now.
    """
    data = inputs[0]
    dim = attrs.dim
    return_counts = attrs.return_counts
    return_inverse = attrs.return_inverse
    is_sorted = attrs.sorted

    assert dim == 0
    assert is_sorted == True
    assert len(data.shape) == 2
                
    #sorted_indices = sort_dim_zero_thrust(data) # Runtime error: illegal memory access
    sorted_indices = _tvm.topi.cuda.lexsort(data)
    #import pdb; pdb.set_trace()
    # adjacent difference
    adjacent_diff = _tvm.topi.cuda.adjacent_diff_dim_zero(data, sorted_indices)
    #import pdb; pdb.set_trace()
    #return [sorted_indices]
    #return [adjacent_diff]
    #return [sorted_indices, adjacent_diff]
    '''
    # generate fake data
    unique_elements_shape = _calc_tensor_shape_like(data)
    inverse_indices_shape = _calc_tensor_shape_like(sorted_indices)
    counts_shape = _calc_tensor_shape_like(sorted_indices2)    
    #ret = [data, unique_elements_shape, sorted_indices, inverse_indices_shape, sorted_indices2, counts_shape]
    ret = [sorted_indices]
    return ret
    '''    
    # cumsum    
    #inc_scan = _tvm.topi.cuda.scan_thrust(adjacent_diff, output_dtype="int64", exclusive=False)
    # scan_thrust seems like to have bugs
    inc_scan = _tvm.topi.cuda.inclusive_scan(adjacent_diff, output_dtype="int64")
    # total number of unique elements
    num_unique_elements = _calc_num_unique(inc_scan)
    #return [adjacent_diff, inc_scan]

    inverse_indices_buf = _tvm.tir.decl_buffer(
        (data.shape[0],), "int64", "inverse_indices_buf", data_alignment=8
    )
    # prepare buffers
    inc_scan_buf = _tvm.tir.decl_buffer((data.shape[0],), "int64", "inc_scan_buf", data_alignment=8)
    data_buf = _tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    unique_elements_buf = _tvm.tir.decl_buffer(data.shape, data.dtype, "unique_elements_buf", data_alignment=8)    
    sorted_indices_buf = _tvm.tir.decl_buffer(sorted_indices.shape, "int64", "sorted_indices_buf", data_alignment=8)
    in_data = [data, sorted_indices, inc_scan]
    in_buffers = [data_buf, sorted_indices_buf, inc_scan_buf]
    if return_counts:
        counts_buf = _tvm.tir.decl_buffer((data.shape[0],), "int64", "counts_buf", data_alignment=8)
        out_data_shape = [data.shape, (data.shape[0],), (data.shape[0],)]
        out_buffers = [unique_elements_buf, inverse_indices_buf, counts_buf]
        out_dtypes = [data.dtype, "int64", "int64"]
        fcompute = lambda ins, outs: _calc_unique_ir(*ins, *outs)
    else:
        out_data_shape = [data.shape, (data.shape[0],)]
        out_buffers = [unique_elements_buf, inverse_indices_buf]
        out_dtypes = [data.dtype, "int64"]
        fcompute = lambda ins, outs: _calc_unique_ir(*ins, *outs, None)
    outs = _tvm.te.extern(
        out_data_shape,
        in_data,
        fcompute,
        dtype=out_dtypes,
        in_buffers=in_buffers,
        out_buffers=out_buffers,        
        name="_calc_unique",
        tag="_calc_unique_gpu",
    )
    unique_elements = outs[0]
    inverse_indices = outs[1]
    if return_counts:
        counts = outs[2]

    #return [inverse_indices, counts]
    #return [adjacent_diff, inc_scan]

    # compute the shape of each output tensor
    #import pdb; pdb.set_trace()
    unique_elements_shape = _calc_tensor_shape(num_unique_elements, unique_elements)
    inverse_indices_shape = _calc_tensor_shape_like(inverse_indices, scalar=True)
    counts_shape = _calc_tensor_shape(num_unique_elements, counts, scalar=True)

    ret = [unique_elements, unique_elements_shape, inverse_indices, inverse_indices_shape]
    if return_counts:
        ret += [counts, counts_shape]
    #import pdb; pdb.set_trace()        
    
    return ret    

@generic_func
def schedule_unique_dim(attrs, outs, target):    
    with target:
        return _topi.generic.schedule_injective(outs)

@schedule_unique_dim.register(["cuda", "gpu"])
def schedule_unique_dim_cuda(attrs, outs, _):    
    outs = [outs] if isinstance(outs, _tvm.te.tensor.Tensor) else outs
    s = _tvm.te.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse(op):
        if _topi.tag.is_injective(op.tag):
            _topi.cuda.injective.schedule_injective_from_existing(s, op.output(0))
        for tensor in op.input_tensors:
            if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                traverse(tensor.op)
        scheduled_ops.append(op)

    for out in outs:
        traverse(out.op)
    return s

_reg.register_schedule("raf.op.tvm.upper_bound.unique_dim", schedule_unique_dim)
#_reg.register_schedule("raf.op.tvm.unique_dim", schedule_unique_dim)
