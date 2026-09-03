import os
import sys
import argparse
import numpy as np
import fsa as F
from dataclasses import dataclass
from typing import Optional

_config_file: Optional[str] = None


def set_config_file(path: str) -> None:
    global _config_file
    _config_file = path


def _reset_allocator_if_configured() -> None:
    if _config_file is None:
        return
    from fsa.config import reset
    reset()
    F.init(_config_file)


@dataclass
class TestResult:
    name: str
    passed: bool
    expected: Optional[np.ndarray] = None
    actual: Optional[np.ndarray] = None
    error_msg: str = ""

def anti_transpose(M: np.ndarray) -> np.ndarray:
    return M.T[::-1, ::-1]


def print_results(results: list[TestResult]):
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name}")
        if not r.passed and r.error_msg:
            print(f"         {r.error_msg}")
    print(f"\n{passed}/{total} tests passed")
    print("=" * 60)
    return passed == total

# Instruction building helpers

def make_mem_stride_fields(stride_bytes: int):
    from fsa.instructions import InstructionField
    inst = InstructionField(stride_bytes, 20, 0, signed=True)
    stride_1 = (inst.value >> 15) & ((1 << 6) - 1)
    stride_2 = inst.value & ((1 << 15) - 1)
    return stride_1, stride_2


def make_dma_load(mem_tile, spad_tile, sem_id, rel_val, rows, cols, e_itemsize):
    from fsa.instructions import (
        DMAInstructionHeader, DMAInstrucionSRAM, DMAInstrucionMem,
        DMAInstruction, DMAFunc
    )
    row_bytes = cols * e_itemsize
    spad_row_bytes = spad_tile.shape[-1] * e_itemsize
    spad_row0 = spad_tile.data_ptr // spad_row_bytes
    stride_1, stride_2 = make_mem_stride_fields(row_bytes)

    return DMAInstruction(
        DMAInstructionHeader(
            semId=sem_id,
            acquireValid=False, acquireSemValue=0,
            releaseValid=True, releaseSemValue=rel_val,
            func=DMAFunc.LD_SRAM.value,
            repeat=rows
        ),
        DMAInstrucionSRAM(
            addr=spad_row0, stride=1, isAccum=False,
            mem_stride_1=stride_1
        ),
        DMAInstrucionMem(
            addr=mem_tile.data_ptr, stride_2=stride_2,
            size=row_bytes
        )
    )


def make_dma_transpose_load(mem_tile, spad_tile, sem_id, rel_val,
                             src_rows, src_cols, e_itemsize):
    from fsa.instructions import (
        DMAInstructionHeader, DMAInstrucionSRAM, DMAInstrucionMem,
        DMAInstruction, DMAFunc
    )
    spad_row_width = spad_tile.shape[-1]
    spad_row_bytes = spad_row_width * e_itemsize
    spad_row0 = spad_tile.data_ptr // spad_row_bytes

    mem_stride = src_cols * e_itemsize
    stride_1, stride_2 = make_mem_stride_fields(mem_stride)

    return DMAInstruction(
        DMAInstructionHeader(
            semId=sem_id,
            acquireValid=False, acquireSemValue=0,
            releaseValid=True, releaseSemValue=rel_val,
            func=DMAFunc.TRANSPOSE_SRAM.value,
            repeat=src_rows * src_cols
        ),
        DMAInstrucionSRAM(
            addr=spad_row0, stride=1, isAccum=False,
            mem_stride_1=stride_1
        ),
        DMAInstrucionMem(
            addr=mem_tile.data_ptr, stride_2=stride_2,
            size=src_rows
        )
    )


def make_dma_store(acc_tile, out_mem, sem_id, acq_val, rel_val,
                    rows, cols, a_itemsize):
    from fsa.instructions import (
        DMAInstructionHeader, DMAInstrucionSRAM, DMAInstrucionMem,
        DMAInstruction, DMAFunc
    )
    row_bytes = cols * a_itemsize
    acc_row_bytes = acc_tile.shape[-1] * a_itemsize
    acc_row0 = acc_tile.data_ptr // acc_row_bytes
    stride_1, stride_2 = make_mem_stride_fields(row_bytes)

    return DMAInstruction(
        DMAInstructionHeader(
            semId=sem_id,
            acquireValid=True, acquireSemValue=acq_val,
            releaseValid=True, releaseSemValue=rel_val,
            func=DMAFunc.ST_SRAM.value,
            repeat=rows
        ),
        DMAInstrucionSRAM(
            addr=acc_row0, stride=1, isAccum=True,
            mem_stride_1=stride_1
        ),
        DMAInstrucionMem(
            addr=out_mem.data_ptr, stride_2=stride_2,
            size=row_bytes
        )
    )


def make_load_stationary(spad_tile, sem_id, acq_val, rel_val, e_itemsize):
    from fsa.instructions import (
        MatrixInstructionHeader, MatrixInstructionSpad, MatrixInstrucionAcc,
        MatrixInstruction, MxFunc
    )
    spad_row_bytes = spad_tile.shape[-1] * e_itemsize
    spad_row0 = spad_tile.data_ptr // spad_row_bytes

    return MatrixInstruction(
        MatrixInstructionHeader(
            semId=sem_id,
            acquireValid=True, acquireSemValue=acq_val,
            releaseValid=True, releaseSemValue=rel_val,
            func=MxFunc.LOAD_STATIONARY.value,
            waitPrevAcc=False
        ),
        MatrixInstructionSpad(
            addr=spad_row0, stride=1,
            revInput=False, revOutput=False, delayOutput=False
        ),
        MatrixInstrucionAcc(addr=0, stride=0, zero=False)
    )


def make_attn_value(spad_tile, acc_tile, sem_id, rel_val, e_itemsize, a_itemsize):
    from fsa.instructions import (
        MatrixInstructionHeader, MatrixInstructionSpad, MatrixInstrucionAcc,
        MatrixInstruction, MxFunc
    )
    spad_row_bytes = spad_tile.shape[-1] * e_itemsize
    acc_row_bytes = acc_tile.shape[-1] * a_itemsize
    spad_row0 = spad_tile.data_ptr // spad_row_bytes
    acc_row0 = acc_tile.data_ptr // acc_row_bytes

    return MatrixInstruction(
        MatrixInstructionHeader(
            semId=sem_id,
            acquireValid=False, acquireSemValue=0,
            releaseValid=True, releaseSemValue=rel_val,
            func=MxFunc.ATTN_VALUE.value,
            waitPrevAcc=False
        ),
        MatrixInstructionSpad(
            addr=spad_row0, stride=1,
            revInput=True, revOutput=False, delayOutput=True
        ),
        MatrixInstrucionAcc(
            addr=acc_row0, stride=1, zero=True
        )
    )

def make_tensor_multiplication(spad_tile, acc_tile, sem_id, rel_val, e_itemsize, a_itemsize, accumulate=False, wait_prev_acc: bool=False):
    from fsa.instructions import (
        MatrixInstructionHeader, MatrixInstructionSpad, MatrixInstrucionAcc,
        MatrixInstruction, MxFunc
    )
    spad_row_bytes = spad_tile.shape[-1] * e_itemsize
    acc_row_bytes = acc_tile.shape[-1] * a_itemsize
    spad_row0 = spad_tile.data_ptr // spad_row_bytes
    acc_row0 = acc_tile.data_ptr // acc_row_bytes

    return MatrixInstruction(
        MatrixInstructionHeader(
            semId=sem_id,
            acquireValid=False, acquireSemValue=0,
            releaseValid=True, releaseSemValue=rel_val,
            func=MxFunc.TENSOR_MULTIPLICATION.value,
            waitPrevAcc=wait_prev_acc
        ),
        MatrixInstructionSpad(
            addr=spad_row0, stride=1,
            revInput=True, revOutput=False, delayOutput=True
        ),
        MatrixInstrucionAcc(
            addr=acc_row0, stride=1, zero=(not accumulate)
        )
    )

def _run_matmul_on_sa(stationary_data, stream_data, engine):
    import fsa as F
    from fsa.instructions import FenceInstruction
    from fsa.kernel import Kernel
    from fsa.config import get_config

    _reset_allocator_if_configured()

    cfg = get_config()
    assert stationary_data.shape == (cfg.sa_rows, cfg.sa_cols)
    assert stream_data.shape == (cfg.sa_rows, cfg.sa_cols)
    e_itemsize = cfg.e_type.itemsize
    a_itemsize = cfg.a_type.itemsize

    output_shape = (stream_data.shape[0], stationary_data.shape[0])
    
    # Allocate on-chip storage
    spad_stream  = F.alloc_spad(stream_data.shape)            
    spad_stationary = F.alloc_spad(stationary_data.shape)            
    acc_out = F.alloc_accumulator(output_shape)

    # Allocate host-visible memory
    stream_mem   = F.from_numpy(stream_data)
    stationary_mem   = F.from_numpy(stationary_data)
    out_mem = F.alloc_mem(output_shape, F.fp32)

    instructions = []

    # Step 1: DMA plain-load stream_data -> spad_stream
    instructions.append(make_dma_load(
        stream_mem, spad_stream, sem_id=0, rel_val=1,
        rows=stream_data.shape[0], cols=stream_data.shape[1], e_itemsize=e_itemsize
    ))

    # Step 2: DMA plain-load stationary_data -> spad_stationary
    instructions.append(make_dma_load(
        stationary_mem, spad_stationary, sem_id=2, rel_val=1,
        rows=stationary_data.shape[0], cols=stationary_data.shape[1], e_itemsize=e_itemsize
    ))

    # Step 3: LoadStationary spad_stationary 
    instructions.append(make_load_stationary(
        spad_stationary, sem_id=2, acq_val=1, rel_val=2, e_itemsize=e_itemsize
    ))

    # Step 4: TensorMultiplication with stream = spad_stream 
    instructions.append(make_tensor_multiplication(
        spad_stream, acc_out, sem_id=1, rel_val=1,
        e_itemsize=e_itemsize, a_itemsize=a_itemsize
    ))

    # Step 5: DMA store acc_out -> out_mem
    instructions.append(make_dma_store(
        acc_out, out_mem, sem_id=1, acq_val=1, rel_val=2,
        rows=output_shape[0], cols=output_shape[1], a_itemsize=a_itemsize
    ))

    instructions.append(FenceInstruction(mx=True, dma=True, stop=True))
    
    kernel = Kernel(instructions=instructions, input=[stream_mem, stationary_mem, out_mem], output=out_mem)
    result_tile = engine.execute(kernel)
    result = F.to_numpy(result_tile)

    return result

def _run_transpose_dma(stream_data, engine):
    import fsa as F
    from fsa.instructions import FenceInstruction
    from fsa.kernel import Kernel
    from fsa.config import get_config

    _reset_allocator_if_configured()

    cfg = get_config()

    original_dtype = stream_data.dtype
    e_dtype = np.dtype("float16")   
    stream_data = stream_data.astype(e_dtype)
    A = stream_data
    e_itemsize = cfg.e_type.itemsize
    a_itemsize = cfg.a_type.itemsize
    output_shape = (stream_data.shape[0], stream_data.shape[1])

    I = np.eye(stream_data.shape[0], dtype=np.float16)

    # Allocate on-chip
    spad_at = F.alloc_spad(stream_data.shape)    
    spad_id = F.alloc_spad(stream_data.shape)    
    acc_out = F.alloc_accumulator(output_shape)

    # Allocate memory
    A_mem = F.from_numpy(A)
    I_mem = F.from_numpy(I)
    out_mem = F.alloc_mem(output_shape, F.fp32)

    instructions = []

    # Step 1: DMA transpose load A -> spad as A^T
    instructions.append(make_dma_transpose_load(
        A_mem, spad_at, sem_id=0, rel_val=1,
        src_rows=stream_data.shape[0], src_cols=stream_data.shape[1], e_itemsize=e_itemsize
    ))

    # Step 2: DMA normal load identity -> spad
    instructions.append(make_dma_load(
        I_mem, spad_id, sem_id=2, rel_val=1,
        rows=stream_data.shape[0], cols=stream_data.shape[1], e_itemsize=e_itemsize
    ))

    # Step 3: LoadStationary A^T 
    instructions.append(make_load_stationary(
        spad_at, sem_id=2, acq_val=1, rel_val=2, e_itemsize=e_itemsize
    ))

    # Step 4: AttentionValue: stream I through SA. With stationary = A.T,
    # SA computes I @ anti_transpose(A.T) = A[::-1, ::-1] = rot180(A) into
    # acc_out.
    instructions.append(make_attn_value(
        spad_id, acc_out, sem_id=1, rel_val=1,
        e_itemsize=e_itemsize, a_itemsize=a_itemsize
    ))

    # Step 5: ST_SRAM -> output
    instructions.append(make_dma_store(
        acc_out, out_mem, sem_id=1, acq_val=1, rel_val=2,
        rows=output_shape[0], cols=output_shape[1], a_itemsize=a_itemsize
    ))

    instructions.append(FenceInstruction(mx=True, dma=True, stop=True))

    kernel = Kernel(instructions=instructions, input=[A_mem, I_mem, out_mem], output=out_mem)
    result_tile = engine.execute(kernel)
    result = F.to_numpy(result_tile)                
   
    return anti_transpose(result).astype(original_dtype)


def mx_matmul(A, B, engine, transpose_a=False, transpose_b=False, transpose_output=False):
    """ Compute A @ B on SA, you can choose a pre-transpose of A, B or the post-transpose of result"""
    if transpose_a:
        stream = _run_transpose_dma(A, engine)
    else:
        stream = A
    stationary = anti_transpose(_run_transpose_dma(B, engine) if transpose_b else B)
    result = _run_matmul_on_sa(stationary, stream, engine)
    return _run_transpose_dma(result, engine) if transpose_output else result

def mx_transpose(A, engine):
    """ Computes the transpose of given Tensor"""
    return _run_transpose_dma(A, engine)


def mx_identity(A, engine):
    """ Returns Tensor. Equivalent to A @ I = A"""
    return A


def mx_row_sum(A, engine):
    """ Computes row-wise sum of Tensor. Returns 1D Vector"""
    from fsa.config import get_config

    cfg = get_config()
    assert A.shape == (cfg.sa_rows, cfg.sa_cols)

    B = np.ones((cfg.sa_rows, cfg.sa_cols), dtype=A.dtype)
    result = mx_matmul(A, B, engine)
    return result[:, 0]

def mx_col_sum(A, engine):
    """ Computes column-wise sum of Tensor. Returns 1D Vector"""
    from fsa.config import get_config

    cfg = get_config()
    assert A.shape == (cfg.sa_rows, cfg.sa_cols)

    B = np.ones((cfg.sa_rows, cfg.sa_cols), dtype=A.dtype)
    result = mx_matmul(B, A, engine)
    return result[0, :]

def mx_sum(A, engine):
    "Computes total sum of Tensor. Returns Scalar"
    return mx_row_sum(A, engine).sum()


def mx_einsum(equation: str, *operands, engine):
    """ Dispatch a 2D contractive einsum pattern. Supported patterns are: 'ij,jk->ik', 'ji,jk->ik', 
'ij,kj->ik', 'ji,kj->ik', 'ij,jk->ki', 'ji,jk->ki', 'ij,kj->ki', 'ji,kj->ki', 
'ij->ji', 'ij->ij', 'ij->i', 'ij->j', 'ij->'"""
    A = operands[0]

    match equation:
        case "ij,jk->ik":
            B = operands[1]
            return mx_matmul(A, B, engine)
        case "ji,jk->ik":
            B = operands[1]
            return mx_matmul(A, B, engine, transpose_a=True)
        case "ij,kj->ik":
            B = operands[1]
            return mx_matmul(A, B, engine, transpose_b=True)
        case "ji,kj->ik":
            B = operands[1]
            return mx_matmul(A, B, engine, transpose_a=True, transpose_b=True)
        case "ij,jk->ki":
            B = operands[1]
            return mx_matmul(A, B, engine, transpose_output=True)
        case "ji,jk->ki":
            B = operands[1]
            return mx_matmul(A, B, engine, transpose_a=True, transpose_output=True)
        case "ij,kj->ki":
            B = operands[1]
            return mx_matmul(A, B, engine, transpose_b=True, transpose_output=True)
        case "ji,kj->ki":
            B = operands[1]
            return mx_matmul(A, B, engine, transpose_a=True, transpose_b=True, transpose_output=True)
        case "ij->ji":
            return mx_transpose(A, engine)
        case "ij->ij":
            return mx_identity(A, engine)
        case "ij->i":
            return mx_row_sum(A, engine)
        case "ij->j":
            return mx_col_sum(A, engine)
        case "ij->":
            return mx_sum(A, engine)
        case _: 
            raise ValueError(f"Unsupported einsum pattern: {equation!r}.")
        
