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

def rot180(M: np.ndarray) -> np.ndarray:
    return M[::-1, ::-1]

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

def make_load_stationary2(spad_tile, sem_id, acq_val, rel_val, e_itemsize):
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
            func=MxFunc.LOAD_STATIONARY2.value,
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

def make_elementwise_muls(spad_tile, acc_tile, sem_id, rel_val, e_itemsize, a_itemsize, accumulate=False, wait_prev_acc: bool=False):
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
            releaseValid=False, releaseSemValue=rel_val,
            func=MxFunc.ELEMENTWISE_MUL.value,
            waitPrevAcc=wait_prev_acc
        ),
        MatrixInstructionSpad(
            addr=spad_row0, stride=1,
            revInput=False, revOutput=False, delayOutput=False
        ),
        MatrixInstrucionAcc(
            addr=acc_row0, stride=1, zero=(not accumulate)
        )
    )

def _run_matmul_on_sa(stationary_data, stream_data, engine, transpose_stream = False):
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
    if transpose_stream == True:
        instructions.append(make_dma_transpose_load(
                stream_mem, spad_stream, sem_id=0, rel_val=1,
                src_rows=stream_data.shape[0], src_cols=stream_data.shape[1], e_itemsize=e_itemsize
            ))
    else:
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
    instructions.append(make_tensor_multiplication(
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


def mx_matmul(A, B, engine, transpose_a = False, transpose_b = False, transpose_output = False):
    """ Compute A @ B on SA, you can choose a pre-transpose of A, B or the post-transpose of result"""
    stream = A
    stationary = rot180(B) if transpose_b else anti_transpose(B)
    result = _run_matmul_on_sa(stationary, stream, engine, transpose_stream=transpose_a)
    return  _run_transpose_dma(result, engine) if transpose_output else result

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

def mx_hadamard(A, B, engine, stream_ones=False):
    import fsa as F
    from fsa.instructions import FenceInstruction
    from fsa.kernel import Kernel
    from fsa.config import get_config

    _reset_allocator_if_configured()

    cfg = get_config()
    assert A.shape == (cfg.sa_rows, cfg.sa_cols)
    assert B.shape == (cfg.sa_rows, cfg.sa_cols)
    e_itemsize = cfg.e_type.itemsize
    a_itemsize = cfg.a_type.itemsize
    e_dtype = np.dtype("float16")

    a_prime = anti_transpose(A.astype(e_dtype))
    b_prime = anti_transpose(B.astype(e_dtype))
    if stream_ones == True:
        identity = np.ones(A.shape, dtype=e_dtype)
    else:
        identity = np.eye(A.shape[0], dtype=e_dtype)

    output_shape = a_prime.shape

    # Host-visible memory
    a_mem = F.from_numpy(a_prime)
    b_mem = F.from_numpy(b_prime)
    i_mem = F.from_numpy(identity)
    out_mem = F.alloc_mem(output_shape, F.fp32)

    # On-chip storage
    a_spad = F.alloc_spad(a_prime.shape)
    b_spad = F.alloc_spad(b_prime.shape)
    i_spad = F.alloc_spad(identity.shape)
    acc_out = F.alloc_accumulator(output_shape)

    instructions = []

    instructions.append(make_dma_load(
        i_mem, i_spad, sem_id=0, rel_val=1,
        rows=identity.shape[0], cols=identity.shape[1], e_itemsize=e_itemsize
    ))

    instructions.append(make_dma_load(
        a_mem, a_spad, sem_id=1, rel_val=1,
        rows=A.shape[0], cols=A.shape[1], e_itemsize=e_itemsize
    ))

    instructions.append(make_dma_load(
        b_mem, b_spad, sem_id=2, rel_val=1,
        rows=B.shape[0], cols=B.shape[1], e_itemsize=e_itemsize
    ))

    instructions.append(make_load_stationary(
        a_spad, sem_id=1, acq_val=1, rel_val=2, e_itemsize=e_itemsize
    ))

    instructions.append(make_load_stationary2(
        b_spad, sem_id=2, acq_val=1, rel_val=2, e_itemsize=e_itemsize
    ))

    instructions.append(make_elementwise_muls(
        a_spad, acc_out, sem_id=5, rel_val=1, e_itemsize=e_itemsize, a_itemsize=a_itemsize
    ))

    instructions.append(make_tensor_multiplication(
        i_spad, acc_out, sem_id=4, rel_val=1,
        e_itemsize=e_itemsize, a_itemsize=a_itemsize
    ))

    instructions.append(make_dma_store(
        acc_out, out_mem, sem_id=4, acq_val=1, rel_val=2,
        rows=output_shape[0], cols=output_shape[1], a_itemsize=a_itemsize
    ))

    instructions.append(FenceInstruction(mx=True, dma=True, stop=True))

    kernel = Kernel(instructions=instructions, input=[a_mem, b_mem, i_mem, out_mem], output=out_mem)
    result_tile = engine.execute(kernel)
    result = F.to_numpy(result_tile)

    return result

def mx_frobenius(A, B, engine):
    result = mx_hadamard(A, B, engine=engine, stream_ones=True)
    return result[0, :].sum()

def normalise(equation: str):
    if "..." in equation:
        raise ValueError(f"ellipsis ('...') is not supported: {equation!r}")
    
    no_whitespace = "".join(equation.split())
    if "->" in no_whitespace:
        inputs, outputs = no_whitespace.split("->")
        inputs = inputs.split(",")
    else:
        inputs = no_whitespace.split(",")
        all_letters = "".join(inputs)
        once = []
        for letter in all_letters:
            if all_letters.count(letter) == 1:
                once.append(letter)
        outputs = "".join(sorted(once))
    
    if len(inputs) > 2:
        raise ValueError(f"at most two operands, got {len(inputs)}: {equation!r}")
    
    letters = set()

    for op in inputs:
        tensors = set()
        if len(op) > 2:
            raise ValueError(f"operand rank > 2: {op!r}")
        if op != "" and not op.isalpha():
            raise ValueError(f"non-letter index in operand {op!r}")
        for letter in op:
            if letter not in letters:
                letters.add(letter)
            if letter not in tensors:
                tensors.add(letter)
            else:
                raise ValueError(f"letter repeated within one operand: {op!r}")
    if len(outputs) > 2:
        raise ValueError(f"output rank > 2: {outputs!r}")
    
    if len(outputs) > 0 and not outputs.isalpha():
        raise ValueError(f"non-letter index in output {outputs!r}")
    if len(set(outputs)) != len(outputs):
        raise ValueError(f"letter repeated in output: {outputs!r}")

    for letter in outputs:
        if letter not in letters:
            raise ValueError(f"output letter {letter!r} not in any input")
    
    descriptors = {}
    replacements = "ijkl"
    replacements_idx = 0

    for op in inputs:
        for letter in op:
            if letter not in descriptors:
                if replacements_idx < len(replacements):
                    descriptors[letter] = replacements[replacements_idx]
                    replacements_idx += 1
                else:
                    raise ValueError("Exceeded the available 'ijkl' replacement letters.")

    def translate(text):
        return "".join(descriptors.get(char, char) for char in text)

    translated_operands = [translate(op) for op in inputs]
    translated_output = translate(outputs)
    
    return ",".join(translated_operands) + "->" + translated_output

def roles_helper(individual_inputs, output):

    a = individual_inputs[0]
    if len(individual_inputs) == 2:
        b = individual_inputs[1]
    else:
        b = ""

    set_a = set(a)

    set_b = set(b)

    all_letters = set_a | set_b

    dict_roles = {}

    for letter in all_letters:
        in_a = letter in set_a
        in_b = letter in set_b
        in_out = letter in output

        if in_a and in_b:
            if in_out:
                dict_roles[letter] = "E"
            else: 
                dict_roles[letter] = "C"

        else:
            if in_out:
                if in_a:
                    dict_roles[letter] = "Fa"
                else:
                    dict_roles[letter] = "Fb"
            else:
                if in_a:
                    dict_roles[letter] = "Ra"
                else:
                    dict_roles[letter] = "Rb"

    return dict_roles

def placement_helper(operand, letters, row, col, n):
    tile = np.zeros((n, n), dtype=operand.dtype)
    transpose = False
    
    if len(letters) == 2:
        rows, cols = operand.shape
        tile[:rows, :cols] = operand
        transpose =  letters != row + col
    elif len(letters) == 1:
        if row == col:
            rows = operand.shape[0]
            for k in range(rows):
                tile[k, k] = operand[k]
        elif letters == row:
            rows = operand.shape[0]
            tile[:rows, 0] = operand
        else:
            cols = operand.shape[0]
            tile[0, :cols] = operand
    elif len(letters) == 0:
        tile[0, 0] = operand
    

    return tile, transpose

def reduction_helper(operand, letters, roles, sizes, n, engine, cast=True):
    kept = ""
    r_letters = ""
    for letter in letters:
        if not roles[letter].startswith("R"):
            kept = kept + letter
        else:
            r_letters = r_letters + letter
    if r_letters == "":
        return operand, letters
    else:
        if len(letters) == 2:
            tile, _ = placement_helper(operand, letters, letters[0], letters[1], n)
            if len(r_letters) == 2:
                result = mx_sum(tile, engine)
            elif r_letters == letters[1]:
                result = mx_row_sum(tile, engine)
            else:
                result = mx_col_sum(tile, engine)
        elif len(letters) == 1:
            tile, _ = placement_helper(operand, letters, None, letters, n)
            result = mx_row_sum(tile, engine)[0]

    dtype = operand.dtype if cast else None
    if len(kept) == 1:
        result = result[:sizes[kept]]
        if cast:
            result = result.astype(dtype)
    if len(kept) == 0:
        result = np.asarray(result, dtype=dtype)

    return result, kept


def mx_einsum(equation: str, *operands, engine):
    """Compute a 2D einsum on the FSA systolic array.

    Supported: every einsum with one or two operands of rank 0, 1 or 2, no
    index repeated within an operand, and an output of rank <= 2 drawn from
    the inputs -- 137 equations up to renaming of letters. Scalars are
    rank-0 operands (an empty subscript, e.g. ',ij->ij').

    Dispatch is by index role, not by pattern:
      1. Each letter gets a role: C (contracted), E (elementwise),
         Fa/Fb (free), Ra/Rb (reduced within one operand).
      2. R letters are summed out first, one chip pass per operand that
         has any (row/column/total sum).
      3. What remains is one core step: a matmul (plain, mat-vec, dot,
         outer, broadcast by diag(v), scaling by s*I, diagonal or trace
         read), a Hadamard (plus a transpose pass if the layouts differ),
         or, for one operand, a transpose or nothing.
    Operand transposes are done on chip (transpose flags); output order is
    handled by swapping the operands, so neither costs an extra pass.

    Any consistent relabelling of the index letters is accepted
    ('ab,bc->ac' is 'ij,jk->ik'), as is numpy's implicit-output form
    ('ij,jk' means 'ij,jk->ik'). Operands may be any shape whose every
    dimension is at most the SA size n: they are placed in n x n tiles and
    the result is cropped to the shape np.einsum would return.

    `engine` is keyword-only. Raises ValueError for anything outside the
    class: more than two operands, rank > 2, repeated indices, ellipsis,
    or inconsistent or too-large dimensions.

    Docstring Co-Authored by Claude Opus 5.5
    """
    operands = [np.asarray(x) for x in operands]
    canonical = normalise(equation)
    
    inputs, outputs = canonical.split("->")
    individual_inputs = inputs.split(",")
        
    if len(individual_inputs) != len(operands):
        raise ValueError(f"{equation!r} expects {len(individual_inputs)} operands, got {len(operands)}")
    
    roles = roles_helper(individual_inputs, outputs)
    reduced_inputs = []

   
    for op in individual_inputs:
        kept = ""
        for letter in op:
            if not roles[letter].startswith("R"):
                kept = kept + letter
        reduced_inputs.append(kept)

        

    reduced_roles = roles_helper(reduced_inputs, outputs)


    values = list(reduced_roles.values())
    n_C = values.count("C")
    n_E = values.count("E")
    n_Fa = values.count("Fa")
    n_Fb = values.count("Fb")

    rule = None
    if len(individual_inputs) == 2:
        if n_E == 2:
            rule = "hadamard"
        elif n_E == 1 and n_C == 1:
            rule = "diag"
        elif n_E == 1 and n_Fa + n_Fb == 1:
            rule = "broadcast"
        elif n_E == 1:
            rule = "vec_hadamard"
        elif n_C == 2:
            rule = "trace"
        elif n_Fa <= 1 and n_Fb <= 1:
            rule = "matmul"
        elif n_Fa == 2 or n_Fb == 2:
            rule = "sI"

    else:
        rule = "unary"
        
    # Unreachable for any equation normalise accepts; kept as a guard.
    if rule is None:
        raise ValueError(f"unsupported einsum pattern {equation!r} (canonical {canonical!r})")

    for subscript, operand in zip(individual_inputs, operands):
        if np.ndim(operand) != len(subscript):
            raise ValueError(f"operand {subscript!r} (canonical) expects ndim {len(subscript)}, got {np.ndim(operand)}")

    from fsa.config import get_config
    cfg = get_config()
    n = cfg.sa_rows

    sizes = {}
    
    for subscript, operand in zip(individual_inputs, operands):
        for letter, size in zip(subscript, operand.shape):
            if size > n:
                raise ValueError(f"dimension of size {size} exceeds SA size {n}")
            elif size < 1:
                raise ValueError(f"cannot have empty operand")
            elif letter in sizes and sizes[letter] != size:
                raise ValueError(
                    f"{equation!r}: index {letter!r} (canonical) has size "
                    f"{sizes[letter]} in one operand and {size} in another")
            else:
                sizes[letter] = size

    new_operands = []
    new_inputs = []
    for operand, letters in zip(operands, individual_inputs):
        reduced, kept = reduction_helper(operand, letters, roles, sizes, n, engine,
                                         cast=len(individual_inputs) == 2)
        new_operands.append(reduced)
        new_inputs.append(kept)

    operands = new_operands
    individual_inputs = new_inputs
    roles = reduced_roles

    if rule in ("matmul", "diag", "trace", "broadcast", "vec_hadamard", "sI"):
        if rule == "matmul":
            c, fa, fb = None, None, None

            for letter, role in roles.items():
                if role == "C":
                    c = letter
                elif role == "Fa":
                    fa = letter
                elif role == "Fb":
                    fb = letter

            swap = fa != None and fb != None and outputs == fb + fa

            if swap:
                first_tile, first_t = placement_helper(operands[1], individual_inputs[1], fb, c, n)
                second_tile, second_t = placement_helper(operands[0], individual_inputs[0], c, fa, n)
                row_letter = fb
            else:
                first_tile, first_t = placement_helper(operands[0], individual_inputs[0], fa, c, n)
                second_tile, second_t = placement_helper(operands[1], individual_inputs[1], c, fb, n)
                row_letter = fa

        elif rule == "diag":
            c, e = None, None
            for letter, role in roles.items():
                if role == "C":
                    c = letter
                elif role == "E":
                    e = letter
         
            first_tile, first_t = placement_helper(operands[1], individual_inputs[1], e, c, n)
            second_tile, second_t = placement_helper(operands[0], individual_inputs[0], c, e, n)
            

        elif rule == "trace":
    
            c1, c2 = individual_inputs[0][0], individual_inputs[0][1]
            first_tile, first_t = placement_helper(operands[1], individual_inputs[1], c1, c2, n)
            second_tile, second_t = placement_helper(operands[0], individual_inputs[0], c2, c1, n)

        elif rule == "broadcast":
            e, f, m = None, None, None

            for letter, role in roles.items():
                if role == "E":
                    e = letter
                elif role == "Fa":
                    f = letter
                    m = 0
                elif role == "Fb":
                    f = letter
                    m = 1

            if outputs == e + f:
                first_tile, first_t = placement_helper(operands[1 - m], individual_inputs[1 - m], e, e, n)
                second_tile, second_t = placement_helper(operands[m], individual_inputs[m], e, f, n)
            else:
                first_tile, first_t = placement_helper(operands[m], individual_inputs[m], f, e, n)
                second_tile, second_t = placement_helper(operands[1 - m], individual_inputs[1 - m], e, e, n)

        elif rule == "vec_hadamard":
            e = None
                       
            for letter, role in roles.items():
                if role == "E":
                    e = letter
            
            first_tile, first_t = placement_helper(operands[0], individual_inputs[0], e, e, n)
            second_tile, second_t = placement_helper(operands[1], individual_inputs[1], e, None, n)
            row_letter = e

        elif rule == "sI":
            fa, fb, m = None, None, None
            
            for letter, role in roles.items():
                if role == "Fa":
                    fa = letter
                    m = 0
                elif role == "Fb":
                    fb = letter
                    m = 1

            first_tile, first_t = placement_helper(operands[m], individual_inputs[m], outputs[0], outputs[1], n)

            second_tile, second_t = np.eye(n, dtype=operands[1-m].dtype) * operands[1-m], False

        result = mx_matmul(first_tile, second_tile, engine, transpose_a=first_t, transpose_b=second_t)

        if rule == "diag":
            dispatch = np.diagonal(result)
        elif rule == "trace":
            dispatch = np.trace(result)
        elif len(outputs) == 2:
            dispatch = result
        elif len(outputs) == 1:
            if outputs == row_letter:
                dispatch = result[:, 0]
            else:
                dispatch = result[0, :]
        else:
            dispatch = result[0, 0]

    elif rule == "hadamard":
        a, b = individual_inputs[0], individual_inputs[1]

        if a == b:
            target = a
        else:
            target = outputs

        A_tile, A_t = placement_helper(operands[0], a, target[0], target[1], n)
        B_tile, B_t = placement_helper(operands[1], b, target[0], target[1], n)    

        if A_t:
            A_tile = mx_transpose(A_tile, engine)   
        if B_t:
            B_tile = mx_transpose(B_tile, engine) 

        result = mx_hadamard(A_tile, B_tile, engine)

        if outputs != target:
            result = mx_transpose(result, engine)

        dispatch = result
    elif rule == "unary":
        reduced_operand = operands[0]
        letters = individual_inputs[0]

        if letters != outputs:
            tile, _ = placement_helper(reduced_operand, letters, letters[0], letters[1], n)
            dispatch = mx_transpose(tile, engine)
        else:
            dispatch  = reduced_operand

    out_shape = []

    for letter in outputs:
        out_shape.append(sizes[letter])

    out_shape = tuple(out_shape)

    if len(out_shape) == 2:
        dispatch = dispatch[:out_shape[0], :out_shape[1]]
    elif len(out_shape) == 1:
        dispatch = dispatch[:out_shape[0]]

    return dispatch