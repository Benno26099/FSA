import os
import sys
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Optional

## Test result tracking
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


def make_mem_stride_fields(stride_bytes: int):
    from fsa.instructions import InstructionField
    inst = InstructionField(stride_bytes, 20, 0, signed=True)
    stride_1 = (inst.value >> 15) & ((1 << 6) - 1)
    stride_2 = inst.value & ((1 << 15) - 1)
    return stride_1, stride_2


def make_dma_load(mem_tile, spad_tile, sem_id, rel_val, rows, cols, e_itemsize):
    """Build a normal LD_SRAM instruction."""
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
    """
    Reads src_rows x src_cols row-major matrix from memory.
    Writes it transposed into the scratchpad as src_cols x src_rows.

    Each AXI burst fetches one element (size=e_itemsize). The memory stride
    jumps by one source row (src_cols * e_itemsize) to walk down a column.
    After collecting src_rows elements, which represents one full transposed row, 
    the LoadQueue flushes the row buffer to the next spad row.

    Total AXI bursts = src_rows * src_cols.
    """
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
            size=src_rows   # one element per burst
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
    """Build a LOAD_STATIONARY MX instruction."""
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


# Test: DMA transpose of a square matrix

def test_transpose_square(engine, sa_rows, sa_cols, label="square"):
    """
    Pipeline:
      1. DMA transpose load: A (NxN row-major) -> spad as A^T (NxN)
      2. DMA normal load: identity I (NxN) -> spad
      3. LoadStationary: load A^T into SA registers
      4. AttentionValue: stream I through SA -> acc = A^T @ I = A^T
      5. ST_SRAM: store acc to memory (fp32)
      6. Verify output == A^T
    """
    import fsa as F
    from fsa.instructions import FenceInstruction
    from fsa.kernel import Kernel
    from fsa.config import get_config

    cfg = get_config()
    N = sa_rows
    assert sa_rows == sa_cols, f"Square test requires sa_rows == sa_cols, got {sa_rows}x{sa_cols}"
    e_itemsize = cfg.e_type.itemsize
    a_itemsize = cfg.a_type.itemsize

    # Input: NxN matrix with recognizable values
    A = np.arange(N * N, dtype=np.float16).reshape(N, N)
    I = np.eye(N, dtype=np.float16)
    # spad will hold A.T after transpose load. The SA readback produces
    # anti_transpose(spad), so expected = anti_transpose(A.T).
    expected = anti_transpose(A.T).astype(np.float32)

    # Allocate on-chip
    spad_at = F.alloc_spad((N, N))    
    spad_id = F.alloc_spad((N, N))    
    acc_out = F.alloc_accumulator((N, N))

    # Allocate memory
    A_mem = F.from_numpy(A)
    I_mem = F.from_numpy(I)
    out_mem = F.alloc_mem((N, N), F.fp32)

    instructions = []

    # Step 1: DMA transpose load A -> spad as A^T
    instructions.append(make_dma_transpose_load(
        A_mem, spad_at, sem_id=0, rel_val=1,
        src_rows=N, src_cols=N, e_itemsize=e_itemsize
    ))

    # Step 2: DMA normal load identity -> spad
    instructions.append(make_dma_load(
        I_mem, spad_id, sem_id=2, rel_val=1,
        rows=N, cols=N, e_itemsize=e_itemsize
    ))

    # Step 3: LoadStationary A^T (waits for identity load = last DMA = sem 2)
    instructions.append(make_load_stationary(
        spad_at, sem_id=2, acq_val=1, rel_val=2, e_itemsize=e_itemsize
    ))

    # Step 4: AttentionValue: stream I through SA -> acc = A^T
    instructions.append(make_attn_value(
        spad_id, acc_out, sem_id=1, rel_val=1,
        e_itemsize=e_itemsize, a_itemsize=a_itemsize
    ))

    # Step 5: ST_SRAM -> output
    instructions.append(make_dma_store(
        acc_out, out_mem, sem_id=1, acq_val=1, rel_val=2,
        rows=N, cols=N, a_itemsize=a_itemsize
    ))

    instructions.append(FenceInstruction(mx=True, dma=True, stop=True))

    kernel = Kernel(instructions=instructions, input=[A_mem, I_mem, out_mem], output=out_mem)
    result_tile = engine.execute(kernel)
    result = F.to_numpy(result_tile)

    passed = np.allclose(result, expected, rtol=1e-2, atol=1e-2)
    error_msg = ""
    if not passed:
        max_err = np.max(np.abs(result - expected))
        error_msg = f"Max error: {max_err}. Got:\n{result}\nExpected:\n{expected}"

    return TestResult(
        name=f"transpose_{label}_{N}x{N}",
        passed=passed, expected=expected, actual=result,
        error_msg=error_msg
    )

# Test: DMA transpose load-only 

def test_transpose_load_readback(engine, sa_rows, sa_cols, values="sequential"):
    """Verify transpose load puts correct values in spad."""
    import fsa as F
    from fsa.instructions import FenceInstruction
    from fsa.kernel import Kernel
    from fsa.config import get_config

    cfg = get_config()
    N = sa_rows
    assert sa_rows == sa_cols
    e_itemsize = cfg.e_type.itemsize
    a_itemsize = cfg.a_type.itemsize

    if values == "sequential":
        A = np.arange(N * N, dtype=np.float16).reshape(N, N)
    elif values == "col_marker":
        A = np.zeros((N, N), dtype=np.float16)
        for i in range(N):
            for j in range(N):
                A[i][j] = j * 10 + i
    else:
        A = np.random.rand(N, N).astype(np.float16)

    I = np.eye(N, dtype=np.float16)
    # spad holds A.T after transpose-load; SA readback gives anti_transpose(spad).
    expected = anti_transpose(A.T).astype(np.float32)

    spad_data = F.alloc_spad((N, N))
    spad_id = F.alloc_spad((N, N))
    acc_out = F.alloc_accumulator((N, N))

    A_mem = F.from_numpy(A)
    I_mem = F.from_numpy(I)
    out_mem = F.alloc_mem((N, N), F.fp32)

    instructions = []

    # Transpose load
    instructions.append(make_dma_transpose_load(
        A_mem, spad_data, sem_id=0, rel_val=1,
        src_rows=N, src_cols=N, e_itemsize=e_itemsize
    ))

    # Normal load identity
    instructions.append(make_dma_load(
        I_mem, spad_id, sem_id=2, rel_val=1,
        rows=N, cols=N, e_itemsize=e_itemsize
    ))

    # LoadStationary the transposed data
    instructions.append(make_load_stationary(
        spad_data, sem_id=2, acq_val=1, rel_val=2, e_itemsize=e_itemsize
    ))

    # Multiply by identity to read back
    instructions.append(make_attn_value(
        spad_id, acc_out, sem_id=1, rel_val=1,
        e_itemsize=e_itemsize, a_itemsize=a_itemsize
    ))

    # Store result
    instructions.append(make_dma_store(
        acc_out, out_mem, sem_id=1, acq_val=1, rel_val=2,
        rows=N, cols=N, a_itemsize=a_itemsize
    ))

    instructions.append(FenceInstruction(mx=True, dma=True, stop=True))

    kernel = Kernel(instructions=instructions, input=[A_mem, I_mem, out_mem], output=out_mem)
    result_tile = engine.execute(kernel)
    result = F.to_numpy(result_tile)

    passed = np.allclose(result, expected, rtol=1e-2, atol=1e-2)
    error_msg = ""
    if not passed:
        max_err = np.max(np.abs(result - expected))
        error_msg = (f"Max error: {max_err}\nInput A:\n{A}\n"
                     f"Got:\n{result}\nExpected A^T:\n{expected}")

    return TestResult(
        name=f"transpose_readback_{values}_{N}x{N}",
        passed=passed, expected=expected, actual=result,
        error_msg=error_msg
    )



# Test: Normal (non-transpose) DMA still works after adding transpose logic

def test_normal_load_unchanged(engine, sa_rows, sa_cols):
    """Regression test: verify normal LD_SRAM path is not broken.

    Loads A directly withpout transpose, then reads back via SA + identity.
    Result should equal A.
    """
    import fsa as F
    from fsa.instructions import FenceInstruction
    from fsa.kernel import Kernel
    from fsa.config import get_config

    cfg = get_config()
    N = sa_rows
    assert sa_rows == sa_cols
    e_itemsize = cfg.e_type.itemsize
    a_itemsize = cfg.a_type.itemsize

    A = np.arange(N * N, dtype=np.float16).reshape(N, N)
    I = np.eye(N, dtype=np.float16)
    # Normal load: spad holds A. SA readback gives anti_transpose(A).
    expected = anti_transpose(A).astype(np.float32)

    spad_a = F.alloc_spad((N, N))
    spad_id = F.alloc_spad((N, N))
    acc_out = F.alloc_accumulator((N, N))

    A_mem = F.from_numpy(A)
    I_mem = F.from_numpy(I)
    out_mem = F.alloc_mem((N, N), F.fp32)

    instructions = []

    # Normal load A (NOT transpose)
    instructions.append(make_dma_load(
        A_mem, spad_a, sem_id=0, rel_val=1,
        rows=N, cols=N, e_itemsize=e_itemsize
    ))

    # Normal load identity
    instructions.append(make_dma_load(
        I_mem, spad_id, sem_id=2, rel_val=1,
        rows=N, cols=N, e_itemsize=e_itemsize
    ))

    # LoadStationary A
    instructions.append(make_load_stationary(
        spad_a, sem_id=2, acq_val=1, rel_val=2, e_itemsize=e_itemsize
    ))

    # AttentionValue: A @ I = A
    instructions.append(make_attn_value(
        spad_id, acc_out, sem_id=1, rel_val=1,
        e_itemsize=e_itemsize, a_itemsize=a_itemsize
    ))

    # Store
    instructions.append(make_dma_store(
        acc_out, out_mem, sem_id=1, acq_val=1, rel_val=2,
        rows=N, cols=N, a_itemsize=a_itemsize
    ))

    instructions.append(FenceInstruction(mx=True, dma=True, stop=True))

    kernel = Kernel(instructions=instructions, input=[A_mem, I_mem, out_mem], output=out_mem)
    result_tile = engine.execute(kernel)
    result = F.to_numpy(result_tile)

    passed = np.allclose(result, expected, rtol=1e-2, atol=1e-2)
    error_msg = ""
    if not passed:
        max_err = np.max(np.abs(result - expected))
        error_msg = f"Max error: {max_err}. Got:\n{result}\nExpected:\n{expected}"

    return TestResult(
        name=f"normal_load_regression_{N}x{N}",
        passed=passed, expected=expected, actual=result,
        error_msg=error_msg
    )

# Test: Transpose of a non-square matrix (tensor)

def test_transpose_nonsquare_padded(engine, sa_rows, sa_cols, src_rows, src_cols):
    """Test transposing a non-square matrix by padding to SA dimensions.
    The matrix is zero-padded to (sa_rows x sa_cols) before loading.
    After transpose, the spad should contain the padded A^T = (sa_cols x sa_rows).
    We verify the top-left (src_cols x src_rows) block matches A^T.
    """
    import fsa as F
    from fsa.instructions import FenceInstruction
    from fsa.kernel import Kernel
    from fsa.config import get_config

    cfg = get_config()
    N_r = sa_rows
    N_c = sa_cols
    assert src_rows <= N_r and src_cols <= N_c, \
        f"Source ({src_rows}x{src_cols}) must fit in SA ({N_r}x{N_c})"
    e_itemsize = cfg.e_type.itemsize
    a_itemsize = cfg.a_type.itemsize

    # Create non-square matrix and pad
    A_small = np.arange(1, src_rows * src_cols + 1, dtype=np.float16).reshape(src_rows, src_cols)
    A_padded = np.zeros((N_r, N_c), dtype=np.float16)
    A_padded[:src_rows, :src_cols] = A_small

    I = np.eye(N_r, N_c, dtype=np.float16)

    # Expected: spad receives A_padded.T after transpose-load; SA readback
    # produces anti_transpose(A_padded.T). With A_small at top-left of
    # A_padded, A_small ends up at the BOTTOM-RIGHT of the readback output.
    expected_full = anti_transpose(A_padded.T).astype(np.float32)

    spad_data = F.alloc_spad((N_c, N_r))  # transposed dimensions
    spad_id = F.alloc_spad((N_r, N_c))
    acc_out = F.alloc_accumulator((N_c, N_r))

    A_mem = F.from_numpy(A_padded)
    I_mem = F.from_numpy(I)
    out_mem = F.alloc_mem((N_c, N_r), F.fp32)

    instructions = []

    # Transpose load padded A
    instructions.append(make_dma_transpose_load(
        A_mem, spad_data, sem_id=0, rel_val=1,
        src_rows=N_r, src_cols=N_c, e_itemsize=e_itemsize
    ))

    # Normal load identity
    instructions.append(make_dma_load(
        I_mem, spad_id, sem_id=2, rel_val=1,
        rows=N_r, cols=N_c, e_itemsize=e_itemsize
    ))

    # LoadStationary transposed data
    instructions.append(make_load_stationary(
        spad_data, sem_id=2, acq_val=1, rel_val=2, e_itemsize=e_itemsize
    ))

    # Multiply by identity to readback
    instructions.append(make_attn_value(
        spad_id, acc_out, sem_id=1, rel_val=1,
        e_itemsize=e_itemsize, a_itemsize=a_itemsize
    ))

    # Store
    instructions.append(make_dma_store(
        acc_out, out_mem, sem_id=1, acq_val=1, rel_val=2,
        rows=N_c, cols=N_r, a_itemsize=a_itemsize
    ))

    instructions.append(FenceInstruction(mx=True, dma=True, stop=True))

    kernel = Kernel(instructions=instructions, input=[A_mem, I_mem, out_mem], output=out_mem)
    result_tile = engine.execute(kernel)
    result = F.to_numpy(result_tile)

    result_block = result[N_r - src_rows:, N_c - src_cols:]
    expected_block = A_small[::-1, ::-1].astype(np.float32)  # rot180

    passed_block = np.allclose(result_block, expected_block, rtol=1e-2, atol=1e-2)
    # Padded region (top rows / left cols of output) should be zero.
    passed_pad = True
    if src_rows < N_r:
        passed_pad = passed_pad and np.allclose(result[:N_r - src_rows, :], 0, atol=1e-2)
    if src_cols < N_c:
        passed_pad = passed_pad and np.allclose(result[:, :N_c - src_cols], 0, atol=1e-2)

    passed = passed_block and passed_pad
    error_msg = ""
    if not passed_block:
        max_err = np.max(np.abs(result_block - expected_block))
        error_msg = (f"Data block mismatch. Max error: {max_err}\n"
                     f"Input A ({src_rows}x{src_cols}):\n{A_small}\n"
                     f"Got block:\n{result_block}\nExpected rot180(A_small):\n{expected_block}")
    if not passed_pad:
        error_msg += "\nPadded region is not zero"

    return TestResult(
        name=f"transpose_nonsquare_{src_rows}x{src_cols}_padded_to_{N_r}x{N_c}",
        passed=passed, expected=expected_block, actual=result_block,
        error_msg=error_msg
    )

# Test: Double transpose (A^T^T == A)

def test_double_transpose(engine, sa_rows, sa_cols):
    """
    Pipeline:
      1. Transpose load A -> spad1 as A^T
      2. Store A^T to memory (via SA + identity readback)
      3. Transpose load A^T (from memory) -> spad2 as A^T^T = A
      4. Store A^T^T to memory (via SA + identity readback)
      5. Verify A^T^T == A
    """
    import fsa as F
    from fsa.instructions import FenceInstruction
    from fsa.kernel import Kernel
    from fsa.config import get_config

    cfg = get_config()
    N = sa_rows
    assert sa_rows == sa_cols
    e_itemsize = cfg.e_type.itemsize
    a_itemsize = cfg.a_type.itemsize

    A = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ], dtype=np.float16)[:N, :N]

    expected = anti_transpose(A).astype(np.float32)

    spad_data = F.alloc_spad((N, N))
    spad_id = F.alloc_spad((N, N))
    acc_out = F.alloc_accumulator((N, N))

    A_mem = F.from_numpy(A)
    I = np.eye(N, dtype=np.float16)
    I_mem = F.from_numpy(I)
    at_mem = F.alloc_mem((N, N), F.fp16)    # intermediate A^T in fp16
    out_mem = F.alloc_mem((N, N), F.fp32)   # final output

    instructions = []

    A_T = A.T.copy()
    AT_mem = F.from_numpy(A_T)

    # Pass 1: transpose load A -> readback -> verify = A^T
    instructions.append(make_dma_transpose_load(
        A_mem, spad_data, sem_id=0, rel_val=1,
        src_rows=N, src_cols=N, e_itemsize=e_itemsize
    ))
    instructions.append(make_dma_load(
        I_mem, spad_id, sem_id=2, rel_val=1,
        rows=N, cols=N, e_itemsize=e_itemsize
    ))
    instructions.append(make_load_stationary(
        spad_data, sem_id=2, acq_val=1, rel_val=2, e_itemsize=e_itemsize
    ))
    instructions.append(make_attn_value(
        spad_id, acc_out, sem_id=1, rel_val=1,
        e_itemsize=e_itemsize, a_itemsize=a_itemsize
    ))

    # Fence between passes — drain MX and DMA
    instructions.append(FenceInstruction(mx=True, dma=True, stop=False))

    # Pass 2: transpose load A^T -> readback -> should = A
    instructions.append(make_dma_transpose_load(
        AT_mem, spad_data, sem_id=3, rel_val=1,
        src_rows=N, src_cols=N, e_itemsize=e_itemsize
    ))
    instructions.append(make_dma_load(
        I_mem, spad_id, sem_id=4, rel_val=1,
        rows=N, cols=N, e_itemsize=e_itemsize
    ))
    instructions.append(make_load_stationary(
        spad_data, sem_id=4, acq_val=1, rel_val=2, e_itemsize=e_itemsize
    ))
    instructions.append(make_attn_value(
        spad_id, acc_out, sem_id=5, rel_val=1,
        e_itemsize=e_itemsize, a_itemsize=a_itemsize
    ))

    # Store final result
    instructions.append(make_dma_store(
        acc_out, out_mem, sem_id=5, acq_val=1, rel_val=2,
        rows=N, cols=N, a_itemsize=a_itemsize
    ))

    instructions.append(FenceInstruction(mx=True, dma=True, stop=True))

    kernel = Kernel(
        instructions=instructions,
        input=[A_mem, AT_mem, I_mem, out_mem],
        output=out_mem
    )
    result_tile = engine.execute(kernel)
    result = F.to_numpy(result_tile)

    passed = np.allclose(result, expected, rtol=1e-2, atol=1e-2)
    error_msg = ""
    if not passed:
        max_err = np.max(np.abs(result - expected))
        error_msg = (f"Double transpose failed. Max error: {max_err}\n"
                     f"Input A:\n{A}\nGot:\n{result}\nExpected A:\n{expected}")

    return TestResult(
        name=f"double_transpose_{N}x{N}",
        passed=passed, expected=expected, actual=result,
        error_msg=error_msg
    )

# Test: Transpose + matmul (A^T @ B)

def test_transpose_matmul(engine, sa_rows, sa_cols):
    """Test computing A^T @ B using DMA transpose + SA matmul.

    Pipeline:
      1. Transpose load A -> spad as A^T
      2. LoadStationary A^T
      3. Normal load B -> spad
      4. AttentionValue: stream B through SA -> acc = A^T @ B
      5. Store result
      6. Verify against numpy A.T @ B
    """
    import fsa as F
    from fsa.instructions import FenceInstruction
    from fsa.kernel import Kernel
    from fsa.config import get_config

    cfg = get_config()
    N = sa_rows
    assert sa_rows == sa_cols
    e_itemsize = cfg.e_type.itemsize
    a_itemsize = cfg.a_type.itemsize

    np.random.seed(123)
    A = np.random.rand(N, N).astype(np.float16)
    B = np.random.rand(N, N).astype(np.float16)
    
    expected = (A.T.astype(np.float32) @ B.astype(np.float32))

    spad_at = F.alloc_spad((N, N))   
    spad_b = F.alloc_spad((N, N))    
    acc_out = F.alloc_accumulator((N, N))

    A_mem = F.from_numpy(A)
    B_mem = F.from_numpy(B)
    out_mem = F.alloc_mem((N, N), F.fp32)

    instructions = []

    # Transpose load A -> A^T in spad
    instructions.append(make_dma_transpose_load(
        A_mem, spad_at, sem_id=0, rel_val=1,
        src_rows=N, src_cols=N, e_itemsize=e_itemsize
    ))

    # Normal load B
    instructions.append(make_dma_load(
        B_mem, spad_b, sem_id=2, rel_val=1,
        rows=N, cols=N, e_itemsize=e_itemsize
    ))

    # LoadStationary A^T
    instructions.append(make_load_stationary(
        spad_at, sem_id=2, acq_val=1, rel_val=2, e_itemsize=e_itemsize
    ))

    # AttentionValue: A^T @ B
    instructions.append(make_attn_value(
        spad_b, acc_out, sem_id=1, rel_val=1,
        e_itemsize=e_itemsize, a_itemsize=a_itemsize
    ))

    # Store
    instructions.append(make_dma_store(
        acc_out, out_mem, sem_id=1, acq_val=1, rel_val=2,
        rows=N, cols=N, a_itemsize=a_itemsize
    ))

    instructions.append(FenceInstruction(mx=True, dma=True, stop=True))

    kernel = Kernel(instructions=instructions, input=[A_mem, B_mem, out_mem], output=out_mem)
    result_tile = engine.execute(kernel)
    result = F.to_numpy(result_tile)

    passed = np.allclose(result, expected, rtol=5e-2, atol=5e-2)
    error_msg = ""
    if not passed:
        max_err = np.max(np.abs(result - expected))
        error_msg = (f"A^T @ B mismatch. Max error: {max_err}\n"
                     f"Got:\n{result}\nExpected:\n{expected}")

    return TestResult(
        name=f"transpose_matmul_{N}x{N}",
        passed=passed, expected=expected, actual=result,
        error_msg=error_msg
    )

# Dry run (no simulator)

def dry_run(sa_rows, sa_cols):
    print(f"Dry run for {sa_rows}x{sa_cols} SA config\n")

    # Square
    N = sa_rows
    A = np.arange(N * N, dtype=np.float16).reshape(N, N)
    print(f"Square transpose {N}x{N}:")
    print(f"  Input A:\n{A}")
    print(f"  Expected A^T:\n{A.T}")

    # Non-square padded
    for sr, sc in [(2, 3), (3, 2), (1, 4), (4, 1)]:
        if sr <= N and sc <= N:
            A_small = np.arange(1, sr * sc + 1, dtype=np.float16).reshape(sr, sc)
            A_padded = np.zeros((N, N), dtype=np.float16)
            A_padded[:sr, :sc] = A_small
            print(f"\n  Non-square {sr}x{sc} (padded to {N}x{N}):")
            print(f"    Input:\n{A_small}")
            print(f"    Padded:\n{A_padded}")
            print(f"    Expected transpose block:\n{A_small.T}")

    # Matmul
    np.random.seed(123)
    A = np.random.rand(N, N).astype(np.float16)
    B = np.random.rand(N, N).astype(np.float16)
    print(f"\nTranspose matmul A^T @ B:")
    print(f"  A:\n{A}")
    print(f"  B:\n{B}")
    print(f"  Expected A^T @ B:\n{A.T.astype(np.float32) @ B.astype(np.float32)}")

# Test registry

def get_tests(engine, sa_rows, sa_cols):
    """Return dict of test_name -> callable returning TestResult."""
    tests = {}

    # Square transpose tests
    tests["square_sequential"] = lambda: test_transpose_square(engine, sa_rows, sa_cols, "sequential")
    tests["readback_sequential"] = lambda: test_transpose_load_readback(engine, sa_rows, sa_cols, "sequential")
    tests["readback_col_marker"] = lambda: test_transpose_load_readback(engine, sa_rows, sa_cols, "col_marker")
    tests["readback_random"] = lambda: test_transpose_load_readback(engine, sa_rows, sa_cols, "random")

    # Regression: normal load still works
    tests["normal_load_regression"] = lambda: test_normal_load_unchanged(engine, sa_rows, sa_cols)

    # Non-square (padded) tests
    non_square_sizes = [(2, 3), (3, 2), (1, sa_cols), (sa_rows, 1), (2, sa_cols), (sa_rows, 2)]
    for sr, sc in non_square_sizes:
        if sr <= sa_rows and sc <= sa_cols:
            # capture loop vars
            tests[f"nonsquare_{sr}x{sc}"] = (lambda r, c: lambda: test_transpose_nonsquare_padded(engine, sa_rows, sa_cols, r, c))(sr, sc)

    # Double transpose
    tests["double_transpose"] = lambda: test_double_transpose(engine, sa_rows, sa_cols)

    # Transpose + matmul
    tests["transpose_matmul"] = lambda: test_transpose_matmul(engine, sa_rows, sa_cols)

    return tests

# Main

def reinit_fsa(config_file: str):
    """Reset and re-initialize FSA state (memory allocators, etc.)."""
    from fsa.config import reset
    reset()
    import fsa as F
    F.init(config_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FSA DMA transpose test suite")
    parser.add_argument("--config", type=str, default="FSA4X4Fp16Config")
    parser.add_argument("--build_dir", type=str, default=None)
    parser.add_argument("--simulator_bin", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="/tmp")
    parser.add_argument("--dry_run", action="store_true", help="Show expected values, no simulation")
    parser.add_argument("--test", type=str, default=None, help="Run a specific test by name")
    parser.add_argument("--list", action="store_true", help="List available tests")
    parser.add_argument("--max_cycles", type=int, default=10000000)
    parser.add_argument("--vcdfile", type=str, default=None,
                        help="Path to write VCD waveform (requires -debug simulator)")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(4, 4)
        sys.exit(0)

    if args.list:
        print("Available tests:")
        for name in get_tests(None, 4, 4).keys():
            print(f"  {name}")
        sys.exit(0)

    import fsa as F
    from fsa.config import get_config

    if args.build_dir is None:
        build_dir = os.path.join("..", "..", "..", "sims", "verilator")
    else:
        build_dir = args.build_dir

    long_name = "chipyard.harness.TestHarness." + args.config
    config_file = os.path.join(
        build_dir, "generated-src", long_name,
        long_name + ".FSAConfig.json"
    )

    if not os.path.isfile(config_file):
        print(f"Config file not found: {config_file}")
        sys.exit(1)

    simulator_bin = args.simulator_bin
    if not simulator_bin or not os.path.isfile(simulator_bin):
        print(f"Simulator binary not found: {simulator_bin}")
        sys.exit(1)

    # Initial init to read config
    F.init(config_file)
    cfg = get_config()
    sa_rows, sa_cols = cfg.sa_rows, cfg.sa_cols

    engine = F.VerilatorSimulator(
        simulator_bin, output_dir=args.output_dir,
        max_cycles=args.max_cycles,
        vcdfile=args.vcdfile,
    )

    # Build test registry (uses lambdas, doesn't allocate yet)
    tests = get_tests(engine, sa_rows, sa_cols)

    if args.test:
        if args.test not in tests:
            print(f"Unknown test: {args.test}")
            print(f"Available: {', '.join(tests.keys())}")
            sys.exit(1)
        # Re-init FSA to get fresh memory allocators
        reinit_fsa(config_file)
        results = [tests[args.test]()]
    else:
        results = []
        for name, test_fn in tests.items():
            print(f"\n{'='*60}")
            print(f"Running: {name}")
            print(f"{'='*60}")
            # Re-init FSA before each test for fresh memory allocators
            reinit_fsa(config_file)
            try:
                results.append(test_fn())
            except Exception as e:
                results.append(TestResult(name=name, passed=False, error_msg=str(e)))

    all_passed = print_results(results)
    sys.exit(0 if all_passed else 1)
