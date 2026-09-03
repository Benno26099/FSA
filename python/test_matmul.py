import os
import sys
import argparse
import numpy as np
import fsa as F
from dataclasses import dataclass
from typing import Optional


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

def test_matmul_identity(engine, sa_rows, sa_cols, label="identity"):
    """
    Pipeline:
      1. DMA plain-load A (NxN)   -> spad_a
      2. DMA plain-load I (NxN)   -> spad_id
      3. LoadStationary spad_a    -> PE registers hold A
      4. TensorMultiplication spad_id -> SA computes I @ anti_transpose(A)
                                         = anti_transpose(A), into acc_out
      5. ST_SRAM: store acc_out to memory (fp32)
      6. Verify result == anti_transpose(A); return pass/fail.
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

    A = np.arange(N * N, dtype=np.float16).reshape(N, N)
    I = np.eye(N, dtype=np.float16)

    # Allocate on-chip storage
    spad_a  = F.alloc_spad((N, N))            
    spad_id = F.alloc_spad((N, N))            
    acc_out = F.alloc_accumulator((N, N))

    # Allocate host-visible memory
    A_mem   = F.from_numpy(A)
    I_mem   = F.from_numpy(I)
    out_mem = F.alloc_mem((N, N), F.fp32)

    expected = I.astype(np.float32) @ (A.astype(np.float32).T)[::-1, ::-1]

    instructions = []

    # Step 1: DMA plain-load A -> spad_a
    instructions.append(make_dma_load(
        A_mem, spad_a, sem_id=0, rel_val=1,
        rows=N, cols=N, e_itemsize=e_itemsize
    ))

    # Step 2: DMA plain-load I -> spad_id
    instructions.append(make_dma_load(
        I_mem, spad_id, sem_id=2, rel_val=1,
        rows=N, cols=N, e_itemsize=e_itemsize
    ))

    # Step 3: LoadStationary A (waits for both DMAs; acquire on sem 2 = last DMA)
    instructions.append(make_load_stationary(
        spad_a, sem_id=2, acq_val=1, rel_val=2, e_itemsize=e_itemsize
    ))

    # Step 4: TensorMultiplication with stream=I. 
    instructions.append(make_tensor_multiplication(
        spad_id, acc_out, sem_id=1, rel_val=1,
        e_itemsize=e_itemsize, a_itemsize=a_itemsize
    ))

    # Step 5: DMA store acc_out -> out_mem
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
        error_msg = (f"Matmul Identity Failed. Max error: {max_err}\n"
                    f"Input A:\n{A}\nGot:\n{result}\nExpected \n{expected}")

    return TestResult(
        name=f"Matmul_Identity_{N}x{N}",
        passed=passed, expected=expected, actual=result,
        error_msg=error_msg
    )

def test_matmul_plain(engine, sa_rows, sa_cols, label="plain"):
    """
    Pipeline:
        1. DMA plain-load A (NxN)   -> spad_a
        2. DMA plain-load B (NxN)   -> spad_b
        3. LoadStationary spad_a    -> PE registers hold A
        4. TensorMultiplication spad_b -> SA computes B @ anti_transpose(A),
                                          into acc_out
        5. ST_SRAM: store acc_out to memory (fp32)
        6. Verify result == B @ anti_transpose(A); return pass/fail.
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

    A = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ], dtype=np.float16)[:N, :N]
    B = np.array([
            [16, 15, 14, 13],
            [12, 11, 10, 9],
            [8, 7, 6, 5],
            [4, 3, 2, 1]
    ], dtype=np.float16)[:N, :N]
    
    expected = B.astype(np.float32) @ (A.astype(np.float32).T)[::-1, ::-1]
    # Allocate on-chip storage
    spad_a  = F.alloc_spad((N, N))            
    spad_b = F.alloc_spad((N, N))            
    acc_out = F.alloc_accumulator((N, N))

    # Allocate host-visible memory
    A_mem   = F.from_numpy(A)
    B_mem   = F.from_numpy(B)
    out_mem = F.alloc_mem((N, N), F.fp32)

    instructions = []

    # Step 1: DMA plain-load A -> spad_a
    instructions.append(make_dma_load(
        A_mem, spad_a, sem_id=0, rel_val=1,
        rows=N, cols=N, e_itemsize=e_itemsize
    ))

    # Step 2: DMA plain-load B -> spad_b
    instructions.append(make_dma_load(
        B_mem, spad_b, sem_id=2, rel_val=1,
        rows=N, cols=N, e_itemsize=e_itemsize
    ))

    # Step 3: LoadStationary A (waits for both DMAs; acquire on sem 2 = last DMA)
    instructions.append(make_load_stationary(
        spad_a, sem_id=2, acq_val=1, rel_val=2, e_itemsize=e_itemsize
    ))

    # Step 4: TensorMultiplication with stream=B. 
    instructions.append(make_tensor_multiplication(
        spad_b, acc_out, sem_id=1, rel_val=1,
        e_itemsize=e_itemsize, a_itemsize=a_itemsize
    ))

    # Step 5: DMA store acc_out -> out_mem
    instructions.append(make_dma_store(
        acc_out, out_mem, sem_id=1, acq_val=1, rel_val=2,
        rows=N, cols=N, a_itemsize=a_itemsize
    ))

    instructions.append(FenceInstruction(mx=True, dma=True, stop=True))
    
    kernel = Kernel(instructions=instructions, input=[A_mem, B_mem, out_mem], output=out_mem)
    result_tile = engine.execute(kernel)
    result = F.to_numpy(result_tile)

    passed = np.allclose(result, expected, rtol=1e-2, atol=1e-2)
    error_msg = ""
    if not passed:
        max_err = np.max(np.abs(result - expected))
        error_msg = (f"Matmul Plain Failed. Max error: {max_err}\n"
                    f"Input A:\n{A}\nGot:\n{result}\nExpected \n{expected}")

    return TestResult(
        name=f"Matmul_Plain_{N}x{N}",
        passed=passed, expected=expected, actual=result,
        error_msg=error_msg
    )

def test_matmul_transpose_A(engine, sa_rows, sa_cols, label="transpose_a"):
    """
    Pipeline:
        1. DMA transpose-load A (NxN)  -> spad_a  (spad_a holds A.T)
        2. DMA plain-load B (NxN)      -> spad_b
        3. LoadStationary spad_a       -> PE registers hold A.T
        4. TensorMultiplication spad_b -> SA computes B @ anti_transpose(A.T)
                                          = B @ A[::-1, ::-1], into acc_out
        5. ST_SRAM: store acc_out to memory (fp32)
        6. Verify result == B @ A[::-1, ::-1]; return pass/fail.
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

    A = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ], dtype=np.float16)[:N, :N]
    B = np.array([
            [16, 15, 14, 13],
            [12, 11, 10, 9],
            [8, 7, 6, 5],
            [4, 3, 2, 1]
    ], dtype=np.float16)[:N, :N]
    
    expected = B.astype(np.float32) @ (A.astype(np.float32))[::-1, ::-1]    
    spad_a  = F.alloc_spad((N, N))            
    spad_b = F.alloc_spad((N, N))           
    acc_out = F.alloc_accumulator((N, N))

    # Allocate host-visible memory
    A_mem   = F.from_numpy(A)
    B_mem   = F.from_numpy(B)
    out_mem = F.alloc_mem((N, N), F.fp32)

    instructions = []

    # Step 1: DMA transpose-load A -> spad_a (spad_a holds A.T)
    instructions.append(make_dma_transpose_load(
        A_mem, spad_a, sem_id=0, rel_val=1,
        src_rows=N, src_cols=N, e_itemsize=e_itemsize
    ))

    # Step 2: DMA plain-load B -> spad_b
    instructions.append(make_dma_load(
        B_mem, spad_b, sem_id=2, rel_val=1,
        rows=N, cols=N, e_itemsize=e_itemsize
    ))

    # Step 3: LoadStationary A 
    instructions.append(make_load_stationary(
        spad_a, sem_id=2, acq_val=1, rel_val=2, e_itemsize=e_itemsize
    ))

    # Step 4: TensorMultiplication with stream=B. 
    instructions.append(make_tensor_multiplication(
        spad_b, acc_out, sem_id=1, rel_val=1,
        e_itemsize=e_itemsize, a_itemsize=a_itemsize
    ))

    # Step 5: DMA store acc_out -> out_mem
    instructions.append(make_dma_store(
        acc_out, out_mem, sem_id=1, acq_val=1, rel_val=2,
        rows=N, cols=N, a_itemsize=a_itemsize
    ))

    instructions.append(FenceInstruction(mx=True, dma=True, stop=True))
    
    kernel = Kernel(instructions=instructions, input=[A_mem, B_mem, out_mem], output=out_mem)
    result_tile = engine.execute(kernel)
    result = F.to_numpy(result_tile)

    passed = np.allclose(result, expected, rtol=1e-2, atol=1e-2)
    error_msg = ""
    if not passed:
        max_err = np.max(np.abs(result - expected))
        error_msg = (f"Matmul Transpose A Failed. Max error: {max_err}\n"
                    f"Input A:\n{A}\nGot:\n{result}\nExpected \n{expected}")

    return TestResult(
        name=f"Matmul_Transpose_A{N}x{N}",
        passed=passed, expected=expected, actual=result,
        error_msg=error_msg
    )

def test_matmul_transpose_B(engine, sa_rows, sa_cols, label="transpose_b"):
    """
    Pipeline:
        1. DMA plain-load A (NxN)        -> spad_a
        2. DMA transpose-load B (NxN)    -> spad_b  (spad_b holds B.T)
        3. LoadStationary spad_a         -> PE registers hold A
        4. TensorMultiplication spad_b   -> SA computes B.T @ anti_transpose(A)
                                            = B.T @ (A.T)[::-1, ::-1], into acc_out
        5. ST_SRAM: store acc_out to memory (fp32)
        6. Verify result == B.T @ anti_transpose(A); return pass/fail.
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

    A = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ], dtype=np.float16)[:N, :N]
    B = np.array([
            [16, 15, 14, 13],
            [12, 11, 10, 9],
            [8, 7, 6, 5],
            [4, 3, 2, 1]
    ], dtype=np.float16)[:N, :N]
    
    expected = B.T @ anti_transpose(A) 
    spad_a  = F.alloc_spad((N, N))            
    spad_b = F.alloc_spad((N, N))            
    acc_out = F.alloc_accumulator((N, N))

    # Allocate host-visible memory
    A_mem   = F.from_numpy(A)
    B_mem   = F.from_numpy(B)
    out_mem = F.alloc_mem((N, N), F.fp32)

    instructions = []

    # Step 1: DMA plain-load A -> spad_a
    instructions.append(make_dma_load(
        A_mem, spad_a, sem_id=0, rel_val=1,
        rows=N, cols=N, e_itemsize=e_itemsize
    ))

    # Step 2: DMA transpose-load B -> spad_b (spad_b holds B.T)
    instructions.append(make_dma_transpose_load(
        B_mem, spad_b, sem_id=2, rel_val=1,
        src_rows=N, src_cols=N, e_itemsize=e_itemsize
    ))

    # Step 3: LoadStationary A 
    instructions.append(make_load_stationary(
        spad_a, sem_id=2, acq_val=1, rel_val=2, e_itemsize=e_itemsize
    ))

    # Step 4: TensorMultiplication with stream=B.
    instructions.append(make_tensor_multiplication(
        spad_b, acc_out, sem_id=1, rel_val=1,
        e_itemsize=e_itemsize, a_itemsize=a_itemsize
    ))

    # Step 5: DMA store acc_out -> out_mem
    instructions.append(make_dma_store(
        acc_out, out_mem, sem_id=1, acq_val=1, rel_val=2,
        rows=N, cols=N, a_itemsize=a_itemsize
    ))

    instructions.append(FenceInstruction(mx=True, dma=True, stop=True))
    
    kernel = Kernel(instructions=instructions, input=[A_mem, B_mem, out_mem], output=out_mem)
    result_tile = engine.execute(kernel)
    result = F.to_numpy(result_tile)

    passed = np.allclose(result, expected, rtol=1e-2, atol=1e-2)
    error_msg = ""
    if not passed:
        max_err = np.max(np.abs(result - expected))
        error_msg = (f"Matmul Transpose B Failed. Max error: {max_err}\n"
                    f"Input A:\n{A}\nGot:\n{result}\nExpected \n{expected}")

    return TestResult(
        name=f"Matmul_Transpose_B{N}x{N}",
        passed=passed, expected=expected, actual=result,
        error_msg=error_msg
    )

def test_matmul_transpose_Both(engine, sa_rows, sa_cols, label="transpose_both"):
    """
    Pipeline:
        1. DMA transpose-load A (NxN)  -> spad_a  (spad_a holds A.T)
        2. DMA transpose-load B (NxN)  -> spad_b  (spad_b holds B.T)
        3. LoadStationary spad_a       -> PE registers hold A.T
        4. TensorMultiplication spad_b -> SA computes B.T @ anti_transpose(A.T)
                                          = B.T @ A[::-1, ::-1], into acc_out
        5. ST_SRAM: store acc_out to memory (fp32)
        6. Verify result == B.T @ A[::-1, ::-1]; return pass/fail.
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

    A = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ], dtype=np.float16)[:N, :N]
    B = np.array([
            [16, 15, 14, 13],
            [12, 11, 10, 9],
            [8, 7, 6, 5],
            [4, 3, 2, 1]
    ], dtype=np.float16)[:N, :N]
    
    expected = B.astype(np.float32).T @ (A.astype(np.float32))[::-1, ::-1]  
    spad_a  = F.alloc_spad((N, N))            
    spad_b = F.alloc_spad((N, N))           
    acc_out = F.alloc_accumulator((N, N))

    # Allocate host-visible memory
    A_mem   = F.from_numpy(A)
    B_mem   = F.from_numpy(B)
    out_mem = F.alloc_mem((N, N), F.fp32)

    instructions = []

    # Step 1: DMA transpose-load A -> spad_a (spad_a holds A.T)
    instructions.append(make_dma_transpose_load(
        A_mem, spad_a, sem_id=0, rel_val=1,
        src_rows=N, src_cols=N, e_itemsize=e_itemsize
    ))

    # Step 2: DMA transpose-load B -> spad_b (spad_b holds B.T)
    instructions.append(make_dma_transpose_load(
        B_mem, spad_b, sem_id=2, rel_val=1,
        src_rows=N, src_cols=N, e_itemsize=e_itemsize
    ))

    # Step 3: LoadStationary A 
    instructions.append(make_load_stationary(
        spad_a, sem_id=2, acq_val=1, rel_val=2, e_itemsize=e_itemsize
    ))

    # Step 4: TensorMultiplication with stream=B.
    instructions.append(make_tensor_multiplication(
        spad_b, acc_out, sem_id=1, rel_val=1,
        e_itemsize=e_itemsize, a_itemsize=a_itemsize
    ))

    # Step 5: DMA store acc_out -> out_mem
    instructions.append(make_dma_store(
        acc_out, out_mem, sem_id=1, acq_val=1, rel_val=2,
        rows=N, cols=N, a_itemsize=a_itemsize
    ))

    instructions.append(FenceInstruction(mx=True, dma=True, stop=True))
    
    kernel = Kernel(instructions=instructions, input=[A_mem, B_mem, out_mem], output=out_mem)
    result_tile = engine.execute(kernel)
    result = F.to_numpy(result_tile)

    passed = np.allclose(result, expected, rtol=1e-2, atol=1e-2)
    error_msg = ""
    if not passed:
        max_err = np.max(np.abs(result - expected))
        error_msg = (f"Matmul Transpose Both Failed. Max error: {max_err}\n"
                    f"Input A:\n{A}\nGot:\n{result}\nExpected \n{expected}")

    return TestResult(
        name=f"Matmul_Transpose_Both{N}x{N}",
        passed=passed, expected=expected, actual=result,
        error_msg=error_msg
    )


def test_matmul_accumulate_two_tiles(engine, sa_rows, sa_cols, label="transpose_both"):
    import fsa as F
    from fsa.instructions import FenceInstruction
    from fsa.kernel import Kernel
    from fsa.config import get_config

    cfg = get_config()
    N = sa_rows
    assert sa_rows == sa_cols, f"Square test requires sa_rows == sa_cols, got {sa_rows}x{sa_cols}"
    e_itemsize = cfg.e_type.itemsize
    a_itemsize = cfg.a_type.itemsize

    A1 = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ], dtype=np.float16)[:N, :N]
    A2 = np.array([
            [4, 3, 2, 1],
            [8, 7, 6, 5],
            [12, 11, 10, 9],
            [16, 15, 14, 13]
    ], dtype=np.float16)[:N, :N]
    B = np.array([
            [16, 15, 14, 13],
            [12, 11, 10, 9],
            [8, 7, 6, 5],
            [4, 3, 2, 1]
    ], dtype=np.float16)[:N, :N]
    
    expected = B.astype(np.float32) @ (A1.astype(np.float32) + A2.astype(np.float32)).T[::-1, ::-1]
    spad_a1  = F.alloc_spad((N, N))            
    spad_a2 = F.alloc_spad((N,N))
    spad_b = F.alloc_spad((N, N))            
    acc_out = F.alloc_accumulator((N, N))

    # Allocate host-visible memory
    A1_mem   = F.from_numpy(A1)
    A2_mem = F.from_numpy(A2)
    B_mem   = F.from_numpy(B)
    out_mem = F.alloc_mem((N, N), F.fp32)

    instructions = []

    # Step 1: DMA plain-load A -> spad_a
    instructions.append(make_dma_load(
        A1_mem, spad_a1, sem_id=0, rel_val=1,
        rows=N, cols=N, e_itemsize=e_itemsize
    ))

    # Step 2: DMA plain-load B -> spad_b
    instructions.append(make_dma_load(
        B_mem, spad_b, sem_id=1, rel_val=1,
        rows=N, cols=N, e_itemsize=e_itemsize
    ))

    # Step 3: LoadStationary A 
    instructions.append(make_load_stationary(
        spad_a1, sem_id=1, acq_val=1, rel_val=2, e_itemsize=e_itemsize
    ))

    # Step 4: TensorMultiplication with stream=B.
    instructions.append(make_tensor_multiplication(
        spad_b, acc_out, sem_id=3, rel_val=1,
        e_itemsize=e_itemsize, a_itemsize=a_itemsize, accumulate=False
    ))

    instructions.append(make_dma_load(
            A2_mem, spad_a2, sem_id=5, rel_val=1,
            rows=N, cols=N, e_itemsize=e_itemsize
        ))

    instructions.append(make_load_stationary(
            spad_a2, sem_id=5, acq_val=1, rel_val=2, e_itemsize=e_itemsize
        ))

    instructions.append(make_tensor_multiplication(
            spad_b, acc_out, sem_id=7, rel_val=1,
            e_itemsize=e_itemsize, a_itemsize=a_itemsize, accumulate=True, wait_prev_acc=True
        ))
    
     # Step 5: DMA store acc_out -> out_mem
    instructions.append(make_dma_store(
            acc_out, out_mem, sem_id=7, acq_val=1, rel_val=2,
            rows=N, cols=N, a_itemsize=a_itemsize
        ))
    
    instructions.append(FenceInstruction(mx=True, dma=True, stop=True))
    
    kernel = Kernel(instructions=instructions, input=[A1_mem, A2_mem, B_mem, out_mem], output=out_mem)
    result_tile = engine.execute(kernel)
    result = F.to_numpy(result_tile)

    passed = np.allclose(result, expected, rtol=1e-2, atol=1e-2)
    error_msg = ""
    if not passed:
        max_err = np.max(np.abs(result - expected))
        error_msg = (f"Matmul Accumulate Two Tiles Failed. Max error: {max_err}\n"
                    f"Input A1:\n{A1}\nGot:\n{result}\nExpected \n{expected}")

    return TestResult(
        name=f"Matmul_Accumulate_Two_Tiles{N}x{N}",
        passed=passed, expected=expected, actual=result,
        error_msg=error_msg
    )
    
def test_matmul_accumulate_four_tiles(engine, sa_rows, sa_cols, label="transpose_both"):
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
    A1 = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ], dtype=np.float16)[:N, :N]
    A2 = np.array([
            [4, 3, 2, 1],
            [8, 7, 6, 5],
            [12, 11, 10, 9],
            [16, 15, 14, 13]
    ], dtype=np.float16)[:N, :N]
    A3 = np.array([
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4]
    ], dtype=np.float16)[:N, :N]
    A4 = np.array([
            [5, 5, 5, 5],
            [6, 6, 6, 6],
            [7, 7, 7, 7],
            [8, 8, 8, 8]
    ], dtype=np.float16)[:N, :N]
    B = np.array([
            [16, 15, 14, 13],
            [12, 11, 10, 9],
            [8, 7, 6, 5],
            [4, 3, 2, 1]
    ], dtype=np.float16)[:N, :N]
    
    expected = B.astype(np.float32) @ (A1.astype(np.float32) + A2.astype(np.float32) + A3.astype(np.float32) + A4.astype(np.float32)).T[::-1, ::-1]
    spad_a1  = F.alloc_spad((N, N))            
    spad_a2 = F.alloc_spad((N, N))
    spad_a3 = F.alloc_spad((N, N))
    spad_a4 = F.alloc_spad((N, N))
    spad_b = F.alloc_spad((N, N))            
    acc_out = F.alloc_accumulator((N, N))

    # Allocate host-visible memory
    A1_mem   = F.from_numpy(A1)
    A2_mem = F.from_numpy(A2)
    A3_mem   = F.from_numpy(A3)
    A4_mem = F.from_numpy(A4)
    B_mem   = F.from_numpy(B)
    out_mem = F.alloc_mem((N, N), F.fp32)

    instructions = []

    # Step 1: DMA plain-load A -> spad_a
    instructions.append(make_dma_load(
        A1_mem, spad_a1, sem_id=0, rel_val=1,
        rows=N, cols=N, e_itemsize=e_itemsize
    ))

    # Step 2: DMA plain-load B -> spad_b
    instructions.append(make_dma_load(
        B_mem, spad_b, sem_id=1, rel_val=1,
        rows=N, cols=N, e_itemsize=e_itemsize
    ))

    # Step 3: LoadStationary A 
    instructions.append(make_load_stationary(
        spad_a1, sem_id=1, acq_val=1, rel_val=2, e_itemsize=e_itemsize
    ))

    # Step 4: TensorMultiplication with stream=B.
    instructions.append(make_tensor_multiplication(
        spad_b, acc_out, sem_id=3, rel_val=1,
        e_itemsize=e_itemsize, a_itemsize=a_itemsize, accumulate=False
    ))

    instructions.append(make_dma_load(
            A2_mem, spad_a2, sem_id=5, rel_val=1,
            rows=N, cols=N, e_itemsize=e_itemsize
        ))

    instructions.append(make_load_stationary(
            spad_a2, sem_id=5, acq_val=1, rel_val=2, e_itemsize=e_itemsize
        ))

    instructions.append(make_tensor_multiplication(
            spad_b, acc_out, sem_id=7, rel_val=1,
            e_itemsize=e_itemsize, a_itemsize=a_itemsize, accumulate=True, wait_prev_acc=True
        ))
    
    instructions.append(make_dma_load(
                A3_mem, spad_a3, sem_id=8, rel_val=1,
                rows=N, cols=N, e_itemsize=e_itemsize
            ))
    
    instructions.append(make_load_stationary(
            spad_a3, sem_id=8, acq_val=1, rel_val=2, e_itemsize=e_itemsize
        ))

    instructions.append(make_tensor_multiplication(
            spad_b, acc_out, sem_id=10, rel_val=1,
            e_itemsize=e_itemsize, a_itemsize=a_itemsize, accumulate=True, wait_prev_acc=True
        ))
    instructions.append(make_dma_load(
            A4_mem, spad_a4, sem_id=11, rel_val=1,
            rows=N, cols=N, e_itemsize=e_itemsize
        ))
        
    instructions.append(make_load_stationary(
            spad_a4, sem_id=11, acq_val=1, rel_val=2, e_itemsize=e_itemsize
        ))

    instructions.append(make_tensor_multiplication(
            spad_b, acc_out, sem_id=13, rel_val=1,
            e_itemsize=e_itemsize, a_itemsize=a_itemsize, accumulate=True, wait_prev_acc=True
        ))
    
        # Step 5: DMA store acc_out -> out_mem
    instructions.append(make_dma_store(
            acc_out, out_mem, sem_id=13, acq_val=1, rel_val=2,
            rows=N, cols=N, a_itemsize=a_itemsize
        ))
    
    instructions.append(FenceInstruction(mx=True, dma=True, stop=True))
    
    kernel = Kernel(instructions=instructions, input=[A1_mem, A2_mem, A3_mem, A4_mem, B_mem, out_mem], output=out_mem)
    result_tile = engine.execute(kernel)
    result = F.to_numpy(result_tile)

    passed = np.allclose(result, expected, rtol=1e-2, atol=1e-2)
    error_msg = ""
    if not passed:
        max_err = np.max(np.abs(result - expected))
        error_msg = (f"Matmul Accumulate Four Tiles Failed. Max error: {max_err}\n"
                    f"Input A1:\n{A1}\nGot:\n{result}\nExpected \n{expected}")

    return TestResult(
        name=f"Matmul_Accumulate_Four_Tiles{N}x{N}",
        passed=passed, expected=expected, actual=result,
        error_msg=error_msg
    )

# Test registration

def get_tests(engine, sa_rows, sa_cols):
    tests = {}
    tests["matmul_identity"] = lambda: test_matmul_identity(engine, sa_rows, sa_cols)
    tests["matmul_plain"] = lambda: test_matmul_plain(engine, sa_rows, sa_cols)
    tests["matmul_transpose_a"] = lambda: test_matmul_transpose_A(engine, sa_rows, sa_cols)
    tests["matmul_transpose_b"] = lambda: test_matmul_transpose_B(engine, sa_rows, sa_cols)
    tests["matmul_transpose_both"] = lambda: test_matmul_transpose_Both(engine, sa_rows, sa_cols)
    tests["matmul_accumulate_two"] = lambda: test_matmul_accumulate_two_tiles(engine, sa_rows, sa_cols)
    tests["matmul_accumulate_four"] = lambda: test_matmul_accumulate_four_tiles(engine, sa_rows, sa_cols)
    return tests

# Main

def reinit_fsa(config_file: str):
    """Reset and re-initialize FSA state (memory allocators, etc.)."""
    from fsa.config import reset
    reset()
    import fsa as F
    F.init(config_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FSA matmul (TENSOR_MULTIPLICATION) test suite")
    parser.add_argument("--config", type=str, default="FSA4X4Fp16Config")
    parser.add_argument("--build_dir", type=str, default=None)
    parser.add_argument("--simulator_bin", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="/tmp")
    parser.add_argument("--test", type=str, default=None, help="Run a specific test by name")
    parser.add_argument("--list", action="store_true", help="List available tests")
    parser.add_argument("--max_cycles", type=int, default=10000000)
    parser.add_argument("--vcdfile", type=str, default=None,
                        help="Path to write VCD waveform (requires -debug simulator)")
    args = parser.parse_args()

    if args.list:
        print("Available tests:")
        for name in get_tests(None, 4, 4).keys():
            print(f"  {name}")
        sys.exit(0)

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

    from fsa.config import get_config
    F.init(config_file)
    cfg = get_config()
    sa_rows, sa_cols = cfg.sa_rows, cfg.sa_cols

    engine = F.VerilatorSimulator(
        simulator_bin, output_dir=args.output_dir,
        max_cycles=args.max_cycles,
        vcdfile=args.vcdfile,
    )

    tests = get_tests(engine, sa_rows, sa_cols)

    if args.test:
        if args.test not in tests:
            print(f"Unknown test: {args.test}")
            print(f"Available: {', '.join(tests.keys())}")
            sys.exit(1)
        reinit_fsa(config_file)
        results = [tests[args.test]()]
    else:
        results = []
        for name, test_fn in tests.items():
            print(f"\n{'='*60}")
            print(f"Running: {name}")
            print(f"{'='*60}")
            reinit_fsa(config_file)
            try:
                results.append(test_fn())
            except Exception as e:
                results.append(TestResult(name=name, passed=False, error_msg=str(e)))

    all_passed = print_results(results)
    sys.exit(0 if all_passed else 1)
