package fsa.dma

import chisel3._
import chisel3.util._
import freechips.rocketchip.amba.axi4._
import fsa.{SRAMNarrowRead, SRAMNarrowWrite}
import fsa.frontend.Semaphore
import fsa.isa.ISA.Constants._
import fsa.utils.{DelayedAssert, Ehr}

abstract class BaseLoadStoreQueue(reqGen: DMARequest, n: Int) extends Module {

  val io = IO(new Bundle {
    val req = Flipped(Decoupled(reqGen))
    val semAcquire = Decoupled(new Semaphore)
    val semRelease = Decoupled(new Semaphore)
    val doSemRelease = Output(Bool())
    val busy = Output(Bool())
    val active = Output(Bool())
  })

  class Entry extends Bundle {
    val req = reqGen.cloneType
    val rRepeat = UInt(DMA_REPEAT_BITS.W)
    val depReady = Bool()
  }

  val entryValid = RegInit(VecInit(Seq.fill(n)(false.B)))
  val entries = Reg(Vec(n, new Entry))

  val enqPtr = Counter(n)
  val acqPtr = Counter(n)
  val deqPtr = Counter(n)

  // acquire semaphore before executing the request
  val acqEntry = entries(acqPtr.value)
  io.semAcquire.valid := entryValid(acqPtr.value) && acqEntry.req.acquireValid
  io.semAcquire.bits.id := acqEntry.req.semId
  io.semAcquire.bits.value := acqEntry.req.acquireSemValue
  when(io.semAcquire.fire || entryValid(acqPtr.value) && !acqEntry.req.acquireValid) {
    acqEntry.req.acquireValid := false.B
    acqEntry.depReady := true.B
    acqPtr.inc()
  }

  def valid(ptr: Counter): Bool = {
    entryValid(ptr.value) && entries(ptr.value).depReady
  }

  // deq request
  val deqEntry = entries(deqPtr.value)
  io.semRelease.valid := valid(deqPtr) && deqEntry.rRepeat === 0.U
  io.semRelease.bits.id := deqEntry.req.semId
  io.semRelease.bits.value := deqEntry.req.releaseSemValue
  io.doSemRelease := deqEntry.req.releaseValid
  when(io.semRelease.fire) {
    entryValid(deqPtr.value) := false.B
    deqPtr.inc()
  }

  // enq request
  when(io.req.fire) {
    entries(enqPtr.value).req := io.req.bits
    entries(enqPtr.value).rRepeat := io.req.bits.repeat
    entries(enqPtr.value).depReady := false.B
    entryValid(enqPtr.value) := true.B
    enqPtr.inc()
  }

  io.req.ready := !entryValid(enqPtr.value)
  io.busy := entryValid(deqPtr.value)
  io.active := valid(deqPtr)
}


class StoreQueue
(
  edge: AXI4EdgeParameters,
  reqGen: DMARequest, nInflight: Int,
  accReadGen: SRAMNarrowRead
) extends BaseLoadStoreQueue(reqGen, nInflight) {

  val aw = IO(Decoupled(new AXI4BundleAW(edge.bundle)))
  val w = IO(Decoupled(new AXI4BundleW(edge.bundle)))
  val b = IO(Flipped(Decoupled(new AXI4BundleB(edge.bundle))))
  val accRead = IO(accReadGen.cloneType)

  val awPtr = Counter(nInflight)
  val rPtr = Counter(nInflight)

  val awEntry = entries(awPtr.value)

  aw.valid := valid(awPtr) && awEntry.req.repeat =/= 0.U
  aw.bits.addr := awEntry.req.memAddr
  aw.bits.addr := awEntry.req.memAddr
  aw.bits.id := 0.U
  aw.bits.len := (awEntry.req.size >> log2Up(edge.slave.beatBytes)).asUInt - 1.U
  aw.bits.size := log2Up(edge.slave.beatBytes).U
  aw.bits.burst := AXI4Parameters.BURST_INCR
  aw.bits.lock := 0.U
  aw.bits.cache := 0.U
  aw.bits.prot := 0.U
  aw.bits.qos := 0.U

  when(aw.fire) {
    awEntry.req.repeat := awEntry.req.repeat - 1.U
    awEntry.req.memAddr := (awEntry.req.memAddr.asSInt + awEntry.req.memStride).asUInt
    when(awEntry.req.repeat === 1.U) {
      awPtr.inc()
    }
  }

  val beatBits = edge.slave.beatBytes * 8
  val nBeats = accRead.nSubBanks

  // sram read
  val rBeatCnt = RegInit(0.U(log2Up(nBeats).W))
  val rLast = rBeatCnt === (nBeats - 1).U
  val writeQueue = Module(new Queue(UInt(beatBits.W), entries = 2, pipe = true))
  DelayedAssert(!writeQueue.io.enq.valid || writeQueue.io.enq.ready)
  val queueCntNext = Mux(writeQueue.io.enq.valid,
    Mux(writeQueue.io.deq.fire, writeQueue.io.count, writeQueue.io.count + 1.U),
    Mux(writeQueue.io.deq.fire, writeQueue.io.count - 1.U, writeQueue.io.count)
  )
  val rEntry = entries(rPtr.value)
  accRead.valid := valid(rPtr) && queueCntNext < writeQueue.entries.U
  accRead.addr := rEntry.req.sramAddr
  accRead.subBankIdx := rBeatCnt
  writeQueue.io.enq.valid := RegNext(accRead.fire, init = false.B)
  writeQueue.io.enq.bits := accRead.data.asUInt
  when(accRead.fire) {
    rBeatCnt := Mux(rLast, 0.U, rBeatCnt + 1.U)
    when(rLast) {
      rEntry.rRepeat := rEntry.rRepeat - 1.U
      rEntry.req.sramAddr := (rEntry.req.sramAddr.asSInt + rEntry.req.sramStride).asUInt
      when(rEntry.rRepeat === 1.U) {
        rPtr.inc()
      }
    }
  }

  // mem write
  val wBeatCnt = RegInit(0.U(log2Up(nBeats).W))
  w.valid := writeQueue.io.deq.valid
  w.bits.data := writeQueue.io.deq.bits
  w.bits.strb := ~0.U(w.bits.strb.getWidth.W)
  w.bits.last := wBeatCnt === (nBeats - 1).U
  writeQueue.io.deq.ready := w.ready
  b.ready := true.B
  assert(b.bits.id === 0.U && b.bits.resp === AXI4Parameters.RESP_OKAY)
  when(w.fire) {
    wBeatCnt := Mux(w.bits.last, 0.U, wBeatCnt + 1.U)
  }

}


class LoadQueue[E <: Data]
(
  edge: AXI4EdgeParameters, reqGen: DMARequest,
  nInflight: Int, spadWriteGen: SRAMNarrowWrite
) extends BaseLoadStoreQueue(reqGen, nInflight) {

  val ar = IO(Decoupled(new AXI4BundleAR(edge.bundle)))
  val r = IO(Flipped(Decoupled(new AXI4BundleR(edge.bundle))))
  val spadWrite = IO(spadWriteGen.cloneType)
  val elemBytes = (spadWrite.elemWidth / 8).U

  /* Row Buffers and Column Count registers 
     rowBuf = Vector of elements that holds the tranposed row. Before flushed to Spad
     colCnt = num elements collected in rowBuf so far
     rowBufValid = flag rowBuf ready to flush 
     ONLY ONE ROW AT A TIME */
  val rowBuf = Reg(Vec(spadWrite.dataSize, UInt(spadWrite.elemWidth.W)))
  val colCnt = RegInit(0.U(log2Up(spadWrite.dataSize).W))
  val rowBufValid = RegInit(false.B)

  /* Per-Entry AR/R-side state */
  val memAddrWidth = edge.bundle.addrBits
  val MAX_COL_BITS = 6
  /* baseAddr = original memory base address for each queue entry. AR mutates the memAddr as it walks
     arColIdx = source coulumn the AR side reads for
     rowCnt = row the current column the AR is on
     rColIdx = independent R-side column tracker. AR advances faster than R. */

  val baseAddr = Reg(Vec(nInflight, UInt(memAddrWidth.W)))
  val arColIdx = RegInit(VecInit(Seq.fill(nInflight)(0.U(MAX_COL_BITS.W))))
  val rowCnt   = RegInit(VecInit(Seq.fill(nInflight)(0.U(DMA_SIZE_BITS.W))))
  val rColIdx  = RegInit(VecInit(Seq.fill(nInflight)(0.U(MAX_COL_BITS.W))))

  /* Initialize per-entry transpose state on enqueue */
  when(io.req.fire) {
    baseAddr(enqPtr.value) := io.req.bits.memAddr
    arColIdx(enqPtr.value) := 0.U
    rowCnt(enqPtr.value) := 0.U
    rColIdx(enqPtr.value) := 0.U
  }

  val arPtr = Counter(nInflight)
  val rPtr = Counter(nInflight)

  // read address
  val arEntry = entries(arPtr.value)
  val arBaseAddr = baseAddr(arPtr.value)
  val arColIdxCur = arColIdx(arPtr.value)
  val arSrcRows = arEntry.req.size
  ar.valid := valid(arPtr) && arEntry.req.repeat =/= 0.U

  /* AR addresses can be element-aligned for transposing, but the
     AXI slave requires beat-aligned addresses. Need to mask off
     the low bits and the per-element offset is recovered later via memAddr LSBs. */ 
  val beatAlignMask = ~((edge.slave.beatBytes - 1).U(memAddrWidth.W))
  ar.bits.addr := Mux(arEntry.req.isTranspose,
                      arEntry.req.memAddr & beatAlignMask,
                      arEntry.req.memAddr)
  ar.bits.id := 0.U
  ar.bits.len := Mux(arEntry.req.isTranspose,0.U, (arEntry.req.size >> log2Up(edge.slave.beatBytes)).asUInt - 1.U)
  ar.bits.size := log2Up(edge.slave.beatBytes).U
  ar.bits.burst := AXI4Parameters.BURST_INCR
  ar.bits.lock := 0.U
  ar.bits.cache := 0.U
  ar.bits.prot := 0.U
  ar.bits.qos := 0.U

  when(ar.fire) {
    arEntry.req.repeat := arEntry.req.repeat - 1.U
    when(arEntry.req.isTranspose) {
      val colDone = rowCnt(arPtr.value) === arSrcRows - 1.U

      when(colDone) {
        // resets row counter as to start at top of rows again
        rowCnt(arPtr.value) := 0.U
        // increments column number
        arColIdx(arPtr.value) := arColIdxCur + 1.U
        // jumps address to start of next column
        arEntry.req.memAddr := arBaseAddr + ((arColIdxCur +& 1.U) * elemBytes)
      } .otherwise {
        // iterate rows
        rowCnt(arPtr.value) := rowCnt(arPtr.value) + 1.U
        arEntry.req.memAddr := (arEntry.req.memAddr.asSInt + arEntry.req.memStride).asUInt
      }
    } .otherwise {
      arEntry.req.memAddr := (arEntry.req.memAddr.asSInt + arEntry.req.memStride).asUInt
    }
    when(arEntry.req.repeat === 1.U) {
      arPtr.inc()
    }
    /* empty requestes (repeat = 0) from the RequestPartitioner that is then routed to port 0 only for the transpose
       without this arPtr will stall perpetually on empty entries and will deadlock next enqueue */
  } .elsewhen(valid(arPtr) && arEntry.req.repeat === 0.U) {
    arPtr.inc()
  }

  val rEntry = entries(rPtr.value)
  val isTranspose = rEntry.req.isTranspose
  
  val nBeats = spadWrite.nSubBanks
  val rBeatCnt = RegInit(0.U(log2Up(nBeats).W))
 
  spadWrite.valid := false.B
  spadWrite.addr := 0.U
  spadWrite.data := DontCare
  spadWrite.subBankIdx := 0.U
  r.ready := false.B

  when(r.fire) {
  assert(r.bits.id === 0.U, "Currently only one ID is supported")
  assert(r.bits.resp === AXI4Parameters.RESP_OKAY, "Currently only OKAY response is supported")
}

  
  when(!isTranspose) {
  spadWrite.valid := r.valid
  spadWrite.addr := entries(rPtr.value).req.sramAddr
  spadWrite.data := r.bits.data.asTypeOf(spadWrite.data)
  spadWrite.subBankIdx := rBeatCnt
  r.ready := spadWrite.ready

  when(r.fire) {
    rBeatCnt := Mux(r.bits.last, 0.U, rBeatCnt + 1.U)
    when(r.bits.last) {
      rEntry.rRepeat := rEntry.rRepeat - 1.U
      rEntry.req.sramAddr := (rEntry.req.sramAddr.asSInt + rEntry.req.sramStride).asUInt
      when(rEntry.rRepeat === 1.U) {
        rPtr.inc()
      }
    }
  }
  } .otherwise {
/* Transpose Path */
  /* Spad not written on every beat, accumulated in rowBuf first */ 
  spadWrite.valid := rowBufValid
  spadWrite.addr  := rEntry.req.sramAddr
  spadWrite.data  := rowBuf
  spadWrite.subBankIdx := 0.U
  
  r.ready := !rowBufValid

  when(r.fire) {
    val beatData = r.bits.data.asTypeOf(Vec(spadWrite.dataSize, UInt(spadWrite.elemWidth.W)))
    /* AR advances memAddr every time it fires. When R returns, memAddr stale.
      rColIdx dervies the element offset within the beat */
    val elemBytesLog2 = log2Up(spadWrite.elemWidth / 8)
    val beatBytesLog2 = log2Up(edge.slave.beatBytes)
    val elemIdxBits = beatBytesLog2 - elemBytesLog2
    val elemIdx = rColIdx(rPtr.value)(elemIdxBits - 1, 0)
    rowBuf(colCnt) := beatData(elemIdx)

    val lastCol = colCnt === (spadWrite.dataSize - 1).U
    colCnt := Mux(lastCol, 0.U, colCnt + 1.U)
    when(lastCol) {
      rowBufValid := true.B
      rColIdx(rPtr.value) := rColIdx(rPtr.value) + 1.U
    }

    rBeatCnt := Mux(r.bits.last, 0.U, rBeatCnt + 1.U)
  }

  when(rowBufValid && spadWrite.ready) {
    rowBufValid := false.B
    rEntry.rRepeat := rEntry.rRepeat - spadWrite.dataSize.U
    rEntry.req.sramAddr := (rEntry.req.sramAddr.asSInt + rEntry.req.sramStride).asUInt
    when(rEntry.rRepeat === spadWrite.dataSize.U) {
      rPtr.inc()
    }
  }
/* Applies to both transpose and !transpose paths. Prevents rPtr deadlock and R misroutes. */
}
  when(valid(rPtr) && rEntry.rRepeat === 0.U) {
    r.ready := false.B
    spadWrite.valid := false.B
    rPtr.inc()
  }
}