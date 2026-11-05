# AI assistance: written with Claude (Anthropic; Claude Code, Claude Opus 5/5.5),
# then reviewed, run and validated by Ben Colbourne. See the thesis's
# Statement on the Use of Generative AI.

import sys
import json
import hashlib
import numpy as np

SEEDS = (42, 100, 69)

def shape_generator(shape, seed):
    rng = np.random.default_rng(seed)
    operand_shape = rng.uniform(low=-1, high=1, size=(shape))
    operand_shape_f16 = operand_shape.astype(np.float16)
    return operand_shape_f16


def make_operands(record, shapes):
    """Operands for one case: random per SEEDS, or as the record's 'operands' list asks.

    'operands' (BLAS3) gives each operand a generator: random, symmetric
    ((X + X^T) / 2), triu / tril, or alias (the same tensor as an earlier operand).
    """
    if "operands" not in record:
        return tuple(shape_generator(shape, seed) for shape, seed in zip(shapes, SEEDS))
    operands = []
    for spec, shape in zip(record["operands"], shapes):
        gen = spec["gen"]
        if gen == "alias":
            operands.append(operands[spec["index"]])
            continue
        x = shape_generator(shape, spec["seed"])
        if gen == "symmetric":
            x = ((x.astype(np.float32) + x.T.astype(np.float32)) / 2).astype(x.dtype)
        elif gen == "triu":
            x = np.triu(x)
        elif gen == "tril":
            x = np.tril(x)
        elif gen != "random":
            raise ValueError(f"unknown operand generator {gen!r}")
        operands.append(x)
    return tuple(operands)


def operand_digest(operands):
    """Short hash of a case's operands, to check two machines generate the same inputs."""
    h = hashlib.sha256()
    for x in operands:
        h.update(str(x.shape).encode())
        h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()[:16]


if __name__ == "__main__":
    # python cases.py <cases.jsonl>: one digest per case; diff the output across machines
    with open(sys.argv[1]) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                shapes = [tuple(s) for s in record["shapes"]]
                print(record["equation"], operand_digest(make_operands(record, shapes)))
