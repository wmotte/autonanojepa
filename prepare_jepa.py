#!/usr/bin/env python3
"""
Data preparation for nanoJEPA Clojure execution prediction.
Generates (expression, result) pairs purely in Python — no Clojure runtime needed.

Task: actual execution prediction — given a full Clojure expression, predict
the embedding of its computed result. Both training and evaluation use real
(expression, result) pairs; no masked-JEPA proxy.

Usage: uv run prepare_jepa.py
"""

import os
import random
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Constants (fixed)
# ---------------------------------------------------------------------------

VOCAB_SIZE = 56  # removed 41 discrete number tokens, added 1 NUM token
MAX_EXPR_LEN = 40
MAX_RESULT_LEN = 8
TIME_BUDGET = 120
N_VAL_PAIRS = 2000

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
VAL_CACHE_PATH = os.path.join(CACHE_DIR, "val_pairs.txt")

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

_SPECIAL = ["PAD", "BOS", "SEP", "EOS"]  # 0-3
_SYNTAX = ["(", ")", "[", "]", "{", "}"]  # 4-9
_OPERATORS = ["+", "-", "*", "/", "mod", "max", "min", "abs", "inc", "dec"]  # 10-19
_HOF = ["map", "filter", "reduce", "count", "first", "last", "rest", "cons"]  # 20-27
_DICT = ["assoc", "get", "keys", "vals", "merge"]  # 28-32
_FORMS = ["let", "fn", "if", "cond", "def"]  # 33-37
_VARS = ["x", "y", "z", "a", "b", "n", "m"]  # 38-44
# Booleans and nil (45-48)
_BOOLEANS = ["true", "false", "nil", "pos?"]  # 45-48 (pos? needed for conditionals)
# Extra Clojure tokens (49-54)
_EXTRA = ["MASK", "->", "->>", "when", "not", "="]  # 49-54
# Numeric placeholder: replaces all discrete number tokens (55)
# Actual integer value is carried in a parallel vals array and encoded via sinusoidal embedding.
_NUM = ["NUM"]  # 55

_ALL_TOKENS = _SPECIAL + _SYNTAX + _OPERATORS + _HOF + _DICT + _FORMS + _VARS + _BOOLEANS + _EXTRA + _NUM

assert len(_ALL_TOKENS) == VOCAB_SIZE, f"Expected {VOCAB_SIZE} tokens, got {len(_ALL_TOKENS)}"

_TOKEN_TO_ID = {tok: i for i, tok in enumerate(_ALL_TOKENS)}
_ID_TO_TOKEN = {i: tok for i, tok in enumerate(_ALL_TOKENS)}

PAD_ID  = _TOKEN_TO_ID["PAD"]
BOS_ID  = _TOKEN_TO_ID["BOS"]
SEP_ID  = _TOKEN_TO_ID["SEP"]
EOS_ID  = _TOKEN_TO_ID["EOS"]
MASK_ID = _TOKEN_TO_ID["MASK"]
NUM_ID  = _TOKEN_TO_ID["NUM"]
MASK_STR = "MASK"


class ClojureVocab:
    """Vocabulary for Clojure expression tokens."""

    def __init__(self):
        self.token_to_id = _TOKEN_TO_ID
        self.id_to_token = _ID_TO_TOKEN
        self.vocab_size = VOCAB_SIZE

    @classmethod
    def from_cache(cls):
        return cls()

    def encode(self, token_list):
        """Encode list of string tokens to list of ints."""
        if isinstance(token_list, str):
            # Try splitting as space-separated
            token_list = token_list.split()
        return [self.token_to_id[t] for t in token_list if t in self.token_to_id]

    def decode(self, ids):
        return [self.id_to_token.get(i, "?") for i in ids if i not in (PAD_ID,)]


# ---------------------------------------------------------------------------
# Python-native Clojure expression evaluator
# ---------------------------------------------------------------------------

def _num_tok(n):
    """Return string token for integer n — no clipping, any integer is valid."""
    return str(int(round(n)))


def _safe_div(a, b):
    if b == 0:
        return 0
    return int(a) // int(b)


def _safe_mod(a, b):
    if b == 0:
        return 0
    return int(a) % int(b)


# ---------------------------------------------------------------------------
# Expression generators (return (expr_tokens, result_tokens))
# ---------------------------------------------------------------------------

def _gen_arithmetic(rng):
    """Family A: arithmetic expressions."""
    ops = ["+", "-", "*", "/", "mod", "max", "min"]

    def rand_num():
        return rng.randint(-5, 15)

    def simple_expr():
        op = rng.choice(ops)
        a, b = rand_num(), rand_num()
        if op == "+":
            val = a + b
        elif op == "-":
            val = a - b
        elif op == "*":
            val = a * b
        elif op == "/":
            val = _safe_div(a, b)
        elif op == "mod":
            val = _safe_mod(a, b)
        elif op == "max":
            val = max(a, b)
        elif op == "min":
            val = min(a, b)
        toks = ["(", op, _num_tok(a), _num_tok(b), ")"]
        return toks, val

    # Nested or simple
    if rng.random() < 0.4:
        # Nested: (+ a (* b c))
        op_outer = rng.choice(["+", "-", "max", "min"])
        a = rand_num()
        inner_toks, inner_val = simple_expr()
        if op_outer == "+":
            val = a + inner_val
        elif op_outer == "-":
            val = a - inner_val
        elif op_outer == "max":
            val = max(a, inner_val)
        elif op_outer == "min":
            val = min(a, inner_val)
        toks = ["(", op_outer, _num_tok(a)] + inner_toks + [")"]
    elif rng.random() < 0.3:
        # abs/inc/dec
        op = rng.choice(["abs", "inc", "dec"])
        a = rand_num()
        if op == "abs":
            val = abs(a)
        elif op == "inc":
            val = a + 1
        elif op == "dec":
            val = a - 1
        toks = ["(", op, _num_tok(a), ")"]
    else:
        toks, val = simple_expr()

    val = max(-10, min(30, int(round(val))))
    result_toks = [_num_tok(val)]
    return toks, result_toks


def _gen_let(rng):
    """Family B: let bindings."""
    var = rng.choice(["x", "y", "z", "a", "b", "n", "m"])
    val_a = rng.randint(-5, 15)
    op = rng.choice(["+", "-", "*", "max", "min", "inc", "dec"])
    b = rng.randint(-3, 10)

    if op in ["inc", "dec"]:
        if op == "inc":
            result = val_a + 1
        else:
            result = val_a - 1
        body_toks = ["(", op, var, ")"]
    else:
        if op == "+":
            result = val_a + b
        elif op == "-":
            result = val_a - b
        elif op == "*":
            result = val_a * b
        elif op == "max":
            result = max(val_a, b)
        elif op == "min":
            result = min(val_a, b)
        body_toks = ["(", op, var, _num_tok(b), ")"]

    result = max(-10, min(30, int(round(result))))
    toks = ["(", "let", "[", var, _num_tok(val_a), "]"] + body_toks + [")"]
    return toks, [_num_tok(result)]


def _gen_hof(rng):
    """Family C: higher-order functions."""
    choice = rng.random()

    if choice < 0.3:
        # (count [a b c ...])
        n = rng.randint(1, 6)
        elems = [rng.randint(-5, 15) for _ in range(n)]
        toks = ["(", "count", "["] + [_num_tok(e) for e in elems] + ["]", ")"]
        result = n
        return toks, [_num_tok(result)]

    elif choice < 0.6:
        # (reduce + [a b c])
        n = rng.randint(2, 5)
        elems = [rng.randint(-3, 10) for _ in range(n)]
        op = rng.choice(["+", "max", "min"])
        if op == "+":
            result = sum(elems)
        elif op == "max":
            result = max(elems)
        elif op == "min":
            result = min(elems)
        toks = ["(", "reduce", op, "["] + [_num_tok(e) for e in elems] + ["]", ")"]
        result = max(-10, min(30, int(round(result))))
        return toks, [_num_tok(result)]

    elif choice < 0.8:
        # (first [a b c]) or (last [a b c])
        n = rng.randint(2, 5)
        elems = [rng.randint(-5, 15) for _ in range(n)]
        fn = rng.choice(["first", "last"])
        result = elems[0] if fn == "first" else elems[-1]
        toks = ["(", fn, "["] + [_num_tok(e) for e in elems] + ["]", ")"]
        return toks, [_num_tok(result)]

    else:
        # (count [a b c]) variant — simple
        n = rng.randint(1, 4)
        elems = [rng.randint(0, 10) for _ in range(n)]
        toks = ["(", "count", "["] + [_num_tok(e) for e in elems] + ["]", ")"]
        return toks, [_num_tok(n)]


def _gen_conditional(rng):
    """Family D: conditionals."""
    a = rng.randint(-8, 15)
    then_val = rng.randint(-5, 15)
    else_val = rng.randint(-5, 15)

    cond_type = rng.random()
    if cond_type < 0.5:
        # (if (pos? a) then else)
        cond_result = a > 0
        toks = ["(", "if", "(", "pos?", _num_tok(a), ")", _num_tok(then_val), _num_tok(else_val), ")"]
    else:
        # (if (> a b) then else) — simplified as (max a 0) > 0
        b = rng.randint(-5, 10)
        op = rng.choice(["+", "max", "min"])
        if op == "+":
            computed = a + b
        elif op == "max":
            computed = max(a, b)
        elif op == "min":
            computed = min(a, b)
        cond_result = computed > 0
        toks = ["(", "if", "(", "pos?", "(", op, _num_tok(a), _num_tok(b), ")", ")", _num_tok(then_val), _num_tok(else_val), ")"]

    result = then_val if cond_result else else_val
    result = max(-10, min(30, int(round(result))))
    return toks, [_num_tok(result)]


def _gen_multi_let(rng):
    """Family E: two-variable let binding.  (let [x 3 y 4] (+ x y)) → 7"""
    var1 = rng.choice(["x", "y", "z", "a", "b"])
    var2 = rng.choice([v for v in ["x", "y", "z", "a", "b", "n", "m"] if v != var1])
    v1 = rng.randint(-5, 12)
    v2 = rng.randint(-5, 12)
    op = rng.choice(["+", "-", "*", "max", "min"])
    if op == "+":
        result = v1 + v2
    elif op == "-":
        result = v1 - v2
    elif op == "*":
        result = v1 * v2
    elif op == "max":
        result = max(v1, v2)
    elif op == "min":
        result = min(v1, v2)
    result = max(-10, min(30, int(round(result))))
    toks = ["(", "let", "[", var1, _num_tok(v1), var2, _num_tok(v2), "]",
            "(", op, var1, var2, ")", ")"]
    return toks, [_num_tok(result)]


def _gen_hof_arithmetic(rng):
    """Family F: arithmetic applied to a HOF result.
    (+ (count [1 2 3]) 2) → 5
    (inc (first [7 3]))   → 8
    """
    choice = rng.random()
    if choice < 0.35:
        # (inc/dec (first/last [...])) or (abs (first [...]))
        n = rng.randint(2, 5)
        elems = [rng.randint(-5, 12) for _ in range(n)]
        acc = rng.choice(["first", "last"])
        inner_val = elems[0] if acc == "first" else elems[-1]
        outer_op = rng.choice(["inc", "dec", "abs"])
        if outer_op == "inc":
            result = inner_val + 1
        elif outer_op == "dec":
            result = inner_val - 1
        else:
            result = abs(inner_val)
        toks = ["(", outer_op, "(", acc, "["] + [_num_tok(e) for e in elems] + ["]", ")", ")"]
    elif choice < 0.7:
        # (+ (count [...]) k)
        n = rng.randint(1, 5)
        elems = [rng.randint(0, 10) for _ in range(n)]
        k = rng.randint(-3, 8)
        outer_op = rng.choice(["+", "-", "*"])
        if outer_op == "+":
            result = n + k
        elif outer_op == "-":
            result = n - k
        else:
            result = n * k
        toks = ["(", outer_op, "(", "count", "["] + [_num_tok(e) for e in elems] + ["]", ")", _num_tok(k), ")"]
    else:
        # (+ (reduce + [...]) k)
        n = rng.randint(2, 4)
        elems = [rng.randint(-3, 8) for _ in range(n)]
        red_val = sum(elems)
        k = rng.randint(-3, 5)
        outer_op = rng.choice(["+", "-"])
        if outer_op == "+":
            result = red_val + k
        else:
            result = red_val - k
        toks = ["(", outer_op, "(", "reduce", "+", "["] + [_num_tok(e) for e in elems] + ["]", ")", _num_tok(k), ")"]
    result = max(-10, min(30, int(round(result))))
    return toks, [_num_tok(result)]


def _gen_let_cond(rng):
    """Family G: let binding + conditional body.
    (let [x 5] (if (pos? x) 1 0)) → 1
    """
    var = rng.choice(["x", "y", "n", "a"])
    v = rng.randint(-8, 12)
    then_v = rng.randint(1, 10)
    else_v = rng.randint(-5, 0)
    # optional: use an arithmetic condition
    if rng.random() < 0.5:
        # (if (pos? var) then else)
        cond_result = v > 0
        body_toks = ["(", "if", "(", "pos?", var, ")", _num_tok(then_v), _num_tok(else_v), ")"]
    else:
        # (if (pos? (op var k)) then else)
        k = rng.randint(-5, 8)
        op = rng.choice(["+", "-"])
        computed = (v + k) if op == "+" else (v - k)
        cond_result = computed > 0
        body_toks = ["(", "if", "(", "pos?", "(", op, var, _num_tok(k), ")", ")", _num_tok(then_v), _num_tok(else_v), ")"]
    result = then_v if cond_result else else_v
    toks = ["(", "let", "[", var, _num_tok(v), "]"] + body_toks + [")"]
    return toks, [_num_tok(result)]


def _gen_deep_arithmetic(rng):
    """Family H: depth-3 arithmetic tree.
    (+ (+ 2 3) (* 4 1)) → 9
    (+  (- a b)  (mod c d)) → ...
    """
    ops = ["+", "-", "*", "max", "min"]

    def eval_op(op, a, b):
        if op == "+":
            return a + b
        elif op == "-":
            return a - b
        elif op == "*":
            return a * b
        elif op == "max":
            return max(a, b)
        elif op == "min":
            return min(a, b)
        return a

    def rand_n():
        return rng.randint(-5, 10)

    op_root = rng.choice(["+", "-", "max", "min"])
    op_l = rng.choice(ops)
    op_r = rng.choice(ops)
    a, b = rand_n(), rand_n()
    c, d = rand_n(), rand_n()
    lv = eval_op(op_l, a, b)
    rv = eval_op(op_r, c, d)
    result = eval_op(op_root, lv, rv)
    result = max(-10, min(30, int(round(result))))
    toks = ["(", op_root,
            "(", op_l, _num_tok(a), _num_tok(b), ")",
            "(", op_r, _num_tok(c), _num_tok(d), ")",
            ")"]
    return toks, [_num_tok(result)]


def _gen_product(rng):
    """Family I: multiplication chains where result is syntactically distant.
    (reduce * [2 3 4]) → 24 — product absent from token list.
    (* 3 (* 2 4)) → 24
    (let [x 4] (* x x)) → 16
    """
    choice = rng.random()
    if choice < 0.4:
        # (reduce * [a b c]) — product unlikely to appear as literal
        n = rng.randint(2, 3)
        elems = [rng.randint(2, 5) for _ in range(n)]
        result = 1
        for e in elems:
            result *= e
        result = max(-10, min(30, result))
        toks = ["(", "reduce", "*", "["] + [_num_tok(e) for e in elems] + ["]", ")"]
        return toks, [_num_tok(result)]
    elif choice < 0.7:
        # (* a (* b c)) — nested multiply
        a = rng.randint(2, 5)
        b = rng.randint(2, 4)
        c = rng.randint(1, 3)
        result = max(-10, min(30, a * b * c))
        toks = ["(", "*", _num_tok(a), "(", "*", _num_tok(b), _num_tok(c), ")", ")"]
        return toks, [_num_tok(result)]
    else:
        # (let [x a] (* x x)) — x squared; result not in expression
        a = rng.randint(2, 5)
        result = max(-10, min(30, a * a))
        toks = ["(", "let", "[", "x", _num_tok(a), "]", "(", "*", "x", "x", ")", ")"]
        return toks, [_num_tok(result)]


def _gen_seq_let(rng):
    """Family J: sequential let where second binding uses first.
    (let [a 3 b (* a 4)] (+ b 2)) → 14 — result absent from tokens.
    """
    var1 = rng.choice(["a", "n", "x", "y"])
    var2 = rng.choice([v for v in ["a", "b", "n", "x", "y", "z"] if v != var1])
    v1 = rng.randint(2, 5)
    op1 = rng.choice(["+", "*"])
    c1 = rng.randint(2, 4)

    v2 = (v1 + c1) if op1 == "+" else (v1 * c1)
    v2 = max(-10, min(30, v2))

    body_op = rng.choice(["+", "-", "*", "inc", "dec"])
    c2 = rng.randint(1, 4)

    if body_op == "inc":
        result = v2 + 1
        body_toks = ["(", "inc", var2, ")"]
    elif body_op == "dec":
        result = v2 - 1
        body_toks = ["(", "dec", var2, ")"]
    elif body_op == "+":
        result = v2 + c2
        body_toks = ["(", "+", var2, _num_tok(c2), ")"]
    elif body_op == "-":
        result = v2 - c2
        body_toks = ["(", "-", var2, _num_tok(c2), ")"]
    else:  # *
        result = v2 * c2
        body_toks = ["(", "*", var2, _num_tok(c2), ")"]

    result = max(-10, min(30, int(round(result))))
    toks = (["(", "let", "[", var1, _num_tok(v1), var2,
             "(", op1, var1, _num_tok(c1), ")", "]"]
            + body_toks + [")"])
    return toks, [_num_tok(result)]


def _gen_hof_cross(rng):
    """Family K: arithmetic applied to a HOF aggregate — result far from inputs.
    (* (reduce + [1 2 3]) 2) → 12
    (mod (reduce + [3 5 7]) 4) → 3
    """
    if rng.random() < 0.5:
        # (* (reduce + [...]) k) — sum then multiply
        n = rng.randint(2, 3)
        elems = [rng.randint(1, 4) for _ in range(n)]
        total = sum(elems)
        k = rng.randint(2, 3)
        result = max(-10, min(30, total * k))
        toks = (["(", "*", "(", "reduce", "+", "["]
                + [_num_tok(e) for e in elems]
                + ["]", ")", _num_tok(k), ")"])
        return toks, [_num_tok(result)]
    else:
        # (mod (reduce + [...]) k) — mod creates surprising small result
        n = rng.randint(2, 4)
        elems = [rng.randint(1, 8) for _ in range(n)]
        total = sum(elems)
        k = rng.randint(3, 7)
        result = _safe_mod(total, k)
        toks = (["(", "mod", "(", "reduce", "+", "["]
                + [_num_tok(e) for e in elems]
                + ["]", ")", _num_tok(k), ")"])
        return toks, [_num_tok(result)]


def _gen_threading(rng):
    """Family L: threading macro. (-> start (op arg) ...) → result"""
    start = rng.randint(-5, 10)
    n_steps = rng.randint(1, 6)  # up to 6 steps (was 5)
    toks = ["(", "->", _num_tok(start)]
    result = start
    for _ in range(n_steps):
        op = rng.choice(["inc", "dec", "+", "-", "*"])  # added *
        if op == "inc":
            result += 1
            toks += ["(", "inc", ")"]
        elif op == "dec":
            result -= 1
            toks += ["(", "dec", ")"]
        elif op == "*":
            k = rng.randint(2, 3)  # small factor to stay in range
            result = result * k
            toks += ["(", "*", _num_tok(k), ")"]
        else:
            k = rng.randint(1, 5)
            result = result + k if op == "+" else result - k
            toks += ["(", op, _num_tok(k), ")"]
        result = max(-10, min(30, result))  # clip each step
    toks.append(")")
    result = max(-10, min(30, result))
    return toks, [_num_tok(result)]


def _gen_when(rng):
    """Family M: when form. (when (pos? x) body) → body or nil"""
    x = rng.randint(-8, 10)
    body_val = rng.randint(-5, 15)
    toks = ["(", "when", "(", "pos?", _num_tok(x), ")", _num_tok(body_val), ")"]
    result_toks = [_num_tok(body_val)] if x > 0 else ["nil"]
    return toks, result_toks


def _gen_equality(rng):
    """Family N: equality check. (= a b) → true/false"""
    choice = rng.random()
    if choice < 0.4:
        a = rng.randint(-5, 10)
        b = a  # equal
    else:
        a = rng.randint(-5, 10)
        b = rng.randint(-5, 10)
    toks = ["(", "=", _num_tok(a), _num_tok(b), ")"]
    return toks, ["true" if a == b else "false"]


def _gen_not(rng):
    """Family O: logical not. (not (pos? x)) → true/false"""
    x = rng.randint(-8, 10)
    toks = ["(", "not", "(", "pos?", _num_tok(x), ")", ")"]
    return toks, ["true" if x <= 0 else "false"]


def _gen_map_fn(rng):
    """Family P: map with anonymous fn — returns actual mapped vector.
    (map (fn [x] (inc x)) [1 2 3]) → [2 3 4]
    Result is a vector: ["[", "2", "3", "4", "]"] — at most 4 elements.
    """
    op = rng.choice(["inc", "dec", "abs", "+", "-", "*"])
    n = rng.randint(2, 4)  # cap at 4 so result fits MAX_RESULT_LEN=8
    elems = [rng.randint(-3, 8) for _ in range(n)]
    if op in ["inc", "dec", "abs"]:
        body_toks = ["(", op, "x", ")"]
        if op == "inc":   mapped = [e + 1 for e in elems]
        elif op == "dec": mapped = [e - 1 for e in elems]
        else:             mapped = [abs(e) for e in elems]
    else:
        k = rng.randint(1, 4)
        body_toks = ["(", op, "x", _num_tok(k), ")"]
        if op == "+":   mapped = [e + k for e in elems]
        elif op == "-": mapped = [e - k for e in elems]
        else:           mapped = [e * k for e in elems]
    mapped = [max(-10, min(30, m)) for m in mapped]
    toks = (["(", "map", "(", "fn", "[", "x", "]"] + body_toks +
            [")", "["] + [_num_tok(e) for e in elems] + ["]", ")"])
    result_toks = ["["] + [_num_tok(m) for m in mapped] + ["]"]
    return toks, result_toks


def _gen_filter_fn(rng):
    """Family Q: filter with anonymous fn — returns actual filtered vector.
    (filter (fn [x] (pos? x)) [1 -2 3 -4]) → [1 3]
    Result is a vector (possibly empty []).  At most 4 input elements.
    """
    n = rng.randint(2, 4)  # cap input so result fits MAX_RESULT_LEN=8
    elems = [rng.randint(-4, 8) for _ in range(n)]
    if rng.random() < 0.6:
        pred_toks = ["(", "pos?", "x", ")"]
        filtered = [e for e in elems if e > 0]
    else:
        op = rng.choice(["+", "-"])
        k = rng.randint(1, 4)
        pred_toks = ["(", "pos?", "(", op, "x", _num_tok(k), ")", ")"]
        filtered = [e for e in elems if (e + k if op == "+" else e - k) > 0]
    filtered = [max(-10, min(30, f)) for f in filtered]
    toks = (["(", "filter", "(", "fn", "[", "x", "]"] + pred_toks +
            [")", "["] + [_num_tok(e) for e in elems] + ["]", ")"])
    result_toks = ["["] + [_num_tok(f) for f in filtered] + ["]"]
    return toks, result_toks


def _gen_reduce_fn(rng):
    """Family R: reduce with anonymous fn. (reduce (fn [a b] op) init [elems]) — 17-20 tokens."""
    op = rng.choice(["+", "*", "max", "min"])
    n = rng.randint(2, 4)
    elems = [rng.randint(1, 6) for _ in range(n)]
    init_val = 0 if op == "+" else 1 if op == "*" else 0 if op == "max" else 30
    result = init_val
    for e in elems:
        if op == "+":   result += e
        elif op == "*": result *= e
        elif op == "max": result = max(result, e)
        elif op == "min": result = min(result, e)
    result = max(-10, min(30, result))
    toks = (["(", "reduce", "(", "fn", "[", "a", "b", "]", "(", op, "a", "b", ")", ")",
             _num_tok(init_val), "["] + [_num_tok(e) for e in elems] + ["]", ")"])
    return toks, [_num_tok(result)]


def _gen_nested_let(rng):
    """Family S: nested let. (let [x v] (let [y (op x k)] (body y))) — 17-20 tokens."""
    var1 = rng.choice(["x", "a", "n"])
    var2 = rng.choice(["y", "b", "m"])
    v1 = rng.randint(-3, 8)
    op1 = rng.choice(["+", "-", "*", "inc", "dec"])
    if op1 in ["inc", "dec"]:
        v2 = v1 + (1 if op1 == "inc" else -1)
        bind2_toks = ["(", op1, var1, ")"]
    else:
        k1 = rng.randint(1, 4)
        v2 = v1 + k1 if op1 == "+" else v1 - k1 if op1 == "-" else v1 * k1
        v2 = max(-10, min(30, v2))
        bind2_toks = ["(", op1, var1, _num_tok(k1), ")"]
    op2 = rng.choice(["+", "-", "*", "inc", "dec"])
    if op2 in ["inc", "dec"]:
        result = v2 + (1 if op2 == "inc" else -1)
        body_toks = ["(", op2, var2, ")"]
    else:
        k2 = rng.randint(1, 4)
        result = v2 + k2 if op2 == "+" else v2 - k2 if op2 == "-" else v2 * k2
        result = max(-10, min(30, result))
        body_toks = ["(", op2, var2, _num_tok(k2), ")"]
    toks = (["(", "let", "[", var1, _num_tok(v1), "]",
             "(", "let", "[", var2] + bind2_toks + ["]"] + body_toks + [")", ")"])
    return toks, [_num_tok(result)]


def _gen_triple_let(rng):
    """Family T: three-binding let. (let [x v1 y v2 z (op x y)] body) — 16-19 tokens."""
    var1, var2, var3 = rng.sample(["x", "y", "z", "a", "b", "n"], 3)
    v1 = rng.randint(-3, 8)
    v2 = rng.randint(-3, 8)
    op_b = rng.choice(["+", "-", "*", "max", "min"])
    v3 = (v1 + v2 if op_b == "+" else v1 - v2 if op_b == "-" else
          v1 * v2 if op_b == "*" else max(v1, v2) if op_b == "max" else min(v1, v2))
    v3 = max(-10, min(30, v3))
    op_body = rng.choice(["+", "-", "inc", "dec"])
    if op_body in ["inc", "dec"]:
        result = v3 + (1 if op_body == "inc" else -1)
        body_toks = ["(", op_body, var3, ")"]
    else:
        k = rng.randint(1, 4)
        result = v3 + k if op_body == "+" else v3 - k
        body_toks = ["(", op_body, var3, _num_tok(k), ")"]
    result = max(-10, min(30, result))
    toks = (["(", "let", "[",
             var1, _num_tok(v1), var2, _num_tok(v2),
             var3, "(", op_b, var1, var2, ")",
             "]"] + body_toks + [")"])
    return toks, [_num_tok(result)]


def _gen_cond_form(rng):
    """Family U: cond with two predicates. (let [x v] (cond p1 e1 p2 e2 true e3)) — 22-25 tokens."""
    var = rng.choice(["x", "n", "a"])
    v = rng.randint(-8, 10)
    e1 = rng.randint(2, 10)
    e2 = rng.randint(-3, 2)
    e3 = rng.randint(-8, -1)
    result = e1 if v > 0 else e2 if v == 0 else e3
    result = max(-10, min(30, result))
    toks = ["(", "let", "[", var, _num_tok(v), "]",
            "(", "cond",
            "(", "pos?", var, ")", _num_tok(e1),
            "(", "pos?", "(", "inc", var, ")", ")", _num_tok(e2),
            "true", _num_tok(e3),
            ")", ")"]
    return toks, [_num_tok(result)]


def _gen_let_hof(rng):
    """Family V: let-bind a HOF result then apply arithmetic.
    (let [x (reduce + [2 3 4])] (* x 2)) → 18
    (let [x (count [1 2 3 4])] (+ x 5)) → 9
    """
    var = rng.choice(["x", "a", "n"])
    if rng.random() < 0.55:
        # (let [var (reduce op [elems])] (arith var k))
        n = rng.randint(2, 4)
        elems = [rng.randint(1, 5) for _ in range(n)]
        red_op = rng.choice(["+", "max", "min"])
        if red_op == "+":    hof_val = sum(elems)
        elif red_op == "max": hof_val = max(elems)
        else:                 hof_val = min(elems)
        hof_toks = ["(", "reduce", red_op, "["] + [_num_tok(e) for e in elems] + ["]", ")"]
    else:
        # (let [var (count [elems])] (arith var k))
        n = rng.randint(1, 6)
        elems = [rng.randint(0, 5) for _ in range(n)]
        hof_val = n
        hof_toks = ["(", "count", "["] + [_num_tok(e) for e in elems] + ["]", ")"]
    body_op = rng.choice(["+", "-", "*", "inc", "dec"])
    if body_op in ["inc", "dec"]:
        result = hof_val + (1 if body_op == "inc" else -1)
        body_toks = ["(", body_op, var, ")"]
    else:
        k = rng.randint(1, 4)
        if body_op == "+":   result = hof_val + k
        elif body_op == "-": result = hof_val - k
        else:                result = hof_val * k
        body_toks = ["(", body_op, var, _num_tok(k), ")"]
    result = max(-10, min(30, int(round(result))))
    toks = ["(", "let", "[", var] + hof_toks + ["]"] + body_toks + [")"]
    return toks, [_num_tok(result)]


def _gen_let_cond_composed(rng):
    """Family W: two-variable let with composed conditional body.
    (let [x 5 y 3] (if (pos? (- x y)) (* x y) 0)) → 15
    """
    var1 = rng.choice(["x", "a"])
    var2 = rng.choice(["y", "b"])
    v1 = rng.randint(1, 8)
    v2 = rng.randint(1, 8)
    cond_op = rng.choice(["+", "-"])
    cond_val = v1 + v2 if cond_op == "+" else v1 - v2
    cond_result = cond_val > 0
    then_op = rng.choice(["*", "+", "-", "max", "min"])
    if then_op == "*":     then_val = v1 * v2
    elif then_op == "+":   then_val = v1 + v2
    elif then_op == "-":   then_val = v1 - v2
    elif then_op == "max": then_val = max(v1, v2)
    else:                  then_val = min(v1, v2)
    then_val = max(-10, min(30, then_val))
    else_val = rng.randint(-5, 5)
    result = then_val if cond_result else else_val
    result = max(-10, min(30, int(round(result))))
    toks = ["(", "let", "[", var1, _num_tok(v1), var2, _num_tok(v2), "]",
            "(", "if", "(", "pos?", "(", cond_op, var1, var2, ")", ")",
            "(", then_op, var1, var2, ")", _num_tok(else_val), ")"]
    return toks, [_num_tok(result)]


def _gen_depth4_arith(rng):
    """Family X: depth-4 arithmetic tree.
    (+ (+ (* 2 3) 4) (- 10 (+ 1 2))) → 11
    """
    ops = ["+", "-", "*", "max", "min"]

    def eval_op(op, a, b):
        if op == "+":     return a + b
        elif op == "-":   return a - b
        elif op == "*":   return a * b
        elif op == "max": return max(a, b)
        else:             return min(a, b)

    def rand_n():
        return rng.randint(-4, 8)

    op_root = rng.choice(["+", "-", "max", "min"])
    # Left branch: (op_l (op_ll a b) c)
    op_l = rng.choice(ops)
    op_ll = rng.choice(["+", "-", "*"])
    a, b, c = rand_n(), rand_n(), rand_n()
    ll_val = eval_op(op_ll, a, b)
    l_val  = eval_op(op_l,  ll_val, c)
    # Right branch: (op_r d (op_rr e f))
    op_r  = rng.choice(["+", "-"])
    op_rr = rng.choice(["+", "-"])
    d, e, f = rand_n(), rand_n(), rand_n()
    rr_val = eval_op(op_rr, e, f)
    r_val  = eval_op(op_r,  d, rr_val)
    result = eval_op(op_root, l_val, r_val)
    result = max(-10, min(30, int(round(result))))
    toks = ["(", op_root,
            "(", op_l, "(", op_ll, _num_tok(a), _num_tok(b), ")", _num_tok(c), ")",
            "(", op_r, _num_tok(d), "(", op_rr, _num_tok(e), _num_tok(f), ")", ")",
            ")"]
    return toks, [_num_tok(result)]


def _gen_var_shadow(rng):
    """Family DD: variable shadowing — inner let rebinds the same name.
    (let [x 3] (let [x (+ x 1)] x))         → 4
    (let [x 5] (let [x (* x 2)] (inc x)))    → 11
    Model must track which binding of x is active in each scope.
    Paren depth 4-5.  ~19-24 tokens.
    """
    var = rng.choice(["x", "n", "a"])
    v1 = rng.randint(1, 8)

    # Inner binding: rebind same var using outer var's value
    op1 = rng.choice(["+", "-", "*", "inc", "dec"])
    if op1 in ["inc", "dec"]:
        v2 = v1 + (1 if op1 == "inc" else -1)
        bind_toks = ["(", op1, var, ")"]
    else:
        k1 = rng.randint(1, 4)
        v2 = v1 + k1 if op1 == "+" else v1 - k1 if op1 == "-" else v1 * k1
        bind_toks = ["(", op1, var, _num_tok(k1), ")"]
    v2 = max(-10, min(30, v2))

    # Body: use the shadowed var (possibly with another op)
    if rng.random() < 0.35:
        result = v2
        body_toks = [var]  # just return the inner var
    else:
        op2 = rng.choice(["+", "-", "*", "inc", "dec"])
        if op2 in ["inc", "dec"]:
            result = v2 + (1 if op2 == "inc" else -1)
            body_toks = ["(", op2, var, ")"]
        else:
            k2 = rng.randint(1, 4)
            result = v2 + k2 if op2 == "+" else v2 - k2 if op2 == "-" else v2 * k2
            body_toks = ["(", op2, var, _num_tok(k2), ")"]

    result = max(-10, min(30, int(round(result))))
    toks = (["(", "let", "[", var, _num_tok(v1), "]",
             "(", "let", "[", var] + bind_toks + ["]"] + body_toks + [")", ")"])
    return toks, [_num_tok(result)]


def _gen_seq_let3(rng):
    """Family EE: three-binding sequential let, each binding uses the previous var.
    (let [a 3 b (* a 2) c (+ b 1)] (inc c)) → 8
    Three-step dependent computation chain; result never appears as a literal.
    ~21-25 tokens.
    """
    var1, var2, var3 = rng.sample(["a", "b", "n", "x", "y", "z"], 3)
    v1 = rng.randint(2, 5)

    # Second binding: f(var1)
    op1 = rng.choice(["+", "*"])
    k1 = rng.randint(2, 3)
    v2 = v1 + k1 if op1 == "+" else v1 * k1
    v2 = max(-10, min(30, v2))

    # Third binding: g(var2)
    op2 = rng.choice(["+", "-", "*"])
    k2 = rng.randint(1, 3)
    v3 = v2 + k2 if op2 == "+" else v2 - k2 if op2 == "-" else v2 * k2
    v3 = max(-10, min(30, v3))

    # Body: h(var3)
    body_op = rng.choice(["+", "-", "inc", "dec"])
    if body_op in ["inc", "dec"]:
        result = v3 + (1 if body_op == "inc" else -1)
        body_toks = ["(", body_op, var3, ")"]
    else:
        k3 = rng.randint(1, 3)
        result = v3 + k3 if body_op == "+" else v3 - k3
        body_toks = ["(", body_op, var3, _num_tok(k3), ")"]

    result = max(-10, min(30, int(round(result))))
    toks = (["(", "let", "[",
             var1, _num_tok(v1),
             var2, "(", op1, var1, _num_tok(k1), ")",
             var3, "(", op2, var2, _num_tok(k2), ")",
             "]"] + body_toks + [")"])
    return toks, [_num_tok(result)]


def _gen_thread_last(rng):
    """Family FF: ->> threading — value inserted as LAST argument.
    (->> 5 (- 3)) → (- 3 5) = -2   (differs from -> where result = (- 5 3) = 2)
    (->> 2 (* 3) (+ 4)) → (* 3 2)=6, (+ 4 6)=10
    Tests whether model distinguishes ->> from -> via argument order.
    ~7-19 tokens.
    """
    start = rng.randint(-3, 8)
    n_steps = rng.randint(1, 4)
    toks = ["(", "->>", _num_tok(start)]
    result = start
    for _ in range(n_steps):
        op = rng.choice(["inc", "dec", "+", "-", "*"])
        if op == "inc":
            result += 1
            toks += ["(", "inc", ")"]
        elif op == "dec":
            result -= 1
            toks += ["(", "dec", ")"]
        elif op == "*":
            k = rng.randint(2, 3)
            result = k * result  # (* k result) — same numerically as (* result k)
            toks += ["(", "*", _num_tok(k), ")"]
        else:
            k = rng.randint(1, 5)
            # ->> inserts value as LAST arg: (->> v (- k)) → (- k v) = k - v
            result = (k + result) if op == "+" else (k - result)
            toks += ["(", op, _num_tok(k), ")"]
        result = max(-10, min(30, result))
    toks.append(")")
    result = max(-10, min(30, result))
    return toks, [_num_tok(result)]


def _gen_let_threading(rng):
    """Family Y: threading macro result bound in let, then used in body.
    (let [x (-> v (+ 2) (* 3))] (+ x 1)) — cross-family composition.
    Paren depth: 4 at threading steps inside let binding.
    """
    var = rng.choice(["x", "a", "n"])
    start = rng.randint(-3, 8)
    n_steps = rng.randint(2, 4)
    thread_val = start
    thread_toks = ["(", "->", _num_tok(start)]
    for _ in range(n_steps):
        op = rng.choice(["inc", "dec", "+", "-", "*"])
        if op == "inc":
            thread_val += 1
            thread_toks += ["(", "inc", ")"]
        elif op == "dec":
            thread_val -= 1
            thread_toks += ["(", "dec", ")"]
        elif op == "*":
            k = rng.randint(2, 3)
            thread_val = thread_val * k
            thread_toks += ["(", "*", _num_tok(k), ")"]
        else:
            k = rng.randint(1, 4)
            thread_val = thread_val + k if op == "+" else thread_val - k
            thread_toks += ["(", op, _num_tok(k), ")"]
        thread_val = max(-10, min(30, thread_val))
    thread_toks.append(")")
    body_op = rng.choice(["+", "-", "*", "inc", "dec"])
    if body_op in ["inc", "dec"]:
        result = thread_val + (1 if body_op == "inc" else -1)
        body_toks = ["(", body_op, var, ")"]
    else:
        k = rng.randint(1, 4)
        if body_op == "+":   result = thread_val + k
        elif body_op == "-": result = thread_val - k
        else:                result = thread_val * k
        body_toks = ["(", body_op, var, _num_tok(k), ")"]
    result = max(-10, min(30, int(round(result))))
    toks = ["(", "let", "[", var] + thread_toks + ["]"] + body_toks + [")"]
    return toks, [_num_tok(result)]


def _gen_threading_cond(rng):
    """Family Z: threading chain used as conditional test.
    (if (pos? (-> v (+ 2) (* 3))) then else) — threading inside if.
    Cross-family: threading feeds into pos? predicate.
    """
    start = rng.randint(-5, 8)
    n_steps = rng.randint(2, 3)
    thread_val = start
    thread_toks = ["(", "->", _num_tok(start)]
    for _ in range(n_steps):
        op = rng.choice(["inc", "+", "-"])
        if op == "inc":
            thread_val += 1
            thread_toks += ["(", "inc", ")"]
        else:
            k = rng.randint(1, 5)
            thread_val = thread_val + k if op == "+" else thread_val - k
            thread_toks += ["(", op, _num_tok(k), ")"]
        thread_val = max(-10, min(30, thread_val))
    thread_toks.append(")")
    cond_result = thread_val > 0
    then_val = rng.randint(1, 12)
    else_val = rng.randint(-8, 0)
    result = then_val if cond_result else else_val
    result = max(-10, min(30, result))
    toks = (["(", "if", "(", "pos?"] + thread_toks +
            [")", _num_tok(then_val), _num_tok(else_val), ")"])
    return toks, [_num_tok(result)]


def _gen_triple_nested_let(rng):
    """Family AA: three nested let forms, each binding depends on outer var.
    (let [x v1] (let [y (op1 x k1)] (let [z (op2 y k2)] (op3 z k3))))
    Paren depth 5 at the innermost binding expression.  ~26-28 tokens.
    """
    var1, var2, var3 = "x", "y", "z"
    v1 = rng.randint(1, 6)

    def pick_op_k(rng, prev_val):
        op = rng.choice(["+", "-", "*", "inc", "dec"])
        if op in ["inc", "dec"]:
            k = None
            nv = prev_val + (1 if op == "inc" else -1)
            toks = ["(", op, None, ")"]  # placeholder for var
        else:
            k = rng.randint(1, 3)
            if op == "+":   nv = prev_val + k
            elif op == "-": nv = prev_val - k
            else:           nv = prev_val * k
            toks = ["(", op, None, _num_tok(k), ")"]
        return op, k, max(-10, min(30, nv)), toks

    op1, k1, v2, btoks1 = pick_op_k(rng, v1)
    op2, k2, v3, btoks2 = pick_op_k(rng, v2)
    op3, k3, result, btoks3 = pick_op_k(rng, v3)

    def fill_var(toks, var):
        return [var if t is None else t for t in toks]

    bind1 = fill_var(btoks1, var1)   # (op1 x k1)
    bind2 = fill_var(btoks2, var2)   # (op2 y k2)
    body  = fill_var(btoks3, var3)   # (op3 z k3)

    result = max(-10, min(30, int(round(result))))
    toks = (["(", "let", "[", var1, _num_tok(v1), "]",
             "(", "let", "[", var2] + bind1 + ["]",
             "(", "let", "[", var3] + bind2 + ["]"] +
             body + [")", ")", ")"])
    return toks, [_num_tok(result)]


def _gen_let_dual_hof(rng):
    """Family BB: two-binding let, each binding a HOF result, combined in body.
    (let [a (reduce + [1 2 3]) b (count [4 5])] (op a b))
    Cross-family: HOF composition inside let.
    """
    var1 = rng.choice(["a", "x"])
    var2 = rng.choice(["b", "y"])

    # First binding: reduce op over small list
    n1 = rng.randint(2, 3)
    elems1 = [rng.randint(1, 4) for _ in range(n1)]
    red_op = rng.choice(["+", "max", "min"])
    if red_op == "+":     hof1 = sum(elems1)
    elif red_op == "max": hof1 = max(elems1)
    else:                 hof1 = min(elems1)
    hof1_toks = ["(", "reduce", red_op, "["] + [_num_tok(e) for e in elems1] + ["]", ")"]

    # Second binding: count of a short list
    n2 = rng.randint(1, 3)
    elems2 = [rng.randint(0, 5) for _ in range(n2)]
    hof2 = n2
    hof2_toks = ["(", "count", "["] + [_num_tok(e) for e in elems2] + ["]", ")"]

    # Body: arithmetic on the two HOF results
    body_op = rng.choice(["+", "-", "*", "max", "min"])
    if body_op == "+":     result = hof1 + hof2
    elif body_op == "-":   result = hof1 - hof2
    elif body_op == "*":   result = hof1 * hof2
    elif body_op == "max": result = max(hof1, hof2)
    else:                  result = min(hof1, hof2)
    result = max(-10, min(30, int(round(result))))
    body_toks = ["(", body_op, var1, var2, ")"]

    toks = (["(", "let", "[",
              var1] + hof1_toks + [
              var2] + hof2_toks + ["]"] + body_toks + [")"])
    return toks, [_num_tok(result)]


def _gen_vector_result(rng):
    """Family CC: expressions returning small vectors (multi-token results).
    (rest [a b c d]) → [b c d]   — result_toks = ["[", "b", "c", "d", "]"]
    (cons a [b c])   → [a b c]   — result_toks = ["[", "a", "b", "c", "]"]
    Larger result vocabulary: result is a sequence, not a scalar.
    """
    if rng.random() < 0.5:
        # (rest [a b c ...]) — drop first element, return rest as vector
        n = rng.randint(2, 4)  # 2-4 output elements (input has n+1)
        elems = [rng.randint(-3, 8) for _ in range(n + 1)]
        toks = ["(", "rest", "["] + [_num_tok(e) for e in elems] + ["]", ")"]
        result_toks = ["["] + [_num_tok(e) for e in elems[1:]] + ["]"]
    else:
        # (cons a [b c d]) — prepend element, return vector
        n = rng.randint(1, 3)  # 1-3 list elements (result has n+1 elements)
        elems = [rng.randint(-3, 8) for _ in range(n)]
        head = rng.randint(-3, 8)
        toks = ["(", "cons", _num_tok(head), "["] + [_num_tok(e) for e in elems] + ["]", ")"]
        result_toks = ["[", _num_tok(head)] + [_num_tok(e) for e in elems] + ["]"]
    return toks, result_toks


def _gen_expr_only(rng):
    """Return only expression tokens from a randomly chosen family."""
    r = rng.random()
    if r < 0.05:   toks, _ = _gen_arithmetic(rng)
    elif r < 0.07: toks, _ = _gen_let(rng)
    elif r < 0.10: toks, _ = _gen_hof(rng)
    elif r < 0.12: toks, _ = _gen_conditional(rng)
    elif r < 0.16: toks, _ = _gen_multi_let(rng)
    elif r < 0.18: toks, _ = _gen_hof_arithmetic(rng)
    elif r < 0.20: toks, _ = _gen_let_cond(rng)
    elif r < 0.22: toks, _ = _gen_deep_arithmetic(rng)
    elif r < 0.26: toks, _ = _gen_product(rng)
    elif r < 0.29: toks, _ = _gen_seq_let(rng)
    elif r < 0.31: toks, _ = _gen_hof_cross(rng)
    elif r < 0.35: toks, _ = _gen_threading(rng)
    elif r < 0.38: toks, _ = _gen_when(rng)
    elif r < 0.41: toks, _ = _gen_equality(rng)
    elif r < 0.44: toks, _ = _gen_not(rng)
    elif r < 0.48: toks, _ = _gen_map_fn(rng)
    elif r < 0.51: toks, _ = _gen_filter_fn(rng)
    elif r < 0.54: toks, _ = _gen_reduce_fn(rng)
    elif r < 0.57: toks, _ = _gen_nested_let(rng)
    elif r < 0.60: toks, _ = _gen_triple_let(rng)
    elif r < 0.63: toks, _ = _gen_cond_form(rng)
    elif r < 0.66: toks, _ = _gen_let_hof(rng)
    elif r < 0.69: toks, _ = _gen_let_cond_composed(rng)
    elif r < 0.72: toks, _ = _gen_depth4_arith(rng)
    elif r < 0.75: toks, _ = _gen_let_threading(rng)
    elif r < 0.78: toks, _ = _gen_threading_cond(rng)
    elif r < 0.81: toks, _ = _gen_triple_nested_let(rng)
    elif r < 0.84: toks, _ = _gen_let_dual_hof(rng)
    elif r < 0.88: toks, _ = _gen_vector_result(rng)
    elif r < 0.92: toks, _ = _gen_var_shadow(rng)
    elif r < 0.96: toks, _ = _gen_seq_let3(rng)
    else:           toks, _ = _gen_thread_last(rng)
    return toks


def _masked_pair(rng):
    """Mask a contiguous span; replace with a single opaque MASK token.

    The model cannot infer span length from the masked expression, forcing it
    to predict span content without positional hints.
    Span length distribution: 1→40%, 2→25%, 3→20%, 4→10%, 5→5%.
    """
    expr_toks = _gen_expr_only(rng)
    n = len(expr_toks)
    max_span = min(5, max(1, n - 1))  # always leave at least one token visible
    p = rng.random()
    if p < 0.40:   span_len = 1
    elif p < 0.65: span_len = 2
    elif p < 0.85: span_len = 3
    elif p < 0.95: span_len = 4
    else:          span_len = 5
    span_len = min(span_len, max_span)
    start = rng.randint(0, n - span_len)
    target_span = expr_toks[start:start + span_len]
    # Single opaque MASK regardless of span length
    masked_expr = expr_toks[:start] + [MASK_STR] + expr_toks[start + span_len:]
    return masked_expr, target_span, span_len


def generate_pair(rng):
    """Generate one actual (expression, result) execution pair.

    Returns (expr_tokens, result_tokens).  All 32 families are included;
    P (map-fn) and Q (filter-fn) now return real vector results.

    Distribution (32 families total):
      Simple  A arithmetic   5%    Complex  P map-fn        4%
              B let          2%             Q filter-fn     3%
              C hof          3%             R reduce-fn     3%
              D conditional  2%             S nested-let    3%
              E multi-let    4%             T triple-let    3%
              F hof-arith    2%             U cond-form     3%
              G let-cond     2%             V let-hof       3%
              H deep-arith   2%             W let-cond-comp 3%
              I product      4%             X depth4-arith  3%
              J seq-let      3%             Y let-thread    3%
              K hof-cross    2%             Z thread-cond   3%
              L threading    4%            AA triple-nlet   3%
              M when         3%            BB let-dual-hof  3%
              N equality     3%            CC vector-result 4%
              O not          3%            DD var-shadow    4%
                                           EE seq-let3      4%
                                           FF thread-last   4%
    """
    r = rng.random()
    if r < 0.05:   return _gen_arithmetic(rng)
    elif r < 0.07: return _gen_let(rng)
    elif r < 0.10: return _gen_hof(rng)
    elif r < 0.12: return _gen_conditional(rng)
    elif r < 0.16: return _gen_multi_let(rng)
    elif r < 0.18: return _gen_hof_arithmetic(rng)
    elif r < 0.20: return _gen_let_cond(rng)
    elif r < 0.22: return _gen_deep_arithmetic(rng)
    elif r < 0.26: return _gen_product(rng)
    elif r < 0.29: return _gen_seq_let(rng)
    elif r < 0.31: return _gen_hof_cross(rng)
    elif r < 0.35: return _gen_threading(rng)
    elif r < 0.38: return _gen_when(rng)
    elif r < 0.41: return _gen_equality(rng)
    elif r < 0.44: return _gen_not(rng)
    elif r < 0.48: return _gen_map_fn(rng)
    elif r < 0.51: return _gen_filter_fn(rng)
    elif r < 0.54: return _gen_reduce_fn(rng)
    elif r < 0.57: return _gen_nested_let(rng)
    elif r < 0.60: return _gen_triple_let(rng)
    elif r < 0.63: return _gen_cond_form(rng)
    elif r < 0.66: return _gen_let_hof(rng)
    elif r < 0.69: return _gen_let_cond_composed(rng)
    elif r < 0.72: return _gen_depth4_arith(rng)
    elif r < 0.75: return _gen_let_threading(rng)
    elif r < 0.78: return _gen_threading_cond(rng)
    elif r < 0.81: return _gen_triple_nested_let(rng)
    elif r < 0.84: return _gen_let_dual_hof(rng)
    elif r < 0.88: return _gen_vector_result(rng)
    elif r < 0.92: return _gen_var_shadow(rng)
    elif r < 0.96: return _gen_seq_let3(rng)
    else:          return _gen_thread_last(rng)


def _parse_int(t):
    """Return (True, int_val) if t is a parseable integer string, else (False, 0)."""
    try:
        return True, int(t)
    except (ValueError, TypeError):
        return False, 0


def encode_pair(vocab, expr_toks, res_toks, max_expr=MAX_EXPR_LEN, max_res=MAX_RESULT_LEN):
    """Encode token lists to padded integer arrays + numeric value arrays.

    Numbers are encoded as NUM_ID in the token array; their actual integer
    values are stored in parallel vals arrays for the sinusoidal embedding.

    Returns (expr_ids, expr_mask, expr_vals, res_ids, res_mask, res_vals).
    """
    def _encode(toks, max_len):
        ids  = [BOS_ID]
        vals = [0]
        for t in toks:
            is_int, iv = _parse_int(t)
            if is_int:
                ids.append(NUM_ID)
                vals.append(iv)
            elif t in vocab.token_to_id:
                ids.append(vocab.token_to_id[t])
                vals.append(0)
        ids  = ids[:max_len]
        vals = vals[:max_len]
        n    = len(ids)
        mask = [1] * n + [0] * (max_len - n)
        ids  = ids  + [PAD_ID] * (max_len - n)
        vals = vals + [0]      * (max_len - n)
        return (
            np.array(ids,  dtype=np.int32),
            np.array(mask, dtype=np.int32),
            np.array(vals, dtype=np.int32),
        )

    ei, em, ev = _encode(expr_toks, max_expr)
    ri, rm, rv = _encode(res_toks,  max_res)
    return ei, em, ev, ri, rm, rv


# ---------------------------------------------------------------------------
# Val pair cache
# ---------------------------------------------------------------------------

def load_val_pairs():
    """Load cached 2000 validation pairs. Generates and caches on first call.
    Returns (expr, expr_mask, expr_vals, res, res_mask, res_vals, mask_len).
    mask_len is the expression complexity bucket (1/2/3).

    Format: plain text, 2000 rows x 145 columns (int32).
    Columns: expr[40] | expr_mask[40] | expr_vals[40] | res[8] | res_mask[8] | res_vals[8] | mask_len[1]
    """
    if not os.path.exists(VAL_CACHE_PATH):
        _generate_val_cache()
    data = np.loadtxt(VAL_CACHE_PATH, dtype=np.int32)
    E = MAX_EXPR_LEN
    R = MAX_RESULT_LEN
    expr      = data[:, :E]
    expr_mask = data[:, E:2*E]
    expr_vals = data[:, 2*E:3*E]
    res       = data[:, 3*E:3*E+R]
    res_mask  = data[:, 3*E+R:3*E+2*R]
    res_vals  = data[:, 3*E+2*R:3*E+3*R]
    mask_len  = data[:, -1]
    return expr, expr_mask, expr_vals, res, res_mask, res_vals, mask_len


def _generate_val_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)
    rng = random.Random(12345)
    vocab = ClojureVocab()
    exprs, expr_masks, expr_vals_list = [], [], []
    ress,  res_masks,  res_vals_list  = [], [], []
    mask_lens = []
    while len(exprs) < N_VAL_PAIRS:
        expr_toks, result_toks = generate_pair(rng)
        e, em, ev, r, rm, rv = encode_pair(vocab, expr_toks, result_toks)
        exprs.append(e)
        expr_masks.append(em)
        expr_vals_list.append(ev)
        ress.append(r)
        res_masks.append(rm)
        res_vals_list.append(rv)
        # Expression complexity bucket: 1=short(≤7), 2=medium(≤14), 3=long(>14)
        n = len(expr_toks)
        mask_lens.append(1 if n <= 7 else 2 if n <= 14 else 3)
    data = np.concatenate([
        np.stack(exprs),
        np.stack(expr_masks),
        np.stack(expr_vals_list),
        np.stack(ress),
        np.stack(res_masks),
        np.stack(res_vals_list),
        np.array(mask_lens, dtype=np.int32)[:, None],
    ], axis=1)
    np.savetxt(VAL_CACHE_PATH, data, fmt="%d")
    print(f"2000 val pairs cached to {VAL_CACHE_PATH}")


# ---------------------------------------------------------------------------
# Training dataloader
# ---------------------------------------------------------------------------

def make_jepa_dataloader(vocab, batch_size):
    """Infinite generator of (expr, expr_mask, expr_vals, res, res_mask, res_vals) batches."""
    rng = random.Random()
    while True:
        exprs, expr_masks, expr_vals_list = [], [], []
        ress,  res_masks,  res_vals_list  = [], [], []
        for _ in range(batch_size):
            et, rt = generate_pair(rng)
            e, em, ev, r, rm, rv = encode_pair(vocab, et, rt)
            exprs.append(e)
            expr_masks.append(em)
            expr_vals_list.append(ev)
            ress.append(r)
            res_masks.append(rm)
            res_vals_list.append(rv)
        yield (
            np.stack(exprs),
            np.stack(expr_masks),
            np.stack(expr_vals_list),
            np.stack(ress),
            np.stack(res_masks),
            np.stack(res_vals_list),
        )


# ---------------------------------------------------------------------------
# Main: cache val pairs and print sample
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    vocab = ClojureVocab()
    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"Token sample: PAD={PAD_ID}, BOS={BOS_ID}, SEP={SEP_ID}, EOS={EOS_ID}")

    # Show sample expressions
    rng = random.Random(42)
    print("\nSample expressions:")
    for i in range(8):
        et, rt = generate_pair(rng)
        print(f"  expr: {' '.join(et):40s}  result: {' '.join(rt)}")

    # Cache val pairs
    if os.path.exists(VAL_CACHE_PATH):
        print(f"\nVal pairs already cached at {VAL_CACHE_PATH}")
    else:
        _generate_val_cache()

    # Verify encode
    test_expr = ["(", "+", "1", "2", ")"]
    encoded = vocab.encode(test_expr)
    print(f"\nEncode test '(+ 1 2)': {encoded}")

    expr, em, res, rm, mask_len = load_val_pairs()
    print(f"Val pairs loaded: expr shape={expr.shape}, res shape={res.shape}")
    print("Done.")
