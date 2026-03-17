#!/usr/bin/env python3
"""
Data preparation for nanoJEPA Clojure execution prediction.
Generates (expression, result) pairs purely in Python — no Clojure runtime needed.
FIXED: do not modify this file (autoresearch loop edits train_jepa.py only).

Usage: uv run prepare_jepa.py
"""

import os
import random
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Constants (fixed)
# ---------------------------------------------------------------------------

VOCAB_SIZE = 96
MAX_EXPR_LEN = 32
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
# Numbers -10 to 30 as tokens: 41 tokens (45-85)
_NUMBERS = [str(i) for i in range(-10, 31)]  # 41 tokens
# Booleans and nil (86-89)
_BOOLEANS = ["true", "false", "nil", "pos?"]  # 86-89 (pos? needed for conditionals)
# Extra Clojure tokens (90-95) — replaces reserved padding
_EXTRA = ["MASK", "->", "->>", "when", "not", "="]  # 90-95

_ALL_TOKENS = _SPECIAL + _SYNTAX + _OPERATORS + _HOF + _DICT + _FORMS + _VARS + _NUMBERS + _BOOLEANS + _EXTRA

assert len(_ALL_TOKENS) == VOCAB_SIZE, f"Expected {VOCAB_SIZE} tokens, got {len(_ALL_TOKENS)}"

_TOKEN_TO_ID = {tok: i for i, tok in enumerate(_ALL_TOKENS)}
_ID_TO_TOKEN = {i: tok for i, tok in enumerate(_ALL_TOKENS)}

PAD_ID = _TOKEN_TO_ID["PAD"]
BOS_ID = _TOKEN_TO_ID["BOS"]
SEP_ID = _TOKEN_TO_ID["SEP"]
EOS_ID = _TOKEN_TO_ID["EOS"]
MASK_ID = _TOKEN_TO_ID["MASK"]
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
    """Return string token for integer n (clipped to vocab range -10..30)."""
    n = max(-10, min(30, int(round(n))))
    return str(n)


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
    n_steps = rng.randint(1, 5)
    toks = ["(", "->", _num_tok(start)]
    result = start
    for _ in range(n_steps):
        op = rng.choice(["inc", "dec", "+", "-"])
        if op == "inc":
            result += 1
            toks += ["(", "inc", ")"]
        elif op == "dec":
            result -= 1
            toks += ["(", "dec", ")"]
        else:
            k = rng.randint(1, 5)
            result = result + k if op == "+" else result - k
            toks += ["(", op, _num_tok(k), ")"]
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
    """Family P: map with anonymous fn. (map (fn [x] body) [elems]) — 14-18 tokens."""
    op = rng.choice(["inc", "dec", "abs", "+", "-", "*"])
    n = rng.randint(2, 5)
    elems = [rng.randint(-3, 8) for _ in range(n)]
    if op in ["inc", "dec", "abs"]:
        body_toks = ["(", op, "x", ")"]
    else:
        k = rng.randint(1, 4)
        body_toks = ["(", op, "x", _num_tok(k), ")"]
    toks = (["(", "map", "(", "fn", "[", "x", "]"] + body_toks +
            [")", "["] + [_num_tok(e) for e in elems] + ["]", ")"])
    return toks, ["nil"]  # result is a collection; ignored in masked-JEPA


def _gen_filter_fn(rng):
    """Family Q: filter with anonymous fn. (filter (fn [x] pred) [elems]) — 16-21 tokens."""
    n = rng.randint(3, 6)
    elems = [rng.randint(-5, 8) for _ in range(n)]
    if rng.random() < 0.5:
        pred_toks = ["(", "pos?", "x", ")"]
    else:
        op = rng.choice(["+", "-"])
        k = rng.randint(1, 4)
        pred_toks = ["(", "pos?", "(", op, "x", _num_tok(k), ")", ")"]
    toks = (["(", "filter", "(", "fn", "[", "x", "]"] + pred_toks +
            [")", "["] + [_num_tok(e) for e in elems] + ["]", ")"])
    return toks, ["nil"]  # result is a collection; ignored in masked-JEPA


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


def _gen_expr_only(rng):
    """Return only expression tokens from a randomly chosen family.

    Short forms (A-O): ~60%   Long forms (P-U, threading): ~40%
      A arithmetic        12%    P map-fn             7%
      B let                8%    Q filter-fn          4%
      C hof                5%    R reduce-fn          3%
      D conditional        4%    S nested-let         5%
      E multi-let          6%    T triple-let         4%
      F hof-arithmetic     4%    U cond-form          3%
      G let-cond           3%    L threading (1-5)    5%
      H deep-arith         5%    M when               3%
      I product            6%    N equality           2%
      J seq-let            5%    O not                2%
      K hof-cross          3%
    """
    r = rng.random()
    if r < 0.12:   toks, _ = _gen_arithmetic(rng)
    elif r < 0.20: toks, _ = _gen_let(rng)
    elif r < 0.25: toks, _ = _gen_hof(rng)
    elif r < 0.29: toks, _ = _gen_conditional(rng)
    elif r < 0.35: toks, _ = _gen_multi_let(rng)
    elif r < 0.39: toks, _ = _gen_hof_arithmetic(rng)
    elif r < 0.42: toks, _ = _gen_let_cond(rng)
    elif r < 0.47: toks, _ = _gen_deep_arithmetic(rng)
    elif r < 0.53: toks, _ = _gen_product(rng)
    elif r < 0.58: toks, _ = _gen_seq_let(rng)
    elif r < 0.61: toks, _ = _gen_hof_cross(rng)
    elif r < 0.68: toks, _ = _gen_map_fn(rng)
    elif r < 0.72: toks, _ = _gen_filter_fn(rng)
    elif r < 0.75: toks, _ = _gen_reduce_fn(rng)
    elif r < 0.80: toks, _ = _gen_nested_let(rng)
    elif r < 0.84: toks, _ = _gen_triple_let(rng)
    elif r < 0.87: toks, _ = _gen_cond_form(rng)
    elif r < 0.92: toks, _ = _gen_threading(rng)
    elif r < 0.95: toks, _ = _gen_when(rng)
    elif r < 0.97: toks, _ = _gen_equality(rng)
    else:           toks, _ = _gen_not(rng)
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
    """Generate one masked (expr_tokens, span_tokens) pair via masked-JEPA.

    The expression has a contiguous span replaced by MASK tokens; the target
    is the original tokens at those positions.  Span length: 60% single-token,
    25% two-token, 15% three-token — single-token pairs are used for class
    accuracy (nearest-neighbour over the 96-token vocabulary).
    """
    masked, span, _ = _masked_pair(rng)
    return masked, span


def encode_pair(vocab, expr_toks, res_toks, max_expr=MAX_EXPR_LEN, max_res=MAX_RESULT_LEN):
    """Encode token lists to padded integer arrays."""
    expr_ids = [BOS_ID] + vocab.encode(expr_toks)
    expr_ids = expr_ids[:max_expr]
    expr_mask = [1] * len(expr_ids) + [0] * (max_expr - len(expr_ids))
    expr_ids = expr_ids + [PAD_ID] * (max_expr - len(expr_ids))

    res_ids = [BOS_ID] + vocab.encode(res_toks)
    res_ids = res_ids[:max_res]
    res_mask = [1] * len(res_ids) + [0] * (max_res - len(res_ids))
    res_ids = res_ids + [PAD_ID] * (max_res - len(res_ids))

    return (
        np.array(expr_ids, dtype=np.int32),
        np.array(expr_mask, dtype=np.int32),
        np.array(res_ids, dtype=np.int32),
        np.array(res_mask, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Val pair cache
# ---------------------------------------------------------------------------

def load_val_pairs():
    """Load cached 2000 validation pairs. Generates and caches on first call.
    Returns (expr, expr_mask, res, res_mask, mask_len).
    mask_len is the number of tokens masked per pair (1/2/3).

    Format: plain text, 2000 rows x 81 columns (int32).
    Columns: expr[32] | expr_mask[32] | res[8] | res_mask[8] | mask_len[1]
    """
    if not os.path.exists(VAL_CACHE_PATH):
        _generate_val_cache()
    data = np.loadtxt(VAL_CACHE_PATH, dtype=np.int32)
    expr      = data[:, :MAX_EXPR_LEN]
    expr_mask = data[:, MAX_EXPR_LEN:2 * MAX_EXPR_LEN]
    res       = data[:, 2 * MAX_EXPR_LEN:2 * MAX_EXPR_LEN + MAX_RESULT_LEN]
    res_mask  = data[:, 2 * MAX_EXPR_LEN + MAX_RESULT_LEN:2 * MAX_EXPR_LEN + 2 * MAX_RESULT_LEN]
    mask_len  = data[:, -1]
    return expr, expr_mask, res, res_mask, mask_len


def _generate_val_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)
    rng = random.Random(12345)
    vocab = ClojureVocab()
    exprs, expr_masks, ress, res_masks, mask_lens = [], [], [], [], []
    for _ in range(N_VAL_PAIRS):
        masked, span, span_len = _masked_pair(rng)
        e, em, r, rm = encode_pair(vocab, masked, span)
        exprs.append(e)
        expr_masks.append(em)
        ress.append(r)
        res_masks.append(rm)
        mask_lens.append(span_len)
    data = np.concatenate([
        np.stack(exprs),
        np.stack(expr_masks),
        np.stack(ress),
        np.stack(res_masks),
        np.array(mask_lens, dtype=np.int32)[:, None],
    ], axis=1)
    np.savetxt(VAL_CACHE_PATH, data, fmt="%d")
    print(f"2000 val pairs cached to {VAL_CACHE_PATH}")


# ---------------------------------------------------------------------------
# Training dataloader
# ---------------------------------------------------------------------------

def make_jepa_dataloader(vocab, batch_size):
    """Infinite generator of (expr, expr_mask, res, res_mask) batches."""
    rng = random.Random()
    while True:
        exprs, expr_masks, ress, res_masks = [], [], [], []
        for _ in range(batch_size):
            et, rt = generate_pair(rng)
            e, em, r, rm = encode_pair(vocab, et, rt)
            exprs.append(e)
            expr_masks.append(em)
            ress.append(r)
            res_masks.append(rm)
        yield (
            np.stack(exprs),
            np.stack(expr_masks),
            np.stack(ress),
            np.stack(res_masks),
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
