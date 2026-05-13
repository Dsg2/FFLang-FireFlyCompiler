"""
ffc/opt.py — FFLang AST optimizer

Passes (run in order):
  1. Dead kernel elimination  — remove kernel declarations never used in run/gpu()
  2. Constant folding         — evaluate BinOp/UnaryOp/Call on literals at compile time
  3. Algebraic simplification — x*1→x, x+0→x, x*0→0, x/1→x, --x→x, not not x→x

GCC -O2 already handles (2) and (3) in the generated C, so the main unique value
here is (1) — eliminating loadkernel() calls for kernels that were declared but
never actually referenced.  The expression passes still help by reducing noise in
the emitted C and catching obvious programmer errors early.
"""

from __future__ import annotations
import math
import operator as _op
from typing import Any, Set

from .ast_nodes import (
    Program, KernelDecl, FuncDef, Assignment, If, While, For,
    Return, Break, Continue, ExprStmt, RunKernel, GlobalStmt,
    BinOp, UnaryOp, Call, Index, Attr, ListLit,
    IntLit, FloatLit, StrLit, BoolLit, Ident,
)

# ── Constant folding ─────────────────────────────────────────

_FOLD_OPS = {
    '+':   _op.add,
    '-':   _op.sub,
    '*':   _op.mul,
    '//':  _op.floordiv,
    '%':   _op.mod,
    '==':  _op.eq,
    '!=':  _op.ne,
    '<':   _op.lt,
    '>':   _op.gt,
    '<=':  _op.le,
    '>=':  _op.ge,
    'and': lambda a, b: bool(a) and bool(b),
    'or':  lambda a, b: bool(a) or bool(b),
}

_FOLD_MATH = {
    'abs':     abs,
    'sqrt':    math.sqrt,
    'floor':   math.floor,
    'ceil':    math.ceil,
    'round':   round,
    'sin':     math.sin,
    'cos':     math.cos,
    'tan':     math.tan,
    'asin':    math.asin,
    'acos':    math.acos,
    'atan':    math.atan,
    'atan2':   math.atan2,
    'radians': math.radians,
    'degrees': math.degrees,
    'exp':     math.exp,
    'log':     math.log,
}


def _is_lit(n) -> bool:
    return isinstance(n, (IntLit, FloatLit, BoolLit))

def _lit_val(n):
    return n.value

def _make_num(value, prefer_float: bool, line: int):
    if prefer_float or isinstance(value, float):
        return FloatLit(float(value), line)
    if isinstance(value, bool):
        return BoolLit(value, line)
    return IntLit(int(value), line)


def fold_expr(node: Any) -> Any:
    """Recursively constant-fold and algebraically simplify an expression."""

    # ── Literals / leaves ────────────────────────────────────
    if isinstance(node, (IntLit, FloatLit, StrLit, BoolLit, Ident)):
        return node

    # ── Attribute access ─────────────────────────────────────
    if isinstance(node, Attr):
        return Attr(fold_expr(node.obj), node.attr, node.line)

    # ── Index ────────────────────────────────────────────────
    if isinstance(node, Index):
        return Index(fold_expr(node.obj),
                     [fold_expr(i) for i in node.indices],
                     node.line)

    # ── List literal ─────────────────────────────────────────
    if isinstance(node, ListLit):
        return ListLit([fold_expr(e) for e in node.elements], node.line)

    # ── Unary ────────────────────────────────────────────────
    if isinstance(node, UnaryOp):
        operand = fold_expr(node.operand)
        if node.op == '-':
            if isinstance(operand, IntLit):   return IntLit(-operand.value,   node.line)
            if isinstance(operand, FloatLit): return FloatLit(-operand.value, node.line)
            # --x → x  (double negation)
            if isinstance(operand, UnaryOp) and operand.op == '-':
                return operand.operand
        if node.op == 'not':
            if isinstance(operand, BoolLit): return BoolLit(not operand.value, node.line)
            # not not x → x (when x is not a literal)
            if isinstance(operand, UnaryOp) and operand.op == 'not':
                return operand.operand
        return UnaryOp(node.op, operand, node.line)

    # ── Binary ───────────────────────────────────────────────
    if isinstance(node, BinOp):
        l = fold_expr(node.left)
        r = fold_expr(node.right)
        op = node.op
        ln = node.line

        # Both sides are numeric literals → fold
        if _is_lit(l) and _is_lit(r):
            lv, rv = _lit_val(l), _lit_val(r)
            is_cmp = op in ('==', '!=', '<', '>', '<=', '>=', 'and', 'or')
            prefer_float = isinstance(l, FloatLit) or isinstance(r, FloatLit)
            try:
                if op == '/' and rv != 0:
                    return _make_num(lv / rv, True, ln)
                if op in _FOLD_OPS and op != '/' and rv != 0:
                    result = _FOLD_OPS[op](lv, rv)
                    if is_cmp:
                        return BoolLit(bool(result), ln)
                    return _make_num(result, prefer_float, ln)
            except (ZeroDivisionError, OverflowError, ValueError):
                pass  # leave un-folded

        # Algebraic identities
        if op == '*':
            if isinstance(r, (IntLit, FloatLit)) and r.value == 1: return l
            if isinstance(l, (IntLit, FloatLit)) and l.value == 1: return r
            if isinstance(r, (IntLit, FloatLit)) and r.value == 0: return r
            if isinstance(l, (IntLit, FloatLit)) and l.value == 0: return l

        if op == '+':
            if isinstance(r, (IntLit, FloatLit)) and r.value == 0: return l
            if isinstance(l, (IntLit, FloatLit)) and l.value == 0: return r

        if op == '-':
            if isinstance(r, (IntLit, FloatLit)) and r.value == 0: return l
            # x - x → 0  (same Ident node)
            if isinstance(l, Ident) and isinstance(r, Ident) and l.name == r.name:
                return IntLit(0, ln)

        if op == '/' and isinstance(r, (IntLit, FloatLit)) and r.value == 1:
            return l

        if op == '**':
            if isinstance(r, (IntLit, FloatLit)) and r.value == 1: return l
            if isinstance(r, (IntLit, FloatLit)) and r.value == 0:
                return IntLit(1, ln)

        return BinOp(l, op, r, ln)

    # ── Call ─────────────────────────────────────────────────
    if isinstance(node, Call):
        folded_args = [fold_expr(a) for a in node.args]

        # Fold pure math calls on all-literal args
        if isinstance(node.func, Ident):
            fname = node.func.name
            if fname in _FOLD_MATH and all(_is_lit(a) for a in folded_args):
                try:
                    result = _FOLD_MATH[fname](*[_lit_val(a) for a in folded_args])
                    return FloatLit(float(result), node.line)
                except (ValueError, ZeroDivisionError, OverflowError):
                    pass

            # int(literal) / float(literal) casts
            if fname == 'int' and len(folded_args) == 1 and _is_lit(folded_args[0]):
                return IntLit(int(_lit_val(folded_args[0])), node.line)
            if fname == 'float' and len(folded_args) == 1 and _is_lit(folded_args[0]):
                return FloatLit(float(_lit_val(folded_args[0])), node.line)

        return Call(node.func, folded_args, node.line)

    return node  # unknown — pass through


def fold_stmt(node: Any) -> Any:
    """Apply fold_expr to every expression inside a statement node."""
    if node is None:
        return None
    if isinstance(node, Assignment):
        return Assignment(fold_expr(node.target), fold_expr(node.value), node.line)
    if isinstance(node, ExprStmt):
        return ExprStmt(fold_expr(node.expr), node.line)
    if isinstance(node, Return):
        return Return(fold_expr(node.value) if node.value else None, node.line)
    if isinstance(node, If):
        return If(
            fold_expr(node.cond),
            [fold_stmt(s) for s in node.body],
            [(fold_expr(c), [fold_stmt(s) for s in b]) for c, b in node.elifs],
            [fold_stmt(s) for s in node.else_body] if node.else_body else None,
            node.line,
        )
    if isinstance(node, While):
        return While(fold_expr(node.cond), [fold_stmt(s) for s in node.body], node.line)
    if isinstance(node, For):
        return For(node.var, fold_expr(node.iter),
                   [fold_stmt(s) for s in node.body], node.line)
    if isinstance(node, FuncDef):
        return FuncDef(node.name, node.params,
                       [fold_stmt(s) for s in node.body],
                       node.param_types, node.line)
    if isinstance(node, RunKernel):
        return RunKernel(
            node.kernel,
            [fold_expr(a) for a in node.args],
            node.outputs,
            fold_expr(node.dispatch) if node.dispatch else None,
            node.line,
        )
    return node  # KernelDecl, Break, Continue, GlobalStmt — unchanged


# ── Dead kernel elimination ───────────────────────────────────

def _collect_kernel_refs(stmts) -> Set[str]:
    """Return the set of kernel names actually used in run / gpu() calls."""
    refs: Set[str] = set()

    def _expr(node):
        if isinstance(node, Call):
            if isinstance(node.func, Ident) and node.func.name == 'gpu':
                if node.args and isinstance(node.args[0], Ident):
                    refs.add(node.args[0].name)
            for a in node.args: _expr(a)
            if isinstance(node.func, Attr): _expr(node.func.obj)
        elif isinstance(node, BinOp):
            _expr(node.left); _expr(node.right)
        elif isinstance(node, UnaryOp):
            _expr(node.operand)
        elif isinstance(node, Index):
            _expr(node.obj)
            for i in node.indices: _expr(i)
        elif isinstance(node, ListLit):
            for e in node.elements: _expr(e)
        elif isinstance(node, Attr):
            _expr(node.obj)

    def _stmts(ss):
        for s in ss:
            if isinstance(s, RunKernel):
                refs.add(s.kernel)
            elif isinstance(s, Assignment):
                _expr(s.value)
            elif isinstance(s, ExprStmt):
                _expr(s.expr)
            elif isinstance(s, Return) and s.value:
                _expr(s.value)
            elif isinstance(s, If):
                _expr(s.cond); _stmts(s.body)
                for _, b in s.elifs: _stmts(b)
                if s.else_body: _stmts(s.else_body)
            elif isinstance(s, While):
                _expr(s.cond); _stmts(s.body)
            elif isinstance(s, For):
                _stmts(s.body)
            elif isinstance(s, FuncDef):
                _stmts(s.body)

    _stmts(stmts)
    return refs


def dead_kernels(stmts: list, verbose: bool = True) -> list:
    """Drop KernelDecl nodes that are never referenced. Returns new list."""
    refs = _collect_kernel_refs(stmts)
    out = []
    for s in stmts:
        if isinstance(s, KernelDecl) and s.name not in refs:
            if verbose:
                print(f'  opt: dropped unused kernel {s.name!r}')
            continue
        out.append(s)
    return out


# ── Public API ────────────────────────────────────────────────

def optimize(program: Program, verbose: bool = True) -> Program:
    """
    Run all optimization passes on a parsed Program and return the result.

    Passes:
      1. Dead kernel elimination
      2. Constant folding + algebraic simplification (two rounds for stability)
    """
    stmts = dead_kernels(program.stmts, verbose)
    stmts = [fold_stmt(s) for s in stmts]   # round 1: fold sub-expressions
    stmts = [fold_stmt(s) for s in stmts]   # round 2: fold results of round 1
    return Program(stmts, program.line)
