from dataclasses import dataclass
from typing import List, Optional, Any


# ── Statements ──────────────────────────────────────────────

@dataclass
class Program:
    stmts: List[Any]
    line: int = 0

@dataclass
class KernelDecl:
    name: str
    cl_file: str
    func_name: str          # function name inside the .cl file
    line: int = 0

@dataclass
class FuncDef:
    name: str
    params: List[str]
    body: List[Any]
    param_types: List[Optional[str]] = None  # explicit type annotation per param, or None
    line: int = 0

@dataclass
class Assignment:
    target: Any             # Ident | Index | Attr
    value: Any
    line: int = 0

@dataclass
class If:
    cond: Any
    body: List[Any]
    elifs: List[Any]        # list of (cond, body) tuples
    else_body: Optional[List[Any]]
    line: int = 0

@dataclass
class While:
    cond: Any
    body: List[Any]
    line: int = 0

@dataclass
class For:
    var: str
    iter: Any               # range() call or list expression
    body: List[Any]
    line: int = 0

@dataclass
class Return:
    value: Optional[Any]
    line: int = 0

@dataclass
class Break:
    line: int = 0

@dataclass
class Continue:
    line: int = 0

@dataclass
class ExprStmt:
    expr: Any
    line: int = 0

@dataclass
class RunKernel:            # run kernel(args) -> out1 [, out2 ...]  [size(n)]
    kernel: str
    args: List[Any]
    outputs: List[Any]      # one or more Idents (must already exist and be lists)
    dispatch: Optional[Any] = None   # explicit dispatch size; defaults to outputs[0].len
    line: int = 0


# ── Expressions ─────────────────────────────────────────────

@dataclass
class BinOp:
    left: Any
    op: str
    right: Any
    line: int = 0

@dataclass
class UnaryOp:
    op: str
    operand: Any
    line: int = 0

@dataclass
class Call:
    func: Any               # Ident | Attr
    args: List[Any]
    line: int = 0

@dataclass
class Index:
    obj: Any
    indices: List[Any]      # one index for list, two for image[x,y]
    line: int = 0

@dataclass
class Attr:
    obj: Any
    attr: str
    line: int = 0

@dataclass
class ListLit:
    elements: List[Any]
    line: int = 0

@dataclass
class IntLit:
    value: int
    line: int = 0

@dataclass
class FloatLit:
    value: float
    line: int = 0

@dataclass
class StrLit:
    value: str
    line: int = 0

@dataclass
class BoolLit:
    value: bool
    line: int = 0

@dataclass
class Ident:
    name: str
    line: int = 0
