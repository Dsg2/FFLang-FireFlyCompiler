"""
ffc/flc.py — FL Kernel Compiler  (.fl → OpenCL .cl)

.fl is a Python-like language for writing GPU kernels.
It compiles to OpenCL C and can be referenced from .ff files:

    kernel mykernel from "myfile.fl"          # uses function name = kernel name
    kernel mykernel = load("myfile.fl", "fn") # explicit function name

Syntax example:

    kernel blur(float[] input, int width, int height) -> float[] output:
        let gx = gid % width
        let gy = gid / width
        let sum = 0.0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                let nx = gx + dx
                let ny = gy + dy
                if nx >= 0 and nx < width and ny >= 0 and ny < height:
                    sum = sum + input[ny * width + nx]
        output = sum / 9.0

Rules:
  - Array params (float[], int[]) are auto-indexed by gid unless given an explicit [i]
  - Assigning to an output param: `out = expr`  →  `out[gid] = expr`
  - `gid` is a built-in int = get_global_id(0)
  - `return` exits the kernel early (no value; kernels are always void)
  - `let var = expr` declares a typed local variable (type inferred)
  - Math: sin cos tan asin acos atan atan2 sqrt fabs abs floor ceil round
          exp log pow min max clamp radians degrees
  - int(x), float(x) — type casts
"""

from .lexer import Lexer, TT

# ── Type tags ────────────────────────────────────────────────
_F   = 'float'
_I   = 'int'
_FA  = 'float[]'
_IA  = 'int[]'

_MATH = {
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
    'sqrt', 'fabs', 'floor', 'ceil', 'round',
    'exp', 'log', 'pow', 'clamp', 'radians', 'degrees',
}
_MATH_MAP = {'abs': 'fabs'}   # rename on output


class FLError(Exception):
    pass


# ── Data structures ──────────────────────────────────────────

class Param:
    __slots__ = ('name', 'typ', 'is_output')
    def __init__(self, name, typ, is_output=False):
        self.name      = name
        self.typ       = typ        # _F / _I / _FA / _IA
        self.is_output = is_output

    @property
    def is_array(self): return self.typ in (_FA, _IA)

    @property
    def c_type(self):
        """C type string for this param (scalar or array element)."""
        return 'float' if self.typ in (_F, _FA) else 'int'


# ── Top-level entry point ────────────────────────────────────

def compile_fl(src: str) -> str:
    """Compile .fl source → OpenCL C string."""
    toks   = Lexer(src).tokenize()
    parser = _Parser(toks)
    kernels = parser.parse_all()
    parts  = []
    for name, params, body_toks in kernels:
        parts.append(_gen_kernel(name, params, body_toks))
    return '\n\n'.join(parts) + '\n'


# ── Parser ───────────────────────────────────────────────────

class _Parser:
    def __init__(self, tokens):
        self.toks = tokens
        self.pos  = 0

    # helpers
    def _peek(self):               return self.toks[self.pos]
    def _advance(self):            t = self.toks[self.pos]; self.pos += 1; return t
    def _check(self, *tts):        return self._peek().type in tts
    def _match(self, *tts):
        if self._check(*tts):      return self._advance()
        return None
    def _expect(self, tt):
        t = self._peek()
        if t.type != tt:
            raise FLError(f'Line {t.line}: expected {tt.name}, got {t.type.name} {t.value!r}')
        return self._advance()
    def _skip_nl(self):
        while self._check(TT.NEWLINE): self._advance()

    def parse_all(self):
        """Return list of (name, params, body_tokens) for every kernel in the file."""
        kernels = []
        while True:
            self._skip_nl()
            if self._check(TT.EOF): break
            kernels.append(self._parse_kernel())
        return kernels

    def _parse_kernel(self):
        self._expect(TT.KERNEL)
        name = self._expect(TT.IDENT).value
        self._expect(TT.LPAREN)

        # Input params
        inputs = []
        if not self._check(TT.RPAREN):
            inputs.append(self._parse_param(False))
            while self._match(TT.COMMA):
                inputs.append(self._parse_param(False))
        self._expect(TT.RPAREN)

        # -> output params
        self._expect(TT.ARROW)
        outputs = [self._parse_param(True)]
        while self._match(TT.COMMA):
            outputs.append(self._parse_param(True))

        self._expect(TT.COLON)
        self._skip_nl()

        body_toks = self._collect_block()
        return name, inputs + outputs, body_toks

    def _parse_param(self, is_output):
        # typename: float | int | float[] | int[]
        type_tok = self._expect(TT.IDENT)
        tname = type_tok.value
        if tname not in ('float', 'int'):
            raise FLError(f'Line {type_tok.line}: unknown type {tname!r} (use float or int)')
        is_arr = False
        if self._check(TT.LBRACKET):
            self._advance()
            self._expect(TT.RBRACKET)
            is_arr = True
        typ = (_FA if tname == 'float' else _IA) if is_arr else (_F if tname == 'float' else _I)
        name = self._expect(TT.IDENT).value
        return Param(name, typ, is_output)

    def _collect_block(self):
        """Consume INDENT … DEDENT and return the inner token list."""
        self._expect(TT.INDENT)
        depth = 1
        toks  = []
        while not self._check(TT.EOF):
            t = self._advance()
            if   t.type == TT.INDENT:  depth += 1
            elif t.type == TT.DEDENT:
                depth -= 1
                if depth == 0: break
            toks.append(t)
        return toks


# ── Code generator ───────────────────────────────────────────

def _gen_kernel(name, params, body_toks):
    # Build OpenCL signature
    parts = []
    for p in params:
        if p.is_array:
            # All array params are __global (no const — input arrays may be
            # sorted or otherwise mutated in-place by the kernel).
            parts.append(f'    __global {p.c_type}* {p.name}')
        else:
            parts.append(f'    {p.c_type} {p.name}')
    sig = ',\n'.join(parts)

    # Track which names are arrays, outputs, and scalar params
    arrays  = {p.name: p.c_type for p in params if p.is_array}
    outputs = {p.name for p in params if p.is_output}
    # Scalar params pre-seeded into scope so their types are known (int vs float)
    scalars = {p.name: (_F if p.typ == _F else _I) for p in params if not p.is_array}

    # Generate body
    gen = _BodyGen(arrays, outputs, scalars)
    body_lines = gen.gen(body_toks)

    lines = [f'__kernel void {name}(\n{sig}\n) {{']
    lines.append('    int _gid = (int)get_global_id(0);')
    for bl in body_lines:
        lines.append('    ' + bl)
    lines.append('}')
    return '\n'.join(lines)


# ── Body code generator ──────────────────────────────────────

class _BodyGen:
    def __init__(self, arrays: dict, outputs: set, scalars: dict = None):
        self._arrays  = arrays   # name → 'float'|'int'
        self._outputs = outputs  # set of output param names
        self._scope   = {'gid': _I}  # name → type tag
        if scalars:
            self._scope.update(scalars)  # pre-seed scalar param types
        self._indent  = 0
        self._lines   = []
        self._toks    = []
        self._pos     = 0

    # ── Token helpers ─────────────────────────────────────────

    def _peek(self):
        return self._toks[self._pos] if self._pos < len(self._toks) else None

    def _advance(self):
        t = self._toks[self._pos]; self._pos += 1; return t

    def _check(self, *tts):
        t = self._peek(); return t is not None and t.type in tts

    def _match(self, *tts):
        if self._check(*tts): return self._advance()
        return None

    def _expect(self, tt):
        t = self._peek()
        if t is None or t.type != tt:
            got = f'{t.type.name} {t.value!r}' if t else 'EOF'
            raise FLError(f'Expected {tt.name}, got {got}')
        return self._advance()

    def _skip_nl(self):
        while self._check(TT.NEWLINE): self._advance()

    # ── Emit helpers ──────────────────────────────────────────

    def _emit(self, s):
        self._lines.append('    ' * self._indent + s)

    # ── Entry point ───────────────────────────────────────────

    def gen(self, tokens) -> list:
        self._toks = tokens
        self._pos  = 0
        self._lines = []
        self._gen_stmts()
        return self._lines

    # ── Statement dispatch ────────────────────────────────────

    def _gen_stmts(self):
        while self._pos < len(self._toks):
            self._skip_nl()
            if self._pos >= len(self._toks): break
            if self._check(TT.DEDENT):       break
            self._gen_stmt()

    def _gen_stmt(self):
        t = self._peek()
        if t is None: return

        if t.type == TT.LET:
            self._advance()
            self._gen_assign(is_let=True)
        elif t.type == TT.IDENT and self._is_assign_ahead():
            self._gen_assign(is_let=False)
        elif t.type == TT.IF:      self._gen_if()
        elif t.type == TT.WHILE:   self._gen_while()
        elif t.type == TT.FOR:     self._gen_for()
        elif t.type == TT.RETURN:
            self._advance(); self._skip_nl()
            self._emit('return;')
        elif t.type == TT.NEWLINE: self._advance()
        else:
            # expression statement (e.g. function call with side effects)
            v, _ = self._expr()
            self._skip_nl()
            self._emit(f'{v};')

    def _is_assign_ahead(self):
        """True if the upcoming token sequence is: IDENT = or IDENT[...] ="""
        i = self._pos + 1
        if i >= len(self._toks): return False
        if self._toks[i].type == TT.ASSIGN: return True
        if self._toks[i].type == TT.LBRACKET:
            depth = 1; i += 1
            while i < len(self._toks) and depth > 0:
                tt = self._toks[i].type
                if tt == TT.LBRACKET: depth += 1
                if tt == TT.RBRACKET: depth -= 1
                i += 1
            return i < len(self._toks) and self._toks[i].type == TT.ASSIGN
        return False

    # ── Assignment ────────────────────────────────────────────

    def _gen_assign(self, is_let=False):
        name = self._expect(TT.IDENT).value

        # Subscript assignment: name[idx] = expr
        if self._check(TT.LBRACKET):
            self._advance()
            idx, _ = self._expr()
            self._expect(TT.RBRACKET)
            self._expect(TT.ASSIGN)
            val, _ = self._expr()
            self._skip_nl()
            self._emit(f'{name}[{idx}] = {val};')
            return

        self._expect(TT.ASSIGN)
        val, vt = self._expr()
        self._skip_nl()

        # Output array param → implicit gid indexing
        if name in self._outputs:
            self._emit(f'{name}[_gid] = {val};')
            return

        # Declare or reassign local
        if name not in self._scope or is_let:
            ct = 'float' if vt == _F else 'int'
            self._scope[name] = vt
            self._emit(f'{ct} {name} = {val};')
        else:
            self._emit(f'{name} = {val};')

    # ── Control flow ──────────────────────────────────────────

    def _gen_if(self):
        self._expect(TT.IF)
        cond, _ = self._expr()
        self._expect(TT.COLON); self._skip_nl()
        self._emit(f'if ({cond}) {{')
        outer = dict(self._scope)          # save outer scope
        self._indent += 1; self._gen_block(); self._indent -= 1
        self._scope = dict(outer)          # restore — if-block locals don't leak out

        while self._check(TT.ELIF):
            self._advance()
            cond, _ = self._expr()
            self._expect(TT.COLON); self._skip_nl()
            self._emit(f'}} else if ({cond}) {{')
            self._indent += 1; self._gen_block(); self._indent -= 1
            self._scope = dict(outer)      # restore after each elif branch too

        if self._check(TT.ELSE):
            self._advance(); self._expect(TT.COLON); self._skip_nl()
            self._emit('} else {')
            self._indent += 1; self._gen_block(); self._indent -= 1
            self._scope = dict(outer)

        self._emit('}')

    def _gen_while(self):
        self._expect(TT.WHILE)
        cond, _ = self._expr()
        self._expect(TT.COLON); self._skip_nl()
        self._emit(f'while ({cond}) {{')
        outer = dict(self._scope)
        self._indent += 1; self._gen_block(); self._indent -= 1
        self._scope = dict(outer)          # loop-body locals don't leak out
        self._emit('}')

    def _gen_for(self):
        self._expect(TT.FOR)
        var = self._expect(TT.IDENT).value
        self._expect(TT.IN)
        fn  = self._expect(TT.IDENT)
        if fn.value != 'range':
            raise FLError(f'Line {fn.line}: FL for-loops must use range()')
        self._expect(TT.LPAREN)
        args = [self._expr()[0]]
        while self._match(TT.COMMA):
            args.append(self._expr()[0])
        self._expect(TT.RPAREN)
        self._expect(TT.COLON); self._skip_nl()

        outer = dict(self._scope)
        self._scope[var] = _I              # loop var visible inside body
        if   len(args) == 1: self._emit(f'for (int {var} = 0; {var} < {args[0]}; {var}++) {{')
        elif len(args) == 2: self._emit(f'for (int {var} = {args[0]}; {var} < {args[1]}; {var}++) {{')
        else:                self._emit(f'for (int {var} = {args[0]}; {var} < {args[1]}; {var} += {args[2]}) {{')
        self._indent += 1; self._gen_block(); self._indent -= 1
        self._scope = dict(outer)          # loop-body locals (and loop var) don't leak out
        self._emit('}')

    def _gen_block(self):
        self._expect(TT.INDENT)
        self._gen_stmts()
        self._match(TT.DEDENT)

    # ── Expression parser (recursive descent) ─────────────────
    # Returns (c_expr_string, type_tag)

    def _expr(self):             return self._or()
    def _or(self):
        l, lt = self._and()
        while self._check(TT.OR):
            self._advance(); r, _ = self._and()
            l = f'({l} || {r})'; lt = _I
        return l, lt
    def _and(self):
        l, lt = self._not()
        while self._check(TT.AND):
            self._advance(); r, _ = self._not()
            l = f'({l} && {r})'; lt = _I
        return l, lt
    def _not(self):
        if self._check(TT.NOT):
            self._advance(); e, _ = self._cmp()
            return f'(!{e})', _I
        return self._cmp()
    def _cmp(self):
        _OPS = {TT.EQ:'==', TT.NEQ:'!=', TT.LT:'<', TT.GT:'>', TT.LTE:'<=', TT.GTE:'>='}
        l, lt = self._add()
        while self._check(*_OPS):
            op = _OPS[self._advance().type]; r, _ = self._add()
            l = f'({l} {op} {r})'; lt = _I
        return l, lt
    def _add(self):
        l, lt = self._mul()
        while self._check(TT.PLUS, TT.MINUS):
            op = '+' if self._advance().type == TT.PLUS else '-'
            r, rt = self._mul()
            lt = _F if _F in (lt, rt) else lt
            l = f'({l} {op} {r})'
        return l, lt
    def _mul(self):
        l, lt = self._unary()
        while self._check(TT.STAR, TT.SLASH, TT.PERCENT):
            tt  = self._advance().type
            op  = {TT.STAR:'*', TT.SLASH:'/', TT.PERCENT:'%'}[tt]
            r, rt = self._unary()
            if op == '%' and _F in (lt, rt):
                l = f'fmod({l}, {r})'; lt = _F
            elif op == '/':
                lt = _F if _F in (lt, rt) else lt
                # Division by a float literal → multiply by reciprocal (faster on GPU)
                import re as _re
                _flit = _re.fullmatch(r'\d+\.\d*f|\d*\.\d+f', r)
                if _flit:
                    fval = float(r.rstrip('f'))
                    if fval != 0.0:
                        recip = 1.0 / fval
                        l = f'({l} * {recip:.9g}f)'
                    else:
                        l = f'({l} / {r})'
                else:
                    l = f'({l} / {r})'
            else:
                lt = _F if _F in (lt, rt) else lt
                l = f'({l} {op} {r})'
        return l, lt
    def _unary(self):
        if self._check(TT.MINUS):
            self._advance(); e, et = self._atom()
            return f'(-{e})', et
        if self._check(TT.NOT):
            self._advance(); e, _ = self._atom()
            return f'(!{e})', _I
        return self._atom()

    def _atom(self):
        t = self._peek()
        if t is None:
            raise FLError('Unexpected end of expression')

        # Integer literal
        if t.type == TT.INT:
            self._advance()
            return str(int(t.value)), _I

        # Float literal
        if t.type == TT.FLOAT:
            self._advance()
            return f'{t.value}f', _F

        # Boolean
        if t.type == TT.TRUE:  self._advance(); return '1', _I
        if t.type == TT.FALSE: self._advance(); return '0', _I

        # Parenthesised expression
        if t.type == TT.LPAREN:
            self._advance()
            e, et = self._expr()
            self._expect(TT.RPAREN)
            return f'({e})', et

        # Identifier, function call, array subscript
        if t.type == TT.IDENT:
            name = t.value
            self._advance()

            # Built-in: gid → _gid
            if name == 'gid':
                return '_gid', _I

            # Type cast or function call
            if self._check(TT.LPAREN):
                self._advance()
                # Type cast: float(x) or int(x)
                if name in ('float', 'int'):
                    arg, _ = self._expr()
                    self._expect(TT.RPAREN)
                    ct  = 'float' if name == 'float' else 'int'
                    ret = _F      if name == 'float' else _I
                    return f'({ct})({arg})', ret
                # len() on an array param is meaningless in a kernel
                if name == 'len':
                    raise FLError(
                        'len() cannot be used on a kernel array param — '
                        'GPU buffers have no runtime length. '
                        'Pass the count as an int parameter instead.')
                # Math built-in
                if name in _MATH or name in _MATH_MAP:
                    fn     = _MATH_MAP.get(name, name)
                    c_args = []
                    if not self._check(TT.RPAREN):
                        av, at = self._expr()
                        # fabs/abs require float — cast int args to avoid ambiguity
                        if fn == 'fabs' and at == _I:
                            av = f'(float)({av})'
                        c_args.append(av)
                        while self._match(TT.COMMA):
                            c_args.append(self._expr()[0])
                    self._expect(TT.RPAREN)
                    # pow(x, n) unrolling: replace with repeated multiplication
                    # (avoids expensive transcendental for small integer exponents)
                    if fn == 'pow' and len(c_args) == 2:
                        import re as _re
                        _exp_match = _re.fullmatch(r'(\d+)(?:\.0*f?)?|(\d+\.\d*)f', c_args[1])
                        if _exp_match:
                            exp = int(float(c_args[1].rstrip('f')))
                            base = c_args[0]
                            if exp == 2:
                                return f'({base} * {base})', _F
                            if exp == 3:
                                return f'({base} * {base} * {base})', _F
                            if exp == 4:
                                return f'({base} * {base} * {base} * {base})', _F
                    return f'{fn}({", ".join(c_args)})', _F
                # User-defined call (fallthrough — assume float return)
                args = []
                if not self._check(TT.RPAREN):
                    args.append(self._expr()[0])
                    while self._match(TT.COMMA):
                        args.append(self._expr()[0])
                self._expect(TT.RPAREN)
                return f'{name}({", ".join(args)})', _F

            # Array subscript: name[idx]
            if self._check(TT.LBRACKET):
                self._advance()
                idx, _ = self._expr()
                self._expect(TT.RBRACKET)
                et = _F if self._arrays.get(name) == 'float' else _I
                return f'{name}[{idx}]', et

            # Plain name — array params without [i] auto-index by gid
            if name in self._arrays:
                et = _F if self._arrays[name] == 'float' else _I
                return f'{name}[_gid]', et

            et = self._scope.get(name, _F)
            return name, et

        raise FLError(f'Line {t.line}: unexpected token {t.type.name} {t.value!r}')
