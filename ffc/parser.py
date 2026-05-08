from typing import List, Optional
from .lexer import Token, TT
from .ast_nodes import *


class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    # ── Helpers ──────────────────────────────────────────────

    def peek(self, offset: int = 0) -> Token:
        i = self.pos + offset
        return self.tokens[i] if i < len(self.tokens) else Token(TT.EOF, '', 0, 0)

    def advance(self) -> Token:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def check(self, *types) -> bool:
        return self.peek().type in types

    def match(self, *types) -> Optional[Token]:
        if self.check(*types):
            return self.advance()
        return None

    def expect(self, tt: TT, hint: str = '') -> Token:
        t = self.advance()
        if t.type != tt:
            msg = f"Line {t.line}:{t.col}: expected {tt.name}, got {t.type.name} ({t.value!r})"
            if hint:
                msg += f' — {hint}'
            raise ParseError(msg)
        return t

    def skip_newlines(self):
        while self.check(TT.NEWLINE):
            self.advance()

    def expect_newline(self):
        if self.check(TT.NEWLINE):
            self.advance()
        elif not self.check(TT.EOF, TT.DEDENT):
            t = self.peek()
            raise ParseError(f"Line {t.line}: expected newline, got {t.type.name} ({t.value!r})")

    # ── Top-level ────────────────────────────────────────────

    def parse(self) -> Program:
        stmts = []
        self.skip_newlines()
        while not self.check(TT.EOF):
            s = self.parse_stmt()
            if s is not None:
                stmts.append(s)
            self.skip_newlines()
        return Program(stmts, line=0)

    def parse_block(self) -> List:
        stmts = []
        self.expect(TT.INDENT)
        self.skip_newlines()
        while not self.check(TT.DEDENT, TT.EOF):
            s = self.parse_stmt()
            if s is not None:
                stmts.append(s)
            self.skip_newlines()
        self.expect(TT.DEDENT)
        return stmts

    # ── Statements ───────────────────────────────────────────

    def parse_stmt(self):
        ln = self.peek().line

        if self.check(TT.NEWLINE):
            self.advance(); return None

        if self.check(TT.KERNEL):
            return self.parse_kernel_decl()
        if self.check(TT.DEF):
            return self.parse_func_def()
        if self.check(TT.IF):
            return self.parse_if()
        if self.check(TT.WHILE):
            return self.parse_while()
        if self.check(TT.FOR):
            return self.parse_for()
        if self.check(TT.RETURN):
            self.advance()
            val = None
            if not self.check(TT.NEWLINE, TT.EOF, TT.DEDENT):
                val = self.parse_expr()
            self.expect_newline()
            return Return(val, line=ln)
        if self.check(TT.BREAK):
            self.advance(); self.expect_newline()
            return Break(line=ln)
        if self.check(TT.CONTINUE):
            self.advance(); self.expect_newline()
            return Continue(line=ln)
        if self.check(TT.RUN):
            return self.parse_run_kernel()
        if self.check(TT.GLOBAL):
            return self.parse_global_stmt()
        if self.check(TT.LET):
            self.advance()  # 'let' is optional sugar

        expr = self.parse_expr()

        if self.check(TT.ASSIGN):
            self.advance()
            value = self.parse_expr()
            self.expect_newline()
            return Assignment(expr, value, line=ln)

        self.expect_newline()
        return ExprStmt(expr, line=ln)

    def parse_kernel_decl(self) -> KernelDecl:
        ln = self.peek().line
        self.expect(TT.KERNEL)
        name = self.expect(TT.IDENT).value

        if self.check(TT.FROM):
            self.advance()
            cl_file = self.expect(TT.STRING).value
            func_name = name
        elif self.check(TT.ASSIGN):
            self.advance()
            self.expect(TT.LOAD, 'expected load("file", "func")')
            self.expect(TT.LPAREN)
            cl_file = self.expect(TT.STRING).value
            self.expect(TT.COMMA)
            func_name = self.expect(TT.STRING).value
            self.expect(TT.RPAREN)
        else:
            t = self.peek()
            raise ParseError(f"Line {t.line}: expected 'from' or '=' after kernel name")

        self.expect_newline()
        return KernelDecl(name, cl_file, func_name, line=ln)

    # Valid type annotation keywords that can appear after ':' in a param list
    _PARAM_TYPE_NAMES = {'int', 'float', 'bool', 'str', 'list', 'image', 'window', 'sound'}

    def _parse_param(self):
        """Parse a single parameter: name [ : typename ]  →  (name, type_str | None)."""
        pname = self.expect(TT.IDENT).value
        ptype = None
        # Check for ': typename' — but only consume the colon if the next IDENT
        # is a recognised type name, so we don't eat the function-body colon.
        if self.check(TT.COLON):
            # Peek ahead: token after the colon
            if (self.pos + 1 < len(self.tokens) and
                    self.tokens[self.pos + 1].type == TT.IDENT and
                    self.tokens[self.pos + 1].value in self._PARAM_TYPE_NAMES):
                self.advance()  # consume ':'
                ptype = self.expect(TT.IDENT).value
        return pname, ptype

    def parse_func_def(self) -> FuncDef:
        ln = self.peek().line
        self.expect(TT.DEF)
        name = self.expect(TT.IDENT).value
        self.expect(TT.LPAREN)
        params = []
        param_types = []
        if not self.check(TT.RPAREN):
            pname, ptype = self._parse_param()
            params.append(pname); param_types.append(ptype)
            while self.match(TT.COMMA):
                pname, ptype = self._parse_param()
                params.append(pname); param_types.append(ptype)
        self.expect(TT.RPAREN)
        self.expect(TT.COLON)
        self.expect_newline()
        body = self.parse_block()
        return FuncDef(name, params, body, param_types=param_types, line=ln)

    def parse_if(self) -> If:
        ln = self.peek().line
        self.expect(TT.IF)
        cond = self.parse_expr()
        self.expect(TT.COLON); self.expect_newline()
        body = self.parse_block()

        elifs = []
        while self.check(TT.ELIF):
            self.advance()
            ec = self.parse_expr()
            self.expect(TT.COLON); self.expect_newline()
            elifs.append((ec, self.parse_block()))

        else_body = None
        if self.check(TT.ELSE):
            self.advance()
            self.expect(TT.COLON); self.expect_newline()
            else_body = self.parse_block()

        return If(cond, body, elifs, else_body, line=ln)

    def parse_while(self) -> While:
        ln = self.peek().line
        self.expect(TT.WHILE)
        cond = self.parse_expr()
        self.expect(TT.COLON); self.expect_newline()
        body = self.parse_block()
        return While(cond, body, line=ln)

    def parse_for(self) -> For:
        ln = self.peek().line
        self.expect(TT.FOR)
        var = self.expect(TT.IDENT).value
        self.expect(TT.IN)
        iter_expr = self.parse_expr()
        self.expect(TT.COLON); self.expect_newline()
        body = self.parse_block()
        return For(var, iter_expr, body, line=ln)

    def parse_global_stmt(self) -> GlobalStmt:
        ln = self.peek().line
        self.expect(TT.GLOBAL)
        names = [self.expect(TT.IDENT).value]
        while self.match(TT.COMMA):
            names.append(self.expect(TT.IDENT).value)
        self.expect_newline()
        return GlobalStmt(names, line=ln)

    def parse_run_kernel(self) -> RunKernel:
        ln = self.peek().line
        self.expect(TT.RUN)
        kernel_name = self.expect(TT.IDENT).value
        self.expect(TT.LPAREN)
        args = []
        if not self.check(TT.RPAREN):
            args.append(self.parse_expr())
            while self.match(TT.COMMA):
                args.append(self.parse_expr())
        self.expect(TT.RPAREN)
        self.expect(TT.ARROW)
        outputs = [self.parse_expr()]
        while self.match(TT.COMMA):
            outputs.append(self.parse_expr())
        # optional:  size(expr)  — explicit dispatch size
        dispatch = None
        if self.check(TT.IDENT) and self.peek().value == 'size':
            self.advance()
            self.expect(TT.LPAREN)
            dispatch = self.parse_expr()
            self.expect(TT.RPAREN)
        self.expect_newline()
        return RunKernel(kernel_name, args, outputs, dispatch, line=ln)

    # ── Expressions (precedence climbing) ────────────────────

    def parse_expr(self):
        return self.parse_or()

    def parse_or(self):
        left = self.parse_and()
        while self.check(TT.OR):
            self.advance()
            right = self.parse_and()
            left = BinOp(left, 'or', right, line=left.line)
        return left

    def parse_and(self):
        left = self.parse_not()
        while self.check(TT.AND):
            self.advance()
            right = self.parse_not()
            left = BinOp(left, 'and', right, line=left.line)
        return left

    def parse_not(self):
        if self.check(TT.NOT):
            ln = self.peek().line; self.advance()
            return UnaryOp('not', self.parse_not(), line=ln)
        return self.parse_cmp()

    def parse_cmp(self):
        left = self.parse_add()
        ops = {TT.EQ:'==', TT.NEQ:'!=', TT.LT:'<', TT.GT:'>',
               TT.LTE:'<=', TT.GTE:'>='}
        while self.peek().type in ops:
            op = ops[self.advance().type]
            right = self.parse_add()
            left = BinOp(left, op, right, line=left.line)
        return left

    def parse_add(self):
        left = self.parse_mul()
        while self.check(TT.PLUS, TT.MINUS):
            op = self.advance().value
            right = self.parse_mul()
            left = BinOp(left, op, right, line=left.line)
        return left

    def parse_mul(self):
        left = self.parse_unary()
        while self.check(TT.STAR, TT.SLASH, TT.PERCENT):
            op = self.advance().value
            right = self.parse_unary()
            left = BinOp(left, op, right, line=left.line)
        return left

    def parse_unary(self):
        if self.check(TT.MINUS):
            ln = self.peek().line; self.advance()
            return UnaryOp('-', self.parse_unary(), line=ln)
        return self.parse_postfix()

    def parse_postfix(self):
        expr = self.parse_primary()
        while True:
            if self.check(TT.DOT):
                self.advance()
                attr = self.expect(TT.IDENT).value
                expr = Attr(expr, attr, line=expr.line)
            elif self.check(TT.LBRACKET):
                self.advance()
                indices = [self.parse_expr()]
                while self.match(TT.COMMA):
                    indices.append(self.parse_expr())
                self.expect(TT.RBRACKET)
                expr = Index(expr, indices, line=expr.line)
            elif self.check(TT.LPAREN):
                self.advance()
                args = []
                if not self.check(TT.RPAREN):
                    args.append(self.parse_expr())
                    while self.match(TT.COMMA):
                        args.append(self.parse_expr())
                self.expect(TT.RPAREN)
                expr = Call(expr, args, line=expr.line)
            else:
                break
        return expr

    def parse_primary(self):
        t = self.peek()

        if t.type == TT.INT:
            self.advance(); return IntLit(int(t.value), line=t.line)
        if t.type == TT.FLOAT:
            self.advance(); return FloatLit(float(t.value), line=t.line)
        if t.type == TT.STRING:
            self.advance(); return StrLit(t.value, line=t.line)
        if t.type == TT.TRUE:
            self.advance(); return BoolLit(True, line=t.line)
        if t.type == TT.FALSE:
            self.advance(); return BoolLit(False, line=t.line)
        if t.type == TT.IDENT:
            self.advance(); return Ident(t.value, line=t.line)

        if t.type == TT.LBRACKET:
            self.advance()
            elems = []
            if not self.check(TT.RBRACKET):
                elems.append(self.parse_expr())
                while self.match(TT.COMMA):
                    elems.append(self.parse_expr())
            self.expect(TT.RBRACKET)
            return ListLit(elems, line=t.line)

        if t.type == TT.LPAREN:
            self.advance()
            expr = self.parse_expr()
            self.expect(TT.RPAREN)
            return expr

        raise ParseError(
            f"Line {t.line}:{t.col}: unexpected token {t.type.name} ({t.value!r})"
        )
