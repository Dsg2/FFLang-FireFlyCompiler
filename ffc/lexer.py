from dataclasses import dataclass
from enum import Enum, auto
from typing import List

class TT(Enum):
    INT = auto(); FLOAT = auto(); STRING = auto()
    KERNEL = auto(); FROM = auto(); LOAD = auto()
    DEF = auto(); RETURN = auto()
    IF = auto(); ELIF = auto(); ELSE = auto()
    WHILE = auto(); FOR = auto(); IN = auto()
    LET = auto(); RUN = auto()
    TRUE = auto(); FALSE = auto()
    AND = auto(); OR = auto(); NOT = auto()
    BREAK = auto(); CONTINUE = auto()
    GLOBAL = auto()
    IDENT = auto()
    PLUS = auto(); MINUS = auto(); STAR = auto()
    SLASH = auto(); PERCENT = auto()
    EQ = auto(); NEQ = auto()
    LT = auto(); GT = auto(); LTE = auto(); GTE = auto()
    ASSIGN = auto(); ARROW = auto(); DOT = auto()
    LPAREN = auto(); RPAREN = auto()
    LBRACKET = auto(); RBRACKET = auto()
    COMMA = auto(); COLON = auto()
    NEWLINE = auto(); INDENT = auto(); DEDENT = auto()
    EOF = auto()

KEYWORDS = {
    'kernel': TT.KERNEL, 'from': TT.FROM, 'load': TT.LOAD,
    'def': TT.DEF, 'return': TT.RETURN,
    'if': TT.IF, 'elif': TT.ELIF, 'else': TT.ELSE,
    'while': TT.WHILE, 'for': TT.FOR, 'in': TT.IN,
    'let': TT.LET, 'run': TT.RUN,
    'true': TT.TRUE, 'false': TT.FALSE,
    'and': TT.AND, 'or': TT.OR, 'not': TT.NOT,
    'break': TT.BREAK, 'continue': TT.CONTINUE,
    'global': TT.GLOBAL,
}

@dataclass
class Token:
    type: TT
    value: str
    line: int
    col: int
    def __repr__(self):
        return f'{self.type.name}({self.value!r})@{self.line}:{self.col}'


class LexError(Exception):
    pass


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.indent_stack = [0]

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []
        lines = self.source.splitlines()

        for line_num, raw_line in enumerate(lines, 1):
            self._lex_line(raw_line, line_num, tokens)

        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            tokens.append(Token(TT.DEDENT, '', len(lines), 0))

        tokens.append(Token(TT.EOF, '', len(lines) + 1, 0))
        return tokens

    def _lex_line(self, line: str, line_num: int, out: List[Token]):
        # Count leading whitespace
        col = 0
        indent = 0
        while col < len(line) and line[col] in (' ', '\t'):
            indent += 4 if line[col] == '\t' else 1
            col += 1

        # Blank line or comment-only: skip
        if col == len(line) or line[col] == '#':
            return

        # Emit INDENT / DEDENT
        current = self.indent_stack[-1]
        if indent > current:
            self.indent_stack.append(indent)
            out.append(Token(TT.INDENT, '', line_num, 0))
        elif indent < current:
            while self.indent_stack[-1] > indent:
                self.indent_stack.pop()
                out.append(Token(TT.DEDENT, '', line_num, 0))
            if self.indent_stack[-1] != indent:
                raise LexError(f"Line {line_num}: inconsistent dedent")

        # Tokenize the rest of the line
        i = col
        while i < len(line):
            c = line[i]

            if c in (' ', '\t'):
                i += 1
                continue

            if c == '#':
                break  # rest of line is comment

            # String literal
            if c in ('"', "'"):
                tok, i = self._lex_string(line, i, line_num)
                out.append(tok)
                continue

            # Number
            if c.isdigit():
                tok, i = self._lex_number(line, i, line_num)
                out.append(tok)
                continue

            # Identifier / keyword
            if c.isalpha() or c == '_':
                tok, i = self._lex_ident(line, i, line_num)
                out.append(tok)
                continue

            # Two-char operators
            two = line[i:i+2]
            if two == '==': out.append(Token(TT.EQ,  '==', line_num, i+1)); i += 2; continue
            if two == '!=': out.append(Token(TT.NEQ, '!=', line_num, i+1)); i += 2; continue
            if two == '<=': out.append(Token(TT.LTE, '<=', line_num, i+1)); i += 2; continue
            if two == '>=': out.append(Token(TT.GTE, '>=', line_num, i+1)); i += 2; continue
            if two == '->': out.append(Token(TT.ARROW,'->',line_num, i+1)); i += 2; continue

            # Single-char
            single = {
                '+': TT.PLUS, '-': TT.MINUS, '*': TT.STAR,
                '/': TT.SLASH, '%': TT.PERCENT,
                '<': TT.LT,   '>': TT.GT,
                '=': TT.ASSIGN, '.': TT.DOT,
                '(': TT.LPAREN,  ')': TT.RPAREN,
                '[': TT.LBRACKET,']': TT.RBRACKET,
                ',': TT.COMMA,   ':': TT.COLON,
            }
            if c in single:
                out.append(Token(single[c], c, line_num, i+1))
                i += 1
                continue

            raise LexError(f"Line {line_num}:{i+1}: unexpected character {c!r}")

        out.append(Token(TT.NEWLINE, '', line_num, len(line)+1))

    def _lex_string(self, line: str, i: int, ln: int):
        quote = line[i]; i += 1
        chars = []
        while i < len(line) and line[i] != quote:
            if line[i] == '\\' and i + 1 < len(line):
                esc = line[i+1]
                chars.append({'n':'\n','t':'\t','\\':'\\','"':'"',"'":"'"}.get(esc, esc))
                i += 2
            else:
                chars.append(line[i]); i += 1
        if i >= len(line):
            raise LexError(f"Line {ln}: unterminated string")
        i += 1  # closing quote
        return Token(TT.STRING, ''.join(chars), ln, i), i

    def _lex_number(self, line: str, i: int, ln: int):
        start = i
        while i < len(line) and line[i].isdigit():
            i += 1
        if i < len(line) and line[i] == '.' and i+1 < len(line) and line[i+1].isdigit():
            i += 1
            while i < len(line) and line[i].isdigit():
                i += 1
            return Token(TT.FLOAT, line[start:i], ln, start+1), i
        return Token(TT.INT, line[start:i], ln, start+1), i

    def _lex_ident(self, line: str, i: int, ln: int):
        start = i
        while i < len(line) and (line[i].isalnum() or line[i] == '_'):
            i += 1
        word = line[start:i]
        tt = KEYWORDS.get(word, TT.IDENT)
        return Token(tt, word, ln, start+1), i
