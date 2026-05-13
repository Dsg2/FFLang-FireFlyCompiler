from typing import Dict, List, Optional
from .ast_nodes import *


# ── Type tags ────────────────────────────────────────────────

INT   = 'int'
FLT   = 'float'
BOOL  = 'bool'
STR   = 'str'
LIST  = 'list'
IMAGE = 'image'
WIN   = 'window'
SND   = 'sound'
VOID  = 'void'
UNK   = 'unknown'

_C_TYPES = {
    INT: 'int64_t', FLT: 'double', BOOL: 'int',
    STR: 'const char*',
    LIST: 'FFList', IMAGE: 'FFImage', WIN: 'FFWindow', SND: 'FFSound',
}

_RETURN_TYPES = {
    'zeros': LIST, 'ones': LIST, 'random': LIST,
    'gpu': LIST,
    'window': WIN, 'image': IMAGE, 'sound': SND,
    'float': FLT, 'int': INT, 'str': STR,
    'len': INT, 'floor': INT, 'ceil': INT, 'round': INT,
    'sum': FLT, 'min': FLT, 'max': FLT, 'abs': FLT, 'sqrt': FLT,
    'sin': FLT, 'cos': FLT, 'tan': FLT,
    'asin': FLT, 'acos': FLT, 'atan': FLT, 'atan2': FLT,
    'radians': FLT, 'degrees': FLT,
    'clamp': FLT, 'time': FLT,
    'load_image': IMAGE, 'load_sound': SND,
    'read_file': STR, 'exists': BOOL,
    'load_list': LIST,
    # String built-ins
    'fmt': STR, 'strcat': STR, 'strlen': INT, 'streq': BOOL,
    'toint': INT, 'tofloat': FLT,
    'strsub': STR, 'strfind': INT, 'strtrim': STR,
}


class Scope:
    def __init__(self, parent: Optional['Scope'] = None):
        self.vars: Dict[str, str] = {}
        self.parent = parent

    def get(self, name: str) -> Optional[str]:
        if name in self.vars:
            return self.vars[name]
        return self.parent.get(name) if self.parent else None

    def set(self, name: str, typ: str):
        self.vars[name] = typ


class CodeGenError(Exception):
    pass


class CodeGen:
    def __init__(self):
        self.lines: List[str] = []
        self.indent = 0
        self.scope = Scope()
        self._tmp = 0
        self._kernel_ids: Dict[str, str] = {}        # ff name -> C var
        self._func_return_types: Dict[str, str] = {}  # user func -> return type tag
        self._func_param_types: Dict[str, List[str]] = {}  # user func -> [param type tags]
        self._pending_frees: List[str] = []  # temp list vars to free at statement end
        self._global_names: Dict[str, str] = {}  # name -> type tag; hoisted to file scope

    # ── Emit helpers ─────────────────────────────────────────

    def emit(self, line: str = ''):
        prefix = '    ' * self.indent
        self.lines.append(prefix + line if line else '')

    def tmp(self, prefix: str = '_t') -> str:
        self._tmp += 1
        return f'{prefix}{self._tmp}'

    def push_scope(self):
        self.scope = Scope(self.scope)

    def pop_scope(self):
        self.scope = self.scope.parent

    # ── Type inference ───────────────────────────────────────

    def typeof(self, node) -> str:
        if isinstance(node, IntLit):   return INT
        if isinstance(node, FloatLit): return FLT
        if isinstance(node, BoolLit):  return BOOL
        if isinstance(node, StrLit):   return STR
        if isinstance(node, ListLit):  return LIST
        if isinstance(node, Ident):
            t = self.scope.get(node.name)
            return t if t else UNK
        if isinstance(node, BinOp):
            if node.op in ('==','!=','<','>','<=','>=','and','or'):
                return BOOL
            lt = self.typeof(node.left)
            rt = self.typeof(node.right)
            if FLT in (lt, rt): return FLT
            return lt
        if isinstance(node, UnaryOp):
            return BOOL if node.op == 'not' else self.typeof(node.operand)
        if isinstance(node, Call):
            fn = node.func
            if isinstance(fn, Ident):
                # random() with no args is a scalar float, not a list
                if fn.name == 'random' and not node.args:
                    return FLT
                return (_RETURN_TYPES.get(fn.name)
                        or self._func_return_types.get(fn.name)
                        or self.scope.get(fn.name)
                        or UNK)
            if isinstance(fn, Attr):
                ot = self.typeof(fn.obj)
                if ot == LIST and fn.attr == 'pop': return FLT
        if isinstance(node, Attr):
            ot = self.typeof(node.obj)
            a  = node.attr
            if ot == WIN:
                if a in ('width','height'): return INT
                if a == 'open':   return BOOL
                if a == 'dt':     return FLT
                if a == 'mouse':  return LIST
            if ot == IMAGE:
                if a in ('width','height'): return INT
                if a == 'pixels': return LIST
            if ot == SND:
                if a == 'samples': return LIST
        if isinstance(node, Index):
            ot = self.typeof(node.obj)
            if ot == LIST:  return FLT
            if ot == IMAGE: return LIST   # pixel = [r,g,b,a]
        return UNK

    def ctype(self, t: str) -> str:
        return _C_TYPES.get(t, 'double')

    # ── Expression generator ─────────────────────────────────
    # Returns a C expression string.  May emit statements first.

    def expr(self, node) -> str:
        if isinstance(node, IntLit):   return str(node.value)
        if isinstance(node, FloatLit): return repr(node.value)
        if isinstance(node, BoolLit):  return '1' if node.value else '0'
        if isinstance(node, StrLit):
            esc = node.value.replace('\\','\\\\').replace('"','\\"').replace('\n','\\n')
            return f'"{esc}"'
        if isinstance(node, Ident):
            return node.name
        if isinstance(node, ListLit):
            return self._list_lit(node)
        if isinstance(node, BinOp):
            return self._binop(node)
        if isinstance(node, UnaryOp):
            return self._unary(node)
        if isinstance(node, Call):
            return self._call(node)
        if isinstance(node, Attr):
            return self._attr(node)
        if isinstance(node, Index):
            return self._index(node)
        raise CodeGenError(f"Unknown expr node {type(node).__name__}")

    def _list_lit(self, node: ListLit) -> str:
        n = len(node.elements)
        v = self.tmp('_lst')
        self.emit(f'FFList {v} = ff_list_new({n});')
        for i, e in enumerate(node.elements):
            self.emit(f'{v}.data[{i}] = (float)({self.expr(e)});')
        self._pending_frees.append(v)
        return v

    def _binop(self, node: BinOp) -> str:
        l = self.expr(node.left)
        r = self.expr(node.right)
        if node.op == '%':
            lt = self.typeof(node.left)
            rt = self.typeof(node.right)
            if FLT in (lt, rt):
                return f'fmod((double)({l}), (double)({r}))'
        op = {'and':'&&','or':'||'}.get(node.op, node.op)
        return f'({l} {op} {r})'

    def _unary(self, node: UnaryOp) -> str:
        op = {'not':'!'}.get(node.op, node.op)
        return f'({op}{self.expr(node.operand)})'

    def _call(self, node: Call) -> str:
        if isinstance(node.func, Ident):
            return self._builtin(node.func.name, node.args)
        if isinstance(node.func, Attr):
            return self._method(node.func.obj, node.func.attr, node.args)
        fn = self.expr(node.func)
        args = ', '.join(self.expr(a) for a in node.args)
        return f'{fn}({args})'

    def _builtin(self, name: str, args: List) -> str:
        def a(i): return self.expr(args[i])

        if name == 'zeros':
            t = self.tmp('_lt'); self.emit(f'FFList {t} = ff_list_zeros((int){a(0)});'); self._pending_frees.append(t); return t
        if name == 'ones':
            t = self.tmp('_lt'); self.emit(f'FFList {t} = ff_list_ones((int){a(0)});');  self._pending_frees.append(t); return t
        if name == 'random':
            if not args:
                # random() → single uniform float in [0, 1)
                return '((double)rand() / ((double)RAND_MAX + 1.0))'
            t = self.tmp('_lt'); self.emit(f'FFList {t} = ff_list_random((int){a(0)});'); self._pending_frees.append(t); return t
        if name == 'len':
            obj = a(0); t = self.typeof(args[0])
            return f'(int64_t){obj}.len' if t in (LIST, IMAGE, SND) else f'(int64_t)strlen({obj})'
        if name == 'float':   return f'(double)({a(0)})'
        if name == 'int':     return f'(int64_t)({a(0)})'
        if name == 'str':
            t = self.typeof(args[0]); v = a(0)
            if t == STR: return v
            if t == INT: return f'ff_str_int((int64_t)({v}))'
            return f'ff_str((double)({v}))'
        # ── String built-ins ──────────────────────────────────
        if name == 'fmt':
            buf = self.tmp('_sb')
            self.emit(f'char* {buf} = _ff_next_str_buf();')
            fmtarg = self.expr(args[0])
            rest = ', '.join(self.expr(args[i]) for i in range(1, len(args)))
            if rest:
                self.emit(f'snprintf({buf}, FF_STR_MAXLEN, {fmtarg}, {rest});')
            else:
                self.emit(f'snprintf({buf}, FF_STR_MAXLEN, "%s", {fmtarg});')
            return buf
        if name == 'strcat':  return f'ff_strcat({a(0)}, {a(1)})'
        if name == 'strlen':  return f'(int64_t)strlen({a(0)})'
        if name == 'streq':   return f'(strcmp({a(0)}, {a(1)}) == 0)'
        if name == 'toint':   return f'(int64_t)atoi({a(0)})'
        if name == 'tofloat': return f'atof({a(0)})'
        if name == 'strsub':
            s0 = a(0)
            if len(args) == 2:
                return f'ff_strsub({s0}, (int64_t)({a(1)}), (int64_t)strlen({s0}))'
            return f'ff_strsub({s0}, (int64_t)({a(1)}), (int64_t)({a(2)}))'
        if name == 'strfind': return f'ff_strfind({a(0)}, {a(1)})'
        if name == 'strtrim': return f'ff_strtrim({a(0)})'
        if name == 'abs':     return f'fabs((double)({a(0)}))'
        if name == 'sqrt':    return f'sqrt((double)({a(0)}))'
        if name == 'floor':   return f'(int64_t)floor((double)({a(0)}))'
        if name == 'ceil':    return f'(int64_t)ceil((double)({a(0)}))'
        if name == 'round':   return f'(int64_t)round((double)({a(0)}))'
        if name == 'sin':     return f'sin((double)({a(0)}))'
        if name == 'cos':     return f'cos((double)({a(0)}))'
        if name == 'tan':     return f'tan((double)({a(0)}))'
        if name == 'asin':    return f'asin((double)({a(0)}))'
        if name == 'acos':    return f'acos((double)({a(0)}))'
        if name == 'atan':    return f'atan((double)({a(0)}))'
        if name == 'atan2':   return f'atan2((double)({a(0)}), (double)({a(1)}))'
        if name == 'radians': return f'((double)({a(0)}) * 0.017453292519943295)'
        if name == 'degrees': return f'((double)({a(0)}) * 57.29577951308232)'
        if name == 'clamp':   return f'ff_clamp((double){a(0)}, (double){a(1)}, (double){a(2)})'
        if name == 'sum':     return f'ff_list_sum({a(0)})'
        if name == 'min':     return f'ff_list_min({a(0)})'
        if name == 'max':     return f'ff_list_max({a(0)})'
        if name == 'time':    return 'ff_time()'
        if name == 'sleep':   return f'(ff_sleep((double){a(0)}), 0)'
        if name == 'play':    return f'(ff_sound_play({a(0)}), 0)'
        if name == 'window':
            w = a(0); h = a(1)
            title = a(2) if len(args) > 2 else '"FFLang"'
            return f'ff_window_new((int){w}, (int){h}, {title})'
        if name == 'image':
            return f'ff_image_new((int){a(0)}, (int){a(1)})'
        if name == 'sound':
            return f'ff_sound_new((int){a(0)})'
        if name == 'load_image':  return f'ff_load_image({a(0)})'
        if name == 'load_sound':  return f'ff_load_sound({a(0)})'
        if name == 'save_image':  return f'(ff_save_image({a(0)}, {a(1)}), 0)'
        if name == 'read_file':   return f'ff_read_file({a(0)})'
        if name == 'exists':      return f'ff_file_exists({a(0)})'
        if name == 'save_list':
            t = self.tmp('_lt')
            self.emit(f'FFList {t} = {a(1)};')
            self.emit(f'ff_save_list({a(0)}, {t});')
            return '0'
        if name == 'load_list':
            t = self.tmp('_lt')
            self.emit(f'FFList {t} = ff_load_list({a(0)});')
            self._pending_frees.append(t)
            return t
        if name == 'write_file':  return f'(ff_write_file({a(0)}, {a(1)}), 0)'
        if name == 'print':
            self._print(args); return '0'
        if name == 'gpu':
            return self._gpu(args)
        if name == 'range':
            # range only valid in for loops; codegen handles it there
            return f'/*range({", ".join(a(i) for i in range(len(args)))})*/'
        # User-defined function
        cargs = ', '.join(self.expr(x) for x in args)
        return f'{name}({cargs})'

    def _method(self, obj_node, attr: str, args: List) -> str:
        obj = self.expr(obj_node)
        ot  = self.typeof(obj_node)
        def a(i): return self.expr(args[i])

        if ot == WIN:
            if attr == 'tick':
                fps = a(0) if args else '0'
                return f'(ff_window_tick(&{obj}, (int)({fps})), 0)'
            if attr == 'draw':
                if len(args) == 1:
                    return f'(ff_window_draw_image(&{obj}, &{a(0)}), 0)'
                return f'(ff_window_draw(&{obj}, {a(0)}.data, (int){a(1)}, (int){a(2)}), 0)'
            if attr == 'key':         return f'ff_window_key(&{obj}, {a(0)})'
            if attr == 'key_pressed': return f'ff_window_key_pressed(&{obj}, {a(0)})'
            if attr == 'mouse_down':  return f'ff_window_mouse_down(&{obj}, (int){a(0)})'

        if ot == IMAGE:
            if attr == 'fill':
                return f'(ff_image_fill(&{obj}, (float){a(0)}, (float){a(1)}, (float){a(2)}, (float){a(3)}), 0)'

        if ot == LIST:
            if attr == 'pin':    return f'(ff_list_pin(&{obj}), 0)'
            if attr == 'unpin':  return f'(ff_list_unpin(&{obj}), 0)'
            if attr == 'sync':   return f'(ff_list_sync(&{obj}), 0)'
            if attr == 'append': return f'(ff_list_append(&{obj}, (float)({a(0)})), 0)'
            if attr == 'extend': return f'(ff_list_extend(&{obj}, {a(0)}), 0)'
            if attr == 'pop':    return f'ff_list_pop(&{obj})'
            if attr == 'resize': return f'(ff_list_resize(&{obj}, (int)({a(0)})), 0)'
            if attr == 'clear':  return f'(ff_list_clear(&{obj}), 0)'

        cargs = ', '.join(self.expr(x) for x in args)
        return f'{obj}_{attr}({cargs})'  # fallback

    def _attr(self, node: Attr) -> str:
        obj = self.expr(node.obj)
        ot  = self.typeof(node.obj)
        a   = node.attr
        if ot == WIN:
            if a == 'open':   return f'ff_window_open(&{obj})'
            if a == 'dt':     return f'ff_window_dt(&{obj})'
            if a == 'width':  return f'(int64_t){obj}.width'
            if a == 'height': return f'(int64_t){obj}.height'
            if a == 'mouse':  return f'ff_window_mouse(&{obj})'
        if ot == IMAGE:
            if a == 'pixels': return f'{obj}.pixels'
            if a == 'width':  return f'(int64_t){obj}.width'
            if a == 'height': return f'(int64_t){obj}.height'
        if ot == SND:
            if a == 'samples': return f'{obj}.samples'
        return f'{obj}.{a}'

    def _index(self, node: Index) -> str:
        obj = self.expr(node.obj)
        ot  = self.typeof(node.obj)
        if ot == LIST:
            idx = f'(int)({self.expr(node.indices[0])})'
            # If indexing a temporary list (e.g. random(1)[0]), extract the
            # element into a scalar now and free the list immediately so it
            # cannot be used-after-free when _flush_pending_frees runs later.
            if obj in self._pending_frees:
                self._pending_frees.remove(obj)
                sv = self.tmp('_sv')
                self.emit(f'double {sv} = {obj}.data[{idx}];')
                self.emit(f'ff_list_free(&{obj});')
                return sv
            return f'{obj}.data[{idx}]'
        if ot == IMAGE and len(node.indices) == 2:
            x = self.expr(node.indices[0])
            y = self.expr(node.indices[1])
            return f'ff_image_get(&{obj}, (int){x}, (int){y})'
        return f'{obj}[{self.expr(node.indices[0])}]'

    def _print(self, args: List):
        if not args:
            self.emit('printf("\\n");')
            return
        parts, cargs = [], []
        for arg in args:
            t = self.typeof(arg)
            v = self.expr(arg)
            if t == LIST:
                self.emit(f'ff_list_print({v});')
                return
            if t == INT:
                parts.append('%lld'); cargs.append(f'(long long)({v})')
            elif t == FLT:
                parts.append('%g');   cargs.append(f'(double)({v})')
            elif t == BOOL:
                parts.append('%s');   cargs.append(f'(({v}) ? "true" : "false")')
            elif t == STR:
                parts.append('%s');   cargs.append(v)
            else:
                parts.append('%g');   cargs.append(f'(double)({v})')
        fmt = ' '.join(parts) + '\\n'
        self.emit(f'printf("{fmt}", {", ".join(cargs)});')

    def _gpu(self, args: List) -> str:
        if not args:
            return 'ff_list_zeros(0)'
        kernel_name = args[0].name if isinstance(args[0], Ident) else '??'
        kid = f'_kernel_{kernel_name}'
        inputs = args[1:]

        if len(inputs) == 2 and all(self.typeof(x) == LIST for x in inputs):
            # Fast path: simple runkernel
            a = self.expr(inputs[0])
            b = self.expr(inputs[1])
            sz = f'{a}.len'
            out = self.tmp('_gpu')
            self.emit(f'FFList {out} = ff_list_zeros({sz});')
            self.emit(f'runkernel({kid}, {sz}, {a}.data, {b}.data, {out}.data);')
            return out

        # General path: advrunkernel
        sz = None
        rb_bufs = []   # list of (cl_mem varname, owned: bool)
        adv_args = []
        for inp in inputs:
            t = self.typeof(inp)
            v = self.expr(inp)
            if t == LIST:
                bv = self.tmp('_rb')
                if sz is None:
                    sz = f'{v}.len'
                # Use the pinned gpu_buf if available; otherwise allocate a temp one
                self.emit(f'cl_mem {bv} = {v}.pinned ? {v}.gpu_buf : loadbuf({v}.data, {v}.len, 1);')
                adv_args.append(f'&{bv}, sizeof(cl_mem)')
                rb_bufs.append((bv, v))
            else:
                sv = self.tmp('_sv')
                if t == INT:
                    self.emit(f'int {sv} = (int)({v});')
                    adv_args.append(f'&{sv}, sizeof(int)')
                else:
                    self.emit(f'float {sv} = (float)({v});')
                    adv_args.append(f'&{sv}, sizeof(float)')

        if sz is None:
            sz = '0'

        out = self.tmp('_gpu')
        out_buf = self.tmp('_obuf')
        self.emit(f'FFList {out} = ff_list_zeros({sz});')
        self.emit(f'cl_mem {out_buf} = loadbuf(NULL, {sz}, 0);')

        adv_args.append(f'&{out_buf}, sizeof(cl_mem)')
        self.emit(f'advrunkernel({kid}, (size_t){sz}, {len(adv_args)}, {", ".join(adv_args)});')

        self.emit(f'readbuf({out_buf}, {out}.data, {sz});')
        for bv, src in rb_bufs:
            self.emit(f'if (!{src}.pinned) freebuf({bv});')
        self.emit(f'freebuf({out_buf});')

        self._pending_frees.append(out)
        return out

    # ── Statement generator ──────────────────────────────────

    def stmt(self, node):
        if node is None:
            return
        if isinstance(node, KernelDecl):
            return  # handled at top level
        if isinstance(node, FuncDef):
            self._func_def(node); return
        if isinstance(node, Assignment):
            self._assign(node); return
        if isinstance(node, If):
            self._if(node); return
        if isinstance(node, While):
            self._while(node); return
        if isinstance(node, For):
            self._for(node); return
        if isinstance(node, Return):
            self._pending_frees = []
            val = self.expr(node.value) if node.value else None
            self._flush_pending_frees(keep=val)
            self.emit(f'return {val};' if val else 'return;'); return
        if isinstance(node, Break):
            self.emit('break;'); return
        if isinstance(node, Continue):
            self.emit('continue;'); return
        if isinstance(node, RunKernel):
            self._run_kernel(node); return
        if isinstance(node, ExprStmt):
            self._expr_stmt(node); return
        if isinstance(node, GlobalStmt):
            return  # handled at file scope; nothing to emit in main()
        raise CodeGenError(f"Unknown stmt node {type(node).__name__}")

    def _flush_pending_frees(self, keep: str = None):
        """Free every temp list accumulated since the last statement boundary,
        except the one being kept (assigned to a variable or returned)."""
        for t in self._pending_frees:
            if t != keep:
                self.emit(f'ff_list_free(&{t});')
        self._pending_frees = []

    def _expr_stmt(self, node: ExprStmt):
        self._pending_frees = []
        # print() already emits statements; handle it directly
        e = node.expr
        if isinstance(e, Call) and isinstance(e.func, Ident) and e.func.name == 'print':
            self._print(e.args)
            self._flush_pending_frees()
            return
        v = self.expr(e)
        t = self.typeof(e)
        if t != VOID and v not in ('0',):
            self.emit(f'(void)({v});')
        self._flush_pending_frees()

    def _expr_uses(self, name: str, node) -> bool:
        """Return True if the expression references the variable 'name'."""
        if node is None: return False
        if isinstance(node, Ident): return node.name == name
        if isinstance(node, BinOp):
            return self._expr_uses(name, node.left) or self._expr_uses(name, node.right)
        if isinstance(node, UnaryOp): return self._expr_uses(name, node.operand)
        if isinstance(node, Call):
            if isinstance(node.func, Attr) and self._expr_uses(name, node.func.obj): return True
            return any(self._expr_uses(name, a) for a in node.args)
        if isinstance(node, Index):
            return self._expr_uses(name, node.obj) or any(self._expr_uses(name, i) for i in node.indices)
        if isinstance(node, Attr): return self._expr_uses(name, node.obj)
        if isinstance(node, ListLit): return any(self._expr_uses(name, e) for e in node.elements)
        return False

    def _assign(self, node: Assignment):
        self._pending_frees = []
        vt  = self.typeof(node.value)
        val = self.expr(node.value)

        if isinstance(node.target, Ident):
            name = node.target.name
            existing = self.scope.get(name)
            ct = self.ctype(vt) if vt != UNK else 'double'
            is_global = name in self._global_names
            if not existing:
                self.scope.set(name, vt)
                if is_global:
                    # File-scope declaration already emitted; just assign
                    self.emit(f'{name} = {val};')
                else:
                    self.emit(f'{ct} {name} = {val};')
            else:
                if existing == LIST and not self._expr_uses(name, node.value):
                    self.emit(f'ff_list_free(&{name});')
                self.emit(f'{name} = {val};')
            # Keep the temp that became this variable; free the rest
            self._flush_pending_frees(keep=val)

        elif isinstance(node.target, Index):
            obj = self.expr(node.target.obj)
            ot  = self.typeof(node.target.obj)
            if ot == LIST:
                idx = self.expr(node.target.indices[0])
                self.emit(f'{obj}.data[(int)({idx})] = (float)({val});')
            elif ot == IMAGE and len(node.target.indices) == 2:
                x = self.expr(node.target.indices[0])
                y = self.expr(node.target.indices[1])
                self.emit(f'ff_image_set(&{obj}, (int){x}, (int){y}, {val});')
            self._flush_pending_frees()

        elif isinstance(node.target, Attr):
            obj = self.expr(node.target.obj)
            ot  = self.typeof(node.target.obj)
            a   = node.target.attr
            if ot == SND and a == 'samples':
                self.emit(f'ff_sound_set_samples(&{obj}, {val});')
            else:
                self.emit(f'{obj}.{a} = {val};')
            self._flush_pending_frees()

    def _scope_mini(self) -> dict:
        """Snapshot the current scope chain as a flat dict for _quick_type."""
        mini = {}
        s = self.scope
        while s:
            for k, v in s.vars.items():
                if k not in mini:
                    mini[k] = v
            s = s.parent
        return mini

    def _hoist_branch_vars(self, all_branches):
        """Hoist scalar (non-list) variables that are first-assigned inside any
        branch to the current scope so they're visible after the if/else.
        Returns nothing; emits declarations and updates self.scope as a side-effect."""
        mini = self._scope_mini()
        hoisted = {}
        for branch in all_branches:
            for s in branch:
                if isinstance(s, Assignment) and isinstance(s.target, Ident):
                    name = s.target.name
                    if name not in mini and name not in hoisted:
                        vt = self._quick_type(s.value, mini)
                        # Only hoist scalars — lists need explicit allocation
                        if vt in (INT, FLT, BOOL, STR):
                            hoisted[name] = vt
        for name, vt in hoisted.items():
            ct = self.ctype(vt)
            default = ('0'    if vt in (INT, BOOL) else
                       'NULL' if vt == STR         else '0.0')
            self.emit(f'{ct} {name} = {default};')
            self.scope.set(name, vt)

    def _if(self, node: If):
        # Hoist first-assigned scalars so they're visible after the if/else
        all_branches = [node.body]
        for _, eb in node.elifs:
            all_branches.append(eb)
        if node.else_body is not None:
            all_branches.append(node.else_body)
        self._hoist_branch_vars(all_branches)

        self.emit(f'if ({self.expr(node.cond)}) {{')
        self.indent += 1
        self.push_scope()
        for s in node.body: self.stmt(s)
        self.pop_scope()
        self.indent -= 1
        for ec, eb in node.elifs:
            self.emit(f'}} else if ({self.expr(ec)}) {{')
            self.indent += 1
            self.push_scope()
            for s in eb: self.stmt(s)
            self.pop_scope()
            self.indent -= 1
        if node.else_body is not None:
            self.emit('} else {')
            self.indent += 1
            self.push_scope()
            for s in node.else_body: self.stmt(s)
            self.pop_scope()
            self.indent -= 1
        self.emit('}')

    def _free_scope_lists(self):
        """Emit ff_list_free for every list variable owned by the current scope.
        Called at the end of loop bodies so per-iteration allocations don't leak."""
        for name, typ in self.scope.vars.items():
            if typ == LIST:
                self.emit(f'ff_list_free(&{name});')

    def _while(self, node: While):
        self.emit(f'while ({self.expr(node.cond)}) {{')
        self.indent += 1
        self.push_scope()
        for s in node.body: self.stmt(s)
        self.pop_scope()
        self.indent -= 1
        self.emit('}')

    def _for(self, node: For):
        it = node.iter
        var = node.var
        self.push_scope()
        if isinstance(it, Call) and isinstance(it.func, Ident) and it.func.name == 'range':
            a = it.args
            if   len(a) == 1: self.emit(f'for (int64_t {var} = 0; {var} < {self.expr(a[0])}; {var}++) {{')
            elif len(a) == 2: self.emit(f'for (int64_t {var} = {self.expr(a[0])}; {var} < {self.expr(a[1])}; {var}++) {{')
            else:              self.emit(f'for (int64_t {var} = {self.expr(a[0])}; {var} < {self.expr(a[1])}; {var} += {self.expr(a[2])}) {{')
            self.scope.set(var, INT)
        else:
            lst = self.expr(it)
            iv  = self.tmp('_i')
            self.emit(f'for (int __{iv} = 0; __{iv} < {lst}.len; __{iv}++) {{')
            self.indent += 1
            self.emit(f'double {var} = {lst}.data[__{iv}];')
            self.indent -= 1
            self.scope.set(var, FLT)
        self.indent += 1
        for s in node.body: self.stmt(s)
        self.pop_scope()
        self.indent -= 1
        self.emit('}')

    def _run_kernel(self, node: RunKernel):
        kid = f'_kernel_{node.kernel}'

        out_exprs = [self.expr(o) for o in node.outputs]
        sz = self.expr(node.dispatch) if node.dispatch else f'{out_exprs[0]}.len'

        # Output buffers: reuse pinned gpu_buf if available, else allocate fresh
        out_bufs = []
        for out in out_exprs:
            ob = self.tmp('_ob')
            self.emit(f'cl_mem {ob} = {out}.pinned ? {out}.gpu_buf : loadbuf({out}.data, {out}.len, 0);')
            out_bufs.append((ob, out))

        rb_bufs = []   # list of (cl_mem varname, src list varname)
        adv_args = []
        for arg in node.args:
            t = self.typeof(arg)
            v = self.expr(arg)
            if t == LIST:
                bv = self.tmp('_rb')
                # Use the pinned gpu_buf if available; otherwise allocate a temp one
                self.emit(f'cl_mem {bv} = {v}.pinned ? {v}.gpu_buf : loadbuf({v}.data, {v}.len, 1);')
                adv_args.append(f'&{bv}, sizeof(cl_mem)')
                rb_bufs.append((bv, v))
            else:
                sv = self.tmp('_sv')
                if t == INT:
                    self.emit(f'int {sv} = (int)({v});')
                    adv_args.append(f'&{sv}, sizeof(int)')
                else:
                    self.emit(f'float {sv} = (float)({v});')
                    adv_args.append(f'&{sv}, sizeof(float)')

        adv_args += [f'&{ob}, sizeof(cl_mem)' for ob, _ in out_bufs]
        self.emit(f'advrunkernel({kid}, (size_t){sz}, {len(adv_args)}, {", ".join(adv_args)});')

        # Readback: only needed for un-pinned outputs (pinned ones stay on GPU until unpin)
        for ob, out in out_bufs:
            self.emit(f'if (!{out}.pinned) readbuf({ob}, {out}.data, {out}.len);')

        # Free only buffers we allocated ourselves (not pinned ones)
        for bv, src in rb_bufs:
            self.emit(f'if (!{src}.pinned) freebuf({bv});')
        for ob, out in out_bufs:
            self.emit(f'if (!{out}.pinned) freebuf({ob});')

        for out_node in node.outputs:
            if isinstance(out_node, Ident):
                self.scope.set(out_node.name, LIST)

    def _quick_type(self, node, mini: dict) -> str:
        """Lightweight type inference that only uses a flat name→type dict.
        Used during the pre-scan pass before the real scope is set up."""
        if isinstance(node, IntLit):   return INT
        if isinstance(node, FloatLit): return FLT
        if isinstance(node, BoolLit):  return BOOL
        if isinstance(node, StrLit):   return STR
        if isinstance(node, ListLit):  return LIST
        if isinstance(node, Ident):
            return mini.get(node.name, UNK)
        if isinstance(node, BinOp):
            if node.op in ('==','!=','<','>','<=','>=','and','or'): return BOOL
            lt = self._quick_type(node.left,  mini)
            rt = self._quick_type(node.right, mini)
            return FLT if FLT in (lt, rt) else lt
        if isinstance(node, UnaryOp):
            return BOOL if node.op == 'not' else self._quick_type(node.operand, mini)
        if isinstance(node, Call):
            fn = node.func
            if isinstance(fn, Ident):
                if fn.name == 'random' and not node.args:
                    return FLT
                return (_RETURN_TYPES.get(fn.name)
                        or self._func_return_types.get(fn.name)
                        or UNK)
        if isinstance(node, Index):
            return FLT if self._quick_type(node.obj, mini) == LIST else UNK
        return UNK

    def _scan_expr_calls(self, node, mini: dict, ptypes: dict):
        """Recursively walk an expression and, for every call to a user function,
        promote parameter slots to LIST when a list argument is detected."""
        if node is None:
            return
        if isinstance(node, Call):
            fn = node.func
            if isinstance(fn, Ident) and fn.name in ptypes:
                slots = ptypes[fn.name]
                for i, arg in enumerate(node.args):
                    if i < len(slots):
                        qt = self._quick_type(arg, mini)
                        if qt in (LIST, STR):
                            slots[i] = qt
            args_to_scan = node.args
            if isinstance(fn, Attr):
                self._scan_expr_calls(fn.obj, mini, ptypes)
            for a in args_to_scan:
                self._scan_expr_calls(a, mini, ptypes)
        elif isinstance(node, BinOp):
            self._scan_expr_calls(node.left,    mini, ptypes)
            self._scan_expr_calls(node.right,   mini, ptypes)
        elif isinstance(node, UnaryOp):
            self._scan_expr_calls(node.operand, mini, ptypes)
        elif isinstance(node, Index):
            self._scan_expr_calls(node.obj, mini, ptypes)
            for idx in node.indices:
                self._scan_expr_calls(idx, mini, ptypes)
        elif isinstance(node, ListLit):
            for e in node.elements:
                self._scan_expr_calls(e, mini, ptypes)

    def _collect_call_types(self, stmts, mini: dict, ptypes: dict):
        """Walk statements, tracking assignments to build mini-scope, and
        record argument types at every user-function call site."""
        for s in stmts:
            if isinstance(s, Assignment):
                # Scan the RHS first so argument types are checked against the
                # pre-assignment mini-scope (e.g. fb = setpixel(fb,...) must
                # see fb as LIST when scanning, not the overwritten type).
                self._scan_expr_calls(s.value, mini, ptypes)
                if isinstance(s.target, Ident):
                    t = self._quick_type(s.value, mini)
                    mini[s.target.name] = t
            elif isinstance(s, ExprStmt):
                self._scan_expr_calls(s.expr, mini, ptypes)
            elif isinstance(s, Return) and s.value:
                self._scan_expr_calls(s.value, mini, ptypes)
            elif isinstance(s, If):
                self._scan_expr_calls(s.cond, mini, ptypes)
                self._collect_call_types(s.body,      dict(mini), ptypes)
                for _, eb in s.elifs:
                    self._collect_call_types(eb,       dict(mini), ptypes)
                if s.else_body:
                    self._collect_call_types(s.else_body, dict(mini), ptypes)
            elif isinstance(s, While):
                self._scan_expr_calls(s.cond, mini, ptypes)
                self._collect_call_types(s.body, dict(mini), ptypes)
            elif isinstance(s, For):
                self._collect_call_types(s.body, dict(mini), ptypes)

    def _collect_global_names(self, funcs: List, main_body: List):
        """Collect names declared 'global' in any function body, infer their
        types from main_body assignments, and populate self._global_names."""
        # Step 1: gather names
        raw: set = set()
        def _scan_body(stmts):
            for s in stmts:
                if isinstance(s, GlobalStmt):
                    raw.update(s.names)
                elif isinstance(s, If):
                    _scan_body(s.body)
                    for _, eb in s.elifs: _scan_body(eb)
                    if s.else_body: _scan_body(s.else_body)
                elif isinstance(s, (While, For)):
                    _scan_body(s.body)
        for f in funcs:
            _scan_body(f.body)

        if not raw:
            return

        # Step 2: infer types from main_body (first-assignment wins)
        mini: dict = {}
        def _scan_main(stmts):
            for s in stmts:
                if isinstance(s, Assignment) and isinstance(s.target, Ident):
                    name = s.target.name
                    if name not in mini:
                        t = self._quick_type(s.value, mini)
                        mini[name] = t if t != UNK else FLT
                elif isinstance(s, If):
                    _scan_main(s.body)
                    for _, eb in s.elifs: _scan_main(eb)
                    if s.else_body: _scan_main(s.else_body)
                elif isinstance(s, (While, For)):
                    _scan_main(s.body)
        _scan_main(main_body)

        for name in raw:
            self._global_names[name] = mini.get(name, FLT)

    def _func_return_type(self, node: FuncDef) -> str:
        """Pre-scan a function body to determine its return type."""
        # Use already-inferred param types so return-type inference is accurate
        ptypes = self._func_param_types.get(node.name, [])
        mini: dict = {p: (ptypes[i] if i < len(ptypes) else FLT)
                      for i, p in enumerate(node.params)}

        def scan(stmts):
            for s in stmts:
                if isinstance(s, Assignment) and isinstance(s.target, Ident):
                    mini[s.target.name] = self._quick_type(s.value, mini)
                elif isinstance(s, Return) and s.value is not None:
                    return self._quick_type(s.value, mini)
                elif isinstance(s, If):
                    r = scan(s.body)
                    if r: return r
                    for _, eb in s.elifs:
                        r = scan(eb)
                        if r: return r
                    if s.else_body:
                        r = scan(s.else_body)
                        if r: return r
                elif isinstance(s, (While, For)):
                    r = scan(s.body)
                    if r: return r
            return None

        return scan(node.body) or FLT   # default: scalar

    def _func_def(self, node: FuncDef):
        rt      = self._func_return_types.get(node.name, FLT)
        c_ret   = self.ctype(rt) if rt not in (UNK, VOID) else 'double'
        ptypes  = self._func_param_types.get(node.name, [FLT] * len(node.params))
        params  = ', '.join(
            f'{self.ctype(t)} {p}' for p, t in zip(node.params, ptypes)
        )
        self.emit(f'static {c_ret} {node.name}({params}) {{')
        self.indent += 1
        self.push_scope()
        for p, t in zip(node.params, ptypes):
            self.scope.set(p, t)
        # Seed global names into function scope so the codegen can resolve their types
        for gname, gtype in self._global_names.items():
            self.scope.set(gname, gtype)
        for s in node.body:
            if isinstance(s, GlobalStmt):
                continue  # already seeded above; no C output needed
            self.stmt(s)
        # Emit a fallback return so C doesn't warn about missing return on non-void
        # functions that have no explicit return (e.g. procedures that sort in-place).
        if c_ret != 'void':
            default = ('ff_list_zeros(0)' if c_ret == 'FFList'
                       else '0' if c_ret in ('int64_t', 'int') else '0.0')
            self.emit(f'return {default};')
        self.pop_scope()
        self.indent -= 1
        self.emit('}')
        self.emit()

    # ── Top-level generate ───────────────────────────────────

    def generate(self, program: Program) -> str:
        kernels   = [s for s in program.stmts if isinstance(s, KernelDecl)]
        funcs     = [s for s in program.stmts if isinstance(s, FuncDef)]
        main_body = [s for s in program.stmts if not isinstance(s, (KernelDecl, FuncDef))]

        self.lines = []
        self.indent = 0

        self.emit('#include "fflang_rt.h"')
        self.emit()

        # Map FFLang annotation names to type tags
        _ANN = {'int': INT, 'float': FLT, 'bool': BOOL, 'str': STR,
                'list': LIST, 'image': IMAGE, 'window': WIN, 'sound': SND}

        # Pass 1 — infer parameter types from call sites.
        # Seed with explicit annotations first; those slots won't be overwritten.
        self._func_param_types = {f.name: [FLT] * len(f.params) for f in funcs}
        for f in funcs:
            anns = f.param_types or []
            for i, ann in enumerate(anns):
                if ann and ann in _ANN:
                    self._func_param_types[f.name][i] = _ANN[ann]

        # Iterate until stable so multi-level chains (A calls B calls C) resolve.
        self._collect_call_types(main_body, {}, self._func_param_types)
        for _ in range(5):   # up to 5 levels deep
            prev = {k: list(v) for k, v in self._func_param_types.items()}
            for f in funcs:
                pmini = {p: self._func_param_types[f.name][i]
                         for i, p in enumerate(f.params)}
                self._collect_call_types(f.body, pmini, self._func_param_types)
            if self._func_param_types == prev:
                break

        # Re-apply annotations — inference must never override explicit types
        for f in funcs:
            anns = f.param_types or []
            for i, ann in enumerate(anns):
                if ann and ann in _ANN:
                    self._func_param_types[f.name][i] = _ANN[ann]

        # Pass 2 — infer return types (now that param types are known)
        for f in funcs:
            self._func_return_types[f.name] = self._func_return_type(f)

        # Pass 3 — collect 'global' declarations and infer their types
        self._collect_global_names(funcs, main_body)

        # File-scope declarations for globals (zero-init; main() fills them in)
        _GLOB_ZERO = {
            INT: '0', FLT: '0.0', BOOL: '0', STR: 'NULL',
            LIST: '{0}', IMAGE: '{0}', WIN: '{0}', SND: '{0}',
        }
        if self._global_names:
            for gname, gtype in self._global_names.items():
                ct   = self.ctype(gtype)
                zero = _GLOB_ZERO.get(gtype, '0')
                self.emit(f'static {ct} {gname} = {zero};')
            self.emit()

        # Forward declarations
        for f in funcs:
            rt     = self._func_return_types[f.name]
            c_ret  = self.ctype(rt) if rt not in (UNK, VOID) else 'double'
            ptypes = self._func_param_types[f.name]
            params = ', '.join(
                f'{self.ctype(t)} {p}' for p, t in zip(f.params, ptypes)
            )
            self.emit(f'static {c_ret} {f.name}({params});')
        if funcs:
            self.emit()

        # Function definitions
        for f in funcs:
            self._func_def(f)

        # main
        self.emit('int main(void) {')
        self.indent += 1

        if kernels:
            self.emit('/* load kernels */')
            for k in kernels:
                var = f'_kernel_{k.name}'
                self._kernel_ids[k.name] = var
                self.emit(f'int {var} = loadkernel("{k.cl_file}", "{k.func_name}");')
                self.emit(f'if ({var} < 0) {{ fprintf(stderr, "kernel {k.name} failed\\n"); return 1; }}')
            self.emit()

        # Register kernel names in scope
        self.scope = Scope()
        for k in kernels:
            self.scope.set(k.name, 'kernel')

        for s in main_body:
            self.stmt(s)

        self.emit()
        self.emit('cleanupkernel();')
        self.emit('return 0;')
        self.indent -= 1
        self.emit('}')

        return '\n'.join(self.lines)
