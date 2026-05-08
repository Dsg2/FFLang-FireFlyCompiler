import sys
import os
import subprocess
import shutil
import json

from .lexer     import Lexer,  LexError
from .parser    import Parser, ParseError
from .codegen   import CodeGen, CodeGenError
from .ast_nodes import KernelDecl, FuncDef
from .flc       import compile_fl, FLError

# ── Path resolution ──────────────────────────────────────────

_FFC_DIR  = os.path.dirname(os.path.abspath(__file__))
FFLANG    = os.path.dirname(_FFC_DIR)          # repo root (parent of ffc/)
_CFG_FILE = os.path.join(_FFC_DIR, 'paths.cfg')

# Common GCC locations to probe before falling back to PATH
_GCC_CANDIDATES = [
    r'C:\msys64\mingw64\bin\gcc.exe',
    r'C:\msys2\mingw64\bin\gcc.exe',
    r'C:\mingw64\bin\gcc.exe',
    r'C:\mingw\bin\gcc.exe',
    r'C:\Program Files\mingw-w64\mingw64\bin\gcc.exe',
]


def _load_cfg() -> dict:
    try:
        return json.load(open(_CFG_FILE)) if os.path.isfile(_CFG_FILE) else {}
    except Exception:
        return {}


def _save_cfg(cfg: dict):
    try:
        json.dump(cfg, open(_CFG_FILE, 'w'), indent=2)
    except Exception as e:
        print(f'ffc: warning: could not save paths.cfg: {e}', file=sys.stderr)


def _auto_gcc() -> str:
    for c in _GCC_CANDIDATES:
        if os.path.isfile(c):
            return c
    return shutil.which('gcc') or ''


def _prompt_path(key: str, label: str, cfg: dict) -> str:
    """Interactively ask the user for a missing path. Saves to cfg if confirmed."""
    while True:
        try:
            val = input(f'  Enter path to {label}: ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return ''
        if not val:
            return ''
        val = os.path.expandvars(os.path.expanduser(val))
        if os.path.exists(val):
            try:
                save = input('  Save for future runs? [Y/n]: ').strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                save = 'n'
            if save != 'n':
                cfg[key] = val
                _save_cfg(cfg)
                print(f'  Saved to {_CFG_FILE}')
            return val
        print(f'  ffc: not found: {val!r}')


def _resolve_paths(need_gcc: bool, need_klt: bool) -> dict:
    """
    Return a dict with keys: gcc, klt, include, lib.

    Resolution order for each path:
      1. Saved override in ffc/paths.cfg
      2. Auto-detected (relative to FFLANG or searched on disk)
      3. Interactive prompt if still missing (only for paths actually needed)
    """
    cfg = _load_cfg()

    # Defaults: everything relative to the repo root so the package is portable
    paths = {
        'gcc':     cfg.get('gcc')     or _auto_gcc(),
        'klt':     cfg.get('klt')     or os.path.join(FFLANG, 'klt.exe'),
        'include': cfg.get('include') or os.path.join(FFLANG, 'include'),
        'lib':     cfg.get('lib')     or os.path.join(FFLANG, 'lib'),
    }

    # Check which required paths are missing
    needed = []
    if need_gcc:
        if not os.path.isfile(paths['gcc']):
            needed.append(('gcc',     'GCC compiler (gcc.exe)'))
        if not os.path.isdir(paths['include']):
            needed.append(('include', f'include/ directory — expected at {paths["include"]}'))
        if not os.path.isdir(paths['lib']):
            needed.append(('lib',     f'lib/ directory — expected at {paths["lib"]}'))
    if need_klt:
        if not os.path.isfile(paths['klt']):
            needed.append(('klt',     f'klt.exe — expected at {paths["klt"]}'))

    if needed:
        print(f'ffc: {len(needed)} path(s) could not be located automatically:', file=sys.stderr)
        for key, label in needed:
            print(f'  missing: {label}', file=sys.stderr)
        print(file=sys.stderr)
        for key, label in needed:
            val = _prompt_path(key, label, cfg)
            if val:
                paths[key] = val

    return paths


# ── Helpers ──────────────────────────────────────────────────

USAGE = """\
Usage: python -m ffc <source.ff> [options]

Options:
  --emit-c    Write generated .c only, do not compile
  --run       Run the executable after a successful build
  --no-opencl Compile without OpenCL (no GPU calls)
  --debug     Add bounds-checking and debug symbols
"""


def die(msg: str, code: int = 1):
    print(f'ffc: {msg}', file=sys.stderr)
    sys.exit(code)


# ── Kernel transpilation ─────────────────────────────────────

def fl_kernels(ast, src_dir: str):
    """Compile any .fl files referenced by kernel declarations using flc.
    Mutates KernelDecl.cl_file: foo.fl → foo.cl in place."""
    for node in ast.stmts:
        if not isinstance(node, KernelDecl):
            continue
        if not node.cl_file.endswith('.fl'):
            continue

        fl_path = os.path.join(src_dir, node.cl_file)
        cl_file = node.cl_file[:-3] + '.cl'
        cl_path = os.path.join(src_dir, cl_file)

        if not os.path.isfile(fl_path):
            die(f'kernel source not found: {fl_path}')

        try:
            src    = open(fl_path, 'r').read()
            cl_src = compile_fl(src)
        except FLError as e:
            print(f'flc error for {node.cl_file}: {e}', file=sys.stderr)
            sys.exit(1)

        with open(cl_path, 'w') as f:
            f.write(cl_src)

        print(f'  flc: {node.cl_file} -> {cl_file}')
        node.cl_file = cl_file


def klt_kernels(ast, src_dir: str, klt_path: str):
    """Run klt on any .kl files referenced by kernel declarations.
    Mutates KernelDecl.cl_file: foo.kl → foo.cl in place."""
    for node in ast.stmts:
        if not isinstance(node, KernelDecl):
            continue
        if not node.cl_file.endswith('.kl'):
            continue

        kl_path = os.path.join(src_dir, node.cl_file)
        cl_file = node.cl_file[:-3] + '.cl'
        cl_path = os.path.join(src_dir, cl_file)

        if not os.path.isfile(kl_path):
            die(f'kernel source not found: {kl_path}')
        if not os.path.isfile(klt_path):
            die(f'klt not found at {klt_path}')

        result = subprocess.run(
            [klt_path, kl_path, '-o', cl_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f'klt error for {node.cl_file}:', file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            sys.exit(1)

        print(f'  klt: {node.cl_file} -> {cl_file}')
        node.cl_file = cl_file


def _has_kl(ast) -> bool:
    return any(isinstance(n, KernelDecl) and n.cl_file.endswith('.kl')
               for n in ast.stmts)


# ── C compilation ────────────────────────────────────────────

def compile_c(c_path: str, exe_path: str, paths: dict,
              debug: bool, no_opencl: bool) -> bool:
    inc  = [f'-I{FFLANG}', f'-I{paths["include"]}']
    opt  = ['-g', '-O0', '-DFF_DEBUG', '-std=c11', '-Wall'] if debug \
           else ['-O2', '-std=c11', '-Wall']
    flags = opt + inc

    lib  = paths['lib'].replace('\\', '/')
    libs = [f'-L{lib}', '-lraylib', '-lgdi32', '-lwinmm', '-lopengl32', '-lm']
    if not no_opencl:
        libs = ['-lOpenCL'] + libs

    # Prepend gcc's own bin dir to PATH so cc1/ld can find their DLLs
    gcc_dir = os.path.dirname(paths['gcc'])
    env = {**os.environ, 'PATH': gcc_dir + os.pathsep + os.environ.get('PATH', '')}

    cmd    = [paths['gcc']] + flags + [c_path, '-o', exe_path] + libs
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print('Compile error:', file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return False
    return True


# ── Entry point ──────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    if not args or args[0] in ('-h', '--help'):
        print(USAGE); return

    src_path  = args[0]
    emit_c    = '--emit-c'    in args
    run_after = '--run'       in args
    no_opencl = '--no-opencl' in args
    debug     = '--debug'     in args

    if not os.path.isfile(src_path):
        die(f'file not found: {src_path}')

    src_path = os.path.abspath(src_path)
    src_dir  = os.path.dirname(src_path)
    source   = open(src_path, 'r', encoding='utf-8').read()
    base     = os.path.splitext(src_path)[0]

    # Lex
    try:
        tokens = Lexer(source).tokenize()
    except LexError as e:
        die(str(e))

    # Parse
    try:
        ast = Parser(tokens).parse()
    except ParseError as e:
        die(str(e))

    # Resolve tool paths — only prompt for what this build actually needs
    paths = _resolve_paths(need_gcc=not emit_c, need_klt=_has_kl(ast))

    # Transpile .fl → .cl
    fl_kernels(ast, src_dir)
    # Transpile .kl → .cl
    klt_kernels(ast, src_dir, paths['klt'])

    # Codegen
    try:
        c_code = CodeGen().generate(ast)
    except CodeGenError as e:
        die(str(e))

    c_path   = base + '.c'
    exe_path = base + '.exe'

    with open(c_path, 'w', encoding='utf-8') as f:
        f.write(c_code)

    if emit_c:
        print(f'wrote {c_path}')
        return

    if not os.path.isfile(paths['gcc']):
        die(f'gcc not found — run again to set the path')

    print(f'compiling {os.path.basename(src_path)} …')
    if not compile_c(c_path, exe_path, paths, debug, no_opencl):
        sys.exit(1)

    print(f'built {exe_path}')

    if run_after:
        print('running …\n')
        subprocess.run([exe_path])


if __name__ == '__main__':
    main()
