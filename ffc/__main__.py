import sys
import os
import subprocess

from .lexer    import Lexer,  LexError
from .parser   import Parser, ParseError
from .codegen  import CodeGen, CodeGenError
from .ast_nodes import KernelDecl
from .flc      import compile_fl, FLError

GCC         = r'C:\msys64\mingw64\bin\gcc.exe'
KLT         = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'klt.exe')
FFLANG      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
RAYLIB_INC  = r'include'   # raylib.h lives here
RAYLIB_LIB  = r'lib'        # libraylib.a lives here

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
            src = open(fl_path, 'r').read()
            cl_src = compile_fl(src)
        except FLError as e:
            print(f'flc error for {node.cl_file}: {e}', file=sys.stderr)
            sys.exit(1)

        with open(cl_path, 'w') as f:
            f.write(cl_src)

        print(f'  flc: {node.cl_file} -> {cl_file}')
        node.cl_file = cl_file


def klt_kernels(ast, src_dir: str):
    """Run klt on any .kl files referenced by kernel declarations.
    Mutates KernelDecl.cl_file: foo.kl → foo.cl in place."""
    for node in ast.stmts:
        if not isinstance(node, KernelDecl):
            continue
        if not node.cl_file.endswith('.kl'):
            continue

        kl_path = os.path.join(src_dir, node.cl_file)
        cl_file  = node.cl_file[:-3] + '.cl'
        cl_path  = os.path.join(src_dir, cl_file)

        if not os.path.isfile(kl_path):
            die(f'kernel source not found: {kl_path}')
        if not os.path.isfile(KLT):
            die(f'klt not found at {KLT}')

        result = subprocess.run(
            [KLT, kl_path, '-o', cl_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f'klt error for {node.cl_file}:', file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            sys.exit(1)

        print(f'  klt: {node.cl_file} -> {cl_file}')
        node.cl_file = cl_file   # codegen sees the .cl name


def compile_c(c_path: str, exe_path: str, debug: bool, no_opencl: bool) -> bool:
    inc   = [f'-I{FFLANG}', f'-I{RAYLIB_INC}']
    opt   = ['-O2', '-std=c11', '-Wall']
    if debug:
        opt = ['-g', '-O0', '-DFF_DEBUG', '-std=c11', '-Wall']
    flags = opt + inc

    rl   = RAYLIB_LIB.replace('\\', '/')
    libs = [f'-L{rl}', '-lraylib', '-lgdi32', '-lwinmm', '-lopengl32', '-lm']
    if not no_opencl:
        libs = ['-lOpenCL'] + libs

    env = {**os.environ, 'PATH': r'C:\msys64\mingw64\bin;' + os.environ.get('PATH', '')}
    cmd = [GCC] + flags + [c_path, '-o', exe_path] + libs
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print('Compile error:', file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return False
    return True


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

    # Transpile any .fl kernel sources → .cl  (Python-based FL compiler)
    fl_kernels(ast, src_dir)
    # Transpile any .kl kernel sources → .cl  (klt.exe)
    klt_kernels(ast, src_dir)

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

    if not os.path.isfile(GCC):
        die(f'gcc not found at {GCC}')

    print(f'compiling {os.path.basename(src_path)} …')
    if not compile_c(c_path, exe_path, debug, no_opencl):
        sys.exit(1)

    print(f'built {exe_path}')

    if run_after:
        print('running …\n')
        subprocess.run([exe_path])


if __name__ == '__main__':
    main()
