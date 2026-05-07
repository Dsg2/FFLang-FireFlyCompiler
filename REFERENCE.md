# FFLang Quick Reference

FFLang transpiles to C via `python -m ffc <source.ff> [options]`.

---

## CLI

```
python -m ffc <source.ff>            # transpile + compile
python -m ffc <source.ff> --emit-c  # emit .c only, skip GCC
python -m ffc <source.ff> --run     # compile then run immediately
python -m ffc <source.ff> --debug   # -g -O0 -DFF_DEBUG (bounds checks)
python -m ffc <source.ff> --no-opencl  # omit -lOpenCL (no GPU calls)
```

Output `.c` and `.exe` are written next to the source file.

---

## Types

| FFLang   | C type        | Notes                              |
|----------|---------------|------------------------------------|
| `int`    | `int64_t`     | integer literal or `int(expr)`     |
| `float`  | `double`      | float literal or `float(expr)`     |
| `bool`   | `int`         | `true` / `false`                   |
| `str`    | `const char*` | immutable; `"..."` or `'...'`      |
| `list`   | `FFList`      | float array; GPU-ready             |
| `image`  | `FFImage`     | RGBA float pixel buffer            |
| `window` | `FFWindow`    | raylib display window              |
| `sound`  | `FFSound`     | PCM mono float buffer              |

Type is inferred from the right-hand side on first assignment.

---

## Variables

```python
x = 42
y = 3.14
name = "hello"
flag = true
let n = 1024    # 'let' is optional sugar, identical to plain assignment
```

---

## Operators

```python
# Arithmetic
x + y   x - y   x * y   x / y   x % y   # % uses fmod() automatically for floats

# Comparison
x == y  x != y  x < y  x > y  x <= y  x >= y

# Logical
x and y    x or y    not x

# Unary
-x
```

---

## Control Flow

### if / elif / else

```python
if x > 10:
    print("big")
elif x == 10:
    print("exact")
else:
    print("small")
```

### while

```python
while flag:
    flag = check()
```

### for / range

```python
for i in range(n):          # 0 .. n-1
    print(i)

for i in range(a, b):       # a .. b-1
    print(i)

for i in range(a, b, step): # a, a+step, ... < b
    print(i)

for v in mylist:            # iterate list elements as float
    print(v)
```

### break / continue

```python
while true:
    if done: break
    if skip: continue
```

---

## Functions

Parameter and return types are inferred from call sites. Use annotations to override inference for `list` params or when inference is ambiguous.

```python
def clamp(v, lo, hi):
    if v < lo: return lo
    if v > hi: return hi
    return v

result = clamp(x, 0.0, 1.0)
```

### Type Annotations

```python
def process(buf: list, scale: float):
    for i in range(len(buf)):
        buf[i] = buf[i] * scale

def add(a: list, b: list) -> list:   # return annotation is optional/informational
    ...
```

Supported annotation names: `int` `float` `bool` `str` `list` `image` `window` `sound`

Without annotations, all un-annotated numeric params are inferred as `float` (double).

---

## Lists

Dynamically resizable float (`float32`) arrays. All lists start with a fixed size but can grow or shrink at runtime.

```python
a = zeros(n)          # n zeros
b = ones(n)           # n ones
c = random(n)         # n uniform [0,1) floats
d = [1.0, 2.0, 3.0]  # literal

d[0]          # read element → float
d[0] = 9.9   # write element

len(d)        # element count → int
sum(d)        # sum of all elements → float
min(d)        # minimum element → float
max(d)        # maximum element → float
print(d)      # prints first 8 elements + total count
```

### Resizing Methods

```python
d.append(val)   # add one float to the end; grows capacity automatically
v = d.pop()     # remove and return the last element → float
d.resize(n)     # set length to n; new elements are zero; never shrinks capacity
d.clear()       # set length to 0; keeps allocation intact (fast reset)
```

**Capacity doubling** — `append` doubles the internal allocation each time it
runs out of space, so a sequence of N appends costs O(N) total, not O(N²).

**GPU note** — `append`, `pop`, `resize`, and `clear` operate on CPU memory
only. If the list is pinned (`a.pin()`), unpin it before mutating, then re-pin.
See **GPU Buffer Pinning** below for the full workflow.

---

## Kernels

Kernel declarations are **module-level only** (not inside functions).

```python
# Infer function name from kernel name
kernel add from "add.cl"

# Explicit function name (when they differ)
kernel blur = load("effects.cl", "gaussian_blur")

# .kl source — compiled by klt.exe to .cl automatically
kernel add from "add.kl"
```

### Running Kernels

**`gpu()`** — simple two-input shorthand; returns a new list:

```python
out = gpu(add, a, b)   # global_size = len(a); new output list same size
```

**`run`** — advanced form; arbitrary mix of list and scalar args; results written into existing buffers:

```python
run kernel_name(arg1, arg2, ...) -> out                  # single output
run kernel_name(arg1, arg2, ...) -> out1, out2           # multiple outputs
run kernel_name(arg1, arg2, ...) -> out1, out2 size(n)   # explicit dispatch size
```

Input lists become `cl_mem` buffers (readable and writable by the kernel); scalars (`int`, `float`) are passed by value.
Output buffers are writable `cl_mem`s; each is read back to CPU after the kernel finishes unless the list is pinned (see below).
The dispatch size defaults to `out1.len`; use `size(n)` to override.
Each output buffer may have a different length — the kernel is responsible for correct indexing.

### GPU Buffer Pinning

By default every `run` statement allocates GPU buffers, uploads data, reads back, and frees — once per call. Pinning a list keeps its GPU buffer alive across calls, eliminating per-frame allocation and upload overhead.

```python
a.pin()    # upload a.data → GPU once; gpu_buf stays resident
a.unpin()  # free GPU buffer
a.sync()   # pull GPU → CPU (needed before reading a's data on CPU
           # when a is a pinned output buffer)
```

**Typical pattern for a render loop:**

```python
buf.pin()
inputs.pin()

while win.open:
    run kernel(inputs) -> buf size(n)   # zero alloc, zero upload/free
    buf.sync()                          # pull result to CPU
    win.draw(buf, w, h)
    win.tick(60)

buf.unpin()
inputs.unpin()
```

**When a list is pinned:**
- `run` inputs: the existing `gpu_buf` is passed directly — no `loadbuf`/`freebuf`
- `run` outputs: the existing `gpu_buf` is used as the output target — no allocation; `readbuf` is skipped (call `.sync()` manually when you need the CPU copy)
- CPU-side mutations (`buf[i] = v`, `buf.clear()`, etc.) are **not** reflected on GPU until you `unpin()` + `pin()` again

---

## Images

```python
img = image(w, h)                    # blank RGBA image, all zeros
img = load_image("photo.png")        # load PNG/JPG

img.width                            # int
img.height                           # int
img.pixels                           # flat FFList of w*h*4 floats (GPU-ready)

img.fill(r, g, b, a)                 # fill all pixels (floats 0.0–1.0)
px = img[x, y]                       # get pixel → list [r, g, b, a]
img[x, y] = [1.0, 0.0, 0.0, 1.0]   # set pixel

save_image(img, "out.png")
```

---

## Windows

```python
win = window(w, h, "Title")    # open window, target 60 FPS

# Main loop pattern
while win.open:
    win.draw(pixel_list, w, h) # draw flat RGBA list as fullscreen texture
    win.draw(img)              # draw FFImage directly
    win.tick()                 # swap buffers + poll events (keep current fps)
    win.tick(30)               # swap buffers + change target fps

win.width          # int
win.height         # int
win.dt             # float: seconds since last tick
win.open           # bool: false when user closes window

win.key("a")               # bool: key held (a-z, 0-9, +, -)
win.key("space")           # named keys: space left right up down enter escape
win.key_pressed("r")       # bool: key just pressed this frame
win.mouse                  # list [x, y]
win.mouse_down(0)          # bool: mouse button held (0=left, 1=right, 2=middle)
```

---

## Sound

```python
snd = sound(n)               # empty mono buffer, n samples, 44100 Hz
snd = load_sound("beep.wav") # load WAV

snd.samples                  # FFList of float samples (GPU-ready)
snd.samples = buf            # replace samples from another list

play(snd)                    # non-blocking playback
sleep(1.1)                   # wait (raylib WaitTime)
```

---

## Built-in Functions

### List / Numeric Construction

| Call            | Returns | Description                      |
|-----------------|---------|----------------------------------|
| `zeros(n)`      | `list`  | n float zeros                    |
| `ones(n)`       | `list`  | n float ones                     |
| `random()`      | `float` | one uniform random float [0, 1)  |
| `random(n)`     | `list`  | n uniform random floats [0, 1)   |
| `range(n)`      | iterable| 0 .. n-1 (for-loop only)         |
| `range(a,b)`    | iterable| a .. b-1                         |
| `range(a,b,s)`  | iterable| a .. b-1 step s                  |

### Math

| Call              | Returns | Description              |
|-------------------|---------|--------------------------|
| `abs(x)`          | `float` | absolute value                        |
| `sqrt(x)`         | `float` | square root                           |
| `floor(x)`        | `int`   | round down to integer                 |
| `ceil(x)`         | `int`   | round up to integer                   |
| `round(x)`        | `int`   | round to nearest integer              |
| `clamp(v, lo, hi)`| `float` | clamp value to [lo, hi]               |
| `sin(x)`          | `float` | sine (radians)                        |
| `cos(x)`          | `float` | cosine (radians)                      |
| `tan(x)`          | `float` | tangent (radians)                     |
| `asin(x)`         | `float` | arc sine → radians                    |
| `acos(x)`         | `float` | arc cosine → radians                  |
| `atan(x)`         | `float` | arc tangent → radians                 |
| `atan2(y, x)`     | `float` | four-quadrant arc tangent → radians   |
| `radians(x)`      | `float` | degrees → radians                     |
| `degrees(x)`      | `float` | radians → degrees                     |
| `sum(list)`       | `float` | sum all elements         |
| `min(list)`       | `float` | minimum element          |
| `max(list)`       | `float` | maximum element          |

### Type Conversion

| Call       | Returns | Description             |
|------------|---------|-------------------------|
| `int(x)`   | `int`   | cast to int64_t         |
| `float(x)` | `float` | cast to double          |
| `str(x)`   | `str`   | *(pass-through only)*   |

### I/O

| Call                  | Returns | Description                     |
|-----------------------|---------|---------------------------------|
| `print(...)`          | —       | print ints, floats, bools, strs, or a list |
| `read_file(path)`     | `str`   | read entire text file           |
| `write_file(path, s)` | —       | write string to file            |
| `load_image(path)`    | `image` | load PNG/JPG into FFImage       |
| `save_image(img, path)`| —      | export FFImage as PNG           |
| `load_sound(path)`    | `sound` | load WAV into FFSound           |

### Multimedia

| Call               | Returns  | Description                  |
|--------------------|----------|------------------------------|
| `window(w, h, title)` | `window` | open display window       |
| `image(w, h)`      | `image`  | blank RGBA image             |
| `sound(n)`         | `sound`  | empty n-sample sound buffer  |
| `play(snd)`        | —        | play sound (non-blocking)    |
| `gpu(k, a, b)`     | `list`   | run simple 2-input kernel    |

### Time

| Call      | Returns | Description                      |
|-----------|---------|----------------------------------|
| `time()`  | `float` | seconds since program start      |
| `sleep(s)`| —       | pause for s seconds              |

### Length

| Call     | Returns | Description                      |
|----------|---------|----------------------------------|
| `len(x)` | `int`   | list element count or string byte length |

---

## Kernel Language (.kl)

`.kl` is a minimal one-liner shorthand compiled to `.cl` by `klt.exe`.
Reference it with `kernel ... from "file.kl"`.

```
kernel add(float[] a, float[] b) -> float[] out:
    out = a + b
```

Every line is a single element-wise expression. Arrays are implicitly indexed by
`get_global_id(0)`. Use `.fl` for anything that needs control flow or local variables.

---

## FL Kernel Language (.fl)

`.fl` is a full Python-like GPU kernel language compiled to `.cl` by the FFLang
toolchain (no external binary required). Reference it the same way as `.kl` or `.cl`:

```python
kernel mykernel from "myfile.fl"              # function name = kernel name
kernel other = load("myfile.fl", "funcname")  # explicit function name
```

Multiple kernels can live in a single `.fl` file and compile into one `.cl`.

### Parameter Types

| Syntax     | C / OpenCL type            | Notes                              |
|------------|----------------------------|------------------------------------|
| `float[]`  | `__global float*`          | input array, auto-indexed by `gid`; may be mutated in-place |
| `int[]`    | `__global int*`            | input int array; may be mutated in-place |
| `float`    | `float`                    | scalar value                       |
| `int`      | `int`                      | scalar integer                     |
| (output)   | `__global float*` / `int*` | arrays listed after `->` are writable output buffers |

### Built-in: `gid`

`gid` is always available as `int` and equals `get_global_id(0)` — the index of
the current work-item.

### Implicit vs Explicit Array Indexing

```python
# Array param used without [i] → automatically indexed by gid
let v = input          # compiles to: float v = input[_gid];

# Explicit index — use any integer expression
let prev = input[gid - 1]
```

### Output Assignment

Assigning to an output param (listed after `->`) writes to `out[gid]`:

```python
output = expr          # compiles to: output[_gid] = expr;
output[i] = expr       # explicit index (e.g. inside a local loop)
```

### Control Flow

All FFLang control-flow works inside `.fl`:

```python
if cond:
    ...
elif cond2:
    ...
else:
    ...

while cond:
    ...

for i in range(n):        # local loop — not a dispatch loop
    ...
for i in range(a, b):
    ...
for i in range(a, b, step):
    ...

return                    # early exit from kernel (no value)
```

### Variable Declarations

```python
let x = expr      # declares a typed local; type is inferred (float or int)
x = expr          # reassigns an existing local or output param
```

### Math Built-ins

`sin` `cos` `tan` `asin` `acos` `atan` `atan2` `sqrt` `fabs` `floor` `ceil`
`round` `exp` `log` `pow` `clamp` `radians` `degrees`

These map directly to OpenCL built-in functions.

### Type Casts

```python
float(x)    # (float)(x)
int(x)      # (int)(x)
```

### Complete Example

```python
# blur.fl — 3×3 box blur kernel
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
```

```python
# main.ff
kernel blur from "blur.fl"

w = 800
h = 600
img = load_image("photo.png")
out = zeros(w * h)
run blur(img.pixels, w, h) -> out
```

### Comparison: .fl vs .kl vs .cl

| Feature              | `.fl`  | `.kl`       | `.cl` (raw) |
|----------------------|--------|-------------|-------------|
| if / else            | ✓      | ✗           | ✓           |
| for / while loops    | ✓      | ✗           | ✓           |
| local variables      | ✓      | ✗           | ✓           |
| int params           | ✓      | ✗           | ✓           |
| multiple kernels/file| ✓      | ✗           | ✓           |
| element-wise ops     | ✓      | ✓ (only)    | ✓           |
| No external tool     | ✓      | ✗ (klt.exe) | ✓ (manual)  |

---

## Generated C Structure

```c
#include "fflang_rt.h"   // FFList, FFImage, FFWindow, FFSound + all helpers

// user function forward declarations + definitions

int main(void) {
    // kernel loads
    int _kernel_add = loadkernel("add.cl", "add");

    // transpiled program statements

    cleanupkernel();
    return 0;
}
```

Runtime header `fflang_rt.h` is header-only and depends on `clbp.h` (OpenCL wrapper) and `raylib.h`.

---

## Complete Example — Vector Addition with GPU

```python
kernel add from "add.kl"

def scale(x, factor):
    return x * factor

n = 1024
a = random(n)
b = random(n)
out = gpu(add, a, b)

print(out)
print(out[0])
print(out[n - 1])
print(sum(out))

for i in range(4):
    print(scale(out[i], 2.0))
```

## Complete Example — Mandelbrot Viewer

```python
kernel mandelbrot = load("mandelbrot.cl", "mandelbrot")

w = 800
h = 600
win = window(w, h, "Mandelbrot")
pixels = zeros(w * h * 4)

zoom = 1.0
cx = -0.5
cy = 0.0

while win.open:
    params = [float(w), float(h), zoom, cx, cy]
    run mandelbrot(pixels, params) -> pixels
    win.draw(pixels, w, h)
    win.tick()

    if win.key("+"):    zoom = zoom * 1.05
    if win.key("-"):    zoom = zoom * 0.95
    if win.key("escape"): break
```

---

## What Is Not Supported (v1)

- Classes or structs
- Multi-file programs / imports
- String operations beyond print / read / write
- 2D kernel dispatch (all kernels are 1D)
- Inline kernel source (kernels must be in `.cl`, `.kl`, or `.fl` files)
- Try/catch or exceptions
- Multi-GPU
- `.fl` kernels: no user-defined helper functions within a kernel file (write raw `.cl` for that)
