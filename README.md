# FFLang-FireFlyCompiler
A transpiler with near-python syntax, focusing on tiny executables and simplicity of use with the same speed as C.
Currently only tested on Win11.

Has RayLib bindings as GUI and OpenCL bindings as GPU acceleration.
Dependencies: Python runtime (tested 3.10 and higher) and C compiler.
```
transpile diagram
           .ff -> .c -> .exe
.fl/.kl -> .cl -^
```
FFLang (.ff/.fl) files are transpiled to C via Python.
.kl OpenCL kernel files are transpiled to .cl via klt.exe.

Example: wave.ff
```python
kernel wave      = load("wave.fl", "wave")
kernel to_pixels = load("wave.fl", "to_pixels")

w = 320
h = 240
n = w * h

a  = zeros(n)      # ping buffer
b  = zeros(n)      # pong buffer
fb = zeros(n * 4)

damp = 0.99
frame = 0

win = window(w, h, "wave")

while win.open:
    # Drop a ripple on left-click — write into whichever buffer is "current"
    if win.mouse_down(0):
        mx = int(win.mouse[0])
        my = int(win.mouse[1])
        if mx >= 1 and mx < w - 1 and my >= 1 and my < h - 1:
            if frame % 2 == 0:
                a[my * w + mx] = 1.0
            else:
                b[my * w + mx] = 1.0

    # Alternate direction each frame — no list swapping needed
    if frame % 2 == 0:
        run wave(a, b, int(w), int(h), damp) -> b size(n)
        run to_pixels(b, int(w), int(h)) -> fb size(n)
    else:
        run wave(b, a, int(w), int(h), damp) -> a size(n)
        run to_pixels(a, int(w), int(h)) -> fb size(n)

    frame = frame + 1
    win.draw(fb, w, h)
    win.tick(60)
```
Example: wave.fl
```python
kernel wave(float[] buf, float[] prev, int w, int h, float damp) -> float[] out:
    x = gid % w
    y = gid / w
    if x == 0 or x == w - 1 or y == 0 or y == h - 1:
        out = 0.0
        return
    u  = buf[gid - 1]
    d  = buf[gid + 1]
    l  = buf[gid - w]
    r  = buf[gid + w]
    v  = (u + d + l + r) * 0.5 - prev[gid]
    out = v * damp

kernel to_pixels(float[] buf, int w, int h) -> float[] fb:
    x = gid % w
    y = gid / w
    v  = buf[gid] * 0.5 + 0.5
    if v < 0.0:
        v = 0.0
    if v > 1.0:
        v = 1.0
    ind = gid * 4
    fb[ind]     = v * v
    fb[ind + 1] = v
    fb[ind + 2] = 1.0 - v * v
    fb[ind + 3] = 1.0
```
Screenshot: wave.exe

![wave.exe allows mouse drags to create ripples](https://github.com/Dsg2/FFLang-FireFlyCompiler/blob/main/examples/wavedemo.png)
