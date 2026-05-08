__kernel void wave(
    __global float* buf,
    __global float* prev,
    int w,
    int h,
    float damp,
    __global float* out
) {
    int _gid = (int)get_global_id(0);
    int x = (_gid % w);
    int y = (_gid / w);
    if (((((x == 0) || (x == (w - 1))) || (y == 0)) || (y == (h - 1)))) {
        out[_gid] = 0.0f;
        return;
    }
    float u = buf[(_gid - 1)];
    float d = buf[(_gid + 1)];
    float l = buf[(_gid - w)];
    float r = buf[(_gid + w)];
    float v = ((((((u + d) + l) + r)) * 0.5f) - prev[_gid]);
    out[_gid] = (v * damp);
}

__kernel void to_pixels(
    __global float* buf,
    int w,
    int h,
    __global float* fb
) {
    int _gid = (int)get_global_id(0);
    int x = (_gid % w);
    int y = (_gid / w);
    float v = ((buf[_gid] * 0.5f) + 0.5f);
    if ((v < 0.0f)) {
        v = 0.0f;
    }
    if ((v > 1.0f)) {
        v = 1.0f;
    }
    int ind = (_gid * 4);
    fb[ind] = (v * v);
    fb[(ind + 1)] = v;
    fb[(ind + 2)] = (1.0f - (v * v));
    fb[(ind + 3)] = 1.0f;
}
