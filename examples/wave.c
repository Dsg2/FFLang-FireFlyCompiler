#include "fflang_rt.h"

int main(void) {
    /* load kernels */
    int _kernel_wave = loadkernel("wave.cl", "wave");
    if (_kernel_wave < 0) { fprintf(stderr, "kernel wave failed\n"); return 1; }
    int _kernel_to_pixels = loadkernel("wave.cl", "to_pixels");
    if (_kernel_to_pixels < 0) { fprintf(stderr, "kernel to_pixels failed\n"); return 1; }

    int64_t w = 320;
    int64_t h = 240;
    int64_t n = (w * h);
    FFList _lt1 = ff_list_zeros((int)n);
    FFList a = _lt1;
    FFList _lt2 = ff_list_zeros((int)n);
    FFList b = _lt2;
    FFList _lt3 = ff_list_zeros((int)(n * 4));
    FFList fb = _lt3;
    double damp = 0.99;
    int64_t frame = 0;
    FFWindow win = ff_window_new((int)w, (int)h, "wave");
    while (ff_window_open(&win)) {
        int64_t mx = 0;
        int64_t my = 0;
        if (ff_window_mouse_down(&win, (int)0)) {
            mx = (int64_t)(ff_window_mouse(&win).data[(int)(0)]);
            my = (int64_t)(ff_window_mouse(&win).data[(int)(1)]);
            if (((((mx >= 1) && (mx < (w - 1))) && (my >= 1)) && (my < (h - 1)))) {
                if (((frame % 2) == 0)) {
                    a.data[(int)(((my * w) + mx))] = (float)(1.0);
                } else {
                    b.data[(int)(((my * w) + mx))] = (float)(1.0);
                }
            }
        }
        if (((frame % 2) == 0)) {
            cl_mem _ob4 = b.pinned ? b.gpu_buf : loadbuf(b.data, b.len, 0);
            cl_mem _rb5 = a.pinned ? a.gpu_buf : loadbuf(a.data, a.len, 1);
            cl_mem _rb6 = b.pinned ? b.gpu_buf : loadbuf(b.data, b.len, 1);
            int _sv7 = (int)((int64_t)(w));
            int _sv8 = (int)((int64_t)(h));
            float _sv9 = (float)(damp);
            advrunkernel(_kernel_wave, (size_t)n, 6, &_rb5, sizeof(cl_mem), &_rb6, sizeof(cl_mem), &_sv7, sizeof(int), &_sv8, sizeof(int), &_sv9, sizeof(float), &_ob4, sizeof(cl_mem));
            if (!b.pinned) readbuf(_ob4, b.data, b.len);
            if (!a.pinned) freebuf(_rb5);
            if (!b.pinned) freebuf(_rb6);
            if (!b.pinned) freebuf(_ob4);
            cl_mem _ob10 = fb.pinned ? fb.gpu_buf : loadbuf(fb.data, fb.len, 0);
            cl_mem _rb11 = b.pinned ? b.gpu_buf : loadbuf(b.data, b.len, 1);
            int _sv12 = (int)((int64_t)(w));
            int _sv13 = (int)((int64_t)(h));
            advrunkernel(_kernel_to_pixels, (size_t)n, 4, &_rb11, sizeof(cl_mem), &_sv12, sizeof(int), &_sv13, sizeof(int), &_ob10, sizeof(cl_mem));
            if (!fb.pinned) readbuf(_ob10, fb.data, fb.len);
            if (!b.pinned) freebuf(_rb11);
            if (!fb.pinned) freebuf(_ob10);
        } else {
            cl_mem _ob14 = a.pinned ? a.gpu_buf : loadbuf(a.data, a.len, 0);
            cl_mem _rb15 = b.pinned ? b.gpu_buf : loadbuf(b.data, b.len, 1);
            cl_mem _rb16 = a.pinned ? a.gpu_buf : loadbuf(a.data, a.len, 1);
            int _sv17 = (int)((int64_t)(w));
            int _sv18 = (int)((int64_t)(h));
            float _sv19 = (float)(damp);
            advrunkernel(_kernel_wave, (size_t)n, 6, &_rb15, sizeof(cl_mem), &_rb16, sizeof(cl_mem), &_sv17, sizeof(int), &_sv18, sizeof(int), &_sv19, sizeof(float), &_ob14, sizeof(cl_mem));
            if (!a.pinned) readbuf(_ob14, a.data, a.len);
            if (!b.pinned) freebuf(_rb15);
            if (!a.pinned) freebuf(_rb16);
            if (!a.pinned) freebuf(_ob14);
            cl_mem _ob20 = fb.pinned ? fb.gpu_buf : loadbuf(fb.data, fb.len, 0);
            cl_mem _rb21 = a.pinned ? a.gpu_buf : loadbuf(a.data, a.len, 1);
            int _sv22 = (int)((int64_t)(w));
            int _sv23 = (int)((int64_t)(h));
            advrunkernel(_kernel_to_pixels, (size_t)n, 4, &_rb21, sizeof(cl_mem), &_sv22, sizeof(int), &_sv23, sizeof(int), &_ob20, sizeof(cl_mem));
            if (!fb.pinned) readbuf(_ob20, fb.data, fb.len);
            if (!a.pinned) freebuf(_rb21);
            if (!fb.pinned) freebuf(_ob20);
        }
        frame = (frame + 1);
        (void)((ff_window_draw(&win, fb.data, (int)w, (int)h), 0));
        (void)((ff_window_tick(&win, (int)(60)), 0));
    }

    cleanupkernel();
    return 0;
}