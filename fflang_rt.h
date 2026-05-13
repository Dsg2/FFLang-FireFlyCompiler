/*
 * fflang_rt.h - FFLang runtime (header-only)
 * Included by every generated .c file.
 * Depends on: clbp.h  raylib.h  CL/cl.h
 */
#ifndef FFLANG_RT_H
#define FFLANG_RT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include "clbp.h"
#include "raylib.h"

/* ── Utility ──────────────────────────────────────────────── */

static inline double ff_clamp(double v, double lo, double hi) {
    return v < lo ? lo : v > hi ? hi : v;
}

static inline double ff_time(void) {
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static inline void ff_sleep(double seconds) {
    WaitTime((float)seconds);   /* raylib */
}

static inline int ff_file_exists(const char* path) {
    FILE* f = fopen(path, "rb");
    if (f) { fclose(f); return 1; }
    return 0;
}

static inline char* ff_read_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return "";
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(sz + 1);
    fread(buf, 1, sz, f); fclose(f); buf[sz] = '\0';
    return buf;
}

static inline void ff_write_file(const char* path, const char* s) {
    FILE* f = fopen(path, "wb");
    if (f) { fputs(s, f); fclose(f); }
}

/* ── String pool ──────────────────────────────────────────── */

#define FF_STR_BUFS   16
#define FF_STR_MAXLEN 512

static char _ff_str_pool[FF_STR_BUFS][FF_STR_MAXLEN];
static int  _ff_str_cur = 0;

static inline char* _ff_next_str_buf(void) {
    char* b = _ff_str_pool[_ff_str_cur % FF_STR_BUFS];
    _ff_str_cur++;
    return b;
}

/* str(x) — number to string; omits decimal point for whole numbers */
static inline const char* ff_str(double x) {
    char* b = _ff_next_str_buf();
    long long xi = (long long)x;
    if ((double)xi == x) snprintf(b, FF_STR_MAXLEN, "%lld", xi);
    else                 snprintf(b, FF_STR_MAXLEN, "%g",   x);
    return b;
}

static inline const char* ff_str_int(int64_t x) {
    char* b = _ff_next_str_buf();
    snprintf(b, FF_STR_MAXLEN, "%lld", (long long)x);
    return b;
}

/* strcat(a, b) */
static inline const char* ff_strcat(const char* a, const char* b) {
    char* buf = _ff_next_str_buf();
    snprintf(buf, FF_STR_MAXLEN, "%s%s", a, b);
    return buf;
}

/* strsub(s, start, n) — n chars starting at index start */
static inline const char* ff_strsub(const char* s, int64_t start, int64_t n) {
    char* buf = _ff_next_str_buf();
    int64_t slen = (int64_t)strlen(s);
    if (start < 0)          start = 0;
    if (start > slen)       start = slen;
    if (n < 0)              n = 0;
    if (start + n > slen)   n = slen - start;
    if (n >= FF_STR_MAXLEN) n = FF_STR_MAXLEN - 1;
    memcpy(buf, s + start, (size_t)n);
    buf[n] = '\0';
    return buf;
}

/* strfind(s, sub) — first index of sub in s, or -1 */
static inline int64_t ff_strfind(const char* s, const char* sub) {
    const char* p = strstr(s, sub);
    return p ? (int64_t)(p - s) : (int64_t)-1;
}

/* strtrim(s) — strip leading and trailing whitespace */
static inline const char* ff_strtrim(const char* s) {
    char* buf = _ff_next_str_buf();
    while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') s++;
    int64_t n = (int64_t)strlen(s);
    while (n > 0 && (s[n-1]==' '||s[n-1]=='\t'||s[n-1]=='\n'||s[n-1]=='\r')) n--;
    if (n >= FF_STR_MAXLEN) n = FF_STR_MAXLEN - 1;
    memcpy(buf, s, (size_t)n);
    buf[n] = '\0';
    return buf;
}

/* ── FFList ───────────────────────────────────────────────── */

typedef struct {
    float*  data;
    int     len;
    int     cap;        /* allocated capacity (>= len) */
    cl_mem  gpu_buf;    /* NULL unless pinned */
    int     pinned;
} FFList;

static inline FFList ff_list_new(int n) {
    FFList l;
    l.len     = n;
    l.cap     = n > 0 ? n : 1;
    l.data    = (float*)calloc(l.cap, sizeof(float));
    l.gpu_buf = NULL;
    l.pinned  = 0;
    return l;
}

static inline FFList ff_list_zeros(int n)  { return ff_list_new(n); }

static inline FFList ff_list_ones(int n) {
    FFList l = ff_list_new(n);
    for (int i = 0; i < n; i++) l.data[i] = 1.0f;
    return l;
}

static inline float _ff_rand_float(void) {
    static uint32_t state = 2463534242u;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (float)(state >> 8) / (float)(1 << 24);
}

static inline FFList ff_list_random(int n) {
    FFList l = ff_list_new(n);
    for (int i = 0; i < n; i++) l.data[i] = _ff_rand_float();
    return l;
}

static inline void ff_list_free(FFList* l) {
    if (l->pinned && l->gpu_buf) { freebuf(l->gpu_buf); l->gpu_buf = NULL; }
    free(l->data); l->data = NULL; l->len = 0; l->cap = 0;
}

static inline void ff_list_append(FFList* l, float val) {
    if (l->len >= l->cap) {
        l->cap = l->cap > 0 ? l->cap * 2 : 8;
        l->data = (float*)realloc(l->data, l->cap * sizeof(float));
    }
    l->data[l->len++] = val;
}

static inline void ff_list_extend(FFList* dst, FFList src) {
    int new_len = dst->len + src.len;
    if (new_len > dst->cap) {
        dst->cap = new_len * 2;
        dst->data = (float*)realloc(dst->data, dst->cap * sizeof(float));
    }
    memcpy(dst->data + dst->len, src.data, src.len * sizeof(float));
    dst->len = new_len;
}

static inline float ff_list_pop(FFList* l) {
    if (l->len <= 0) { fprintf(stderr, "fflang: pop from empty list\n"); return 0.0f; }
    return l->data[--l->len];
}

static inline void ff_list_resize(FFList* l, int n) {
    if (n > l->cap) {
        l->data = (float*)realloc(l->data, n * sizeof(float));
        l->cap = n;
    }
    memset(l->data, 0, n * sizeof(float));
    l->len = n;
}

static inline void ff_list_clear(FFList* l) {
    l->len = 0;
}

static inline void ff_list_pin(FFList* l) {
    if (l->pinned) return;
    /* READ_WRITE buffer, data uploaded — works as both input and output */
    l->gpu_buf = pinbuf(l->data, l->len);
    l->pinned  = 1;
}

static inline void ff_list_unpin(FFList* l) {
    if (!l->pinned) return;
    freebuf(l->gpu_buf); l->gpu_buf = NULL; l->pinned = 0;
}

/* Pull GPU→CPU for a pinned list (e.g. before win.draw or CPU reads) */
static inline void ff_list_sync(FFList* l) {
    if (!l->pinned || !l->gpu_buf) return;
    readbuf(l->gpu_buf, l->data, l->len);
}

static inline double ff_list_sum(FFList l) {
    double s = 0; for (int i = 0; i < l.len; i++) s += l.data[i]; return s;
}
static inline double ff_list_min(FFList l) {
    double m = l.len ? l.data[0] : 0;
    for (int i = 1; i < l.len; i++) if (l.data[i] < m) m = l.data[i];
    return m;
}
static inline double ff_list_max(FFList l) {
    double m = l.len ? l.data[0] : 0;
    for (int i = 1; i < l.len; i++) if (l.data[i] > m) m = l.data[i];
    return m;
}

static inline void ff_list_print(FFList l) {
    printf("[");
    int show = l.len > 8 ? 8 : l.len;
    for (int i = 0; i < show; i++) {
        printf("%g%s", l.data[i], i < show-1 ? ", " : "");
    }
    if (l.len > 8) printf(", ... (%d total)", l.len);
    printf("]\n");
}

/* ── List serialization ───────────────────────────────────── */

static inline void ff_save_list(const char* path, FFList l) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "fflang: cannot open '%s' for writing\n", path); return; }
    int32_t n = (int32_t)l.len;
    fwrite(&n, sizeof(int32_t), 1, f);
    fwrite(l.data, sizeof(float), (size_t)n, f);
    fclose(f);
}

static inline FFList ff_load_list(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "fflang: cannot open '%s' for reading\n", path); return ff_list_new(0); }
    int32_t n = 0;
    fread(&n, sizeof(int32_t), 1, f);
    FFList l = ff_list_new(n);
    fread(l.data, sizeof(float), (size_t)n, f);
    l.len = n;
    fclose(f);
    return l;
}

/* ── FFImage ──────────────────────────────────────────────── */

typedef struct {
    FFList  pixels;     /* flat RGBA floats, w*h*4 elements */
    int     width;
    int     height;
    Texture2D _tex;     /* raylib texture, initialised on first draw */
    int     _tex_ready;
} FFImage;

static inline FFImage ff_image_new(int w, int h) {
    FFImage img;
    img.pixels    = ff_list_new(w * h * 4);
    img.width     = w;
    img.height    = h;
    img._tex_ready = 0;
    memset(&img._tex, 0, sizeof(Texture2D));
    return img;
}

static inline FFImage ff_load_image(const char* path) {
    Image ri = LoadImage(path);
    ImageFormat(&ri, PIXELFORMAT_UNCOMPRESSED_R32G32B32A32);
    FFImage img = ff_image_new(ri.width, ri.height);
    memcpy(img.pixels.data, ri.data, ri.width * ri.height * 4 * sizeof(float));
    UnloadImage(ri);
    return img;
}

static inline void ff_save_image(FFImage img, const char* path) {
    Image ri;
    ri.data    = img.pixels.data;
    ri.width   = img.width;
    ri.height  = img.height;
    ri.mipmaps = 1;
    ri.format  = PIXELFORMAT_UNCOMPRESSED_R32G32B32A32;
    ExportImage(ri, path);
}

static inline void ff_image_fill(FFImage* img, float r, float g, float b, float a) {
    int n = img->width * img->height;
    for (int i = 0; i < n; i++) {
        img->pixels.data[i*4+0] = r;
        img->pixels.data[i*4+1] = g;
        img->pixels.data[i*4+2] = b;
        img->pixels.data[i*4+3] = a;
    }
}

static inline FFList ff_image_get(FFImage* img, int x, int y) {
    FFList px = ff_list_new(4);
    int base = (y * img->width + x) * 4;
    px.data[0] = img->pixels.data[base+0];
    px.data[1] = img->pixels.data[base+1];
    px.data[2] = img->pixels.data[base+2];
    px.data[3] = img->pixels.data[base+3];
    return px;
}

static inline void ff_image_set(FFImage* img, int x, int y, FFList px) {
    int base = (y * img->width + x) * 4;
    img->pixels.data[base+0] = px.len > 0 ? px.data[0] : 0;
    img->pixels.data[base+1] = px.len > 1 ? px.data[1] : 0;
    img->pixels.data[base+2] = px.len > 2 ? px.data[2] : 0;
    img->pixels.data[base+3] = px.len > 3 ? px.data[3] : 1;
}

/* Convert float RGBA [0,1] to raylib Image and upload as texture */
static inline void _ff_image_upload(FFImage* img) {
    int n = img->width * img->height;
    unsigned char* bytes = (unsigned char*)malloc(n * 4);
    for (int i = 0; i < n * 4; i++) {
        float v = img->pixels.data[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        bytes[i] = (unsigned char)(v * 255.0f);
    }
    if (!img->_tex_ready) {
        Image ri = { bytes, img->width, img->height, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 };
        img->_tex = LoadTextureFromImage(ri);
        img->_tex_ready = 1;
    } else {
        UpdateTexture(img->_tex, bytes);
    }
    free(bytes);
}

/* ── FFWindow ─────────────────────────────────────────────── */

typedef struct {
    int       width;
    int       height;
    double    _last_t;
    int       _fps;        /* current target fps; 0 = not yet set */
    Texture2D _tex;        /* persistent draw texture for win.draw() */
    int       _tex_w;      /* dimensions of _tex (0 = not created) */
    int       _tex_h;
} FFWindow;

static inline FFWindow ff_window_new(int w, int h, const char* title) {
    InitWindow(w, h, title);
    SetTargetFPS(60);
    FFWindow win;
    win.width   = w;  win.height = h;
    win._last_t = ff_time();
    win._fps    = 60;
    win._tex_w  = 0;  win._tex_h = 0;
    memset(&win._tex, 0, sizeof(Texture2D));
    BeginDrawing();
    ClearBackground(BLACK);
    return win;
}

static inline int    ff_window_open(FFWindow* w)  { (void)w; return !WindowShouldClose(); }
static inline double ff_window_dt(FFWindow* w) {
    double now = ff_time(); double dt = now - w->_last_t;
    w->_last_t = now; return dt;
}

static inline void ff_window_tick(FFWindow* w, int fps) {
    /* Only call SetTargetFPS when the value actually changes */
    if (fps > 0 && fps != w->_fps) {
        SetTargetFPS(fps);
        w->_fps = fps;
    }
    EndDrawing();
    BeginDrawing();
    ClearBackground(BLACK);
}

/* Convert float RGBA pixels to a u8 byte array */
static inline unsigned char* _ff_pixels_to_bytes(float* pixels, int n) {
    unsigned char* bytes = (unsigned char*)malloc(n * 4);
    for (int i = 0; i < n * 4; i++) {
        float v = pixels[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        bytes[i] = (unsigned char)(v * 255.0f);
    }
    return bytes;
}

static inline void ff_window_draw(FFWindow* w, float* pixels, int iw, int ih) {
    unsigned char* bytes = _ff_pixels_to_bytes(pixels, iw * ih);
    /* Create the texture once; UpdateTexture on subsequent frames */
    if (w->_tex_w != iw || w->_tex_h != ih) {
        if (w->_tex_w != 0) UnloadTexture(w->_tex);
        Image ri = { bytes, iw, ih, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 };
        w->_tex   = LoadTextureFromImage(ri);
        w->_tex_w = iw;
        w->_tex_h = ih;
    } else {
        UpdateTexture(w->_tex, bytes);
    }
    free(bytes);
    DrawTexturePro(w->_tex,
        (Rectangle){0,0,(float)iw,(float)ih},
        (Rectangle){0,0,(float)w->width,(float)w->height},
        (Vector2){0,0}, 0.0f, WHITE);
}

static inline void ff_window_draw_image(FFWindow* w, FFImage* img) {
    /* Delegate through ff_window_draw so the same persistent texture is used */
    ff_window_draw(w, img->pixels.data, img->width, img->height);
}

static inline int ff_window_key(FFWindow* w, const char* key) {
    (void)w;
    if (!key || !key[0]) return 0;
    /* Map common single-char and named keys */
    if (strlen(key) == 1) {
        char c = key[0];
        if (c >= 'a' && c <= 'z') return IsKeyDown(KEY_A + (c - 'a'));
        if (c >= '0' && c <= '9') return IsKeyDown(KEY_ZERO + (c - '0'));
        if (c == '+') return IsKeyDown(KEY_EQUAL);
        if (c == '-') return IsKeyDown(KEY_MINUS);
    }
    if (strcmp(key,"space")==0)  return IsKeyDown(KEY_SPACE);
    if (strcmp(key,"left")==0)   return IsKeyDown(KEY_LEFT);
    if (strcmp(key,"right")==0)  return IsKeyDown(KEY_RIGHT);
    if (strcmp(key,"up")==0)     return IsKeyDown(KEY_UP);
    if (strcmp(key,"down")==0)   return IsKeyDown(KEY_DOWN);
    if (strcmp(key,"enter")==0)  return IsKeyDown(KEY_ENTER);
    if (strcmp(key,"escape")==0) return IsKeyDown(KEY_ESCAPE);
    return 0;
}

static inline int ff_window_key_pressed(FFWindow* w, const char* key) {
    (void)w;
    if (!key || !key[0]) return 0;
    if (strlen(key) == 1) {
        char c = key[0];
        if (c >= 'a' && c <= 'z') return IsKeyPressed(KEY_A + (c - 'a'));
    }
    if (strcmp(key,"space")==0)  return IsKeyPressed(KEY_SPACE);
    if (strcmp(key,"escape")==0) return IsKeyPressed(KEY_ESCAPE);
    return 0;
}

static inline int ff_window_mouse_down(FFWindow* w, int btn) {
    (void)w; return IsMouseButtonDown(btn);
}

static inline FFList ff_window_mouse(FFWindow* w) {
    (void)w;
    Vector2 mp = GetMousePosition();
    FFList l = ff_list_new(2);
    l.data[0] = mp.x; l.data[1] = mp.y;
    return l;
}

/* ── FFSound ──────────────────────────────────────────────── */

typedef struct {
    FFList  samples;
    int     rate;
    Wave    _wave;
    Sound   _snd;
    int     _loaded;
} FFSound;

static inline FFSound ff_sound_new(int n) {
    if (!IsAudioDeviceReady()) InitAudioDevice();
    FFSound s; s.samples = ff_list_new(n); s.rate = 44100; s._loaded = 0;
    return s;
}

static inline FFSound ff_load_sound(const char* path) {
    if (!IsAudioDeviceReady()) InitAudioDevice();
    FFSound s;
    s._wave   = LoadWave(path);
    s.rate    = s._wave.sampleRate;
    s.samples = ff_list_new(s._wave.frameCount);
    /* Copy mono float data */
    WaveFormat(&s._wave, s.rate, 32, 1);
    float* src = (float*)s._wave.data;
    for (unsigned i = 0; i < s._wave.frameCount; i++) s.samples.data[i] = src[i];
    s._snd    = LoadSoundFromWave(s._wave);
    s._loaded = 1;
    return s;
}

static inline void ff_sound_set_samples(FFSound* s, FFList buf) {
    if (s->_loaded) { UnloadSound(s->_snd); UnloadWave(s->_wave); }
    s->samples = buf;
    s->_wave.frameCount  = buf.len;
    s->_wave.sampleRate  = s->rate;
    s->_wave.sampleSize  = 32;
    s->_wave.channels    = 1;
    s->_wave.data        = buf.data;
    s->_snd   = LoadSoundFromWave(s->_wave);
    s->_loaded = 1;
}

static inline void ff_sound_play(FFSound s) {
    if (s._loaded) PlaySound(s._snd);
}

#endif /* FFLANG_RT_H */
