/*
 * brain_core.c  —  Spiking Neural Network Brain
 *
 * 400k LIF neurons, Hebbian readout, 5fps SDL2 visualiser with
 * sampled rendering (1/SAMPLE_RATIO neurons/frame to save CPU ticks),
 * chat dialog via local ollama.  No SDL2_ttf needed (embedded font).
 *
 * Build:  gcc -O2 -o brain_core brain_core.c -lSDL2 -lpthread -lm
 * Deps:   sudo apt install libsdl2-dev
 * Ollama: ensure 'ollama serve' is running
 */

#define _POSIX_C_SOURCE 200809L
#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

/* ════════════════════════════════════════════════════════════════════════════
 *  CONSTANTS
 * ════════════════════════════════════════════════════════════════════════════ */

#define N         399424    /* total neurons  (GRID×GRID ≈400k)*/
#define FAN       32        /* recurrent fan-in per neuron     */
#define OUT       256       /* readout dimensions              */
#define GRID      632       /* sqrt(N): 632*632 = 399424       */
#define NEIGH_R   50        /* neighbourhood radius            */

/* Temporal self-attention — mirrors ATTN-11 (PDP-11 transformer) constants:
 *   ATTN_SEQ = SEQ.LN = 8   (temporal context window, not token positions)
 *   ATTN_D   = D.MODL = 16  (query/key projection dimension)              */
#define ATTN_SEQ  8         /* temporal attention window (readout history) */
#define ATTN_D    16        /* Q/K projection dim for temporal attention   */

#define RENDER_FPS          4               /* visualiser target FPS           */
#define RENDER_INTERVAL_MS  (1000/RENDER_FPS) /* 250ms between frames          */

#define WIN_W     1280
#define WIN_H     720
#define NP_X      10        /* neuron panel offset x           */
#define NP_Y      28        /* neuron panel offset y           */
/* Texture is GRID×GRID (1000×1000), displayed scaled to NP_W×NP_W.
 * CELL no longer sets pixel size — SDL scales the texture to fit.  */
#define NP_W      690       /* neuron panel display size (px)  */

#define CP_X      730       /* chat panel left edge            */
#define CP_W      (WIN_W - CP_X - 6)

/* text rendering */
#define TS        2         /* text scale (2 → 16x16 glyphs)  */
#define GW        (8*TS)    /* glyph width  in pixels          */
#define GH        (8*TS)    /* glyph height in pixels          */
#define LH        (GH+3)    /* line height                     */
#define CPL       (CP_W/GW) /* chars per line in chat          */

#define MAX_CHAT  80        /* stored chat entries             */
#define LCHAT     256       /* max chars per entry             */
#define INPUT_MAX 510

#define STATE_FILE "brain_state.bin"
#define LOG_FILE   "brain_core_log.txt"
#define MON_INTERVAL_S 5   /* monitor writes a snapshot every N seconds */
#define MODEL      "gemma3:1b"

#define MODE_LLM   0    /* Ctrl-L: LLM replies, neurons learn from response */
#define MODE_BRAIN 1    /* Ctrl-B: neurons reply directly, no LLM           */

/* ════════════════════════════════════════════════════════════════════════════
 *  EMBEDDED 8×8 BITMAP FONT  (ASCII 32–126, 95 glyphs)
 *  Each glyph: 8 rows, one byte per row, bit7 = leftmost pixel.
 * ════════════════════════════════════════════════════════════════════════════ */
static const uint8_t F8[95][8] = {
/* 32 spc */{0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
/* 33 !   */{0x18,0x3C,0x3C,0x18,0x18,0x00,0x18,0x00},
/* 34 "   */{0x66,0x66,0x24,0x00,0x00,0x00,0x00,0x00},
/* 35 #   */{0x6C,0x6C,0xFE,0x6C,0xFE,0x6C,0x6C,0x00},
/* 36 $   */{0x18,0x3E,0x60,0x3C,0x06,0x7C,0x18,0x00},
/* 37 %   */{0x00,0xC6,0xCC,0x18,0x30,0x66,0xC6,0x00},
/* 38 &   */{0x38,0x6C,0x38,0x76,0xDC,0xCC,0x76,0x00},
/* 39 '   */{0x18,0x18,0x30,0x00,0x00,0x00,0x00,0x00},
/* 40 (   */{0x0C,0x18,0x30,0x30,0x30,0x18,0x0C,0x00},
/* 41 )   */{0x30,0x18,0x0C,0x0C,0x0C,0x18,0x30,0x00},
/* 42 *   */{0x00,0x66,0x3C,0xFF,0x3C,0x66,0x00,0x00},
/* 43 +   */{0x00,0x18,0x18,0x7E,0x18,0x18,0x00,0x00},
/* 44 ,   */{0x00,0x00,0x00,0x00,0x18,0x18,0x30,0x00},
/* 45 -   */{0x00,0x00,0x00,0x7E,0x00,0x00,0x00,0x00},
/* 46 .   */{0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x00},
/* 47 /   */{0x06,0x0C,0x18,0x30,0x60,0xC0,0x80,0x00},
/* 48 0   */{0x3C,0x66,0x6E,0x76,0x66,0x66,0x3C,0x00},
/* 49 1   */{0x18,0x38,0x18,0x18,0x18,0x18,0x7E,0x00},
/* 50 2   */{0x3C,0x66,0x06,0x1C,0x30,0x60,0x7E,0x00},
/* 51 3   */{0x3C,0x66,0x06,0x1C,0x06,0x66,0x3C,0x00},
/* 52 4   */{0x06,0x0E,0x1E,0x66,0x7F,0x06,0x06,0x00},
/* 53 5   */{0x7E,0x60,0x7C,0x06,0x06,0x66,0x3C,0x00},
/* 54 6   */{0x3C,0x66,0x60,0x7C,0x66,0x66,0x3C,0x00},
/* 55 7   */{0x7E,0x06,0x0C,0x18,0x30,0x30,0x30,0x00},
/* 56 8   */{0x3C,0x66,0x66,0x3C,0x66,0x66,0x3C,0x00},
/* 57 9   */{0x3C,0x66,0x66,0x3E,0x06,0x66,0x3C,0x00},
/* 58 :   */{0x00,0x18,0x18,0x00,0x18,0x18,0x00,0x00},
/* 59 ;   */{0x00,0x18,0x18,0x00,0x18,0x18,0x30,0x00},
/* 60 <   */{0x0C,0x18,0x30,0x60,0x30,0x18,0x0C,0x00},
/* 61 =   */{0x00,0x00,0x7E,0x00,0x7E,0x00,0x00,0x00},
/* 62 >   */{0x60,0x30,0x18,0x0C,0x18,0x30,0x60,0x00},
/* 63 ?   */{0x3C,0x66,0x06,0x1C,0x18,0x00,0x18,0x00},
/* 64 @   */{0x3C,0x66,0x6E,0x6A,0x6E,0x60,0x3C,0x00},
/* 65 A   */{0x18,0x3C,0x66,0x7E,0x66,0x66,0x66,0x00},
/* 66 B   */{0x7C,0x66,0x66,0x7C,0x66,0x66,0x7C,0x00},
/* 67 C   */{0x3C,0x66,0x60,0x60,0x60,0x66,0x3C,0x00},
/* 68 D   */{0x78,0x6C,0x66,0x66,0x66,0x6C,0x78,0x00},
/* 69 E   */{0x7E,0x60,0x60,0x78,0x60,0x60,0x7E,0x00},
/* 70 F   */{0x7E,0x60,0x60,0x78,0x60,0x60,0x60,0x00},
/* 71 G   */{0x3C,0x66,0x60,0x6E,0x66,0x66,0x3C,0x00},
/* 72 H   */{0x66,0x66,0x66,0x7E,0x66,0x66,0x66,0x00},
/* 73 I   */{0x3C,0x18,0x18,0x18,0x18,0x18,0x3C,0x00},
/* 74 J   */{0x1E,0x0C,0x0C,0x0C,0x6C,0x6C,0x38,0x00},
/* 75 K   */{0x66,0x6C,0x78,0x70,0x78,0x6C,0x66,0x00},
/* 76 L   */{0x60,0x60,0x60,0x60,0x60,0x60,0x7E,0x00},
/* 77 M   */{0xC6,0xEE,0xFE,0xD6,0xC6,0xC6,0xC6,0x00},
/* 78 N   */{0xC6,0xE6,0xF6,0xDE,0xCE,0xC6,0xC6,0x00},
/* 79 O   */{0x3C,0x66,0x66,0x66,0x66,0x66,0x3C,0x00},
/* 80 P   */{0x7C,0x66,0x66,0x7C,0x60,0x60,0x60,0x00},
/* 81 Q   */{0x3C,0x66,0x66,0x66,0x76,0x3C,0x0E,0x00},
/* 82 R   */{0x7C,0x66,0x66,0x7C,0x78,0x6C,0x66,0x00},
/* 83 S   */{0x3C,0x66,0x60,0x3C,0x06,0x66,0x3C,0x00},
/* 84 T   */{0x7E,0x18,0x18,0x18,0x18,0x18,0x18,0x00},
/* 85 U   */{0x66,0x66,0x66,0x66,0x66,0x66,0x3C,0x00},
/* 86 V   */{0x66,0x66,0x66,0x66,0x66,0x3C,0x18,0x00},
/* 87 W   */{0xC6,0xC6,0xC6,0xD6,0xFE,0xEE,0xC6,0x00},
/* 88 X   */{0x66,0x66,0x3C,0x18,0x3C,0x66,0x66,0x00},
/* 89 Y   */{0x66,0x66,0x66,0x3C,0x18,0x18,0x18,0x00},
/* 90 Z   */{0x7E,0x06,0x0C,0x18,0x30,0x60,0x7E,0x00},
/* 91 [   */{0x3C,0x30,0x30,0x30,0x30,0x30,0x3C,0x00},
/* 92 \   */{0x80,0xC0,0x60,0x30,0x18,0x0C,0x06,0x00},
/* 93 ]   */{0x3C,0x0C,0x0C,0x0C,0x0C,0x0C,0x3C,0x00},
/* 94 ^   */{0x10,0x38,0x6C,0xC6,0x00,0x00,0x00,0x00},
/* 95 _   */{0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xFF},
/* 96 `   */{0x30,0x18,0x0C,0x00,0x00,0x00,0x00,0x00},
/* 97 a   */{0x00,0x00,0x3C,0x06,0x3E,0x66,0x3E,0x00},
/* 98 b   */{0x60,0x60,0x7C,0x66,0x66,0x66,0x7C,0x00},
/* 99 c   */{0x00,0x00,0x3C,0x60,0x60,0x60,0x3C,0x00},
/*100 d   */{0x06,0x06,0x3E,0x66,0x66,0x66,0x3E,0x00},
/*101 e   */{0x00,0x00,0x3C,0x66,0x7E,0x60,0x3C,0x00},
/*102 f   */{0x1C,0x30,0x30,0x7C,0x30,0x30,0x30,0x00},
/*103 g   */{0x00,0x00,0x3E,0x66,0x66,0x3E,0x06,0x3C},
/*104 h   */{0x60,0x60,0x7C,0x66,0x66,0x66,0x66,0x00},
/*105 i   */{0x18,0x00,0x38,0x18,0x18,0x18,0x3C,0x00},
/*106 j   */{0x06,0x00,0x06,0x06,0x06,0x66,0x3C,0x00},
/*107 k   */{0x60,0x60,0x66,0x6C,0x78,0x6C,0x66,0x00},
/*108 l   */{0x38,0x18,0x18,0x18,0x18,0x18,0x3C,0x00},
/*109 m   */{0x00,0x00,0xCC,0xFE,0xD6,0xC6,0xC6,0x00},
/*110 n   */{0x00,0x00,0x7C,0x66,0x66,0x66,0x66,0x00},
/*111 o   */{0x00,0x00,0x3C,0x66,0x66,0x66,0x3C,0x00},
/*112 p   */{0x00,0x00,0x7C,0x66,0x66,0x7C,0x60,0x60},
/*113 q   */{0x00,0x00,0x3E,0x66,0x66,0x3E,0x06,0x06},
/*114 r   */{0x00,0x00,0x6C,0x76,0x60,0x60,0x60,0x00},
/*115 s   */{0x00,0x00,0x3E,0x60,0x3C,0x06,0x7C,0x00},
/*116 t   */{0x18,0x18,0x7E,0x18,0x18,0x18,0x0E,0x00},
/*117 u   */{0x00,0x00,0x66,0x66,0x66,0x66,0x3E,0x00},
/*118 v   */{0x00,0x00,0x66,0x66,0x66,0x3C,0x18,0x00},
/*119 w   */{0x00,0x00,0xC6,0xD6,0xFE,0x7C,0x6C,0x00},
/*120 x   */{0x00,0x00,0x66,0x3C,0x18,0x3C,0x66,0x00},
/*121 y   */{0x00,0x00,0x66,0x66,0x3E,0x06,0x3C,0x00},
/*122 z   */{0x00,0x00,0x7E,0x0C,0x18,0x30,0x7E,0x00},
/*123 {   */{0x0E,0x18,0x18,0x70,0x18,0x18,0x0E,0x00},
/*124 |   */{0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x00},
/*125 }   */{0x70,0x18,0x18,0x0E,0x18,0x18,0x70,0x00},
/*126 ~   */{0x76,0xDC,0x00,0x00,0x00,0x00,0x00,0x00},
};

/* draw one glyph at pixel (x,y) with given ARGB colour */
static void draw_char(SDL_Renderer *rdr, char c, int x, int y,
                      uint8_t r, uint8_t g, uint8_t b) {
    if (c < 32 || c > 126) c = '?';
    const uint8_t *bm = F8[(uint8_t)c - 32];
    SDL_SetRenderDrawColor(rdr, r, g, b, 255);
    for (int row = 0; row < 8; row++) {
        uint8_t bits = bm[row];
        for (int col = 0; col < 8; col++) {
            if (bits & (0x80u >> col)) {
                SDL_Rect rect = {x + col*TS, y + row*TS, TS, TS};
                SDL_RenderFillRect(rdr, &rect);
            }
        }
    }
}

/* draw string, returns x after last char */
static int draw_str(SDL_Renderer *rdr, const char *s, int x, int y,
                    uint8_t r, uint8_t g, uint8_t b) {
    for (; *s; s++, x += GW) draw_char(rdr, *s, x, y, r, g, b);
    return x;
}

/* draw string with line-wrap inside [x0, x0+max_w], advancing *py */
static void draw_wrapped(SDL_Renderer *rdr, const char *s,
                          int x0, int *py, int max_w,
                          uint8_t r, uint8_t g, uint8_t b) {
    int cpl = max_w / GW;
    if (cpl < 1) cpl = 1;
    int len = (int)strlen(s);
    int i = 0;
    while (i < len) {
        int n = (len - i < cpl) ? len - i : cpl;
        /* try break at space */
        if (i + n < len) {
            int sp = -1;
            for (int k = i+n-1; k >= i; k--)
                if (s[k] == ' ') { sp = k; break; }
            if (sp >= i) n = sp - i + 1;
        }
        char buf[256]; int blen = n < 255 ? n : 255;
        memcpy(buf, s+i, blen); buf[blen] = '\0';
        draw_str(rdr, buf, x0, *py, r, g, b);
        *py += LH;
        i += n;
    }
    if (len == 0) *py += LH;
}

/* ════════════════════════════════════════════════════════════════════════════
 *  BRAIN
 * ════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float    v[N];           /* membrane voltage                */
    float    thr[N];         /* adaptive threshold              */
    float    avg[N];         /* firing-rate estimate            */
    uint8_t  spk[N];         /* spike this step                 */

    uint32_t idx[N][FAN];    /* pre-synaptic indices (uint32: N can exceed 65535) */
    float    W[N][FAN];      /* recurrent weights               */

    float    Wr[OUT][N];     /* readout weights                 */
    float    br[OUT];        /* readout bias                    */
    float    out[OUT];       /* last readout                    */

    /* Temporal self-attention (ATTN-11 inspired) */
    float    out_hist[ATTN_SEQ][OUT]; /* ring buffer of recent readout states  */
    int      hist_idx;                /* next write position in ring buffer    */
    float    Wq_r[ATTN_D][OUT];      /* query projection: OUT → ATTN_D        */
    float    Wk_r[ATTN_D][OUT];      /* key   projection: OUT → ATTN_D        */
    float    attn_out[OUT];           /* attention-weighted temporal readout   */

    int      n_train;
    int      n_interact;

    char     vocab[OUT][32];   /* best word label for each readout slot  */
    uint8_t  pre_spk[N];       /* spike snapshot saved before LLM call   */

    /* memory formation tracking */
    uint64_t steps;            /* total simulation steps                 */
    float    spike_rate;       /* rolling average spike rate  0..1       */
    int      vocab_count[OUT]; /* activation count per readout slot      */
    float    Wr_norm[OUT];     /* L2 norm of Wr per slot (learned str.)  */
    float    Wr_delta_acc;     /* accumulated |ΔWr| across all learning  */
    uint32_t consol_strong;    /* synapses |W|>0.5 (memory-formed)       */
    uint32_t consol_weak;      /* synapses |W|<0.5 (still plastic)       */
} Brain;

/* ════════════════════════════════════════════════════════════════════════════
 *  APP STATE
 * ════════════════════════════════════════════════════════════════════════════ */
#define HIST 6
typedef struct {
    Brain           *brain;
    pthread_mutex_t  bmtx;
    pthread_mutex_t  cmtx;
    volatile int     run;
    volatile int     llm_busy;

    /* stats snapshot (updated via trylock — never blocks render) */
    uint64_t snap_steps;
    float    snap_spike_rate;
    int      snap_train;
    int      snap_interact;
    int      snap_vocab_count[OUT];
    float    snap_Wr_norm[OUT];
    float    snap_Wr_delta;
    char     snap_vocab[OUT][32];
    uint32_t snap_consol_strong;
    uint32_t snap_consol_weak;

    /* chat */
    char  chat[MAX_CHAT][LCHAT];
    int   chat_role[MAX_CHAT];   /* 0=user 1=brain 2=sys */
    int   cn;

    /* LLM conversation history */
    char  hu[HIST][256];
    char  ha[HIST][256];
    int   hn;

    /* input */
    char  inp[INPUT_MAX+1];
    int   inl;

    int           mode;        /* MODE_LLM or MODE_BRAIN  */
    volatile int  chat_loop;   /* 1 while /c loop running */

    /* monitor counters — written by LLM/loop threads, read by monitor */
    volatile int  ollama_ok;   /* successful ollama queries               */
    volatile int  ollama_fail; /* failed ollama queries                   */
} App;

/* ════════════════════════════════════════════════════════════════════════════
 *  RNG
 * ════════════════════════════════════════════════════════════════════════════ */
static float randu(void){ return (float)rand()/((float)RAND_MAX+1.f); }
static float randn_f(void){
    static int hv=0; static float nx;
    if(hv){hv=0;return nx;}
    float u=randu()+1e-9f,v=randu()+1e-9f;
    float r=sqrtf(-2.f*logf(u));
    float a=6.28318530f*v;
    nx=r*sinf(a); hv=1;
    return r*cosf(a);
}

/* ════════════════════════════════════════════════════════════════════════════
 *  BRAIN INIT / SAVE / LOAD
 * ════════════════════════════════════════════════════════════════════════════ */
static Brain *brain_new(void){
    Brain *b=(Brain*)calloc(1,sizeof(Brain));
    if(!b){perror("calloc brain");exit(1);}
    for(int i=0;i<N;i++){
        b->thr[i]=1.5f;
        for(int j=0;j<FAN;j++){
            b->idx[i][j]=(uint32_t)((unsigned)rand()%(unsigned)N);
            b->W[i][j]=0.1f*randn_f();
        }
    }
    for(int o=0;o<OUT;o++)
        for(int i=0;i<N;i++)
            b->Wr[o][i]=0.05f*randn_f();
    /* temporal attention projections — random init, fixed (no backprop needed) */
    for(int d=0;d<ATTN_D;d++)
        for(int o=0;o<OUT;o++){
            b->Wq_r[d][o]=0.05f*randn_f();
            b->Wk_r[d][o]=0.05f*randn_f();
        }
    return b;
}
static void brain_save(const Brain *b){
    FILE *f=fopen(STATE_FILE,"wb");
    if(!f){fprintf(stderr,"Cannot save brain\n");return;}
    fwrite(b->W, sizeof(b->W),1,f);
    fwrite(b->Wr,sizeof(b->Wr),1,f);
    fwrite(b->br,sizeof(b->br),1,f);
    fwrite(&b->n_train,   sizeof(int),1,f);
    fwrite(&b->n_interact,sizeof(int),1,f);
    fwrite(b->vocab,        sizeof(b->vocab),       1,f);
    fwrite(&b->steps,       sizeof(b->steps),       1,f);
    fwrite(b->vocab_count,  sizeof(b->vocab_count), 1,f);
    fwrite(&b->Wr_delta_acc,sizeof(float),          1,f);
    fwrite(b->Wq_r,         sizeof(b->Wq_r),        1,f);
    fwrite(b->Wk_r,         sizeof(b->Wk_r),        1,f);
    fclose(f);
}
static void brain_load(Brain *b){
    FILE *f=fopen(STATE_FILE,"rb");
    if(!f)return;
    fread(b->W, sizeof(b->W),1,f);
    fread(b->Wr,sizeof(b->Wr),1,f);
    fread(b->br,sizeof(b->br),1,f);
    fread(&b->n_train,   sizeof(int),1,f);
    fread(&b->n_interact,sizeof(int),1,f);
    fread(b->vocab,        sizeof(b->vocab),       1,f);
    fread(&b->steps,       sizeof(b->steps),       1,f);
    fread(b->vocab_count,  sizeof(b->vocab_count), 1,f);
    fread(&b->Wr_delta_acc,sizeof(float),          1,f);
    fread(b->Wq_r,         sizeof(b->Wq_r),        1,f); /* 0 on old saves → uniform attn */
    fread(b->Wk_r,         sizeof(b->Wk_r),        1,f);
    fclose(f);
    /* recompute Wr_norm from loaded weights */
    for(int o=0;o<OUT;o++){
        float norm=0.f;
        for(int i=0;i<N;i++) norm+=b->Wr[o][i]*b->Wr[o][i];
        b->Wr_norm[o]=sqrtf(norm);
    }
    printf("[brain] loaded state  train=%d  interact=%d  steps=%llu\n",
           b->n_train, b->n_interact, (unsigned long long)b->steps);
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SIMULATION STEP
 * ════════════════════════════════════════════════════════════════════════════ */
static void push_readout_hist(Brain *b); /* forward decl — defined after hebbian */

static float _pre[N], _neigh[N];   /* global scratch — only one thread steps */

/* drive/noise levels — low at idle, boosted during forward pass */
static float g_drive = 0.04f;   /* idle: enough firing to be visible       */
static float g_noise = 0.02f;   /* idle: light noise for variation         */

static void brain_step(Brain *b, const float *ext){
    for(int i=0;i<N;i++) _pre[i]=(float)b->spk[i];

    /* sliding-window neighbour sum O(N) */
    float ws=0.f;
    for(int k=-NEIGH_R;k<=NEIGH_R;k++) ws+=_pre[((k%N)+N)%N];
    for(int i=0;i<N;i++){
        _neigh[i]=(ws-_pre[i])*0.02f;
        ws-=_pre[((i-NEIGH_R)%N+N)%N];
        ws+=_pre[(i+NEIGH_R+1)%N];
    }

    int sc=0;
    for(int i=0;i<N;i++){
        float syn=0.f;
        for(int j=0;j<FAN;j++) syn+=_pre[b->idx[i][j]]*b->W[i][j];
        float tot=syn+_neigh[i]+g_drive+g_noise*randn_f()+(ext?ext[i]:0.f);
        b->v[i]=b->v[i]*0.97f+tot;
        b->spk[i]=(b->v[i]>=b->thr[i])?1:0;
        if(b->spk[i]){b->v[i]=0.f; sc++;}
        b->avg[i]=0.98f*b->avg[i]+0.02f*_pre[i];
        float d=(b->avg[i]-0.05f)*0.5f;
        b->thr[i]=fmaxf(0.5f,fminf(2.5f,b->thr[i]+d));
    }
    b->steps++;
    b->spike_rate=0.995f*b->spike_rate+0.005f*((float)sc/(float)N);

    /* synaptic consolidation every 1000 steps — bimodal weight distribution:
     * strong synapses (|W|>0.5) grow toward ±1 (memory formation)
     * weak   synapses (|W|<0.5) decay toward  0 (forgetting)           */
    if(b->steps%1000==0){
        uint32_t ns=0,nw=0;
        for(int i=0;i<N;i++){
            for(int j=0;j<FAN;j++){
                float w=b->W[i][j];
                float aw=fabsf(w);
                if(aw>0.35f){
                    b->W[i][j]+=(w>0.f?1.f:-1.f)*0.002f*(1.f-aw);
                    ns++;
                } else {
                    b->W[i][j]*=0.9995f;  /* slow decay ~1400 steps half-life */
                    nw++;
                }
            }
        }
        b->consol_strong=ns;
        b->consol_weak=nw;
    }
}

/* ════════════════════════════════════════════════════════════════════════════
 *  ENCODING / FORWARD PASS / HEBBIAN
 * ════════════════════════════════════════════════════════════════════════════ */
static float _stim[N];

static void encode_text(const char *txt,float *stim){
    memset(stim,0,N*sizeof(float));
    for(int i=0;txt[i];i++){
        unsigned h=(unsigned)((unsigned char)txt[i]*2654435761u^(unsigned)(i*12345u));
        stim[h%N]+=0.5f;
    }
}
static void forward_pass(Brain *b,const char *txt,int steps){
    encode_text(txt,_stim);
    g_drive=0.05f; g_noise=0.03f;          /* boost: neurons respond to input */
    for(int s=0;s<steps;s++) brain_step(b,_stim);
    g_drive=0.04f; g_noise=0.02f;          /* restore idle levels             */
    for(int o=0;o<OUT;o++){
        float x=b->br[o];
        for(int i=0;i<N;i++) x+=b->Wr[o][i]*(float)b->spk[i];
        b->out[o]=tanhf(x);
    }
    push_readout_hist(b);                  /* update temporal attention ctx   */
}

/* deep_encode: multiple stimulus cycles with rest gaps for stronger pattern
 * formation. Each cycle: 80 steps with stim + 20 steps free decay.
 * Used during learning for better retention than a single forward_pass.    */
static void deep_encode(Brain *b, const char *txt, int cycles){
    encode_text(txt,_stim);
    g_drive=0.05f; g_noise=0.03f;
    for(int c=0;c<cycles;c++){
        for(int s=0;s<80;s++) brain_step(b,_stim);
        for(int s=0;s<20;s++) brain_step(b,NULL);  /* brief rest — pattern settles */
    }
    g_drive=0.04f; g_noise=0.02f;
    for(int o=0;o<OUT;o++){
        float x=b->br[o];
        for(int i=0;i<N;i++) x+=b->Wr[o][i]*(float)b->spk[i];
        b->out[o]=tanhf(x);
    }
    push_readout_hist(b);                  /* update temporal attention ctx   */
}

/* brain_replay: replay stored spike pattern into recurrent W.
 * Consolidates short-term spike pattern into long-term synaptic structure.
 * Analogous to hippocampal replay during sleep.                           */
static uint8_t _replay_pat[N];
static void brain_replay(Brain *b, int reps){
    memcpy(_replay_pat, b->spk, N);
    float lr=0.0008f;
    for(int r=0;r<reps;r++){
        brain_step(b,NULL);
        for(int i=0;i<N;i++){
            if(!b->spk[i]) continue;
            for(int j=0;j<FAN;j++){
                if(_replay_pat[b->idx[i][j]]){
                    b->W[i][j]+=lr;
                    if(b->W[i][j]>1.f) b->W[i][j]=1.f;
                }
            }
        }
    }
}
/* Temporal self-attention over readout history.
 *
 * Direct application of ATTN-11's dot-product attention (ATTN.MAC / LAYER.MAC)
 * but over readout TIME STEPS instead of token sequence positions:
 *
 *   Q = Wq_r . out_current               (project current state to query)
 *   K[t] = Wk_r . out_hist[t]            (project each history step to key)
 *   score[t] = Q.K[t] / sqrt(ATTN_D)     (scaled dot-product — SQRTSH equiv.)
 *   A[t] = softmax(score[t])             (ACTFN.MAC SFTMX equivalent)
 *   attn_out = Σ A[t] * out_hist[t]      (value = raw history, no V projection)
 *
 * ATTN_SEQ=8 and ATTN_D=16 deliberately mirror ATTN-11's SEQ.LN=8 / D.MODL=16.
 * attn_out is blended into brain_reply() as a temporal context residual.       */
static void attn_readout(Brain *b){
    float q[ATTN_D];
    for(int d=0;d<ATTN_D;d++){
        float s=0.f;
        for(int o=0;o<OUT;o++) s+=b->Wq_r[d][o]*b->out[o];
        q[d]=s;
    }
    float scores[ATTN_SEQ];
    float scale=1.f/sqrtf((float)ATTN_D);   /* 1/sqrt(16) = 0.25 — like SQRTSH=2 */
    for(int t=0;t<ATTN_SEQ;t++){
        int ti=(b->hist_idx - ATTN_SEQ + t + ATTN_SEQ*2) % ATTN_SEQ;
        float k[ATTN_D];
        for(int d=0;d<ATTN_D;d++){
            float s=0.f;
            for(int o=0;o<OUT;o++) s+=b->Wk_r[d][o]*b->out_hist[ti][o];
            k[d]=s;
        }
        float dot=0.f;
        for(int d=0;d<ATTN_D;d++) dot+=q[d]*k[d];
        scores[t]=dot*scale;
    }
    /* softmax */
    float mx=scores[0];
    for(int t=1;t<ATTN_SEQ;t++) if(scores[t]>mx) mx=scores[t];
    float sum=0.f;
    for(int t=0;t<ATTN_SEQ;t++){scores[t]=expf(scores[t]-mx);sum+=scores[t];}
    for(int t=0;t<ATTN_SEQ;t++) scores[t]/=sum;
    /* weighted sum of history (V = out_hist, no separate projection needed) */
    memset(b->attn_out,0,OUT*sizeof(float));
    for(int t=0;t<ATTN_SEQ;t++){
        int ti=(b->hist_idx - ATTN_SEQ + t + ATTN_SEQ*2) % ATTN_SEQ;
        float a=scores[t];
        for(int o=0;o<OUT;o++) b->attn_out[o]+=a*b->out_hist[ti][o];
    }
}

/* Push current out[] into the temporal ring buffer, then recompute attention */
static void push_readout_hist(Brain *b){
    memcpy(b->out_hist[b->hist_idx], b->out, OUT*sizeof(float));
    b->hist_idx=(b->hist_idx+1)%ATTN_SEQ;
    attn_readout(b);
}

static void hebbian_learn(Brain *b,const char *resp){
    float target[OUT]={0};
    char tmp[512]; strncpy(tmp,resp,511); tmp[511]='\0';
    for(char *tok=strtok(tmp," \t\n");tok;tok=strtok(NULL," \t\n")){
        unsigned h=0;
        for(int i=0;tok[i];i++) h=h*31+(unsigned char)tok[i];
        int slot=(int)(h%OUT);
        target[slot]=1.f;
        /* store clean label — strip non-ASCII (curly quotes etc.) */
        if(b->vocab[slot][0]=='\0'){
            char clean[32]; int ci=0;
            for(int ki=0;tok[ki]&&ci<31;ki++)
                if((unsigned char)tok[ki]>=32&&(unsigned char)tok[ki]<127
                   &&tok[ki]!='.'&&tok[ki]!=','&&tok[ki]!=':'&&tok[ki]!=';')
                    clean[ci++]=tok[ki];
            clean[ci]='\0';
            if(ci>0) strncpy(b->vocab[slot],clean,31);
        }
    }

    /* recompute current output for error signal */
    for(int o=0;o<OUT;o++){
        float x=b->br[o];
        for(int i=0;i<N;i++) x+=b->Wr[o][i]*(float)b->spk[i];
        b->out[o]=tanhf(x);
    }

    /* Delta rule: ΔWr = lr*(target-out)*spk
     * Error-driven: when concept already learned (out≈target), err≈0 → no update.
     * When novel (target=1, out≈0), err≈1 → strong update.
     * Also updates bias br[] to encode concept base activation rate.           */
    float lr  =0.003f;   /* 6x stronger than pure Hebbian was                  */
    float lr_b=0.005f;   /* bias learning rate                                  */
    for(int o=0;o<OUT;o++){
        float err=target[o]-b->out[o];
        if(fabsf(err)<0.05f) continue;    /* already learned — skip             */
        if(target[o]>0.f) b->vocab_count[o]++;
        float delta=0.f;
        for(int i=0;i<N;i++){
            float dw=lr*err*(float)b->spk[i];
            b->Wr[o][i]+=dw;
            delta+=fabsf(dw);
            if(b->Wr[o][i]> 1.f) b->Wr[o][i]= 1.f;
            if(b->Wr[o][i]<-1.f) b->Wr[o][i]=-1.f;
        }
        b->Wr_delta_acc+=delta;
        /* bias encodes how often concept is active — lowers recall threshold   */
        b->br[o]+=lr_b*err;
        if(b->br[o]> 2.f) b->br[o]= 2.f;
        if(b->br[o]<-2.f) b->br[o]=-2.f;
        /* recompute L2 norm for this slot */
        float norm=0.f;
        for(int i=0;i<N;i++) norm+=b->Wr[o][i]*b->Wr[o][i];
        b->Wr_norm[o]=sqrtf(norm);
    }
    b->n_train++;
}

/* Associate user spike pattern (pre_spk) with LLM response spike pattern.
 * Uses deep_encode for stronger response pattern, then:
 *   Hebbian (LTP): pre AND post fire → strengthen W
 *   anti-Hebbian (LTD): pre fires, post silent → weaken W (specificity)
 * This creates directed pathways: user concept → response concept.        */
static void brain_associate(Brain *b, const char *resp){
    deep_encode(b, resp, 2);     /* settle strongly on response — 2 cycles     */
    float lr     =0.001f;        /* 5x stronger than before                    */
    float lr_anti=0.0002f;       /* mild LTD for specificity                   */
    for(int i=0;i<N;i++){
        for(int j=0;j<FAN;j++){
            int pre_fired =(int)b->pre_spk[b->idx[i][j]];
            int post_fired=(int)b->spk[i];
            if(post_fired && pre_fired){
                b->W[i][j]+=lr;       /* LTP: both active → strengthen         */
                if(b->W[i][j]>1.f) b->W[i][j]=1.f;
            } else if(!post_fired && pre_fired){
                b->W[i][j]-=lr_anti;  /* LTD: pre active, post silent → weaken */
                if(b->W[i][j]<-1.f) b->W[i][j]=-1.f;
            }
        }
    }
}

/* Generate a text reply from the current readout state.
 * Takes the top-K active readout slots, looks up their vocab labels,
 * concatenates into a sentence.                                            */
static void brain_reply(Brain *b, char *out, int out_sz){
    /* find top 6 readout units with known vocab labels */
    #define REPLY_K 6
    int   best_o[REPLY_K]; float best_v[REPLY_K];
    for(int k=0;k<REPLY_K;k++){best_o[k]=-1;best_v[k]=-1e9f;}
    for(int o=0;o<OUT;o++){
        if(b->vocab[o][0]=='\0') continue;   /* skip unlabelled slots */
        /* blend current readout with attention-weighted temporal context
         * (residual mix: 0.6 now + 0.4 history — like ATTN-11's Y = O + X) */
        float v=0.6f*b->out[o]+0.4f*b->attn_out[o];
        for(int k=0;k<REPLY_K;k++){
            if(v>best_v[k]){
                /* shift down */
                for(int m=REPLY_K-1;m>k;m--){best_o[m]=best_o[m-1];best_v[m]=best_v[m-1];}
                best_o[k]=o; best_v[k]=v; break;
            }
        }
    }
    /* build output string */
    int off=0;
    for(int k=0;k<REPLY_K;k++){
        if(best_o[k]<0||best_v[k]<0.0f) continue;  /* skip weak/empty */
        off+=snprintf(out+off,out_sz-off,"%s%s",
                      (off>0?" ":""), b->vocab[best_o[k]]);
        if(off>=out_sz-1) break;
    }
    if(off==0) snprintf(out,out_sz,"...");  /* no vocab learned yet */
}

/* ════════════════════════════════════════════════════════════════════════════
 *  OLLAMA HTTP CLIENT  (raw POSIX sockets, no libcurl)
 * ════════════════════════════════════════════════════════════════════════════ */
static void json_esc(const char *in,char *out,int sz){
    int j=0;
    for(int i=0;in[i]&&j<sz-4;i++){
        char c=in[i];
        if(c=='"')      {out[j++]='\\';out[j++]='"';}
        else if(c=='\\'){out[j++]='\\';out[j++]='\\';}
        else if(c=='\n'){out[j++]='\\';out[j++]='n';}
        else if(c=='\r'){out[j++]='\\';out[j++]='r';}
        else             out[j++]=c;
    }
    out[j]='\0';
}
static void json_str(const char *json,const char *key,char *out,int sz){
    char srch[128]; snprintf(srch,sizeof(srch),"\"%s\":\"",key);
    const char *p=strstr(json,srch);
    if(!p){out[0]='\0';return;}
    p+=strlen(srch);
    int j=0;
    while(*p&&j<sz-1){
        if(*p=='\\'){
            if(!*++p) break;
            if(*p=='n') out[j++]='\n';
            else        out[j++]=*p;
            p++;
        } else if(*p=='"') break;
        else out[j++]=*p++;
    }
    out[j]='\0';
}
static int ollama_query(const char *prompt,char *out,int out_sz){
    out[0]='\0';
    char ep[4096]; json_esc(prompt,ep,sizeof(ep));
    char body[8192];
    snprintf(body,sizeof(body),
        "{\"model\":\"%s\",\"prompt\":\"%s\",\"stream\":false}",MODEL,ep);

    int fd=socket(AF_INET,SOCK_STREAM,0);
    if(fd<0) return -1;
    struct timeval tv={20,0};
    setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&tv,sizeof(tv));
    struct sockaddr_in addr={0};
    addr.sin_family=AF_INET;
    addr.sin_port=htons(11434);
    inet_pton(AF_INET,"127.0.0.1",&addr.sin_addr);
    if(connect(fd,(struct sockaddr*)&addr,sizeof(addr))<0){
        close(fd);
        snprintf(out,out_sz,"[ollama not running — start with: ollama serve]");
        return -1;
    }
    char req[12288];
    int rlen=snprintf(req,sizeof(req),
        "POST /api/generate HTTP/1.0\r\n"
        "Host: localhost:11434\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "\r\n%s",strlen(body),body);
    send(fd,req,rlen,0);

    static char raw[131072];
    int tot=0,n;
    while((n=(int)recv(fd,raw+tot,sizeof(raw)-tot-1,0))>0) tot+=n;
    raw[tot]='\0';
    close(fd);

    char *bdy=strstr(raw,"\r\n\r\n");
    if(!bdy){snprintf(out,out_sz,"[bad HTTP response]");return -1;}
    bdy+=4;
    json_str(bdy,"response",out,out_sz);
    if(!out[0]){
        json_str(bdy,"error",out,out_sz);
        if(!out[0]) snprintf(out,out_sz,"[no response from model]");
    }
    return 0;
}

/* ════════════════════════════════════════════════════════════════════════════
 *  CHAT HELPER
 * ════════════════════════════════════════════════════════════════════════════ */
static void chat_add(App *app,const char *line,int role){
    pthread_mutex_lock(&app->cmtx);
    int idx=app->cn%MAX_CHAT;
    snprintf(app->chat[idx],LCHAT,"%s",line);
    app->chat_role[idx]=role;
    app->cn++;
    pthread_mutex_unlock(&app->cmtx);
}

/* ════════════════════════════════════════════════════════════════════════════
 *  LLM THREAD
 * ════════════════════════════════════════════════════════════════════════════ */
typedef struct{App *app; char ut[512];}LLMArg;
typedef struct{App *app; char seed[512];}LoopArg;

static void *llm_thread(void *arg){
    LLMArg *a=(LLMArg*)arg;
    App    *app=a->app;
    char    ut[512]; strncpy(ut,a->ut,511); ut[511]='\0';
    free(a);

    /* deep encode user input — stronger pattern than single forward_pass */
    pthread_mutex_lock(&app->bmtx);
    deep_encode(app->brain,ut,2);
    memcpy(app->brain->pre_spk, app->brain->spk, N);

    /* find top-5 concepts by Wr_norm + active concept for prompt */
    char best_word[32]=""; float best=-1e9f;
    int top5[5]={-1,-1,-1,-1,-1}; float top5n[5]={0,0,0,0,0};
    char weak_word[32]="";        float weak_n=1e9f;
    for(int o=0;o<OUT;o++){
        if(app->brain->out[o]>best){
            best=app->brain->out[o];
            strncpy(best_word,app->brain->vocab[o][0]?app->brain->vocab[o]:"?",31);
        }
        if(!app->brain->vocab[o][0]) continue;
        float n=app->brain->Wr_norm[o];
        if(n<weak_n){weak_n=n;strncpy(weak_word,app->brain->vocab[o],31);}
        for(int k=0;k<5;k++){
            if(n>top5n[k]){
                for(int m=4;m>k;m--){top5[m]=top5[m-1];top5n[m]=top5n[m-1];}
                top5[k]=o;top5n[k]=n;break;
            }
        }
    }
    char concepts[192]=""; int coff=0;
    for(int k=0;k<5;k++){
        if(top5[k]<0) break;
        coff+=snprintf(concepts+coff,sizeof(concepts)-coff,"%s%s",
                       k?", ":"",app->brain->vocab[top5[k]]);
    }
    pthread_mutex_unlock(&app->bmtx);

    /* build prompt — brain's actual learned concepts shape the response */
    char prompt[4096]; int off=0;
    off+=snprintf(prompt+off,sizeof(prompt)-off,
        "You are a spiking neural network brain that learns and retains knowledge. "
        "Your strongest learned concepts: %s. "
        "Your weakest concept (needs reinforcement): %s. "
        "Active concept from this input: %s. "
        "Reply in 1-3 sentences, using your learned concepts where relevant.\n\n",
        concepts[0]?concepts:"(none yet)",
        weak_word[0]?weak_word:"(none yet)",
        best_word[0]?best_word:"unknown");
    for(int i=(app->hn>HIST?app->hn-HIST:0);i<app->hn;i++){
        int k=i%HIST;
        off+=snprintf(prompt+off,sizeof(prompt)-off,
            "User: %s\nBrain: %s\n",app->hu[k],app->ha[k]);
    }
    off+=snprintf(prompt+off,sizeof(prompt)-off,"User: %s\nBrain:",ut);

    char resp[2048];
    if(ollama_query(prompt,resp,sizeof(resp))<0) app->ollama_fail++;
    else app->ollama_ok++;

    /* store history */
    int k=app->hn%HIST;
    strncpy(app->hu[k],ut,  255); app->hu[k][255]='\0';
    strncpy(app->ha[k],resp,255); app->ha[k][255]='\0';
    app->hn++;

    /* Full learning sequence under lock:
     * 1. Re-encode user text (sim_thread ran freely during ollama — spk is stale)
     * 2. Delta-rule Wr update toward response words
     * 3. Associate user spike pattern → response spike pattern in W
     * 4. Replay: write response pattern into recurrent W (consolidation)        */
    pthread_mutex_lock(&app->bmtx);
    deep_encode(app->brain,ut,2);            /* restore user spike pattern       */
    hebbian_learn(app->brain,resp);          /* delta rule: Wr → response words  */
    brain_associate(app->brain,resp);        /* W: user pattern → resp pattern   */
    brain_replay(app->brain,15);             /* consolidate into recurrent W     */
    app->brain->n_interact++;
    pthread_mutex_unlock(&app->bmtx);

    /* add to chat */
    char line[LCHAT];
    snprintf(line,LCHAT,"Brain: %.230s",resp);
    chat_add(app,line,1);

    /* autosave */
    pthread_mutex_lock(&app->bmtx);
    brain_save(app->brain);
    pthread_mutex_unlock(&app->bmtx);

    app->llm_busy=0;
    return NULL;
}

/* ════════════════════════════════════════════════════════════════════════════
 *  BRAIN-MODE THREAD  — neurons reply directly, no LLM
 * ════════════════════════════════════════════════════════════════════════════ */
static void *brain_thread(void *arg){
    LLMArg *a=(LLMArg*)arg;
    App    *app=a->app;
    char    ut[512]; strncpy(ut,a->ut,511); ut[511]='\0';
    free(a);

    pthread_mutex_lock(&app->bmtx);
    forward_pass(app->brain,ut,80);
    char reply[256]="";
    brain_reply(app->brain,reply,sizeof(reply));
    app->brain->n_interact++;
    pthread_mutex_unlock(&app->bmtx);

    char line[LCHAT];
    snprintf(line,LCHAT,"Brain: %s",reply);
    chat_add(app,line,1);

    /* save after every interaction */
    pthread_mutex_lock(&app->bmtx);
    brain_save(app->brain);
    pthread_mutex_unlock(&app->bmtx);

    app->llm_busy=0;
    return NULL;
}

/* ════════════════════════════════════════════════════════════════════════════
 *  /c LEARNING LOOP — LLM teaches brain in an autonomous back-and-forth
 *
 *  LLM generates a response to current topic  (role 3 = orange)
 *  Brain learns from it via Hebbian + associate
 *  Brain replies using its readout vocab      (role 1 = cyan)
 *  Brain reply becomes next topic for LLM
 *  Repeat until /stop or window closes
 * ════════════════════════════════════════════════════════════════════════════ */
static void *chat_loop_thread(void *arg){
    LoopArg *la=(LoopArg*)arg;
    App     *app=la->app;
    char     topic[512];
    strncpy(topic,la->seed,511); topic[511]='\0';
    free(la);

    char line[LCHAT];
    snprintf(line,LCHAT,"System: /c loop started — topic: %.180s",topic);
    chat_add(app,line,2);

    int round=0;
    while(app->chat_loop && app->run){

        /* ── Find weak/strong concepts to guide teaching ─────── */
        pthread_mutex_lock(&app->bmtx);
        char strong_str[128]=""; int soff=0;
        char weak_word[32]="";   float weak_n=1e9f;
        int  top3[3]={-1,-1,-1}; float top3n[3]={0,0,0};
        for(int o=0;o<OUT;o++){
            if(!app->brain->vocab[o][0]) continue;
            float n=app->brain->Wr_norm[o];
            if(n<weak_n){weak_n=n;strncpy(weak_word,app->brain->vocab[o],31);}
            for(int k=0;k<3;k++){
                if(n>top3n[k]){
                    for(int m=2;m>k;m--){top3[m]=top3[m-1];top3n[m]=top3n[m-1];}
                    top3[k]=o;top3n[k]=n;break;
                }
            }
        }
        for(int k=0;k<3;k++){
            if(top3[k]<0) break;
            soff+=snprintf(strong_str+soff,sizeof(strong_str)-soff,"%s%s",
                           k?", ":"",app->brain->vocab[top3[k]]);
        }
        pthread_mutex_unlock(&app->bmtx);

        /* ── Every 3rd round target the weakest concept ──────── */
        if((round%3)==2 && weak_word[0])
            strncpy(topic,weak_word,sizeof(topic)-1);

        /* ── LLM turn: teach toward weak concept, build on strong */
        char prompt[2048];
        snprintf(prompt,sizeof(prompt),
            "You are teaching a spiking neural network. "
            "The brain's strong concepts: %s. "
            "It needs to learn: \"%s\". "
            "The brain said: \"%s\". "
            "In 1-2 sentences, teach it clearly and connect it to what it knows.",
            strong_str[0]?strong_str:"(learning)",
            weak_word[0]?weak_word:topic,
            topic);

        char llm_resp[1024]="[no response]";
        if(ollama_query(prompt,llm_resp,sizeof(llm_resp))<0) app->ollama_fail++;
        else app->ollama_ok++;
        if(!app->chat_loop||!app->run) break;

        snprintf(line,LCHAT,"LLM: %.230s",llm_resp);
        chat_add(app,line,3);                       /* orange = LLM teaching */

        /* ── Full learning sequence ───────────────────────────── */
        pthread_mutex_lock(&app->bmtx);
        deep_encode(app->brain,topic,2);            /* settle strongly on topic    */
        memcpy(app->brain->pre_spk,app->brain->spk,N);
        hebbian_learn(app->brain,llm_resp);         /* delta rule Wr update        */
        brain_associate(app->brain,llm_resp);       /* W: topic → response pattern */
        brain_replay(app->brain,15);                /* consolidate into W          */
        app->brain->n_interact++;
        pthread_mutex_unlock(&app->bmtx);

        /* ── Brain replies using learned readout ─────────────── */
        char brain_resp[256]="...";
        pthread_mutex_lock(&app->bmtx);
        deep_encode(app->brain,llm_resp,1);         /* settle on LLM response      */
        brain_reply(app->brain,brain_resp,sizeof(brain_resp));
        pthread_mutex_unlock(&app->bmtx);

        snprintf(line,LCHAT,"Brain: %s",brain_resp);
        chat_add(app,line,1);                       /* cyan = brain                */

        /* ── Save every exchange ─────────────────────────────── */
        pthread_mutex_lock(&app->bmtx);
        brain_save(app->brain);
        pthread_mutex_unlock(&app->bmtx);

        /* next topic: brain reply if meaningful, else LLM resp */
        strncpy(topic, brain_resp[0]&&brain_resp[0]!='.'?brain_resp:llm_resp,
                sizeof(topic)-1);
        round++;

        /* pace: 1s between rounds */
        struct timespec ts={1,0};
        nanosleep(&ts,NULL);
    }

    chat_add(app,"System: /c loop stopped.",2);
    app->chat_loop=0;
    app->llm_busy =0;
    return NULL;
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SIM THREAD  (~1000 steps/sec)
 * ════════════════════════════════════════════════════════════════════════════ */
static void *sim_thread(void *arg){
    App *app=(App*)arg;
    struct timespec ts={0,1000000};  /* 1ms — lets LLM thread acquire bmtx */
    while(app->run){
        pthread_mutex_lock(&app->bmtx);
        brain_step(app->brain,NULL);
        pthread_mutex_unlock(&app->bmtx);
        nanosleep(&ts,NULL);  /* yield: at 400k neurons step is no longer self-pacing */
    }
    return NULL;
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SUBMIT INPUT
 * ════════════════════════════════════════════════════════════════════════════ */
static void submit(App *app){
    if(!app->inl||app->llm_busy) return;

    /* ── slash commands ─────────────────────────────────────────────── */
    if(!strcmp(app->inp,"/save")){
        pthread_mutex_lock(&app->bmtx); brain_save(app->brain);
        pthread_mutex_unlock(&app->bmtx);
        chat_add(app,"System: Brain saved.",2);
        app->inp[0]='\0'; app->inl=0; return;
    }
    if(!strcmp(app->inp,"/stats")){
        char s[LCHAT];
        pthread_mutex_lock(&app->bmtx);
        snprintf(s,LCHAT,"System: train=%d  interact=%d  neurons=%d  vocab_slots=%d",
                 app->brain->n_train,app->brain->n_interact,N,
                 (int)(app->mode)); /* reuse slot for mode display */
        /* count filled vocab slots */
        int vfilled=0;
        for(int o=0;o<OUT;o++) if(app->brain->vocab[o][0]) vfilled++;
        snprintf(s,LCHAT,"System: train=%d  interact=%d  vocab=%d/256",
                 app->brain->n_train,app->brain->n_interact,vfilled);
        pthread_mutex_unlock(&app->bmtx);
        chat_add(app,s,2); app->inp[0]='\0'; app->inl=0; return;
    }
    if(!strcmp(app->inp,"/b")){
        app->mode=MODE_BRAIN;
        chat_add(app,"System: /b — BRAIN mode. Neurons reply directly.",2);
        app->inp[0]='\0'; app->inl=0; return;
    }
    if(!strcmp(app->inp,"/l")){
        app->mode=MODE_LLM;
        chat_add(app,"System: /l — LLM mode. Neurons learn from each reply.",2);
        app->inp[0]='\0'; app->inl=0; return;
    }
    if(!strncmp(app->inp,"/c",2)&&(app->inp[2]=='\0'||app->inp[2]==' ')){
        if(app->llm_busy){
            chat_add(app,"System: Busy — type /stop first.",2);
            app->inp[0]='\0'; app->inl=0; return;
        }
        app->chat_loop=1; app->llm_busy=1;
        LoopArg *la=(LoopArg*)malloc(sizeof(LoopArg));
        la->app=app;
        const char *seed=(app->inp[2]==' '&&app->inp[3])?
                          app->inp+3:"what is learning?";
        strncpy(la->seed,seed,511); la->seed[511]='\0';
        pthread_t t;
        pthread_create(&t,NULL,chat_loop_thread,la);
        pthread_detach(t);
        app->inp[0]='\0'; app->inl=0; return;
    }
    if(!strcmp(app->inp,"/stop")){
        app->chat_loop=0;
        chat_add(app,"System: Stopping loop...",2);
        app->inp[0]='\0'; app->inl=0; return;
    }

    char line[LCHAT];
    snprintf(line,LCHAT,"You: %.230s",app->inp);
    chat_add(app,line,0);

    LLMArg *a=(LLMArg*)malloc(sizeof(LLMArg));
    a->app=app;
    strncpy(a->ut,app->inp,511); a->ut[511]='\0';
    app->inp[0]='\0'; app->inl=0;
    app->llm_busy=1;

    pthread_t t;
    void *fn = (app->mode==MODE_BRAIN) ? brain_thread : llm_thread;
    pthread_create(&t,NULL,fn,a);
    pthread_detach(t);
}

/* ════════════════════════════════════════════════════════════════════════════
 *  RENDERING
 * ════════════════════════════════════════════════════════════════════════════ */

/* filled progress bar — val 0..1, w pixels wide */
static void draw_bar(SDL_Renderer *rdr, float val, int x, int y, int w,
                     uint8_t r, uint8_t g, uint8_t b){
    int fill=(int)(fminf(1.f,fmaxf(0.f,val))*(float)w);
    SDL_SetRenderDrawColor(rdr,28,28,48,255);
    SDL_Rect bg={x,y,w,GH-2}; SDL_RenderFillRect(rdr,&bg);
    if(fill>0){
        SDL_SetRenderDrawColor(rdr,r,g,b,255);
        SDL_Rect fg={x,y,fill,GH-2}; SDL_RenderFillRect(rdr,&fg);
    }
}

static void render_stats(SDL_Renderer *rdr, App *app){
    /* snapshot brain state — trylock never blocks render */
    if(pthread_mutex_trylock(&app->bmtx)==0){
        Brain *b=app->brain;
        app->snap_steps       = b->steps;
        app->snap_spike_rate  = b->spike_rate;
        app->snap_train       = b->n_train;
        app->snap_interact    = b->n_interact;
        app->snap_Wr_delta    = b->Wr_delta_acc;
        app->snap_consol_strong=b->consol_strong;
        app->snap_consol_weak  =b->consol_weak;
        memcpy(app->snap_vocab_count,b->vocab_count,OUT*sizeof(int));
        memcpy(app->snap_Wr_norm,   b->Wr_norm,    OUT*sizeof(float));
        memcpy(app->snap_vocab,     b->vocab,       sizeof(b->vocab));
        pthread_mutex_unlock(&app->bmtx);
    }

    /* panel background */
    SDL_SetRenderDrawColor(rdr,6,8,16,255);
    SDL_Rect bg={0,0,NP_X+NP_W+10,WIN_H};
    SDL_RenderFillRect(rdr,&bg);

    int x=NP_X, y=4;
    char buf[160];

    /* ── Header ─────────────────────────────────────────────────── */
    const char *mstr=app->chat_loop?"[/c LOOP]":
                     (app->mode==MODE_BRAIN)?"[/b BRAIN]":"[/l LLM]";
    uint8_t hr=app->chat_loop?220:(app->mode==MODE_BRAIN)?80:100;
    uint8_t hg=app->chat_loop?160:(app->mode==MODE_BRAIN)?220:200;
    uint8_t hb=app->chat_loop?40 :(app->mode==MODE_BRAIN)?80:255;
    snprintf(buf,sizeof(buf),"BRAIN CORE  %s%s",
             mstr, app->llm_busy?" [THINKING]":"");
    draw_str(rdr,buf,x,y,hr,hg,hb); y+=LH+1;
    SDL_SetRenderDrawColor(rdr,50,50,90,255);
    SDL_RenderDrawLine(rdr,x,y,NP_X+NP_W,y); y+=5;

    /* ── ACTIVITY ───────────────────────────────────────────────── */
    draw_str(rdr,"ACTIVITY",x,y,160,130,255); y+=LH;
    snprintf(buf,sizeof(buf),"  Steps:        %llu",
             (unsigned long long)app->snap_steps);
    draw_str(rdr,buf,x,y,200,200,200); y+=LH;
    snprintf(buf,sizeof(buf),"  Spike rate:   %.3f%%",
             app->snap_spike_rate*100.f);
    draw_str(rdr,buf,x,y,200,200,200); y+=LH;
    draw_bar(rdr,app->snap_spike_rate*20.f,x+4,y,NP_W-8,80,200,100);
    y+=GH;
    snprintf(buf,sizeof(buf),"  Interactions: %d    Training events: %d",
             app->snap_interact,app->snap_train);
    draw_str(rdr,buf,x,y,200,200,200); y+=LH;
    snprintf(buf,sizeof(buf),"  Drive: %.3f   Noise: %.3f",g_drive,g_noise);
    draw_str(rdr,buf,x,y,140,140,160); y+=LH;
    snprintf(buf,sizeof(buf),"  Temporal attn: %d steps x d=%d  [ATTN-11]",
             ATTN_SEQ,ATTN_D);
    draw_str(rdr,buf,x,y,120,160,140); y+=LH+3;

    SDL_SetRenderDrawColor(rdr,35,35,65,255);
    SDL_RenderDrawLine(rdr,x,y,NP_X+NP_W,y); y+=5;

    /* ── MEMORY CONSOLIDATION ───────────────────────────────────── */
    draw_str(rdr,"MEMORY CONSOLIDATION",x,y,255,180,80); y+=LH;
    uint32_t total_s=app->snap_consol_strong+app->snap_consol_weak;
    float spc=total_s>0?100.f*(float)app->snap_consol_strong/(float)total_s:0.f;
    uint32_t total_syn=(uint32_t)N*(uint32_t)FAN;
    snprintf(buf,sizeof(buf),"  Strong |W|>0.5: %7u / %u  (%.3f%%)",
             app->snap_consol_strong, total_syn,
             100.f*(float)app->snap_consol_strong/(float)total_syn);
    draw_str(rdr,buf,x,y,80,220,120); y+=LH;
    snprintf(buf,sizeof(buf),"  Weak   |W|<0.5: %7u         (%.2f%%)",
             app->snap_consol_weak,
             100.f*(float)app->snap_consol_weak/(float)total_syn);
    draw_str(rdr,buf,x,y,160,160,160); y+=LH;
    draw_str(rdr,"  Formation:",x,y,140,140,140);
    draw_bar(rdr,spc/100.f,x+13*GW,y,NP_W-14*GW,60,180,80); y+=LH+3;

    SDL_SetRenderDrawColor(rdr,35,35,65,255);
    SDL_RenderDrawLine(rdr,x,y,NP_X+NP_W,y); y+=5;

    /* ── READOUT CONCEPTS ───────────────────────────────────────── */
    int filled=0;
    float max_norm=0.001f;
    for(int o=0;o<OUT;o++){
        if(app->snap_vocab[o][0]) filled++;
        if(app->snap_Wr_norm[o]>max_norm) max_norm=app->snap_Wr_norm[o];
    }
    snprintf(buf,sizeof(buf),"READOUT CONCEPTS  %d/%d filled  (%.1f%%)",
             filled,OUT,100.f*filled/OUT);
    draw_str(rdr,buf,x,y,160,130,255); y+=LH;

    /* vocab fill bar */
    draw_bar(rdr,(float)filled/OUT,x,y,NP_W-4,100,80,200);  y+=GH+2;

    /* header — columns match data row layout below */
    #define BAR_W 130
    #define COL_BAR  (x+25*GW)            /* bar starts here */
    #define COL_CNT  (COL_BAR+BAR_W+2)    /* count starts here */
    draw_str(rdr,"  #  Slot  Word          ",x,y,100,100,130);
    draw_str(rdr,"Strength",COL_BAR,y,100,100,130);
    draw_str(rdr,"Cnt",COL_CNT,y,100,100,130); y+=LH;

    /* sort top SHOW_N slots by Wr_norm */
    #define SHOW_N 18
    int sidx[OUT]; for(int o=0;o<OUT;o++) sidx[o]=o;
    for(int k=0;k<SHOW_N&&k<OUT;k++){
        int best=k;
        for(int j=k+1;j<OUT;j++)
            if(app->snap_Wr_norm[sidx[j]]>app->snap_Wr_norm[sidx[best]])
                best=j;
        int tmp=sidx[k]; sidx[k]=sidx[best]; sidx[best]=tmp;
    }
    for(int k=0;k<SHOW_N;k++){
        if(y+LH>WIN_H-50) break;
        int o=sidx[k];
        if(!app->snap_vocab[o][0]&&app->snap_Wr_norm[o]<0.1f) continue;
        const char *word=app->snap_vocab[o][0]?app->snap_vocab[o]:"---";
        snprintf(buf,sizeof(buf),"  %2d  %3d  %-13s",k+1,o,word);
        draw_str(rdr,buf,x,y,200,200,200);
        draw_bar(rdr,app->snap_Wr_norm[o]/max_norm,COL_BAR,y,BAR_W,80,160,220);
        snprintf(buf,sizeof(buf),"%5d",app->snap_vocab_count[o]);
        draw_str(rdr,buf,COL_CNT,y,160,160,160);
        y+=LH;
    }
    y+=3;

    if(y<WIN_H-52){
        SDL_SetRenderDrawColor(rdr,35,35,65,255);
        SDL_RenderDrawLine(rdr,x,y,NP_X+NP_W,y); y+=5;

        /* ── LEARNING DYNAMICS ──────────────────────────────── */
        draw_str(rdr,"LEARNING DYNAMICS",x,y,255,120,120); y+=LH;
        snprintf(buf,sizeof(buf),"  Hebbian events:    %d",app->snap_train);
        draw_str(rdr,buf,x,y,200,200,200); y+=LH;
        snprintf(buf,sizeof(buf),"  dWr accumulated:   %.1f",app->snap_Wr_delta);
        draw_str(rdr,buf,x,y,200,200,200); y+=LH;
        if(app->snap_train>0){
            float rate=app->snap_Wr_delta/(float)app->snap_train;
            snprintf(buf,sizeof(buf),"  dWr per event:     %.4f",rate);
            draw_str(rdr,buf,x,y,200,200,200); y+=LH;
        }
        /* find most active concept */
        int best_o=-1,best_cnt=0;
        for(int o=0;o<OUT;o++)
            if(app->snap_vocab_count[o]>best_cnt&&app->snap_vocab[o][0])
                {best_cnt=app->snap_vocab_count[o];best_o=o;}
        if(best_o>=0){
            snprintf(buf,sizeof(buf),"  Top concept:       \"%s\" (slot %d, %d activations)",
                     app->snap_vocab[best_o],best_o,best_cnt);
            draw_str(rdr,buf,x,y,220,200,100); y+=LH;
        }
        /* % improvement: Wr_delta per n_train vs baseline */
        if(app->snap_train>0&&y+LH<WIN_H-4){
            float pct=fminf(100.f,app->snap_Wr_delta/(float)N*1000.f);
            snprintf(buf,sizeof(buf),"  Learning progress: %.2f%%",pct);
            draw_str(rdr,buf,x,y,200,200,100);
            draw_bar(rdr,pct/100.f,x+22*GW,y,140,200,200,80);
            y+=LH;
        }
    }
}

static void render_chat(SDL_Renderer *rdr, App *app, int frame){
    /* panel bg */
    SDL_SetRenderDrawColor(rdr,12,12,22,255);
    SDL_Rect bg={CP_X,0,CP_W+6,WIN_H};
    SDL_RenderFillRect(rdr,&bg);
    SDL_SetRenderDrawColor(rdr,50,50,90,255);
    SDL_RenderDrawLine(rdr,CP_X,0,CP_X,WIN_H);

    draw_str(rdr,"CHAT",CP_X+4,4,200,180,50);

    /* input box */
    int iy=WIN_H-38;
    SDL_SetRenderDrawColor(rdr,25,25,45,255);
    SDL_Rect ib={CP_X+3,iy-3,CP_W,LH+8};
    SDL_RenderFillRect(rdr,&ib);
    SDL_SetRenderDrawColor(rdr,70,70,110,255);
    SDL_RenderDrawRect(rdr,&ib);

    if(app->llm_busy){
        draw_str(rdr,"Thinking...",CP_X+6,iy,180,180,50);
    } else {
        /* scroll: show tail of input so cursor is always visible */
        int max_chars=(CP_W-12)/GW - 2;  /* chars that fit after "> " */
        if(max_chars<1) max_chars=1;
        int start=(app->inl>max_chars)?(app->inl-max_chars):0;
        char disp[INPUT_MAX+4];
        snprintf(disp,sizeof(disp),"> %s",app->inp+start);
        if((frame/25)&1) strncat(disp,"_",sizeof(disp)-strlen(disp)-1);
        draw_str(rdr,disp,CP_X+6,iy,220,220,220);
    }

    /* chat lines — newest at bottom */
    pthread_mutex_lock(&app->cmtx);
    int n=app->cn;
    /* figure out how many lines we can fit */
    int avail_y=iy-4;
    #define VCAP 60
    int vis[VCAP]; int vc=0;
    memset(vis,0,sizeof(vis));
    /* collect from newest backwards */
    for(int i=n-1;i>=0&&vc<VCAP;i--) vis[vc++]=i;
    /* measure from bottom to find oldest visible */
    int start=vc-1;
    int cur_y=avail_y;
    for(int vi=0;vi<vc;vi++){
        int idx=vis[vi]%MAX_CHAT;
        int len=(int)strlen(app->chat[idx]);
        int cpl2=CP_W/GW; if(cpl2<1)cpl2=1;
        int lines=(len+cpl2-1)/cpl2; if(lines<1)lines=1;
        cur_y-=lines*LH+2;
        if(cur_y<28){start=vi-1;break;}
    }
    if(start<0) start=0;
    /* render from oldest visible to newest */
    int py=28;
    for(int vi=start;vi>=0;vi--){
        int idx=vis[vi]%MAX_CHAT;
        int role=app->chat_role[idx];
        uint8_t cr,cg,cb;
        if(role==0){cr=80;cg=220;cb=100;}       /* green   = user        */
        else if(role==1){cr=80;cg=200;cb=220;}  /* cyan    = brain reply  */
        else if(role==3){cr=220;cg=160;cb=60;}  /* orange  = LLM teaching */
        else{cr=140;cg=140;cb=160;}             /* grey    = system       */
        draw_wrapped(rdr,app->chat[idx],CP_X+6,&py,CP_W-8,cr,cg,cb);
        if(py>=iy-4) break;
    }
    pthread_mutex_unlock(&app->cmtx);
}

/* ════════════════════════════════════════════════════════════════════════════
 *  MONITOR  — samples brain state every MON_INTERVAL_S seconds and writes
 *  a structured diagnostic log to LOG_FILE.  Runs in its own thread so it
 *  never blocks the sim or render loops.
 * ════════════════════════════════════════════════════════════════════════════ */

/* Snapshot all monitoring data under bmtx, then release before writing. */
static void monitor_write(App *app, FILE *lf){
    char ts[32];
    {
        time_t now=time(NULL);
        struct tm *tm=localtime(&now);
        strftime(ts,sizeof(ts),"%Y-%m-%d %H:%M:%S",tm);
    }

    /* ── snapshot under lock ─────────────────────────────────────── */
    pthread_mutex_lock(&app->bmtx);
    Brain *b=app->brain;

    uint64_t steps      = b->steps;
    float    spike_rate = b->spike_rate;
    int      n_train    = b->n_train;
    int      n_interact = b->n_interact;
    float    Wr_delta   = b->Wr_delta_acc;
    uint32_t cs         = b->consol_strong;
    uint32_t cw         = b->consol_weak;
    int      mode       = app->mode;
    int      looping    = app->chat_loop;
    int      ok         = app->ollama_ok;
    int      fail       = app->ollama_fail;

    int vfill=0;
    for(int o=0;o<OUT;o++) if(b->vocab[o][0]) vfill++;

    /* top-5 by Wr_norm */
    int   top5[5]={-1,-1,-1,-1,-1};
    float top5n[5]={0,0,0,0,0};
    char  top5w[5][32]; memset(top5w,0,sizeof(top5w));

    /* weakest */
    float min_norm=1e9f; int min_o=-1; char min_word[32]="";

    for(int o=0;o<OUT;o++){
        if(!b->vocab[o][0]) continue;
        float n=b->Wr_norm[o];
        if(n<min_norm){min_norm=n;min_o=o;strncpy(min_word,b->vocab[o],31);min_word[31]='\0';}
        for(int k=0;k<5;k++){
            if(n>top5n[k]){
                for(int m=4;m>k;m--){
                    top5[m]=top5[m-1]; top5n[m]=top5n[m-1];
                    memcpy(top5w[m],top5w[m-1],32);
                }
                top5[k]=o; top5n[k]=n;
                strncpy(top5w[k],b->vocab[o],31); top5w[k][31]='\0';
                break;
            }
        }
    }

    /* most-activated concept */
    int best_cnt=0; int best_o=-1; char best_word[32]="";
    for(int o=0;o<OUT;o++)
        if(b->vocab_count[o]>best_cnt&&b->vocab[o][0])
            {best_cnt=b->vocab_count[o];best_o=o;strncpy(best_word,b->vocab[o],31);}

    /* quick brain reply — reads only, safe under lock */
    char test_reply[256]="";
    brain_reply(b,test_reply,sizeof(test_reply));

    /* attention entropy proxy — variance of attn_out vs out */
    float attn_var=0.f, out_mean=0.f;
    for(int o=0;o<OUT;o++) out_mean+=b->out[o];
    out_mean/=OUT;
    for(int o=0;o<OUT;o++){
        float d=b->attn_out[o]-out_mean;
        attn_var+=d*d;
    }
    attn_var/=OUT;

    pthread_mutex_unlock(&app->bmtx);

    /* ── derived metrics ─────────────────────────────────────────── */
    uint32_t total_syn=(uint32_t)N*(uint32_t)FAN;
    float strong_pct  = total_syn>0 ? 100.f*(float)cs/(float)total_syn : 0.f;
    float vocab_pct   = 100.f*(float)vfill/(float)OUT;
    float dwr_per_ev  = n_train>0 ? Wr_delta/(float)n_train : 0.f;

    /* ── write log ───────────────────────────────────────────────── */
    fprintf(lf,"\n[%s] ========== BRAIN MONITOR ==========\n",ts);
    fprintf(lf,"  Mode:            %s%s\n",
            mode==MODE_BRAIN?"BRAIN":"LLM", looping?" + /c LOOP":"");
    fprintf(lf,"  Steps:           %llu\n",(unsigned long long)steps);

    /* ACTIVITY */
    fprintf(lf,"\n--- ACTIVITY ---\n");
    fprintf(lf,"  Spike rate:      %.4f%%",spike_rate*100.f);
    if     (spike_rate<0.0005f)  fprintf(lf,"  [DEAD — neurons silent; check drive/connections]\n");
    else if(spike_rate>0.15f)    fprintf(lf,"  [EPILEPTIC — runaway firing; threshold adaptation may be failing]\n");
    else if(spike_rate<0.005f)   fprintf(lf,"  [VERY SPARSE — may be ok for sparse coding but watch vocab fill]\n");
    else if(spike_rate<0.02f)    fprintf(lf,"  [SPARSE — healthy LIF range]\n");
    else                         fprintf(lf,"  [OK]\n");
    fprintf(lf,"  Interactions:    %d\n",n_interact);
    fprintf(lf,"  Training events: %d\n",n_train);

    /* VOCABULARY */
    fprintf(lf,"\n--- VOCABULARY / READOUT ---\n");
    fprintf(lf,"  Vocab fill:      %d/%d (%.1f%%)",vfill,OUT,vocab_pct);
    if     (vfill==0)             fprintf(lf,"  [EMPTY — no interactions yet]\n");
    else if(vocab_pct<10.f)       fprintf(lf,"  [VERY LOW — needs more interactions]\n");
    else if(vocab_pct<30.f)       fprintf(lf,"  [LOW — learning proceeding, keep interacting]\n");
    else if(vocab_pct>90.f)       fprintf(lf,"  [SATURATED — slot collisions likely; diversity may be lost]\n");
    else                          fprintf(lf,"  [OK]\n");

    fprintf(lf,"  Top concepts:    ");
    int any=0;
    for(int k=0;k<5;k++){
        if(top5[k]<0) break;
        fprintf(lf,"%s%s(%.3f)",any?", ":"",top5w[k],top5n[k]);
        any=1;
    }
    if(!any) fprintf(lf,"(none)");
    fprintf(lf,"\n");

    if(best_o>=0)
        fprintf(lf,"  Most activated:  \"%s\" — %d times (slot %d)\n",
                best_word,best_cnt,best_o);
    if(min_o>=0)
        fprintf(lf,"  Weakest concept: \"%s\"  norm=%.4f (slot %d)  [needs reinforcement]\n",
                min_word,min_norm,min_o);

    /* LEARNING DYNAMICS */
    fprintf(lf,"\n--- LEARNING DYNAMICS ---\n");
    fprintf(lf,"  dWr accumulated: %.2f\n",Wr_delta);
    fprintf(lf,"  dWr per event:   %.4f",dwr_per_ev);
    if     (n_train==0)           fprintf(lf,"  [NO LEARNING YET — send messages to start]\n");
    else if(dwr_per_ev<0.1f)      fprintf(lf,"  [WEAK — error signal small; inputs too similar or weights saturating]\n");
    else if(dwr_per_ev>500.f)     fprintf(lf,"  [VERY STRONG — rapid weight changes; watch for oscillation]\n");
    else                          fprintf(lf,"  [OK]\n");

    /* MEMORY CONSOLIDATION */
    fprintf(lf,"\n--- MEMORY CONSOLIDATION ---\n");
    fprintf(lf,"  Strong |W|>0.35: %u / %u (%.4f%%)",cs,total_syn,strong_pct);
    if     (strong_pct<0.05f)     fprintf(lf,"  [EARLY — almost no consolidated synapses yet]\n");
    else if(strong_pct<1.f)       fprintf(lf,"  [FORMING — consolidation underway]\n");
    else if(strong_pct>40.f)      fprintf(lf,"  [HIGH — risk of plasticity exhaustion]\n");
    else                          fprintf(lf,"  [OK]\n");
    fprintf(lf,"  Weak   |W|<0.35: %u\n",cw);

    /* ATTENTION */
    fprintf(lf,"\n--- TEMPORAL ATTENTION (ATTN-11 style, seq=%d d=%d) ---\n",
            ATTN_SEQ,ATTN_D);
    fprintf(lf,"  attn_out variance: %.6f",attn_var);
    if(attn_var<1e-6f)            fprintf(lf,"  [FLAT — attention weights uniform; not differentiating history]\n");
    else if(attn_var<0.001f)      fprintf(lf,"  [LOW — weak temporal discrimination]\n");
    else                          fprintf(lf,"  [OK — temporal context is shaping output]\n");

    /* BRAIN REPLY */
    fprintf(lf,"\n--- BRAIN REPLY TEST ---\n");
    fprintf(lf,"  Reply:           \"%s\"",test_reply);
    if(strcmp(test_reply,"...")==0) fprintf(lf,"  [MUTE — no vocab yet; LLM mode needed first]\n");
    else if(strlen(test_reply)<4)   fprintf(lf,"  [THIN — very few concepts available]\n");
    else                            fprintf(lf,"  [OK]\n");

    /* OLLAMA */
    fprintf(lf,"\n--- OLLAMA LLM ---\n");
    fprintf(lf,"  Queries OK:      %d\n",ok);
    fprintf(lf,"  Queries FAIL:    %d",fail);
    if(fail>0 && ok==0)             fprintf(lf,"  [ALL FAILING — run: ollama serve]\n");
    else if(fail>0 && fail>ok/2)    fprintf(lf,"  [FREQUENT FAILURES — ollama may be overloaded or timing out]\n");
    else if(fail>0)                 fprintf(lf,"  [SOME FAILURES — transient; ok if rare]\n");
    else                            fprintf(lf,"\n");

    /* ISSUES SUMMARY */
    fprintf(lf,"\n--- ISSUES / SUGGESTIONS ---\n");
    int issues=0;
    if(spike_rate<0.0005f)
        {fprintf(lf,"  !! Neurons silent — check g_drive and connectivity.\n");issues++;}
    if(spike_rate>0.15f)
        {fprintf(lf,"  !! Epileptic firing — threshold adaptation is not converging.\n");issues++;}
    if(vfill==0 && steps>5000)
        {fprintf(lf,"  !! No vocab after %llu steps — try sending a message.\n",(unsigned long long)steps);issues++;}
    if(n_train>0 && dwr_per_ev<0.1f)
        {fprintf(lf,"  !  Weak learning (dWr/event=%.4f) — vary inputs or check spike activity.\n",dwr_per_ev);issues++;}
    if(fail>0 && ok==0)
        {fprintf(lf,"  !! All ollama queries fail — LLM mode non-functional. Run: ollama serve\n");issues++;}
    if(vfill>=(OUT-5))
        {fprintf(lf,"  !  Readout slots nearly full (%d/256) — hash collisions will overwrite old concepts.\n",vfill);issues++;}
    if(strong_pct>40.f)
        {fprintf(lf,"  !  High synapse consolidation (%.1f%%) — plasticity may be exhausted; new learning slow.\n",strong_pct);issues++;}
    if(attn_var<1e-6f && n_interact>5)
        {fprintf(lf,"  !  Attention output flat — temporal context not contributing to replies.\n");issues++;}
    if(strcmp(test_reply,"...")==0 && n_interact>10)
        {fprintf(lf,"  !  Brain still mute after %d interactions — Wr learning may be too weak.\n",n_interact);issues++;}
    if(issues==0)
        fprintf(lf,"  (none — brain appears healthy)\n");

    fprintf(lf,"=====================================\n");
    fflush(lf);
}

static void *monitor_thread(void *arg){
    App *app=(App*)arg;
    FILE *lf=fopen(LOG_FILE,"a");
    if(!lf){perror("monitor: cannot open "LOG_FILE);return NULL;}

    {
        char ts[32];
        time_t now=time(NULL);
        struct tm *tm=localtime(&now);
        strftime(ts,sizeof(ts),"%Y-%m-%d %H:%M:%S",tm);
        fprintf(lf,"\n\n[%s] === Brain Core Monitor Started (interval=%ds) ===\n",
                ts, MON_INTERVAL_S);
        fflush(lf);
    }

    while(app->run){
        /* sleep in 1s chunks so we notice run=0 promptly */
        for(int i=0;i<MON_INTERVAL_S&&app->run;i++){
            struct timespec ts={1,0};
            nanosleep(&ts,NULL);
        }
        if(!app->run) break;
        monitor_write(app,lf);
    }

    {
        char ts[32];
        time_t now=time(NULL);
        struct tm *tm=localtime(&now);
        strftime(ts,sizeof(ts),"%Y-%m-%d %H:%M:%S",tm);
        fprintf(lf,"[%s] === Brain Core Monitor Stopped ===\n",ts);
    }
    fclose(lf);
    return NULL;
}

/* ════════════════════════════════════════════════════════════════════════════
 *  MAIN
 * ════════════════════════════════════════════════════════════════════════════ */
int main(int argc,char *argv[]){
    (void)argc;(void)argv;
    srand((unsigned)time(NULL));

    App *app=(App*)calloc(1,sizeof(App));
    if(!app){perror("calloc app");return 1;}
    app->run=1;
    pthread_mutex_init(&app->bmtx,NULL);
    pthread_mutex_init(&app->cmtx,NULL);

    app->brain=brain_new();
    brain_load(app->brain);

    if(SDL_Init(SDL_INIT_VIDEO)<0){
        fprintf(stderr,"SDL_Init: %s\n",SDL_GetError());return 1;}

    SDL_Window *win=SDL_CreateWindow(
        "Brain Core - Spiking Neural Network",
        SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,
        WIN_W,WIN_H,SDL_WINDOW_SHOWN);
    if(!win){fprintf(stderr,"SDL_CreateWindow: %s\n",SDL_GetError());return 1;}

    SDL_Renderer *rdr=SDL_CreateRenderer(win,-1,
        SDL_RENDERER_ACCELERATED);  /* no vsync — we gate at RENDER_FPS */
    if(!rdr){fprintf(stderr,"SDL_CreateRenderer: %s\n",SDL_GetError());return 1;}

    pthread_t stid, mtid;
    pthread_create(&stid,NULL,sim_thread,app);
    pthread_create(&mtid,NULL,monitor_thread,app);

    SDL_StartTextInput();

    chat_add(app,"System: Brain online. Type a message and press Enter.",2);
    chat_add(app,"System: /l=LLM mode  /b=Brain mode  /c [topic]=teach loop",2);
    chat_add(app,"System: /stop=end loop  /save  /stats  Ctrl-Q=quit",2);

    int frame=0;
    uint32_t last_render=0;
    SDL_Event ev;
    while(app->run){
        while(SDL_PollEvent(&ev)){
            if(ev.type==SDL_QUIT){app->run=0;break;}
            if(ev.type==SDL_KEYDOWN){
                SDL_Keycode k=ev.key.keysym.sym;
                SDL_Keymod  mod=SDL_GetModState();
                if(k==SDLK_RETURN||k==SDLK_KP_ENTER){submit(app);}
                else if(k==SDLK_BACKSPACE){if(app->inl>0)app->inp[--app->inl]='\0';}
                else if(k==SDLK_ESCAPE||(k==SDLK_q&&(mod&KMOD_CTRL))){app->run=0;}
            }
            if(ev.type==SDL_TEXTINPUT){
                /* filter control chars — Ctrl+key combos can leak \x01-\x1f */
                if((unsigned char)ev.text.text[0]>=32){
                    int len=(int)strlen(ev.text.text);
                    if(app->inl+len<INPUT_MAX){
                        memcpy(app->inp+app->inl,ev.text.text,len);
                        app->inl+=len; app->inp[app->inl]='\0';
                    }
                }
            }
        }

        /* 5 fps render gate — yield CPU between frames */
        uint32_t now=SDL_GetTicks();
        if((now-last_render)>=RENDER_INTERVAL_MS){
            SDL_SetRenderDrawColor(rdr,8,8,18,255);
            SDL_RenderClear(rdr);
            render_stats(rdr,app);
            render_chat(rdr,app,frame);
            SDL_RenderPresent(rdr);
            frame++;
            last_render=now;
        } else {
            SDL_Delay(5);  /* yield to sim/LLM threads between render frames */
        }
    }

    SDL_StopTextInput();
    app->run=0;
    pthread_join(stid,NULL);
    pthread_join(mtid,NULL);
    brain_save(app->brain);
    printf("[brain] state saved. bye.\n");

    SDL_DestroyRenderer(rdr);
    SDL_DestroyWindow(win);
    SDL_Quit();
    pthread_mutex_destroy(&app->bmtx);
    pthread_mutex_destroy(&app->cmtx);
    free(app->brain);
    free(app);
    return 0;
}
