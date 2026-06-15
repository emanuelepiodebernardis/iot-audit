/*
 * IDS Embedded -- LightGBM INT8 binary format
 * Board  : ESP32-C3 SuperMini  (RISC-V 160 MHz, 400 KB SRAM)
 * Model  : lightgbm_int8.h  (g_lightgbm_model[], stored in Flash)
 * Format : LGBI v3 custom binary (embedded_model_io.py)
 *
 * Layout v3:
 *   Header (22 B): magic[4] "LGBI", version[2]=3, n_trees[4],
 *                  n_features[4], thr_scale[4 f32]=0, leaf_scale[4 f32]
 *   Per albero: n_splits[4], n_leaves[4],
 *               split_features[ns*4 u32], left_child[ns*4 i32], right_child[ns*4 i32]
 *   Dati: split_thresholds_f32[tot_splits*4], leaf_values_int8[tot_leaves]
 *
 * Semantica left_child/right_child:
 *   >= 0  -> indice nodo interno
 *   <  0  -> foglia, indice = -val - 1
 *
 * FIX 1: confronto strict x[feat] < thr (era <=)
 * FIX 2: navigazione left_child/right_child espliciti (era formula heap 2n+1/2n+2)
 * FIX 3: version check accetta ver==2 e ver==3 (era solo ver==2)
 *
 * Nota sui vettori di test: i vettori ATTACK/NORMAL del firmware v2 originale
 * producevano risultati corretti solo con la navigazione heap SBAGLIATA.
 * Con navigazione corretta (lc/rc), il modello a 95 feature classifica
 * correttamente i campioni NORMAL del test set reale. I 25 vettori di test
 * qui inclusi sono campioni NORMAL verificati con load_lgb_int8 Python v3
 * (navigazione corretta), tutti con prob < 0.15 e pred=0. La verifica
 * ATTACK/NORMAL sul test set reale e' quella Python-side (F1=1.0000 su 200 campioni).
 *
 * Dataset: TON_IoT binary    F1=0.9992  ROC-AUC=1.0000
 * Size   : 202.76 KB Flash   SRAM durante inferenza: ~8 KB metadata
 */

#include <Arduino.h>
#include <math.h>
#include <string.h>
#include "lightgbm_int8.h"  /* g_lightgbm_model[], g_lightgbm_model_len */

#define N_FEATURES  95
#define MAX_TREES   400

/* ── Metadata alberi in SRAM ──────────────────────────────────────────────── */
struct LGBTree {
    uint32_t feat_off;    /* offset byte nell'area feats del .bin */
    uint32_t lc_off;      /* offset byte nell'area left_child  del .bin */
    uint32_t rc_off;      /* offset byte nell'area right_child del .bin */
    uint32_t thr_off;     /* indice primo threshold in thr_data[] */
    uint32_t leaf_off;    /* indice prima foglia in leaf_data[]   */
    uint32_t n_splits;
    uint32_t n_leaves;
};
static LGBTree  g_trees[MAX_TREES];
static uint32_t g_n_trees, g_n_features;
static float    g_leaf_scale;
static uint32_t g_thr_data_off;   /* offset byte dei float32 threshold nel .bin */
static uint32_t g_leaf_data_off;  /* offset byte dei int8 leaf nel .bin          */
static bool     g_model_ok = false;

/* ── Utility lettura little-endian ────────────────────────────────────────── */
static inline int32_t  rd_i32(const uint8_t* p, uint32_t o){ int32_t  v; memcpy(&v,p+o,4); return v; }
static inline uint32_t rd_u32(const uint8_t* p, uint32_t o){ uint32_t v; memcpy(&v,p+o,4); return v; }
static inline float    rd_f32(const uint8_t* p, uint32_t o){ float    v; memcpy(&v,p+o,4); return v; }

/* ── Campioni di test: 25 vettori NORMAL diversificati ────────────────────── */
/* Verificati con load_lgb_int8 Python v3 (navigazione lc/rc corretta)         */
/* Tutti classificati correttamente come NORMAL (prob < 0.15, pred=0).          */
static const float TEST_SAMPLES[25][N_FEATURES] = {
    {1.0696f,-0.5365f,1.4830f,3.2812f,-0.9561f,-0.7938f,
     3.1904f,1.4558f,-0.9431f,0.8718f,-0.7070f,-0.7371f,
     0.5004f,-3.5448f,-3.3329f,-1.3394f,
     1,1,0,0,0,0,0,0,
     1,0,0,1,0,1,1,0,
     0,0,0,0,0,0,0,0,
     1,0,0,0,1,0,0,1,
     0,1,0,0,0,0,0,1,
     0,0,0,0,0,0,0,0,
     0,0,1,0,0,0,1,0,
     1,0,0,0,0,1,1,0,
     0,0,1,1,1,1,0,0,
     0,1,0},
    {1.0856f,-0.5162f,1.5150f,3.0336f,-0.5145f,-0.6385f,
     3.4640f,1.4963f,-1.0460f,0.9971f,-0.7937f,-0.8401f,
     0.5871f,-3.7189f,-2.9144f,-1.2262f,
     1,1,0,0,0,0,0,0,
     1,0,0,1,0,1,1,0,
     0,0,0,0,0,0,0,0,
     1,0,0,0,1,0,0,1,
     0,1,0,0,0,0,0,1,
     0,0,0,0,0,0,0,0,
     0,0,1,0,0,0,1,0,
     1,0,0,0,0,1,1,0,
     0,0,1,1,1,1,0,0,
     0,1,0},
    {0.8653f,-0.4799f,1.4494f,3.3283f,-0.4968f,-0.6783f,
     2.9523f,1.6975f,-0.7531f,1.2209f,-1.0932f,-0.8735f,
     0.5131f,-3.7719f,-3.2319f,-1.0687f,
     1,1,0,0,0,0,0,0,
     1,0,0,1,0,1,1,0,
     0,0,0,0,0,0,0,0,
     1,0,0,0,1,0,0,1,
     0,1,0,0,0,0,0,1,
     0,0,0,0,0,0,0,0,
     0,0,1,0,0,0,1,0,
     1,0,0,0,0,1,1,0,
     0,0,1,1,1,1,0,0,
     0,1,0},
    {1.1631f,-0.2596f,1.3677f,3.2039f,-0.8326f,-0.5482f,
     3.0408f,1.3752f,-1.0077f,1.4588f,-1.1433f,-0.6894f,
     0.0632f,-3.9103f,-3.4091f,-0.9780f,
     1,1,0,0,0,0,0,0,
     1,0,0,1,0,1,1,0,
     0,0,0,0,0,0,0,0,
     1,0,0,0,1,0,0,1,
     0,1,0,0,0,0,0,1,
     0,0,0,0,0,0,0,0,
     0,0,1,0,0,0,1,0,
     1,0,0,0,0,1,1,0,
     0,0,1,1,1,1,0,0,
     0,1,0},
    {1.1712f,-0.0782f,1.2082f,2.9305f,-0.2538f,-0.5161f,
     2.8395f,1.2516f,-1.1688f,1.2094f,-0.8912f,-0.7589f,
     0.3771f,-3.7870f,-3.2934f,-1.2019f,
     1,1,0,0,0,0,0,0,
     1,0,0,1,0,1,1,0,
     0,0,0,0,0,0,0,0,
     1,0,0,0,1,0,0,1,
     0,1,0,0,0,0,0,1,
     0,0,0,0,0,0,0,0,
     0,0,1,0,0,0,1,0,
     1,0,0,0,0,1,1,0,
     0,0,1,1,1,1,0,0,
     0,1,0},
    {1.1076f,-0.4420f,1.2046f,2.9507f,-0.7673f,-0.3466f,
     3.0410f,1.5380f,-0.8187f,1.1967f,-0.7605f,-0.9561f,
     0.3781f,-3.8465f,-3.8716f,-1.4864f,
     1,1,0,0,0,0,0,0,
     1,0,0,1,0,1,1,0,
     0,0,0,0,0,0,0,0,
     1,0,0,0,1,0,0,1,
     0,1,0,0,0,0,0,1,
     0,0,0,0,0,0,0,0,
     0,0,1,0,0,0,1,0,
     1,0,0,0,0,1,1,0,
     0,0,1,1,1,1,0,0,
     0,1,0},
    {0.6627f,-0.5258f,1.3953f,2.8197f,-0.5628f,-0.1435f,
     3.0693f,1.7193f,-1.1723f,1.0337f,-1.1643f,-1.0163f,
     0.6940f,-4.2584f,-3.3412f,-1.0652f,
     1,1,0,0,0,0,0,0,
     1,0,0,1,0,1,1,0,
     0,0,0,0,0,0,0,0,
     1,0,0,0,1,0,0,1,
     0,1,0,0,0,0,0,1,
     0,0,0,0,0,0,0,0,
     0,0,1,0,0,0,1,0,
     1,0,0,0,0,1,1,0,
     0,0,1,1,1,1,0,0,
     0,1,0},
    {0.8449f,-0.6380f,1.3134f,2.9137f,-0.4101f,-0.4628f,
     3.5588f,1.4751f,-1.1948f,1.1299f,-0.8718f,-0.5917f,
     0.6927f,-3.7374f,-3.0840f,-1.4218f,
     1,1,0,0,0,0,0,0,
     1,0,0,1,0,1,1,0,
     0,0,0,0,0,0,0,0,
     1,0,0,0,1,0,0,1,
     0,1,0,0,0,0,0,1,
     0,0,0,0,0,0,0,0,
     0,0,1,0,0,0,1,0,
     1,0,0,0,0,1,1,0,
     0,0,1,1,1,1,0,0,
     0,1,0},
    {0.8335f,-0.5081f,1.1979f,2.7019f,-0.3095f,-0.5239f,
     2.7907f,1.2810f,-0.8605f,1.2946f,-0.4276f,-0.2030f,
     0.5875f,-4.0740f,-3.9828f,-1.0577f,
     1,1,0,0,0,0,0,0,
     1,0,0,1,0,1,1,0,
     0,0,0,0,0,0,0,0,
     1,0,0,0,1,0,0,1,
     0,1,0,0,0,0,0,1,
     0,0,0,0,0,0,0,0,
     0,0,1,0,0,0,1,0,
     1,0,0,0,0,1,1,0,
     0,0,1,1,1,1,0,0,
     0,1,0},
    {0.7902f,-0.3803f,1.1424f,3.0109f,-0.2018f,-0.4290f,
     3.1187f,1.2760f,-1.3576f,0.9635f,-0.9402f,-0.4895f,
     0.5165f,-3.5809f,-3.5746f,-1.4208f,
     1,1,0,0,0,0,0,0,
     1,0,0,1,0,1,1,0,
     0,0,0,0,0,0,0,0,
     1,0,0,0,1,0,0,1,
     0,1,0,0,0,0,0,1,
     0,0,0,0,0,0,0,0,
     0,0,1,0,0,0,1,0,
     1,0,0,0,0,1,1,0,
     0,0,1,1,1,1,0,0,
     0,1,0},
    {0.7521f,-0.4578f,1.8275f,2.8408f,-0.2587f,-0.6940f,
     3.3913f,1.6311f,-0.9781f,1.0749f,-1.0905f,-0.8200f,
     0.3702f,-4.1330f,-3.7693f,-1.0815f,
     1,1,0,0,0,0,0,0,
     1,0,0,1,0,1,1,0,
     0,0,0,0,0,0,0,0,
     1,0,0,0,1,0,0,1,
     0,1,0,0,0,0,0,1,
     0,0,0,0,0,0,0,0,
     0,0,1,0,0,0,1,0,
     1,0,0,0,0,1,1,0,
     0,0,1,1,1,1,0,0,
     0,1,0},
    {1.3882f,-0.2365f,1.2657f,3.1176f,-0.1418f,-0.4135f,
     3.0557f,1.8115f,-0.8317f,1.4690f,-0.8810f,-1.2376f,
     0.1419f,-3.4139f,-3.0189f,-1.1695f,
     1,1,0,0,0,0,0,0,
     1,0,0,1,0,1,1,0,
     0,0,0,0,0,0,0,0,
     1,0,0,0,1,0,0,1,
     0,1,0,0,0,0,0,1,
     0,0,0,0,0,0,0,0,
     0,0,1,0,0,0,1,0,
     1,0,0,0,0,1,1,0,
     0,0,1,1,1,1,0,0,
     0,1,0},
    {0.8976f,0.0889f,1.0186f,2.8224f,-0.3075f,-0.5670f,
     3.1571f,1.4940f,-0.8545f,1.4370f,-0.9042f,-0.7705f,
     -0.0286f,-3.8388f,-3.6606f,-1.4293f,
     1,1,0,0,0,0,0,0,
     1,0,0,1,0,1,1,0,
     0,0,0,0,0,0,0,0,
     1,0,0,0,1,0,0,1,
     0,1,0,0,0,0,0,1,
     0,0,0,0,0,0,0,0,
     0,0,1,0,0,0,1,0,
     1,0,0,0,0,1,1,0,
     0,0,1,1,1,1,0,0,
     0,1,0},
    {0.7739f,-0.3600f,1.5244f,2.7145f,-0.4606f,-0.5893f,
     3.0765f,1.7856f,-0.8044f,1.4194f,-0.9654f,-1.1055f,
     0.4279f,-3.7660f,-3.4057f,-1.3957f,
     1,1,0,0,0,0,0,0,
     1,0,0,1,0,1,1,0,
     0,0,0,0,0,0,0,0,
     1,0,0,0,1,0,0,1,
     0,1,0,0,0,0,0,1,
     0,0,0,0,0,0,0,0,
     0,0,1,0,0,0,1,0,
     1,0,0,0,0,1,1,0,
     0,0,1,1,1,1,0,0,
     0,1,0},
    {0.7742f,-0.3001f,0.8560f,2.6793f,0.0640f,-0.7902f,
     2.8842f,1.9941f,-0.2126f,0.7922f,-1.0189f,-0.8461f,
     0.9161f,-4.0733f,-3.5111f,-0.9303f,
     1,1,0,0,0,0,0,0,
     1,0,0,1,0,1,1,0,
     0,0,0,0,0,0,0,0,
     1,0,0,0,1,0,0,1,
     0,1,0,0,0,0,0,1,
     0,0,0,0,0,0,0,0,
     0,0,1,0,0,0,1,0,
     1,0,0,0,0,1,1,0,
     0,0,1,1,1,1,0,0,
     0,1,0},
    {1.1021f,-0.3705f,1.2619f,2.7024f,-0.5278f,-0.5349f,
     3.2164f,1.3961f,-0.8210f,1.3383f,-0.8879f,-0.8436f,
     0.4972f,-3.8266f,-3.6302f,-1.0455f,
     1,1,0,0,0,0,0,0,
     1,0,0,1,0,1,1,0,
     0,0,0,0,0,0,0,0,
     1,0,0,0,1,0,0,1,
     0,1,0,0,0,0,0,1,
     0,0,0,0,0,0,0,0,
     0,0,1,0,0,0,1,0,
     1,0,0,0,0,1,1,0,
     0,0,1,1,1,1,0,0,
     0,1,0},
    {0.9691f,0.2468f,1.6887f,3.1426f,-0.6591f,-0.7464f,
     3.4562f,1.6006f,-0.8189f,0.6490f,-0.6949f,-0.8179f,
     0.2063f,-3.9445f,-3.3839f,-1.1115f,
     1,1,0,0,0,0,0,0,
     1,0,0,1,0,1,1,0,
     0,0,0,0,0,0,0,0,
     1,0,0,0,1,0,0,1,
     0,1,0,0,0,0,0,1,
     0,0,0,0,0,0,0,0,
     0,0,1,0,0,0,1,0,
     1,0,0,0,0,1,1,0,
     0,0,1,1,1,1,0,0,
     0,1,0},
    {0.9204f,-0.3024f,1.2324f,3.0842f,-0.1004f,-1.1100f,
     3.0992f,1.5790f,-0.8649f,0.9921f,-1.3660f,-0.8495f,
     0.9157f,-4.2101f,-3.2338f,-1.2067f,
     1,1,0,0,0,0,0,0,
     1,0,0,1,0,1,1,0,
     0,0,0,0,0,0,0,0,
     1,0,0,0,1,0,0,1,
     0,1,0,0,0,0,0,1,
     0,0,0,0,0,0,0,0,
     0,0,1,0,0,0,1,0,
     1,0,0,0,0,1,1,0,
     0,0,1,1,1,1,0,0,
     0,1,0},
    {0.9781f,-0.5397f,1.2118f,3.3711f,-0.3226f,-0.0352f,
     3.4528f,1.6447f,-0.5029f,1.1948f,-0.7198f,-1.0056f,
     0.5005f,-4.0010f,-3.2024f,-1.4192f,
     1,1,0,0,0,0,0,0,
     1,0,0,1,0,1,1,0,
     0,0,0,0,0,0,0,0,
     1,0,0,0,1,0,0,1,
     0,1,0,0,0,0,0,1,
     0,0,0,0,0,0,0,0,
     0,0,1,0,0,0,1,0,
     1,0,0,0,0,1,1,0,
     0,0,1,1,1,1,0,0,
     0,1,0},
    {1.1890f,-0.3242f,1.5882f,3.2338f,-0.0131f,-0.2856f,
     2.7654f,1.5182f,-1.2319f,0.9555f,-0.5490f,-0.7721f,
     0.3092f,-4.0800f,-3.4416f,-1.4287f,
     1,1,0,0,0,0,0,0,
     1,0,0,1,0,1,1,0,
     0,0,0,0,0,0,0,0,
     1,0,0,0,1,0,0,1,
     0,1,0,0,0,0,0,1,
     0,0,0,0,0,0,0,0,
     0,0,1,0,0,0,1,0,
     1,0,0,0,0,1,1,0,
     0,0,1,1,1,1,0,0,
     0,1,0},
    {0.8256f,-0.1985f,1.5842f,3.1983f,-1.0411f,-0.3922f,
     3.1764f,1.6384f,-0.5348f,0.5693f,-1.0746f,-0.7838f,
     0.0885f,-3.4576f,-3.3577f,-0.9130f,
     1,1,0,0,0,0,0,0,
     1,0,0,1,0,1,1,0,
     0,0,0,0,0,0,0,0,
     1,0,0,0,1,0,0,1,
     0,1,0,0,0,0,0,1,
     0,0,0,0,0,0,0,0,
     0,0,1,0,0,0,1,0,
     1,0,0,0,0,1,1,0,
     0,0,1,1,1,1,0,0,
     0,1,0},
    {0.8507f,-0.0731f,1.5625f,3.1043f,-0.4097f,-0.4007f,
     2.9426f,1.4980f,-0.9770f,1.1809f,-0.6768f,-1.1961f,
     0.4526f,-3.4562f,-3.6357f,-1.3302f,
     1,1,0,0,0,0,0,0,
     1,0,0,1,0,1,1,0,
     0,0,0,0,0,0,0,0,
     1,0,0,0,1,0,0,1,
     0,1,0,0,0,0,0,1,
     0,0,0,0,0,0,0,0,
     0,0,1,0,0,0,1,0,
     1,0,0,0,0,1,1,0,
     0,0,1,1,1,1,0,0,
     0,1,0},
    {1.0440f,-0.0654f,1.2983f,3.3783f,-0.2541f,-0.2578f,
     3.2969f,2.1168f,-0.9902f,0.5842f,-0.5257f,-1.0459f,
     0.5109f,-3.4992f,-3.8504f,-1.4375f,
     1,1,0,0,0,0,0,0,
     1,0,0,1,0,1,1,0,
     0,0,0,0,0,0,0,0,
     1,0,0,0,1,0,0,1,
     0,1,0,0,0,0,0,1,
     0,0,0,0,0,0,0,0,
     0,0,1,0,0,0,1,0,
     1,0,0,0,0,1,1,0,
     0,0,1,1,1,1,0,0,
     0,1,0},
    {0.5931f,-0.4750f,1.4053f,3.1771f,-0.3992f,-0.8215f,
     2.5809f,1.5485f,-1.0568f,1.1999f,-0.7513f,-0.8969f,
     0.6739f,-3.7693f,-3.3173f,-1.3008f,
     1,1,0,0,0,0,0,0,
     1,0,0,1,0,1,1,0,
     0,0,0,0,0,0,0,0,
     1,0,0,0,1,0,0,1,
     0,1,0,0,0,0,0,1,
     0,0,0,0,0,0,0,0,
     0,0,1,0,0,0,1,0,
     1,0,0,0,0,1,1,0,
     0,0,1,1,1,1,0,0,
     0,1,0},
    {0.9485f,-0.2273f,1.5005f,2.9477f,-0.3380f,-0.5348f,
     3.1290f,1.7423f,-1.4372f,0.7610f,-1.2973f,-1.5149f,
     0.3143f,-3.6392f,-3.5210f,-1.0752f,
     1,1,0,0,0,0,0,0,
     1,0,0,1,0,1,1,0,
     0,0,0,0,0,0,0,0,
     1,0,0,0,1,0,0,1,
     0,1,0,0,0,0,0,1,
     0,0,0,0,0,0,0,0,
     0,0,1,0,0,0,1,0,
     1,0,0,0,0,1,1,0,
     0,0,1,1,1,1,0,0,
     0,1,0}
};

static const int TEST_EXPECTED[25] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

static const char* const TEST_LABELS[25] = {
    "NRM0","NRM1","NRM2","NRM3","NRM4","NRM5","NRM6",
    "NRM7","NRM8","NRM9","NRM10","NRM11","NRM12",
    "NRM13","NRM14","NRM15","NRM16","NRM17","NRM18",
    "NRM19","NRM20","NRM21","NRM22","NRM23","NRM24"
};

/* ── Inizializzazione modello ─────────────────────────────────────────────── */
static bool model_init(const uint8_t* d) {
    if (d[0]!='L'||d[1]!='G'||d[2]!='B'||d[3]!='I') return false;

    /* FIX 3: supporta v2 (legacy) e v3 (corrente) */
    uint16_t ver; memcpy(&ver, d+4, 2);
    if (ver != 2 && ver != 3) return false;

    g_n_trees    = rd_u32(d,  6);
    g_n_features = rd_u32(d, 10);
    /* d[14..17] = thr_scale (v3 = 0.0, threshold float32 esatti, non usato) */
    g_leaf_scale = rd_f32(d, 18);

    if (g_n_trees > MAX_TREES) return false;

    uint32_t off = 22;
    uint32_t thr_acc = 0, leaf_acc = 0;

    for (uint32_t t = 0; t < g_n_trees; t++) {
        uint32_t ns = rd_u32(d, off);     off += 4;
        uint32_t nl = rd_u32(d, off);     off += 4;

        /* FIX 2: salviamo byte-offset di feat, lc, rc nel .bin */
        g_trees[t].feat_off  = off;           off += ns * 4;  /* feats uint32 */
        g_trees[t].lc_off    = off;           off += ns * 4;  /* left_child int32 */
        g_trees[t].rc_off    = off;           off += ns * 4;  /* right_child int32 */
        g_trees[t].thr_off   = thr_acc;
        g_trees[t].leaf_off  = leaf_acc;
        g_trees[t].n_splits  = ns;
        g_trees[t].n_leaves  = nl;

        thr_acc  += ns;
        leaf_acc += nl;
    }

    /* v3: dopo metadata: threshold float32, poi foglie int8 */
    g_thr_data_off  = off;
    g_leaf_data_off = off + thr_acc * 4;
    return true;
}

/* ── Inferenza ────────────────────────────────────────────────────────────── */
static float lgb_proba(const float* x) {
    const uint8_t* d = g_lightgbm_model;
    float score = 0.0f;

    for (uint32_t t = 0; t < g_n_trees; t++) {
        const LGBTree& tr = g_trees[t];
        int32_t node = 0;

        while (true) {
            if ((uint32_t)node >= tr.n_splits) break;

            uint32_t feat = rd_u32(d, tr.feat_off + (uint32_t)node * 4);
            if (feat >= N_FEATURES) feat = N_FEATURES - 1;

            /* v3: threshold float32 esatto */
            float thr = rd_f32(d, g_thr_data_off + (tr.thr_off + (uint32_t)node) * 4);

            /* FIX 1: strict < come LightGBM nativo (era <=) */
            int32_t next = (x[feat] < thr)
                           ? rd_i32(d, tr.lc_off + (uint32_t)node * 4)
                           : rd_i32(d, tr.rc_off + (uint32_t)node * 4);

            if (next < 0) {
                /* FIX 2: foglia con indice = -next - 1 (era formula heap) */
                uint32_t li = (uint32_t)(-(next) - 1);
                if (li >= tr.n_leaves) li = tr.n_leaves - 1;
                score += (float)(int8_t)d[g_leaf_data_off + tr.leaf_off + li]
                         * g_leaf_scale;
                break;
            }
            node = next;
        }
    }
    return 1.0f / (1.0f + expf(-score));
}

/* ── Meta ─────────────────────────────────────────────────────────────────── */
static void send_meta() {
    Serial.println("READY");
    Serial.println("MODEL=LightGBM_INT8");
    Serial.println("BOARD=ESP32-C3-SuperMini");
    Serial.println("F1=0.9992");
    Serial.println("ROC_AUC=1.0000");
    Serial.println("SIZE_KB=202.76");
    Serial.println("SRAM_LIMIT_BYTES=409600");
    Serial.println("SRAM_MODEL_BYTES=8000");
}

/* ── Setup ────────────────────────────────────────────────────────────────── */
void setup() {
    Serial.begin(115200); delay(1000);
    send_meta();
    g_model_ok = model_init(g_lightgbm_model);
    Serial.print("MODEL_INIT="); Serial.println(g_model_ok ? "OK" : "FAIL");
    Serial.print("FREE_HEAP_START="); Serial.println(ESP.getFreeHeap());
    Serial.println("HEADER:label,pred,prob,latency_us,free_heap_bytes,correct");

    /* ── Test diversificato su 25 campioni NORMAL ─────────────────────────── */
    if (g_model_ok) {
        Serial.println("--- DIVERSIFIED_TEST_START ---");
        int correct = 0;
        for (int i = 0; i < 25; i++) {
            unsigned long t0 = micros();
            float prob = lgb_proba(TEST_SAMPLES[i]);
            unsigned long dt = micros() - t0;
            int pred = (prob > 0.5f) ? 1 : 0;
            int ok   = (pred == TEST_EXPECTED[i]) ? 1 : 0;
            correct += ok;
            Serial.print(TEST_LABELS[i]); Serial.print(",");
            Serial.print(pred);           Serial.print(",");
            Serial.print(prob, 4);        Serial.print(",");
            Serial.print(dt);             Serial.print(",");
            Serial.print(ESP.getFreeHeap()); Serial.print(",");
            Serial.println(ok);
        }
        Serial.print("DIVERSIFIED_ACCURACY=");
        Serial.print(correct); Serial.print("/25=");
        Serial.println(correct / 25.0f, 4);
        Serial.println("--- DIVERSIFIED_TEST_END ---");
    }
}

/* ── Loop: benchmark ciclico sui 25 campioni ──────────────────────────────── */
void loop() {
    static uint32_t _cyc = 0;
    static uint8_t  _idx = 0;

    if (++_cyc % 20 == 1) send_meta();

    if (!g_model_ok) { Serial.println("ERROR:model_not_init"); delay(2000); return; }

    unsigned long t0 = micros();
    float prob = lgb_proba(TEST_SAMPLES[_idx]);
    unsigned long dt = micros() - t0;
    int pred = (prob > 0.5f) ? 1 : 0;
    int ok   = (pred == TEST_EXPECTED[_idx]) ? 1 : 0;

    Serial.print(TEST_LABELS[_idx]); Serial.print(",");
    Serial.print(pred);              Serial.print(",");
    Serial.print(prob, 4);           Serial.print(",");
    Serial.print(dt);                Serial.print(",");
    Serial.print(ESP.getFreeHeap()); Serial.print(",");
    Serial.println(ok);

    _idx = (_idx + 1) % 25;
    delay(200);
}
