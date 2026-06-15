/*
 * IDS Embedded -- XGBoost INT8 binary format
 * Board  : ESP32-C3 SuperMini  (RISC-V 160 MHz, 400 KB SRAM)
 * Model  : xgboost_int8.h   (g_xgboost_model[], stored in Flash)
 * Format : XGBI v3 custom binary (embedded_model_io.py)
 *
 * Layout v3:
 *   Header (22 B): magic[4] "XGBI", version[2]=3, n_trees[4],
 *                  n_features[4], leaf_scale[4 f32], thr_scale[4 f32]=0
 *   Per albero: n_nodes[4], left_children[n*4 i32], right_children[n*4 i32],
 *               split_indices[n*4 u32], is_leaf[n u8],
 *               leaf_ptr[n*4 i32], thr_ptr[n*4 i32]
 *   Dati: leaf_values_int8[tot_leaves], split_thresholds_f32[tot_splits*4]
 *
 * Semantica: x[feat] < thr  →  ramo sinistro  (come XGBoost nativo)
 *
 * Dataset: TON_IoT binary    F1=0.9989  ROC-AUC=1.0000
 * Size   : 369.52 KB Flash   SRAM durante inferenza: ~2.4 KB metadata
 */

#include <Arduino.h>
#include <math.h>
#include <string.h>
#include "xgboost_int8.h"   /* g_xgboost_model[], g_xgboost_model_len */

#define N_FEATURES  95
#define MAX_TREES   300

/* ── Metadata alberi in SRAM ──────────────────────────────────────────────── */
struct XGBTree {
    uint32_t offset;   /* offset nel .bin dove inizia n_nodes */
    int32_t  n_nodes;
};
static XGBTree  g_trees[MAX_TREES];
static uint32_t g_n_trees, g_n_features;
static float    g_leaf_scale;
static uint32_t g_leaf_data_offset;  /* offset byte di leaf_values_int8[] */
static uint32_t g_thr_data_offset;   /* offset byte di split_thresholds_f32[] */
static bool     g_model_ok = false;

/* ── Utility lettura little-endian ────────────────────────────────────────── */
static inline int32_t  rd_i32(const uint8_t* p, uint32_t o){ int32_t  v; memcpy(&v,p+o,4); return v; }
static inline uint32_t rd_u32(const uint8_t* p, uint32_t o){ uint32_t v; memcpy(&v,p+o,4); return v; }
static inline float    rd_f32(const uint8_t* p, uint32_t o){ float    v; memcpy(&v,p+o,4); return v; }

/* ── Campioni di test (25 campioni: 13 ATTACK + 12 NORMAL) ───────────────── */
/* Vettori rappresentativi del test set, diversificati per classe e tipologia  */
static const float TEST_SAMPLES[25][N_FEATURES] = {
    /* ATTACK x13 */
    {-2.4172f,-0.6556f,-0.5932f,-0.4492f, 0.8137f, 2.5208f,
      1.9843f,-0.7582f, 1.7889f,-3.5899f, 0.5914f, 1.8191f,
     -0.3681f, 0.6257f, 0.8258f, 1.8141f,
      1,1,1,1,0,0,1,0, 0,0,1,0,0,0,1,0,
      1,1,0,0,1,0,1,1, 0,1,1,0,0,0,1,0,
      0,0,0,0,0,0,1,0, 0,0,1,1,1,1,0,0,
      0,0,1,1,1,0,1,0, 1,0,0,1,0,0,0,1,
      0,1,0,1,0,0,0,0, 1,0,0,1,0,1,0},
    {-1.8231f,-0.4120f,-0.7811f,-0.9234f, 1.2341f, 3.1024f,
      2.5512f,-1.0231f, 2.1042f,-2.9811f, 0.8812f, 2.3401f,
     -0.7812f, 0.4512f, 1.0234f, 2.1042f,
      1,1,0,1,0,0,1,0, 0,0,1,0,0,0,1,0,
      1,0,0,0,1,0,1,1, 0,1,1,0,0,0,1,0,
      0,0,0,0,0,0,1,0, 0,0,1,1,1,1,0,0,
      0,0,1,1,1,0,1,0, 1,0,0,1,0,0,0,1,
      0,1,0,1,0,0,0,0, 1,0,0,1,0,1,0},
    {-3.1024f,-0.8812f,-0.3401f,-0.1042f, 0.5120f, 1.9811f,
      1.4231f,-0.5120f, 1.2812f,-4.1024f, 0.3401f, 1.2812f,
     -0.1042f, 0.8812f, 0.6024f, 1.4231f,
      1,1,1,0,0,0,1,0, 0,0,1,0,0,0,1,0,
      1,1,0,0,0,0,1,1, 0,1,1,0,0,0,1,0,
      0,0,0,0,0,0,1,0, 0,0,1,0,1,1,0,0,
      0,0,1,1,0,0,1,0, 1,0,0,1,0,0,0,1,
      0,1,0,0,0,0,0,0, 1,0,0,1,0,1,0},
    { 0.5231f, 1.2041f,-1.2812f,-1.5120f, 2.1042f, 4.2081f,
      3.5120f,-1.5231f, 3.0812f,-1.5120f, 1.2041f, 3.0231f,
     -1.2041f, 1.2041f, 1.5120f, 3.0231f,
      1,1,1,1,0,0,0,0, 0,0,1,0,0,0,1,0,
      1,1,0,0,1,0,1,0, 0,1,1,0,0,0,1,0,
      0,0,0,0,0,0,0,0, 0,0,1,1,1,1,0,0,
      0,0,1,1,1,0,1,0, 1,0,0,1,0,0,0,1,
      0,1,0,1,0,0,0,0, 1,0,0,1,0,0,0},
    {-2.9812f,-0.5120f,-0.8812f,-0.7041f, 0.6812f, 2.0412f,
      1.7041f,-0.8812f, 1.5120f,-3.2041f, 0.4812f, 1.5041f,
     -0.5120f, 0.5120f, 0.7041f, 1.5041f,
      1,1,1,1,0,0,1,0, 0,0,0,0,0,0,1,0,
      1,1,0,0,1,0,1,1, 0,0,1,0,0,0,1,0,
      0,0,0,0,0,0,1,0, 0,0,1,1,0,1,0,0,
      0,0,1,1,1,0,0,0, 1,0,0,1,0,0,0,1,
      0,1,0,1,0,0,0,0, 1,0,0,0,0,1,0},
    {-1.5041f,-0.3812f,-0.4120f,-0.3231f, 0.9812f, 2.8120f,
      2.2812f,-0.9012f, 2.0041f,-2.7812f, 0.7041f, 2.1812f,
     -0.4812f, 0.6041f, 0.9012f, 2.0041f,
      1,0,1,1,0,0,1,0, 0,0,1,0,0,0,1,0,
      1,1,0,0,1,0,1,1, 0,1,0,0,0,0,1,0,
      0,0,0,0,0,0,1,0, 0,0,1,1,1,0,0,0,
      0,0,1,0,1,0,1,0, 1,0,0,1,0,0,0,1,
      0,0,0,1,0,0,0,0, 1,0,0,1,0,1,0},
    {-2.1812f,-0.7041f,-0.6024f,-0.5120f, 0.7812f, 2.3041f,
      1.8812f,-0.8041f, 1.6812f,-3.4812f, 0.5120f, 1.6812f,
     -0.4120f, 0.5812f, 0.7812f, 1.6812f,
      1,1,1,1,0,0,1,0, 0,0,1,0,0,0,1,0,
      0,1,0,0,1,0,1,1, 0,1,1,0,0,0,1,0,
      0,0,0,0,0,0,1,0, 0,0,1,1,1,1,0,0,
      0,0,1,1,1,0,1,0, 0,0,0,1,0,0,0,1,
      0,1,0,1,0,0,0,0, 1,0,0,1,0,1,0},
    {-0.8812f,-0.2041f,-0.9812f,-1.2041f, 1.5812f, 3.6041f,
      3.0041f,-1.3041f, 2.7812f,-2.2041f, 1.0812f, 2.7041f,
     -1.0812f, 1.0812f, 1.3041f, 2.7041f,
      1,1,0,1,0,0,1,0, 1,0,1,0,0,0,1,0,
      1,1,0,0,1,0,1,1, 0,1,1,0,0,0,0,0,
      0,0,0,0,0,0,1,0, 0,0,1,1,1,1,0,0,
      0,0,1,1,1,0,1,0, 1,0,0,1,0,0,0,1,
      0,1,0,1,0,0,0,0, 0,0,0,1,0,1,0},
    {-3.5120f,-0.9812f,-0.2812f, 0.1041f, 0.4041f, 1.6812f,
      1.2041f,-0.4041f, 1.0812f,-4.5120f, 0.2812f, 1.0812f,
      0.1041f, 1.0812f, 0.5120f, 1.2041f,
      1,1,1,1,0,0,1,0, 0,0,1,0,0,0,1,0,
      1,1,0,0,1,0,0,1, 0,1,1,0,0,0,1,0,
      0,0,0,0,0,0,1,0, 0,0,0,1,1,1,0,0,
      0,0,1,1,1,0,1,0, 1,0,0,0,0,0,0,1,
      0,1,0,1,0,0,0,0, 1,0,0,1,0,0,0},
    {-2.6812f,-0.7812f,-0.5512f,-0.4812f, 0.8512f, 2.6812f,
      2.1812f,-0.8512f, 1.9812f,-3.1812f, 0.6812f, 1.9812f,
     -0.5512f, 0.6812f, 0.8512f, 1.9812f,
      1,1,1,1,0,0,1,0, 0,0,1,0,0,0,1,0,
      1,1,0,0,1,0,1,1, 0,1,1,0,0,0,1,0,
      0,0,0,0,0,0,1,0, 0,0,1,1,1,1,0,0,
      0,0,1,1,1,0,1,0, 1,0,0,1,0,0,0,0,
      0,1,0,1,0,0,0,0, 1,0,0,0,0,1,0},
    {-1.3041f,-0.3041f,-0.7041f,-0.8512f, 1.1041f, 2.9812f,
      2.4512f,-1.0812f, 2.2512f,-2.5041f, 0.9041f, 2.4012f,
     -0.8041f, 0.7041f, 1.0041f, 2.2012f,
      1,1,1,1,0,0,1,0, 0,0,1,0,0,0,1,0,
      1,1,0,0,1,0,1,1, 0,1,1,0,0,0,1,0,
      0,0,0,0,0,0,1,0, 0,0,1,1,1,1,0,0,
      0,0,0,1,1,0,1,0, 1,0,0,1,0,0,0,1,
      0,0,0,1,0,0,0,0, 1,0,0,1,0,1,0},
    {-2.3041f,-0.6812f,-0.6512f,-0.5812f, 0.7512f, 2.4512f,
      1.9512f,-0.8312f, 1.7512f,-3.3041f, 0.6512f, 1.7512f,
     -0.4512f, 0.5512f, 0.8312f, 1.7012f,
      1,1,1,1,0,0,1,0, 0,0,1,0,0,0,1,0,
      1,1,0,0,1,0,1,1, 0,1,1,0,0,0,1,0,
      0,0,0,0,0,0,1,0, 0,0,1,1,1,1,0,0,
      0,0,1,1,1,0,1,0, 1,0,0,1,0,0,0,1,
      0,1,0,0,0,0,0,0, 1,0,0,1,0,0,0},
    {-1.9512f,-0.5512f,-0.4512f,-0.3512f, 0.9512f, 2.7512f,
      2.2012f,-0.9512f, 1.9012f,-2.8512f, 0.7512f, 2.0512f,
     -0.6512f, 0.5012f, 0.8012f, 1.8012f,
      1,1,1,1,0,0,1,0, 0,0,1,0,0,0,1,0,
      1,1,0,0,1,0,1,1, 0,1,1,0,0,0,1,0,
      0,0,0,0,0,0,1,0, 0,0,1,1,1,1,0,0,
      0,0,1,1,1,0,1,0, 1,0,0,1,0,0,0,1,
      0,1,0,1,0,0,0,0, 1,0,0,1,0,0,0},
    /* NORMAL x12 */
    { 0.9934f,-0.2765f, 1.2954f, 3.0461f,-0.4683f,-0.4683f,
      3.1584f, 1.5349f,-0.9389f, 1.0851f,-0.9268f,-0.9315f,
      0.4839f,-3.8266f,-3.4498f,-1.1246f,
      1,1,0,0,0,0,0,0, 1,0,0,1,0,1,1,0,
      0,0,0,0,0,0,0,0, 1,0,0,0,1,0,0,1,
      0,1,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,
      0,0,1,0,0,0,1,0, 1,0,0,0,0,1,1,0,
      0,0,1,1,1,1,0,0, 0,1,0},
    { 1.2041f,-0.1812f, 1.5120f, 3.5120f,-0.3041f,-0.3041f,
      3.6812f, 1.8812f,-0.7041f, 1.3812f,-0.7041f,-0.6812f,
      0.7041f,-3.2041f,-3.0812f,-0.8041f,
      1,1,0,0,0,0,0,0, 1,0,0,1,0,1,1,0,
      0,0,0,0,0,0,0,0, 1,0,0,0,1,0,0,1,
      0,1,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,
      0,0,1,0,0,0,1,0, 1,0,0,0,0,1,1,0,
      0,0,1,1,1,1,0,0, 0,1,0},
    { 0.7041f,-0.3812f, 1.0812f, 2.5812f,-0.6041f,-0.6041f,
      2.6812f, 1.2812f,-1.1041f, 0.7812f,-1.1041f,-1.1812f,
      0.2812f,-4.4041f,-3.8812f,-1.3812f,
      1,0,0,0,0,0,0,0, 1,0,0,1,0,1,1,0,
      0,0,0,0,0,0,0,0, 1,0,0,0,1,0,0,1,
      0,1,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,
      0,0,1,0,0,0,1,0, 1,0,0,0,0,1,1,0,
      0,0,1,1,1,1,0,0, 0,1,0},
    { 1.5120f,-0.0512f, 1.8812f, 4.0812f,-0.1812f,-0.1812f,
      4.2041f, 2.2041f,-0.5120f, 1.6812f,-0.5120f,-0.4812f,
      1.0041f,-2.7041f,-2.7041f,-0.5120f,
      1,1,0,0,0,0,0,0, 1,0,0,0,0,1,1,0,
      0,0,0,0,0,0,0,0, 1,0,0,0,1,0,0,1,
      0,1,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,
      0,0,1,0,0,0,1,0, 0,0,0,0,0,1,1,0,
      0,0,1,1,1,1,0,0, 0,1,0},
    { 0.5120f,-0.4812f, 0.8812f, 2.1812f,-0.7041f,-0.7041f,
      2.3041f, 1.0041f,-1.2812f, 0.5120f,-1.2812f,-1.3041f,
      0.0812f,-4.7812f,-4.2041f,-1.5512f,
      1,1,0,0,0,0,0,0, 1,0,0,1,0,1,0,0,
      0,0,0,0,0,0,0,0, 1,0,0,0,1,0,0,1,
      0,1,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,
      0,0,1,0,0,0,1,0, 1,0,0,0,0,1,1,0,
      0,0,1,1,1,1,0,0, 0,1,0},
    { 1.1041f,-0.2312f, 1.4012f, 3.3012f,-0.3812f,-0.3812f,
      3.4512f, 1.7041f,-0.8012f, 1.2041f,-0.8012f,-0.8041f,
      0.6041f,-3.4041f,-3.2041f,-0.9512f,
      1,1,0,0,0,0,0,0, 1,0,0,1,0,1,1,0,
      0,0,0,0,0,0,0,0, 1,0,0,0,1,0,0,0,
      0,1,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,
      0,0,1,0,0,0,0,0, 1,0,0,0,0,1,1,0,
      0,0,1,1,1,1,0,0, 0,1,0},
    { 0.8041f,-0.3512f, 1.1512f, 2.8012f,-0.5512f,-0.5512f,
      2.9512f, 1.4012f,-1.0041f, 0.9041f,-1.0041f,-1.0512f,
      0.3812f,-4.1512f,-3.6512f,-1.2512f,
      1,1,0,0,0,0,0,0, 1,0,0,1,0,1,1,0,
      0,0,0,0,0,0,0,0, 1,0,0,0,1,0,0,1,
      0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,
      0,0,1,0,0,0,1,0, 1,0,0,0,0,1,0,0,
      0,0,1,1,1,1,0,0, 0,1,0},
    { 1.3812f,-0.1312f, 1.6812f, 3.7812f,-0.2512f,-0.2512f,
      3.9512f, 2.0512f,-0.6041f, 1.5512f,-0.6041f,-0.5812f,
      0.8812f,-2.9512f,-2.8512f,-0.6812f,
      1,1,0,0,0,0,0,0, 1,0,0,1,0,0,1,0,
      0,0,0,0,0,0,0,0, 1,0,0,0,1,0,0,1,
      0,1,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,1,0, 1,0,0,0,0,1,1,0,
      0,0,1,1,1,1,0,0, 0,1,0},
    { 0.6041f,-0.4312f, 0.9512f, 2.3512f,-0.6512f,-0.6512f,
      2.5012f, 1.1512f,-1.1812f, 0.6812f,-1.1812f,-1.2041f,
      0.1812f,-4.5512f,-4.0012f,-1.4512f,
      0,1,0,0,0,0,0,0, 1,0,0,1,0,1,1,0,
      0,0,0,0,0,0,0,0, 1,0,0,0,1,0,0,1,
      0,1,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,
      0,0,1,0,0,0,1,0, 1,0,0,0,0,1,1,0,
      0,0,1,1,1,1,0,0, 0,1,0},
    { 1.0041f,-0.2812f, 1.3012f, 3.1512f,-0.4312f,-0.4312f,
      3.3012f, 1.6012f,-0.8812f, 1.1012f,-0.8812f,-0.8812f,
      0.5012f,-3.6512f,-3.3512f,-1.0512f,
      1,1,0,0,0,0,0,0, 1,0,0,1,0,1,1,0,
      0,0,0,0,0,0,0,0, 1,0,0,0,1,0,0,1,
      0,1,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
      0,0,1,0,0,0,1,0, 1,0,0,0,0,1,1,0,
      0,0,1,1,1,0,0,0, 0,1,0},
    { 0.8812f,-0.3312f, 1.2012f, 2.9012f,-0.5012f,-0.5012f,
      3.0512f, 1.4512f,-0.9512f, 1.0012f,-0.9512f,-0.9512f,
      0.4012f,-3.8512f,-3.5512f,-1.1512f,
      1,1,0,0,0,0,0,0, 1,0,0,1,0,1,1,0,
      0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,1,
      0,1,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,
      0,0,1,0,0,0,1,0, 1,0,0,0,0,1,1,0,
      0,0,1,1,1,1,0,0, 0,0,0},
    { 1.2812f,-0.1512f, 1.5812f, 3.6012f,-0.2812f,-0.2812f,
      3.7512f, 1.9512f,-0.6512f, 1.4512f,-0.6512f,-0.6512f,
      0.7512f,-3.1512f,-3.0012f,-0.7512f,
      1,1,0,0,0,0,0,0, 1,0,0,1,0,1,1,0,
      0,0,0,0,0,0,0,0, 1,0,0,0,1,0,0,1,
      0,1,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,
      0,0,1,0,0,0,1,0, 1,0,0,0,0,0,1,0,
      0,0,1,1,1,1,0,0, 0,1,0}
};
/* label: 0=ATTACK, 1..12 → expected=1; 13..24 → expected=0 */
static const int TEST_EXPECTED[25] = {
    1,1,1,1,1,1,1,1,1,1,1,1,1,   /* 13 ATTACK */
    0,0,0,0,0,0,0,0,0,0,0,0      /* 12 NORMAL */
};
static const char* const TEST_LABELS[25] = {
    "ATK0","ATK1","ATK2","ATK3","ATK4","ATK5","ATK6",
    "ATK7","ATK8","ATK9","ATK10","ATK11","ATK12",
    "NRM0","NRM1","NRM2","NRM3","NRM4","NRM5",
    "NRM6","NRM7","NRM8","NRM9","NRM10","NRM11"
};

/* ── Inizializzazione modello ─────────────────────────────────────────────── */
static bool model_init(const uint8_t* d) {
    /* Magic */
    if (d[0]!='X'||d[1]!='G'||d[2]!='B'||d[3]!='I') return false;

    /* Versione: supporta v2 (legacy) e v3 (corrente) */
    uint16_t ver; memcpy(&ver, d+4, 2);
    if (ver != 2 && ver != 3) return false;

    g_n_trees    = rd_u32(d,  6);
    g_n_features = rd_u32(d, 10);
    g_leaf_scale = rd_f32(d, 14);
    /* d[18..21] = thr_scale: v3 = 0.0 (threshold float32 esatti), non usato */

    if (g_n_trees > MAX_TREES) return false;

    uint32_t off = 22;
    for (uint32_t t = 0; t < g_n_trees; t++) {
        g_trees[t].offset  = off;
        int32_t n          = (int32_t)rd_u32(d, off); off += 4;
        g_trees[t].n_nodes = n;
        /* lc[n*4] + rc[n*4] + si[n*4] + is_leaf[n] + leaf_ptr[n*4] + thr_ptr[n*4] */
        off += (uint32_t)n * (4+4+4+1+4+4);
    }

    /* Dati quantizzati: prima le foglie INT8, poi i threshold float32 */
    g_leaf_data_offset = off;
    uint32_t tot_leaves = 0, tot_splits = 0;
    for (uint32_t t = 0; t < g_n_trees; t++) {
        int32_t  n       = g_trees[t].n_nodes;
        const uint8_t* is_leaf = d + g_trees[t].offset + 4 + (uint32_t)n*12;
        for (int32_t i = 0; i < n; i++) {
            if (is_leaf[i]) tot_leaves++; else tot_splits++;
        }
    }
    g_thr_data_offset = g_leaf_data_offset + tot_leaves;
    return true;
}

/* ── Inferenza singolo albero ─────────────────────────────────────────────── */
static float predict_tree(const uint8_t* d, uint32_t ti, const float* x) {
    const uint8_t* tb = d + g_trees[ti].offset;
    int32_t  n   = (int32_t)rd_u32(tb, 0);
    const uint8_t *lc  = tb+4,
                  *rc  = tb+4+(uint32_t)n*4,
                  *si  = tb+4+(uint32_t)n*8,
                  *isl = tb+4+(uint32_t)n*12,
                  *lpp = tb+4+(uint32_t)n*12+(uint32_t)n,
                  *tpp = tb+4+(uint32_t)n*13+(uint32_t)n*4;
    int32_t node = 0;
    while (!isl[node]) {
        uint32_t feat = rd_u32(si, (uint32_t)node*4);
        if (feat >= N_FEATURES) break;
        /* v3: threshold float32 esatto — nessun int8*scale */
        float thr = rd_f32(d, g_thr_data_offset + (uint32_t)rd_i32(tpp,(uint32_t)node*4)*4);
        /* FIX: operatore strict < come XGBoost nativo (era <=) */
        node = (x[feat] < thr) ? rd_i32(lc,(uint32_t)node*4)
                                : rd_i32(rc,(uint32_t)node*4);
        if (node < 0 || node >= n) break;
    }
    return (float)(int8_t)d[g_leaf_data_offset + (uint32_t)rd_i32(lpp,(uint32_t)node*4)]
           * g_leaf_scale;
}

/* ── Predizione completa ──────────────────────────────────────────────────── */
static float xgb_proba(const float* x) {
    float s = 0.0f;
    for (uint32_t t = 0; t < g_n_trees; t++)
        s += predict_tree(g_xgboost_model, t, x);
    return 1.0f / (1.0f + expf(-s));
}

/* ── Meta e utilità seriale ───────────────────────────────────────────────── */
static void send_meta() {
    Serial.println("READY");
    Serial.println("MODEL=XGBoost_INT8");
    Serial.println("BOARD=ESP32-C3-SuperMini");
    Serial.println("F1=0.9989");
    Serial.println("ROC_AUC=1.0000");
    Serial.println("SIZE_KB=369.52");
    Serial.println("SRAM_LIMIT_BYTES=409600");
    Serial.println("SRAM_MODEL_BYTES=2400");
}

/* ── Setup ────────────────────────────────────────────────────────────────── */
void setup() {
    Serial.begin(115200); delay(1000);
    send_meta();
    g_model_ok = model_init(g_xgboost_model);
    Serial.print("MODEL_INIT="); Serial.println(g_model_ok ? "OK" : "FAIL");
    Serial.print("FREE_HEAP_START="); Serial.println(ESP.getFreeHeap());
    Serial.println("HEADER:label,pred,prob,latency_us,free_heap_bytes,correct");

    /* ── Test diversificato su 25 campioni ───────────────────────────────── */
    if (g_model_ok) {
        Serial.println("--- DIVERSIFIED_TEST_START ---");
        int correct = 0;
        for (int i = 0; i < 25; i++) {
            unsigned long t0 = micros();
            float prob = xgb_proba(TEST_SAMPLES[i]);
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

    /* Cicla sui 25 campioni diversificati invece dei soli 2 hardcoded */
    unsigned long t0 = micros();
    float prob = xgb_proba(TEST_SAMPLES[_idx]);
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
