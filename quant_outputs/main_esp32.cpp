/*
 * =========================================================================
 * IDS Embedded — ESP32-C3 con TFLite Micro
 * Modello: MLP (64-32-1) quantizzato INT8
 * =========================================================================
 * Dataset:  TON_IoT  |  Features: 95  |  F1=0.9959  |  ROC-AUC=0.9993
 *
 * Dimensione float32 originale: 121.69 KB
 * Dimensione TFLite INT8:        13.03 KB  →  compressione 9.34x
 * SRAM stimata runtime:          ~31 KB (tensor arena)
 * Limite ESP32-C3:               400 KB   →  RIENTRA ✓
 *
 * File richiesti nel progetto:
 *   mlp.h      — array C con i 13344 byte del modello INT8
 *   mlp.tflite — file binario (per riferimento / rigenerazione header)
 *
 * Wokwi: board ESP32-C3 DevKit, libreria TFLite Micro for ESP32
 * =========================================================================
 */

#include <Arduino.h>
#include "mlp.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

/* ── Configurazione ─────────────────────────────────────────── */
#define N_FEATURES      95
#define TENSOR_ARENA_KB 48          /* 48 KB arena TFLite Micro */
#define LED_ATTACK       8          /* GPIO8 — rosso */
#define LED_NORMAL       9          /* GPIO9 — verde */

/* Parametri quantizzazione calibrati durante la conversione INT8 */
#define INPUT_SCALE      0.0078740f
#define INPUT_ZP        (-1)
#define OUTPUT_SCALE     0.00390625f
#define OUTPUT_ZP      (-128)

/* ── Allocazioni statiche (evitano stack overflow su ESP32) ─── */
static uint8_t tensor_arena[TENSOR_ARENA_KB * 1024];

using MlpResolver = tflite::MicroMutableOpResolver<4>;
static MlpResolver resolver;

static const tflite::Model*     tfl_model    = nullptr;
static tflite::MicroInterpreter* interpreter  = nullptr;
static TfLiteTensor*             input_tensor = nullptr;
static TfLiteTensor*             output_tensor= nullptr;

/* ── Feature di test (stesso campione di main_arduino.ino) ─── */
float sample_attack[N_FEATURES] = {
   1.23f, -0.45f,  2.10f,  3.85f,  0.39f, -0.10f,
   1.47f,  2.92f,  0.38f,  1.93f,  0.04f, -0.02f,
   0.08f,  0.04f, -0.03f,  0.02f,
   0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
   0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
};

float sample_normal[N_FEATURES] = {
  -0.12f, -0.45f,  0.82f, -0.91f, -0.38f,  0.00f,
  -0.75f, -0.82f, -0.71f, -0.63f, -0.02f,  0.01f,
  -0.04f, -0.02f,  0.01f, -0.01f,
   0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
};

/* ─────────────────────────────────────────────────────────── */

/*
 * quantize_input() — converte float32 → int8 con i parametri calibrati.
 * Formula: q = clamp(round(x / scale) + zero_point, -128, 127)
 */
static int8_t quantize(float x) {
  int q = (int)roundf(x / INPUT_SCALE) + INPUT_ZP;
  if (q < -128) q = -128;
  if (q >  127) q =  127;
  return (int8_t)q;
}

/*
 * dequantize_output() — converte int8 output → probabilità float32.
 * Formula: p = (q - zero_point) * scale  →  mappato in [0, 1]
 */
static float dequantize(int8_t q) {
  return (float)(q - OUTPUT_ZP) * OUTPUT_SCALE;
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  pinMode(LED_ATTACK, OUTPUT);
  pinMode(LED_NORMAL, OUTPUT);

  Serial.println("===========================================");
  Serial.println("IDS Embedded — ESP32-C3 + TFLite Micro");
  Serial.println("Modello: MLP INT8 (F1=0.9959, ROC-AUC=0.9993)");
  Serial.println("121.69 KB float32  →  13.03 KB INT8  (9.34x)");
  Serial.println("===========================================");

  /* Carica il modello dall'array in mlp.h */
  tfl_model = tflite::GetModel(g_mlp_model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERRORE: versione schema TFLite incompatibile!");
    while (true) {}
  }

  /* Registra solo le operazioni usate dal MLP */
  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddLogistic();   /* sigmoid dell'output */
  resolver.AddQuantize();

  /* Crea e alloca l'interprete */
  static tflite::MicroInterpreter static_interpreter(
    tfl_model, resolver, tensor_arena, sizeof(tensor_arena));
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERRORE: AllocateTensors() fallito!");
    while (true) {}
  }

  input_tensor  = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.print("Arena usata: ");
  Serial.print(interpreter->arena_used_bytes());
  Serial.println(" byte");

  /* Verifica tipo input (deve essere int8) */
  if (input_tensor->type != kTfLiteInt8) {
    Serial.println("AVVISO: input tensor non è int8 — modello non fully quantizzato");
  }

  Serial.println("Modello caricato. Avvio classificazione...");
}

void classify(float* features, const char* label, int expected) {
  /* 1. Quantizza input float32 → int8 */
  for (int i = 0; i < N_FEATURES; i++) {
    input_tensor->data.int8[i] = quantize(features[i]);
  }

  /* 2. Inferenza */
  unsigned long t0 = micros();
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERRORE: Invoke() fallito!");
    return;
  }
  unsigned long dt = micros() - t0;

  /* 3. Dequantizza output int8 → probabilità */
  int8_t raw    = output_tensor->data.int8[0];
  float  prob   = dequantize(raw);
  int    pred   = (raw > OUTPUT_ZP) ? 1 : 0;  /* soglia su int8: > zero_point */

  Serial.print("["); Serial.print(label); Serial.print("]");
  Serial.print(" prob="); Serial.print(prob, 4);
  Serial.print(pred == 1 ? "  ATTACK" : "  normal");
  Serial.print(pred == expected ? "  OK" : "  ERRORE");
  Serial.print("  latenza="); Serial.print(dt); Serial.println(" us");

  digitalWrite(LED_ATTACK, pred == 1 ? HIGH : LOW);
  digitalWrite(LED_NORMAL,  pred == 0 ? HIGH : LOW);
}

void loop() {
  Serial.println("--- Classificazione campioni ---");
  classify(sample_attack, "ATTACK", 1);
  classify(sample_normal, "NORMAL", 0);
  Serial.println();
  delay(3000);
}
