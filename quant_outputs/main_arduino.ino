/*
 * =========================================================================
 * IDS Embedded — Arduino Mega 2560
 * Modelli: Logistic Regression (logreg.c) e Decision Tree (tree_depth5.c)
 * =========================================================================
 * Dataset:  TON_IoT  |  Features: 95 (preprocessate)
 *
 * IMPORTANTE — differenza API tra i due modelli:
 *   logreg.c:     double score(double* input)         → logit grezzo
 *                 int    predict(double* input)        → 0/1
 *   tree_depth5.c: void  score(double* input, double* output[2])
 *                 int    predict(double* input)        → 0/1
 *
 *   Entrambi espongono predict(input) con la stessa firma → usare quella.
 *
 * Per scegliere il modello: decommentare UNO solo dei due #define.
 * =========================================================================
 */

/* ── Scegliere il modello ─────────────────────────────────── */
#define USE_LOGISTIC_REGRESSION
/* #define USE_DECISION_TREE */

/* ── Costanti ─────────────────────────────────────────────── */
#define N_FEATURES  95
#define LED_ATTACK  13    /* LED built-in Mega → attack */
#define LED_NORMAL   2    /* LED verde opzionale → normal */

/* ── Prototipo unificato — stesso in entrambi i file .c ───── */
extern int predict(double * input);

#ifdef USE_LOGISTIC_REGRESSION
  extern double score(double * input);       /* logit grezzo */
  extern double score_proba(double * input); /* sigmoid [0,1] */
  #define MODEL_NAME  "Logistic Regression"
  #define MODEL_F1    0.9900
  #define MODEL_ROC   0.9945
  #define MODEL_KB    3.20
#else
  /* Per DT score() ha firma diversa: non esposta direttamente qui */
  #define MODEL_NAME  "Decision Tree (depth=5)"
  #define MODEL_F1    0.9943
  #define MODEL_ROC   0.9856
  #define MODEL_KB    4.76
#endif

/* ── Feature di test (campione TON_IoT preprocessato) ────────
 * Ordine: [0..15] numeriche StandardScaler, [16..94] OHE
 * Tutti i valori sono post-trasformazione DataFramePreprocessor.
 */
double sample_attack[N_FEATURES] = {
  /* numeriche [0..15] — valori tipici traffico malevolo */
   1.23, -0.45,  2.10,  3.85,  0.39, -0.10,
   1.47,  2.92,  0.38,  1.93,  0.04, -0.02,
   0.08,  0.04, -0.03,  0.02,
  /* OHE [16..94] — proto=tcp, vari flag */
   0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
};

double sample_normal[N_FEATURES] = {
  /* numeriche [0..15] — valori tipici traffico normale */
  -0.12, -0.45,  0.82, -0.91, -0.38,  0.00,
  -0.75, -0.82, -0.71, -0.63, -0.02,  0.01,
  -0.04, -0.02,  0.01, -0.01,
  /* OHE — proto=udp, conn_state=SF */
   0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
};

/* ─────────────────────────────────────────────────────────── */

void setup() {
  Serial.begin(115200);
  while (!Serial) {}
  pinMode(LED_ATTACK, OUTPUT);
  pinMode(LED_NORMAL, OUTPUT);

  Serial.println(F("==========================================="));
  Serial.print(F("IDS Embedded — "));
  Serial.println(F(MODEL_NAME));
  Serial.print(F("F1="));  Serial.print(MODEL_F1, 4);
  Serial.print(F("  ROC-AUC="));  Serial.println(MODEL_ROC, 4);
  Serial.print(F("Dimensione modello: "));
  Serial.print(MODEL_KB);  Serial.println(F(" KB  (limite: 8 KB SRAM)"));
  Serial.println(F("==========================================="));
}

void classify(double* features, const char* label, int expected) {
  unsigned long t0 = micros();
  int pred = predict(features);
  unsigned long dt = micros() - t0;

  /* Opzionale per LR: stampa anche la probabilità */
#ifdef USE_LOGISTIC_REGRESSION
  double prob = score_proba(features);
  Serial.print(F("["));  Serial.print(label);
  Serial.print(F("] prob="));  Serial.print(prob, 4);
#else
  Serial.print(F("["));  Serial.print(label);
  Serial.print(F("] pred="));
#endif

  Serial.print(pred == 1 ? F("  ATTACK") : F("  normal"));
  Serial.print(pred == expected ? F("  OK") : F("  ERRORE"));
  Serial.print(F("  latenza="));  Serial.print(dt);  Serial.println(F(" us"));

  digitalWrite(LED_ATTACK, pred == 1 ? HIGH : LOW);
  digitalWrite(LED_NORMAL,  pred == 0 ? HIGH : LOW);
}

void loop() {
  Serial.println(F("--- Classificazione campioni ---"));
  classify(sample_attack, "ATTACK", 1);
  classify(sample_normal, "NORMAL", 0);
  Serial.println();
  delay(3000);
}
