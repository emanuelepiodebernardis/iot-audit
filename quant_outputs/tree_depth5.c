#include <string.h>
void score(double * input, double * output) {
    double var0[2];
    if (input[17] <= 0.5) {
        if (input[46] <= 0.5) {
            if (input[2] <= -0.0005463569250423461) {
                if (input[4] <= -0.01473047723993659) {
                    if (input[0] <= -0.311388298869133) {
                        memcpy(var0, (double[]){0.010526315789473684, 0.9894736842105263}, 2 * sizeof(double));
                    } else {
                        memcpy(var0, (double[]){0.0, 1.0}, 2 * sizeof(double));
                    }
                } else {
                    memcpy(var0, (double[]){1.0, 0.0}, 2 * sizeof(double));
                }
            } else {
                if (input[12] <= 3.7529101371765137) {
                    memcpy(var0, (double[]){1.0, 0.0}, 2 * sizeof(double));
                } else {
                    memcpy(var0, (double[]){0.0, 1.0}, 2 * sizeof(double));
                }
            }
        } else {
            if (input[44] <= 0.5) {
                if (input[2] <= -0.013936611358076334) {
                    if (input[0] <= 1.1380477547645569) {
                        memcpy(var0, (double[]){0.024290112897707834, 0.9757098871022921}, 2 * sizeof(double));
                    } else {
                        memcpy(var0, (double[]){1.0, 0.0}, 2 * sizeof(double));
                    }
                } else {
                    if (input[8] <= -0.007050476968288422) {
                        memcpy(var0, (double[]){0.973170731707317, 0.026829268292682926}, 2 * sizeof(double));
                    } else {
                        memcpy(var0, (double[]){0.05759162303664921, 0.9424083769633508}, 2 * sizeof(double));
                    }
                }
            } else {
                if (input[1] <= -0.30616264045238495) {
                    if (input[0] <= -2.1080856323242188) {
                        memcpy(var0, (double[]){1.0, 0.0}, 2 * sizeof(double));
                    } else {
                        memcpy(var0, (double[]){0.06818181818181818, 0.9318181818181818}, 2 * sizeof(double));
                    }
                } else {
                    if (input[2] <= -0.014080236665904522) {
                        memcpy(var0, (double[]){0.9496996996996997, 0.0503003003003003}, 2 * sizeof(double));
                    } else {
                        memcpy(var0, (double[]){0.9949427338985571, 0.0050572661014428085}, 2 * sizeof(double));
                    }
                }
            }
        }
    } else {
        if (input[6] <= 0.9597744941711426) {
            if (input[53] <= 0.5) {
                if (input[54] <= 0.5) {
                    memcpy(var0, (double[]){1.0, 0.0}, 2 * sizeof(double));
                } else {
                    memcpy(var0, (double[]){0.0, 1.0}, 2 * sizeof(double));
                }
            } else {
                if (input[15] <= 11.247903928160667) {
                    if (input[83] <= 0.5) {
                        memcpy(var0, (double[]){0.9826086956521739, 0.017391304347826087}, 2 * sizeof(double));
                    } else {
                        memcpy(var0, (double[]){0.007467195743337692, 0.9925328042566623}, 2 * sizeof(double));
                    }
                } else {
                    if (input[80] <= 0.5) {
                        memcpy(var0, (double[]){0.9757575757575757, 0.024242424242424242}, 2 * sizeof(double));
                    } else {
                        memcpy(var0, (double[]){0.0, 1.0}, 2 * sizeof(double));
                    }
                }
            }
        } else {
            if (input[6] <= 1.1225665807724) {
                if (input[28] <= 0.5) {
                    if (input[8] <= 0.14442132413387299) {
                        memcpy(var0, (double[]){0.9998742928975487, 0.0001257071024512885}, 2 * sizeof(double));
                    } else {
                        memcpy(var0, (double[]){0.0, 1.0}, 2 * sizeof(double));
                    }
                } else {
                    memcpy(var0, (double[]){0.0, 1.0}, 2 * sizeof(double));
                }
            } else {
                if (input[4] <= 0.03080033604055643) {
                    if (input[38] <= 0.5) {
                        memcpy(var0, (double[]){0.016286644951140065, 0.9837133550488599}, 2 * sizeof(double));
                    } else {
                        memcpy(var0, (double[]){1.0, 0.0}, 2 * sizeof(double));
                    }
                } else {
                    if (input[1] <= 0.14698276668787003) {
                        memcpy(var0, (double[]){1.0, 0.0}, 2 * sizeof(double));
                    } else {
                        memcpy(var0, (double[]){0.0, 1.0}, 2 * sizeof(double));
                    }
                }
            }
        }
    }
    memcpy(output, var0, 2 * sizeof(double));
}
