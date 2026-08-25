/* On-device animal gate (TFLite Micro). Fail-open: if init fails the
 * caller must treat every motion frame as transmit-worthy. */
#ifndef CAMLOGGER_GATE_H_
#define CAMLOGGER_GATE_H_

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GATE_IMG 96   /* model input: GATE_IMG x GATE_IMG x 3, int8 */

bool gate_init(void);
/* img: 96*96*3 int8 (uint8 pixel ^ 0x80). Returns animal probability
 * 0..1, or -1.0 on inference failure. */
float gate_score(const int8_t *img);

#ifdef __cplusplus
}
#endif

#endif
