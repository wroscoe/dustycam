/* Sample uploader: ships (96x96 frame, person scores) pairs to the LAN
 * collection server so images can be matched with predictions later. */
#ifndef PERSON_DETECTION_UPLOADER_H_
#define PERSON_DETECTION_UPLOADER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void uploader_init(void);
/* img: int8 quantized tensor input (converted back to uint8 internally). */
void uploader_submit(const int8_t *img, float person_score,
                     float no_person_score);

#ifdef __cplusplus
}
#endif

#endif
