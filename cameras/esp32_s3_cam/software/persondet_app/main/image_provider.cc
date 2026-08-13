/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "string.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#if (CONFIG_TFLITE_USE_BSP)
#include "bsp/esp-bsp.h"
#endif

#include "esp_heap_caps.h"
#include "esp_log.h"

#include "app_camera_esp.h"
#include "esp_camera.h"
#include "model_settings.h"
#include "image_provider.h"
#include "esp_main.h"

static const char* TAG = "app_camera";
static uint16_t* display_buf;

// Get the camera module ready
TfLiteStatus InitCamera() {
#if CLI_ONLY_INFERENCE
  ESP_LOGI(TAG, "CLI_ONLY_INFERENCE enabled, skipping camera init");
  return kTfLiteOk;
#endif
// if display support is present, initialise display buf
#if DISPLAY_SUPPORT
  if (display_buf == NULL) {
    // Size of display_buf:
    // Frame 96x96 from camera is extrapolated to 192x192. RGB565 pixel format -> 2 bytes per pixel
    display_buf = (uint16_t *) heap_caps_malloc(96 * 2 * 96 * 2 * sizeof(uint16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  }
  if (display_buf == NULL) {
    ESP_LOGE(TAG, "Couldn't allocate display buffer");
    return kTfLiteError;
  }
#endif // DISPLAY_SUPPORT

#if ESP_CAMERA_SUPPORTED
  int ret = app_camera_init();
  if (ret != 0) {
    MicroPrintf("Camera init failed\n");
    return kTfLiteError;
  }
  MicroPrintf("Camera Initialized\n");
#else
  ESP_LOGE(TAG, "Camera not supported for this device");
#endif
  return kTfLiteOk;
}

void *image_provider_get_display_buf()
{
  return (void *) display_buf;
}

// Get an image from the camera module
TfLiteStatus GetImage(int image_width, int image_height, int channels, int8_t* image_data) {
#if ESP_CAMERA_SUPPORTED
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    ESP_LOGE(TAG, "Camera capture failed");
    return kTfLiteError;
  }

#if DISPLAY_SUPPORT
  // In case if display support is enabled, we initialise camera in rgb mode
  // Hence, we need to convert this data to grayscale to send it to tf model
  // For display we extra-polate the data to 192X192

  // point to the last quarter of buffer
  uint16_t* cam_buf = display_buf + (96 * 96 * 3);
  memcpy((uint8_t*)cam_buf, fb->buf, fb->len);
  esp_camera_fb_return(fb);

  for (int i = 0; i < kNumRows; i++) {
    for (int j = 0; j < kNumCols; j++) {
      uint16_t inference_pixel = cam_buf[i * kNumCols + j];

      // for inference
      uint8_t hb = inference_pixel & 0xFF;
      uint8_t lb = inference_pixel >> 8;
      uint8_t r = (lb & 0x1F) << 3;
      uint8_t g = ((hb & 0x07) << 5) | ((lb & 0xE0) >> 3);
      uint8_t b = (hb & 0xF8);

      /**
       * Gamma corected rgb to greyscale formula: Y = 0.299R + 0.587G + 0.114B
       * for effiency we use some tricks on this + quantize to [-128, 127]
       */
      int8_t grey_pixel = ((305 * r + 600 * g + 119 * b) >> 10) - 128;

      image_data[i * kNumCols + j] = grey_pixel;
    }
  }

  // for display
  lv_draw_sw_rgb565_swap(cam_buf, 96 * 96);
  for (int i = 0; i < kNumRows; i++) {
    for (int j = 0; j < kNumCols; j++) {
      uint16_t pixel = cam_buf[i * kNumCols + j];
      display_buf[2 * i * kNumCols * 2 + 2 * j] = pixel;
      display_buf[2 * i * kNumCols * 2 + 2 * j + 1] = pixel;
      display_buf[(2 * i + 1) * kNumCols * 2 + 2 * j] = pixel;
      display_buf[(2 * i + 1) * kNumCols * 2 + 2 * j + 1] = pixel;
    }
  }
#else // DISPLAY_SUPPORT
  MicroPrintf("Image Captured\n");
  // GC0308 only produces real frames at QVGA/VGA (96x96 and QQVGA modes
  // output noise). Capture QVGA (320x240), center-crop 192x192, 2x2
  // average down to the model's 96x96, quantize to int8.
  {
    // RGB565 QVGA (proven working on GC0308) -> luma -> center-crop
    // 192x192 -> 2x2 average -> 96x96 int8.
    const uint8_t *src = (const uint8_t *) fb->buf;
    const int stride = 320 * 2;
    const int ox = (320 - 192) / 2, oy = (240 - 192) / 2;
    for (int y = 0; y < image_height; y++) {
      const uint8_t *r0 = src + (oy + 2 * y) * stride + ox * 2;
      const uint8_t *r1 = r0 + stride;
      for (int x = 0; x < image_width; x++) {
        int sum = 0;
        const uint8_t *px[4] = { r0 + 4 * x, r0 + 4 * x + 2,
                                 r1 + 4 * x, r1 + 4 * x + 2 };
        for (int k = 0; k < 4; k++) {
          uint16_t v = ((uint16_t)px[k][0] << 8) | px[k][1];  /* big-endian */
          int r = (v >> 11) & 0x1F, g = (v >> 5) & 0x3F, b = v & 0x1F;
          sum += (299 * ((r * 255) / 31) + 587 * ((g * 255) / 63)
                  + 114 * ((b * 255) / 31)) / 1000;
        }
        image_data[y * image_width + x] = (uint8_t)(sum >> 2) ^ 0x80;
      }
    }
  }

  esp_camera_fb_return(fb);
#endif // DISPLAY_SUPPORT
  /* here the esp camera can give you grayscale image directly */
  return kTfLiteOk;
#else
  return kTfLiteError;
#endif
}
