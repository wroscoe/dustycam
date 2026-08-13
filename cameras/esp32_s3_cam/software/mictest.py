# Mic test for Waveshare ESP32-S3-CAM: ES7210 dual-mic ADC -> I2S -> RMS levels
# Pins (Waveshare 06_esp_sr.ino, ESP32 perspective): I2C SDA=8 SCL=7; I2S MCLK=10 SCLK=11 WS=12 DIN=13
import struct
import math
import time
from machine import Pin, I2C, I2S

i2c = I2C(1, sda=Pin(8), scl=Pin(7), freq=100000)
print("I2C scan:", [hex(a) for a in i2c.scan()])

ADDR = 0x40  # ES7210, AD1/AD0=00

def wr(reg, val):
    i2c.writeto_mem(ADDR, reg, bytes([val]))

def rd(reg):
    return i2c.readfrom_mem(ADDR, reg, 1)[0]

def upd(reg, mask, val):
    wr(reg, (rd(reg) & (0xFF ^ mask)) | (val & mask))

def es7210_init(gain=10):  # gain index: 8=24dB, 10=30dB
    # reset + core setup (esp-adf es7210_adc_init sequence)
    wr(0x00, 0xFF)
    wr(0x00, 0x41)
    wr(0x01, 0x3F)
    wr(0x09, 0x30)
    wr(0x0A, 0x30)
    wr(0x23, 0x2A)
    wr(0x22, 0x0A)
    wr(0x20, 0x0A)
    wr(0x21, 0x2A)
    upd(0x08, 0x01, 0x00)        # I2S slave
    wr(0x40, 0x43)               # analog power
    wr(0x41, 0x70)               # mic12 bias 2.87V
    wr(0x42, 0x70)               # mic34 bias 2.87V
    wr(0x07, 0x20)
    wr(0x02, 0xC1)
    # sample config: 16kHz with MCLK=4.096MHz (256fs): reg02=0xC1 osr=0x20 lrck=0x0100
    wr(0x02, 0xC1)
    wr(0x07, 0x20)
    wr(0x04, 0x01)
    wr(0x05, 0x00)
    # SDP: 16-bit, I2S normal
    v = (rd(0x11) & 0x1F) | 0x60
    wr(0x11, v)
    wr(0x11, rd(0x11) & 0xFC)
    # mic select: MIC1 + MIC2
    for i in range(4):
        upd(0x43 + i, 0x10, 0x00)
    wr(0x4B, 0xFF)
    wr(0x4C, 0xFF)
    for greg in (0x43, 0x44):
        upd(0x01, 0x0B, 0x00)
        wr(0x4B, 0x00)
        upd(greg, 0x10, 0x10)
        upd(greg, 0x0F, gain)
    wr(0x12, 0x00)               # no TDM (2 mics)
    # start
    wr(0x06, 0x00)
    wr(0x40, 0x43)
    for preg in (0x47, 0x48, 0x49, 0x4A):
        wr(preg, 0x08)

es7210_init(gain=10)
print("ES7210 chip id regs:", hex(rd(0x3D)) if True else "")
print("ES7210 initialized, starting I2S capture...")

RATE = 16000
# this build's I2S lacks mck output; generate ~4.096MHz MCLK (256*fs) with LEDC PWM
from machine import PWM
mclk = PWM(Pin(10), freq=RATE * 256)
mclk.duty_u16(32768)
print("MCLK via PWM:", mclk.freq(), "Hz")
audio = I2S(0, sck=Pin(11), ws=Pin(12), sd=Pin(13),
            mode=I2S.RX, bits=16, format=I2S.STEREO, rate=RATE, ibuf=32000)

CHUNK_FRAMES = 1600            # 100 ms
buf = bytearray(CHUNK_FRAMES * 4)
fmt = "<%dh" % (CHUNK_FRAMES * 2)

print("t_s  micL_rms  micR_rms  micL_peak  micR_peak")
t0 = time.ticks_ms()
# aggregate per ~0.5s (5 chunks)
while time.ticks_diff(time.ticks_ms(), t0) < 45000:
    sl = sr = 0
    pl = pr = 0
    n = 0
    for _ in range(5):
        audio.readinto(buf)
        s = struct.unpack(fmt, buf)
        for i in range(0, len(s), 2):
            v = s[i]
            w = s[i + 1]
            sl += v * v
            sr += w * w
            if v > pl:
                pl = v
            if w > pr:
                pr = w
        n += len(s) // 2
    print("%5.1f  %8.1f  %8.1f  %6d  %6d" % (
        time.ticks_diff(time.ticks_ms(), t0) / 1000,
        math.sqrt(sl / n), math.sqrt(sr / n), pl, pr))
audio.deinit()
print("done")
