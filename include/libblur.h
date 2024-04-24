#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

enum class FastBlurChannels {
  Channels3,
  Channels4,
};

template<typename T = void>
struct Vec;

extern "C" {

void box_blur(const Vec<uint8_t> *src,
              uint32_t src_stride,
              Vec<uint8_t> *dst,
              uint32_t dst_stride,
              uint32_t width,
              uint32_t height,
              uint32_t radius,
              FastBlurChannels channels);

void box_blur_u16(const Vec<uint16_t> *src,
                  uint32_t src_stride,
                  Vec<uint16_t> *dst,
                  uint32_t dst_stride,
                  uint32_t width,
                  uint32_t height,
                  uint32_t radius,
                  FastBlurChannels channels);

void tent_blur(const Vec<uint8_t> *src,
               uint32_t src_stride,
               Vec<uint8_t> *dst,
               uint32_t dst_stride,
               uint32_t width,
               uint32_t height,
               uint32_t radius,
               FastBlurChannels channels);

void tent_blur_u16(const Vec<uint16_t> *src,
                   uint32_t src_stride,
                   Vec<uint16_t> *dst,
                   uint32_t dst_stride,
                   uint32_t width,
                   uint32_t height,
                   uint32_t radius,
                   FastBlurChannels channels);

void gaussian_box_blur(const Vec<uint8_t> *src,
                       uint32_t src_stride,
                       Vec<uint8_t> *dst,
                       uint32_t dst_stride,
                       uint32_t width,
                       uint32_t height,
                       uint32_t radius,
                       FastBlurChannels channels);

void gaussian_box_blur_u16(const Vec<uint16_t> *src,
                           uint32_t src_stride,
                           Vec<uint16_t> *dst,
                           uint32_t dst_stride,
                           uint32_t width,
                           uint32_t height,
                           uint32_t radius,
                           FastBlurChannels channels);

void fast_gaussian(Vec<uint8_t> *bytes,
                   uint32_t stride,
                   uint32_t width,
                   uint32_t height,
                   uint32_t radius,
                   FastBlurChannels channels);

void fast_gaussian_u16(Vec<uint16_t> *bytes,
                       uint32_t stride,
                       uint32_t width,
                       uint32_t height,
                       uint32_t radius,
                       FastBlurChannels channels);

void gaussian_blur(const Vec<uint8_t> *src,
                   uint32_t src_stride,
                   Vec<uint8_t> *dst,
                   uint32_t dst_stride,
                   uint32_t width,
                   uint32_t height,
                   uint32_t kernel_size,
                   float sigma,
                   FastBlurChannels channels);

void gaussian_blur_u16(const Vec<uint16_t> *src,
                       uint32_t src_stride,
                       Vec<uint16_t> *dst,
                       uint32_t dst_stride,
                       uint32_t width,
                       uint32_t height,
                       uint32_t kernel_size,
                       float sigma,
                       FastBlurChannels channels);

void median_blur(const Vec<uint8_t> *src,
                 uint32_t src_stride,
                 Vec<uint8_t> *dst,
                 uint32_t dst_stride,
                 uint32_t width,
                 uint32_t height,
                 uint32_t radius,
                 FastBlurChannels median_channels);

void fast_gaussian_next(Vec<uint8_t> *bytes,
                        uint32_t stride,
                        uint32_t width,
                        uint32_t height,
                        uint32_t radius,
                        FastBlurChannels channels);

void fast_gaussian_next_u16(Vec<uint16_t> *bytes,
                            uint32_t stride,
                            uint32_t width,
                            uint32_t height,
                            uint32_t radius,
                            FastBlurChannels channels);

} // extern "C"
