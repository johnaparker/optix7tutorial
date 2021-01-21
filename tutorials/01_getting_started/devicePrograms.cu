#include <optix_device.h>
#include "launchParams.h"


using namespace optix7tutorial;

namespace optix7tutorial {
  /*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
  extern "C" __constant__ LaunchParams optixLaunchParams;
  
  extern "C" __global__ void __closesthit__radiance() {}
  extern "C" __global__ void __anyhit__radiance() {}
  extern "C" __global__ void __miss__radiance() {}

  extern "C" __global__ void __raygen__renderFrame() {
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    int ymax = optixLaunchParams.fbSize.y;

    float t = (float)iy/ymax;
    const float3 color = (1-t)*make_float3(1.0,1.0,1.0) + t*make_float3(.3,.5,1);
    const int3 rgb = make_int3(color.x*255, color.y*255, color.z*255);

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    const uint32_t rgba = 0xff000000
      | (rgb.x<<0) | (rgb.y<<8) | (rgb.z<<16);

    // and write to frame buffer ...
    const uint32_t fbIndex = ix+iy*optixLaunchParams.fbSize.x;
    optixLaunchParams.colorBuffer[fbIndex] = rgba;
  }
}
