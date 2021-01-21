#pragma once

#include <sutil/sutil.h>
#include <sutil/vec_math.h>

namespace optix7tutorial {
  struct LaunchParams {
    struct {
      uint32_t *colorBuffer;
      int2     size;
    } frame;
    
    struct {
      float3 position;
      float3 direction;
      float3 horizontal;
      float3 vertical;
    } camera;

    OptixTraversableHandle traversable;
  };
}
