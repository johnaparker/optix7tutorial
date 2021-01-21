#pragma once

#include <sutil/sutil.h>
#include <sutil/vec_math.h>

namespace optix7tutorial {
  struct LaunchParams
  {
    uint32_t *colorBuffer;
    int2      fbSize;
  };

}
