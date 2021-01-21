// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "scene.h"
#include "glfWindow/CameraWindow.h"

namespace optix7tutorial {
  extern "C" int main(int ac, char **av)
  {
    try {
      Camera camera = { /*from*/make_float3(-10.f,2.f,-12.f),
                        /* at */make_float3(0.f,0.f,0.f),
                        /* up */make_float3(0.f,1.f,0.f) };

      // something approximating the scale of the world, so the
      // camera knows how much to move for any given user interaction:
      const float worldScale = 10.f;

      CameraWindow window("Optix 7 Tutorial",
                          camera,worldScale, make_int2(800,600));
      window.run();
      
    } catch (std::runtime_error& e) {
      std::cout << "FATAL ERROR: " << e.what() << std::endl;
      exit(1);
    }
    return 0;
  }
}
}
