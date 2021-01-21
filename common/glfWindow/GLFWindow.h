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

#pragma once

#include <string>
#include <assert.h>
#include <sutil/sutil.h>

// glfw framework
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace optix7tutorial {


  struct GLFWindow {
    GLFWindow(const std::string &title, const int2 size=make_int2(1200,800));
    ~GLFWindow();

    /*! put pixels on the screen ... */
    virtual void draw()
    { /* empty - to be subclassed by user */ }

    /*! callback that window got resized */
    virtual void resize(const int2 &newSize)
    { /* empty - to be subclassed by user */ }

    virtual void key(int key, int mods)
    {}
    
    /*! callback that window got resized */
    virtual void mouseMotion(const int2 &newPos)
    {}
    
    /*! callback that window got resized */
    virtual void mouseButton(int button, int action, int mods)
    {}

    inline int2 getMousePos() const
    {
      double x,y;
      glfwGetCursorPos(handle,&x,&y);
      return make_int2((int)x, (int)y);
    }
    
    /*! re-render the frame - typically part of draw(), but we keep
      this a separate function so render() can focus on optix
      rendering, and now have to deal with opengl pixel copies
      etc */
    virtual void render() 
    { /* empty - to be subclassed by user */ }

    /*! opens the actual window, and runs the window's events to
      completion. This function will only return once the window
      gets closed */
    void run();

    /*! the glfw window handle */
    GLFWwindow *handle { nullptr };
  };

}
