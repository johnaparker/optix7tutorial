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

// glfw framework
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <sutil/sutil.h>
#include <sutil/Matrix.h>
#include <sutil/vec_math.h>
#include "GLFWindow.h"
#include <memory>

namespace optix7tutorial {
  struct CameraFrame {
    CameraFrame(const float worldScale): motionSpeed(worldScale) {}
    
    float3 getPOI() const { 
      return position - poiDistance * frame.getCol(2);
    }
      
    /*! re-compute all orientation related fields from given
      'user-style' camera parameters */
    void setOrientation(/* camera origin    : */const float3 &origin,
                        /* point of interest: */const float3 &interest,
                        /* up-vector        : */const float3 &up) {
      position = origin;
      upVector = up;
      //frame.setCol(2, (interest==origin) ? make_float3(0,0,1)
        //: [> negative because we use NEGATIZE z axis <] - normalize(interest - origin));
      frame.setCol(2, -normalize(interest - origin));
      frame.setCol(0, cross(up,frame.getCol(2)));
      if (dot(frame.getCol(0),frame.getCol(0)) < 1e-8f)
        frame.setCol(0, make_float3(0,1,0));
      else
        frame.setCol(0, normalize(frame.getCol(0)));
      // frame.vx
      //   = (fabs(dot(up,frame.vz)) < 1e-6f)
      //   ? vec3f(0,1,0)
      //   : normalize(cross(up,frame.vz));
      frame.setCol(1, normalize(cross(frame.getCol(2),frame.getCol(0))));
      poiDistance = length(interest-origin);
      forceUpFrame();
    }
      
    /*! tilt the frame around the z axis such that the y axis is "facing upwards" */
    void forceUpFrame() {
      // frame.vz remains unchanged
      if (fabsf(dot(frame.getCol(2),upVector)) < 1e-6f)
        // looking along upvector; not much we can do here ...
        return;
      frame.setCol(0, normalize(cross(upVector,frame.getCol(2))));
      frame.setCol(1, normalize(cross(frame.getCol(2),frame.getCol(0))));
      modified = true;
    }

    void setUpVector(const float3 &up) {
        upVector = up; forceUpFrame();
    }

    inline float computeStableEpsilon(float f) const {
      return abs(f) * float(1./(1<<21));
    }
                               
    inline float computeStableEpsilon(const float3 v) const {
      return fmax(fmax(computeStableEpsilon(v.x),
                     computeStableEpsilon(v.y)),
                 computeStableEpsilon(v.z));
    }

    inline float3 get_from() const { return position; }
    inline float3 get_at() const { return getPOI(); }
    inline float3 get_up() const { return upVector; }
      
    sutil::Matrix3x3      frame         = sutil::Matrix3x3::identity();
    float3         position      { 0,-1,0 };
    /*! distance to the 'point of interst' (poi); e.g., the point we
      will rotate around */
    float         poiDistance   { 1.f };
    float3         upVector      { 0,1,0 };
    /* if set to true, any change to the frame will always use to
       upVector to 'force' the frame back upwards; if set to false,
       the upVector will be ignored */
    bool          forceUp       { true };

    /*! multiplier how fast the camera should move in world space
      for each unit of "user specifeid motion" (ie, pixel
      count). Initial value typically should depend on the world
      size, but can also be adjusted. This is actually something
      that should be more part of the manipulator widget(s), but
      since that same value is shared by multiple such widgets
      it's easiest to attach it to the camera here ...*/
    float         motionSpeed   { 1.f };
    
    /*! gets set to true every time a manipulator changes the camera
      values */
    bool          modified      { true };
  };



  // ------------------------------------------------------------------
  /*! abstract base class that allows to manipulate a renderable
    camera */
  struct CameraFrameManip {
    CameraFrameManip(CameraFrame *cameraFrame)
      : cameraFrame(cameraFrame)
    {}
    
    /*! this gets called when the user presses a key on the keyboard ... */
    virtual void key(int key, int mods) {
      CameraFrame &fc = *cameraFrame;
      
      switch(key) {
      case '+':
      case '=':
        fc.motionSpeed *= 1.5f;
        break;
      case '-':
      case '_':
        fc.motionSpeed /= 1.5f;
        break;
      case 'C':
        break;
      }
    }

    virtual void strafe(const float3 &howMuch) {
      cameraFrame->position += howMuch;
      cameraFrame->modified =  true;
    }
    /*! strafe, in screen space */
    virtual void strafe(const float2 &howMuch) {
      strafe(+howMuch.x*cameraFrame->frame.getCol(0)
             -howMuch.y*cameraFrame->frame.getCol(1));
    }

    virtual void move(const float step) = 0;
    virtual void rotate(const float dx, const float dy) = 0;
    
    // /*! this gets called when the user presses a key on the keyboard ... */
    // virtual void special(int key, const vec2i &where) { };
    
    /*! mouse got dragged with left button pressedn, by 'delta'
      pixels, at last position where */
    virtual void mouseDragLeft  (const float2 &delta) {
      rotate(delta.x * degrees_per_drag_fraction,
             delta.y * degrees_per_drag_fraction);
    }
    
    /*! mouse got dragged with left button pressedn, by 'delta'
      pixels, at last position where */
    virtual void mouseDragMiddle(const float2 &delta) {
      strafe(delta*pixels_per_move*cameraFrame->motionSpeed);
    }

    /*! mouse got dragged with left button pressedn, by 'delta'
      pixels, at last position where */
    virtual void mouseDragRight (const float2 &delta) {
      move(delta.y*pixels_per_move*cameraFrame->motionSpeed);
    }

    // /*! mouse button got either pressed or released at given location */
    // virtual void mouseButtonLeft  (const vec2i &where, bool pressed) {}
      
    // /*! mouse button got either pressed or released at given location */
    // virtual void mouseButtonMiddle(const vec2i &where, bool pressed) {}
      
    // /*! mouse button got either pressed or released at given location */
    // virtual void mouseButtonRight (const vec2i &where, bool pressed) {}
      
  protected:
    CameraFrame *cameraFrame;
    const float kbd_rotate_degrees        {  10.f };
    const float degrees_per_drag_fraction { 150.f };
    const float pixels_per_move           {  10.f };
  };


  struct GLFCameraWindow : public GLFWindow {
    GLFCameraWindow(const std::string &title,
                    const float3 &camera_from,
                    const float3 &camera_at,
                    const float3 &camera_up,
                    const float worldScale,
                    const int2 size=make_int2(1200,800))
      : GLFWindow(title, size),
        cameraFrame(worldScale)
    {
      cameraFrame.setOrientation(camera_from,camera_at,camera_up);
      enableFlyMode();
      enableInspectMode();
    }

    void enableFlyMode();
    void enableInspectMode();
    
    // /*! put pixels on the screen ... */
    // virtual void draw()
    // { /* empty - to be subclassed by user */ }

    // /*! callback that window got resized */
    // virtual void resize(const vec2i &newSize)
    // { /* empty - to be subclassed by user */ }

    virtual void key(int key, int mods) override {
      switch(key) {
      case 'f':
      case 'F':
        if (flyModeManip) cameraFrameManip = flyModeManip;
        break;
      case 'i':
      case 'I':
        if (inspectModeManip) cameraFrameManip = inspectModeManip;
        break;
      default:
        if (cameraFrameManip)
          cameraFrameManip->key(key,mods);
      }
    }
    
    /*! callback that window got resized */
    virtual void mouseMotion(const int2 &newPos) override {
      int2 windowSize;
      glfwGetWindowSize(handle, &windowSize.x, &windowSize.y);
      
      if (isPressed.leftButton && cameraFrameManip)
        cameraFrameManip->mouseDragLeft(make_float2(newPos-lastMousePos)/make_float2(windowSize));
      if (isPressed.rightButton && cameraFrameManip)
        cameraFrameManip->mouseDragRight(make_float2(newPos-lastMousePos)/make_float2(windowSize));
      if (isPressed.middleButton && cameraFrameManip)
        cameraFrameManip->mouseDragMiddle(make_float2(newPos-lastMousePos)/make_float2(windowSize));
      lastMousePos = newPos;
      /* empty - to be subclassed by user */
    }
    
    /*! callback that window got resized */
    virtual void mouseButton(int button, int action, int mods) override {
      const bool pressed = (action == GLFW_PRESS);
      switch(button) {
      case GLFW_MOUSE_BUTTON_LEFT:
        isPressed.leftButton = pressed;
        break;
      case GLFW_MOUSE_BUTTON_MIDDLE:
        isPressed.middleButton = pressed;
        break;
      case GLFW_MOUSE_BUTTON_RIGHT:
        isPressed.rightButton = pressed;
        break;
      }
      lastMousePos = getMousePos();
    }

    // /*! mouse got dragged with left button pressedn, by 'delta'
    //   pixels, at last position where */
    // virtual void mouseDragLeft  (const vec2i &where, const vec2i &delta) {}

    // /*! mouse got dragged with left button pressedn, by 'delta'
    //   pixels, at last position where */
    // virtual void mouseDragRight (const vec2i &where, const vec2i &delta) {}

    // /*! mouse got dragged with left button pressedn, by 'delta'
    //   pixels, at last position where */
    // virtual void mouseDragMiddle(const vec2i &where, const vec2i &delta) {}

    
    /*! a (global) pointer to the currently active window, so we can
      route glfw callbacks to the right GLFWindow instance (in this
      simplified library we only allow on window at any time) */
    // static GLFWindow *current;

    struct {
      bool leftButton { false }, middleButton { false }, rightButton { false };
    } isPressed;
    int2 lastMousePos = { -1,-1 };

    friend struct CameraFrameManip;

    CameraFrame cameraFrame;
    std::shared_ptr<CameraFrameManip> cameraFrameManip;
    std::shared_ptr<CameraFrameManip> inspectModeManip;
    std::shared_ptr<CameraFrameManip> flyModeManip;
  };


  // ------------------------------------------------------------------
  /*! camera manipulator with the following traits
      
    - there is a "point of interest" (POI) that the camera rotates
    around.  (we track this via poiDistance, the point is then
    thta distance along the fame's z axis)
      
    - we can restrict the minimum and maximum distance camera can be
    away from this point
      
    - we can specify a max bounding box that this poi can never
    exceed (validPoiBounds).
      
    - we can restrict whether that point can be moved (by using a
    single-point valid poi bounds box
      
    - left drag rotates around the object

    - right drag moved forward, backward (within min/max distance
    bounds)

    - middle drag strafes left/right/up/down (within valid poi
    bounds)
      
  */
  struct InspectModeManip : public CameraFrameManip {

    InspectModeManip(CameraFrame *cameraFrame)
      : CameraFrameManip(cameraFrame)
    {}
      
  private:
    /*! helper function: rotate camera frame by given degrees, then
      make sure the frame, poidistance etc are all properly set,
      the widget gets notified, etc */
    virtual void rotate(const float deg_u, const float deg_v) override {
      float rad_u = -M_PI/180.f*deg_u;
      float rad_v = -M_PI/180.f*deg_v;

      CameraFrame &fc = *cameraFrame;
      
      const float3 poi  = fc.getPOI();
      fc.frame
        = sutil::make_matrix3x3(sutil::Matrix4x4::rotate(rad_u, fc.frame.getCol(1)))
        * sutil::make_matrix3x3(sutil::Matrix4x4::rotate(rad_v, fc.frame.getCol(0)))
        * fc.frame;

      if (fc.forceUp) fc.forceUpFrame();

      fc.position = poi + fc.poiDistance * fc.frame.getCol(2);
      fc.modified = true;
    }
      
    /*! helper function: move forward/backwards by given multiple of
      motion speed, then make sure the frame, poidistance etc are
      all properly set, the widget gets notified, etc */
    virtual void move(const float step) override {
      const float3 poi = cameraFrame->getPOI();
      // inspectmode can't get 'beyond' the look-at point:
      const float minReqDistance = 0.1f * cameraFrame->motionSpeed;
      cameraFrame->poiDistance   = fmax(minReqDistance,cameraFrame->poiDistance-step);
      cameraFrame->position      = poi + cameraFrame->poiDistance * cameraFrame->frame.getCol(2);
      cameraFrame->modified      = true;
    }
  };

  // ------------------------------------------------------------------
  /*! camera manipulator with the following traits

    - left button rotates the camera around the viewer position

    - middle button strafes in camera plane
      
    - right buttton moves forward/backwards
      
  */
  struct FlyModeManip : public CameraFrameManip {

    FlyModeManip(CameraFrame *cameraFrame)
      : CameraFrameManip(cameraFrame)
    {}
      
  private:
    /*! helper function: rotate camera frame by given degrees, then
      make sure the frame, poidistance etc are all properly set,
      the widget gets notified, etc */
    virtual void rotate(const float deg_u, const float deg_v) override
    {
      float rad_u = -M_PI/180.f*deg_u;
      float rad_v = -M_PI/180.f*deg_v;

      CameraFrame &fc = *cameraFrame;
      
      //const vec3f poi  = fc.getPOI();
      fc.frame
        = sutil::make_matrix3x3(sutil::Matrix4x4::rotate(rad_u, fc.frame.getCol(1)))
        * sutil::make_matrix3x3(sutil::Matrix4x4::rotate(rad_v, fc.frame.getCol(0)))
        * fc.frame;

      if (fc.forceUp) fc.forceUpFrame();

      fc.modified = true;
    }
      
    /*! helper function: move forward/backwards by given multiple of
      motion speed, then make sure the frame, poidistance etc are
      all properly set, the widget gets notified, etc */
    virtual void move(const float step) override
    {
      cameraFrame->position    += step * cameraFrame->frame.getCol(2);
      cameraFrame->modified     = true;
    }
  };

  inline void GLFCameraWindow::enableFlyMode()
  {
    flyModeManip     = std::make_shared<FlyModeManip>(&cameraFrame);
    cameraFrameManip = flyModeManip;
  }
  
  inline void GLFCameraWindow::enableInspectMode()
  {
    inspectModeManip = std::make_shared<InspectModeManip>(&cameraFrame);
    cameraFrameManip = inspectModeManip;
  }
