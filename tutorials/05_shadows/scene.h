#pragma once
#include "cuBuffer.h"
#include "launchParams.h"
#include <iostream>

namespace optix7tutorial {
  enum RayType {
      RAY_TYPE_RADIANCE=0,
      RAY_TYPE_OCCLUSION=1,
      RAY_TYPE_COUNT
  };

  struct RayGenData {};
  
  struct MissData {};
  
  struct HitGroupData {
      enum Type {
          SPHERE                = 0,
          TRIANGLE_MESH         = 1,
      };

      struct Sphere {
        float radius;
      };

      struct TriangleMesh {
        float3* vertex;
        int3* index;
      };

      union {
          Sphere sphere;
          TriangleMesh triangle_mesh;
      } geometry;

      Type  type;

      struct Material {
        float ambient = .2;
        float diffuse = .8;
        float specular = .1;
        int shininess = 32;
      } material; 
  };

  struct Camera {
    /*! camera position - *from* where we are looking */
    float3 from;
    /*! which point we are looking *at* */
    float3 at;
    /*! general up-vector */
    float3 up;
  };

  /*! a sample OptiX-7 renderer that demonstrates how to set up
      context, module, programs, pipeline, SBT, etc, and perform a
      valid launch that renders some pixel (using a simple test
      pattern, in this case */
  class Scene {
  public:
    /*! constructor - performs all setup, including initializing
      optix, creates module, pipeline, programs, SBT, etc. */
    Scene();

    /*! render one frame */
    void render();

    /*! resize frame buffer to given resolution */
    void resize(const int2 &newSize);

    /*! download the rendered color buffer */
    void download_pixels(uint32_t h_pixels[]);

    /*! set camera to render with */
    void setCamera(const Camera &camera);

  protected:
    /*! helper function that initializes optix and checks for errors */
    void initOptix();
  
    /*! creates and configures a optix device context (in this simple
      example, only for the primary GPU device) */
    void createContext();

    /*! creates the module that contains all the programs we are going
      to use. in this simple example, we use a single module from a
      single .cu file, using a single embedded ptx string */
    void createPTXModule();
    
    /*! does all setup for the raygen program(s) we are going to use */
    void createRaygenPrograms();
    
    /*! does all setup for the miss program(s) we are going to use */
    void createMissPrograms();
    
    /*! does all setup for the hitgroup program(s) we are going to use */
    void createHitGroupPrograms();

    /*! assembles the full pipeline of all programs */
    void createPipeline();

    /*! constructs the shader binding table */
    void createSBT();

    /*! build an acceleration structure for the given triangle mesh */
    OptixTraversableHandle buildIAS();
    OptixTraversableHandle buildGAS(OptixBuildInput& build_input, cuBuffer& asBuffer);
    void buildSphereGAS();
    void buildTriangleGAS();

  protected:
    /*! @{ CUDA device context and stream that optix pipeline will run
        on, as well as device properties for this device */
    CUcontext          cudaContext;
    CUstream           stream;
    cudaDeviceProp     deviceProps;
    /*! @} */

    //! the optix context that our pipeline will run in.
    OptixDeviceContext optixContext;

    /*! @{ the pipeline we're building */
    OptixPipeline               pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions    pipelineLinkOptions = {};
    /*! @} */

    /*! @{ the module that contains out device programs */
    OptixModule                 module;
    OptixModuleCompileOptions   moduleCompileOptions = {};
    /* @} */

    /*! vector of all our program(group)s, and the SBT built around
        them */
    std::vector<OptixProgramGroup> raygenPGs;
    cuBuffer raygenRecordsBuffer;
    std::vector<OptixProgramGroup> missPGs;
    cuBuffer missRecordsBuffer;
    std::vector<OptixProgramGroup> hitgroupPGs;
    cuBuffer hitgroupRecordsBuffer;
    OptixShaderBindingTable sbt = {};

    /*! @{ our launch parameters, on the host, and the buffer to store
        them on the device */
    LaunchParams launchParams;
    cuBuffer   launchParamsBuffer;
    /*! @} */

    cuBuffer colorBuffer;

    /*! the camera we are to render with. */
    Camera lastSetCamera;

    /*! the model we are going to trace rays against */
    cuBuffer aabbBuffer;

    /*! the model we are going to trace rays against */
    cuBuffer vertexBuffer;
    cuBuffer indexBuffer;

    OptixTraversableHandle sphere_gas;
    OptixTraversableHandle triangle_gas;

    //! buffer that keeps the (final, compacted) accel structure
    cuBuffer sphereSbtIndex;
    cuBuffer triangleSbtIndex;
    cuBuffer sphereBuffer;
    cuBuffer triangleBuffer;
    cuBuffer IASBuffer;
  };

}
