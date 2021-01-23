#include "scene.h"
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>
#include <cstring>

namespace optix7tutorial {

  extern "C" char embedded_ptx_code[];

  template <typename T>
  struct SbtRecord {
      __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
      T data;
  };
  
  typedef SbtRecord<RayGenData>     RayGenSbtRecord;
  typedef SbtRecord<MissData>       MissSbtRecord;
  typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;

  /*! constructor - performs all setup, including initializing
    optix, creates module, pipeline, programs, SBT, etc. */
  Scene::Scene() {
    initOptix();
    createContext();
    createPTXModule();
    createRaygenPrograms();
    createMissPrograms();
    createHitGroupPrograms();

    buildSphereGAS();
    buildTriangleGAS();
    //launchParams.traversable = sphere_gas;
    launchParams.traversable = buildIAS();

    createPipeline();
    createSBT();

    launchParams.lightPos = make_float3(0,5,3);
    launchParamsBuffer.alloc(sizeof(launchParams));
  }

  /*! helper function that initializes optix and checks for errors */
  void Scene::initOptix() {
    // -------------------------------------------------------
    // check for available optix7 capable devices
    // -------------------------------------------------------
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
      throw std::runtime_error("no CUDA capable devices found!");

    // -------------------------------------------------------
    // initialize optix
    // -------------------------------------------------------
    OPTIX_CHECK(optixInit());
  }

  /*! creates and configures a optix device context (in this simple
      example, only for the primary GPU device) */
  void Scene::createContext() {
    // for this sample, do everything on one device
    const int deviceID = 0;
    CUDA_CHECK(SetDevice(deviceID));
    CUDA_CHECK(StreamCreate(&stream));
      
    cudaGetDeviceProperties(&deviceProps, deviceID);
      
    CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
    if( cuRes != CUDA_SUCCESS ) 
      fprintf( stderr, "Error querying current context: error code %d\n", cuRes );
      
    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
  }

  /*! creates the module that contains all the programs we are going
      to use. in this simple example, we use a single module from a
      single .cu file, using a single embedded ptx string */
  void Scene::createPTXModule() {
    moduleCompileOptions.maxRegisterCount  = 50;
    moduleCompileOptions.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipelineCompileOptions.usesMotionBlur     = false;
    pipelineCompileOptions.numPayloadValues   = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags     = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";
      
    pipelineLinkOptions.maxTraceDepth          = 2;
      
    const std::string ptxCode = embedded_ptx_code;
      
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
                                         &moduleCompileOptions,
                                         &pipelineCompileOptions,
                                         ptxCode.c_str(),
                                         ptxCode.size(),
                                         log,&sizeof_log,
                                         &module
                                         ));
  }
    


  /*! does all setup for the raygen program(s) we are going to use */
  void Scene::createRaygenPrograms() {
    // we do a single ray gen program in this example:
    raygenPGs.resize(1);
      
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module            = module;           
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &raygenPGs[0]
                                        ));
  }
    
  /*! does all setup for the miss program(s) we are going to use */
  void Scene::createMissPrograms() {
    // we do a single ray gen program in this example:
    missPGs.resize(1);
      
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module            = module;           
    pgDesc.miss.entryFunctionName = "__miss__radiance";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &missPGs[0]
                                        ));
  }
    
  /*! does all setup for the hitgroup program(s) we are going to use */
  void Scene::createHitGroupPrograms() {
    hitgroupPGs.resize(2);
      
    {
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH            = module;           
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__sphere";
    pgDesc.hitgroup.moduleAH            = module;           
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";
    pgDesc.hitgroup.moduleIS            = module;
    pgDesc.hitgroup.entryFunctionNameIS = "__intersection__is";

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &hitgroupPGs[0]
                                        ));
    }

    {
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH            = module;           
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__triangle";
    pgDesc.hitgroup.moduleAH            = module;           
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &hitgroupPGs[1]
                                        ));
    }
  }
    

  /*! assembles the full pipeline of all programs */
  void Scene::createPipeline() {
    std::vector<OptixProgramGroup> programGroups;
    for (auto pg : raygenPGs)
      programGroups.push_back(pg);
    for (auto pg : missPGs)
      programGroups.push_back(pg);
    for (auto pg : hitgroupPGs)
      programGroups.push_back(pg);
      
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixPipelineCreate(optixContext,
                                    &pipelineCompileOptions,
                                    &pipelineLinkOptions,
                                    programGroups.data(),
                                    (int)programGroups.size(),
                                    log,&sizeof_log,
                                    &pipeline
                                    ));

    OPTIX_CHECK(optixPipelineSetStackSize
                (/* [in] The pipeline to configure the stack size for */
                 pipeline, 
                 /* [in] The direct stack size requirement for direct
                    callables invoked from IS or AH. */
                 2*1024,
                 /* [in] The direct stack size requirement for direct
                    callables invoked from RG, MS, or CH.  */                 
                 2*1024,
                 /* [in] The continuation stack requirement. */
                 2*1024,
                 /* [in] The maximum depth of a traversable graph
                    passed to trace. */
                 1));
  }


  /*! constructs the shader binding table */
  void Scene::createSBT() {
    // ------------------------------------------------------------------
    // build raygen records
    // ------------------------------------------------------------------
    std::vector<RayGenSbtRecord> raygenRecords;
    for (int i=0;i<raygenPGs.size();i++) {
      RayGenSbtRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i],&rec));
      raygenRecords.push_back(rec);
    }
    raygenRecordsBuffer.alloc_and_copy_to_device(raygenRecords);
    sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------
    std::vector<MissSbtRecord> missRecords;
    for (int i=0;i<missPGs.size();i++) {
      MissSbtRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i],&rec));
      missRecords.push_back(rec);
    }
    missRecordsBuffer.alloc_and_copy_to_device(missRecords);
    sbt.missRecordBase          = missRecordsBuffer.d_pointer();
    sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    sbt.missRecordCount         = (int)missRecords.size();

    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------

    std::vector<HitGroupSbtRecord> hitgroupRecords;
    {
      HitGroupSbtRecord rec;
      rec.data.geometry.sphere.radius = 1.5f;
      rec.data.material.specular = 0.3;
      rec.data.type = HitGroupData::Type::SPHERE;
      OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[0],&rec));
      hitgroupRecords.push_back(rec);
    }
    {
      HitGroupSbtRecord rec;
      rec.data.geometry.triangle_mesh.index    = (int3*)indexBuffer.d_pointer();
      rec.data.geometry.triangle_mesh.vertex   = (float3*)vertexBuffer.d_pointer();
      rec.data.type = HitGroupData::Type::TRIANGLE_MESH;
      OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[1],&rec));
      hitgroupRecords.push_back(rec);
    }

    hitgroupRecordsBuffer.alloc_and_copy_to_device(hitgroupRecords);
    sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    sbt.hitgroupRecordCount         = hitgroupRecords.size();
  }

  void Scene::buildTriangleGAS() {
    // ==================================================================
    // triangle inputs
    // ==================================================================
    std::vector<float3> vertices;
    std::vector<int3> indices;
    float W = 4.0;
    vertices.push_back(make_float3(-W,-3,W));
    vertices.push_back(make_float3(-W, -3,-W));
    vertices.push_back(make_float3(W, -3,-W));
    vertices.push_back(make_float3(W, -3,W));
    indices.push_back(make_int3(0,1,2));
    indices.push_back(make_int3(0,2,3));

    vertexBuffer.alloc_and_copy_to_device(vertices);
    indexBuffer.alloc_and_copy_to_device(indices);
    
    OptixBuildInput triangleInput = {};
    triangleInput.type
      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // create local variables, because we need a *pointer* to the
    // device pointers
    CUdeviceptr d_vertices = vertexBuffer.d_pointer();
    CUdeviceptr d_indices  = indexBuffer.d_pointer();
      
    triangleInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput.triangleArray.vertexStrideInBytes = sizeof(float3);
    triangleInput.triangleArray.numVertices         = (int)vertices.size();
    triangleInput.triangleArray.vertexBuffers       = &d_vertices;
    
    triangleInput.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput.triangleArray.indexStrideInBytes  = sizeof(int3);
    triangleInput.triangleArray.numIndexTriplets    = (int)indices.size();
    triangleInput.triangleArray.indexBuffer         = d_indices;
    
    uint32_t triangleInputFlags[1] = { 0 };
    triangleSbtIndex.alloc_and_copy_to_device(std::vector<uint32_t>{1,1});
    
    // in this example we have one SBT entry, and no per-primitive
    // materials:
    triangleInput.triangleArray.flags               = triangleInputFlags;
    triangleInput.triangleArray.numSbtRecords               = 2;
    triangleInput.triangleArray.sbtIndexOffsetBuffer        = triangleSbtIndex.d_pointer(); 
    triangleInput.triangleArray.sbtIndexOffsetSizeInBytes   = sizeof( uint32_t ); 
    triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0; 

    triangle_gas = buildGAS(triangleInput, triangleBuffer);
  }

  void Scene::buildSphereGAS() {
    // ==================================================================
    // sphere inputs
    // ==================================================================
    std::vector<float> aabb = {-1.5f, -1.5f, -1.5f, 1.5f, 1.5f, 1.5f};
    aabbBuffer.alloc_and_copy_to_device(aabb);
    CUdeviceptr d_aabb_buffer = aabbBuffer.d_pointer();
                
    OptixBuildInput aabb_input = {};
    aabb_input.type                               = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.customPrimitiveArray.aabbBuffers   = &d_aabb_buffer;
    aabb_input.customPrimitiveArray.numPrimitives = 1;

    sphereSbtIndex.alloc_and_copy_to_device(std::vector<uint32_t>{0});

    uint32_t aabb_input_flags[1]                  = {OPTIX_GEOMETRY_FLAG_NONE};
    aabb_input.customPrimitiveArray.flags         = aabb_input_flags;
    aabb_input.customPrimitiveArray.numSbtRecords = 2;
    aabb_input.customPrimitiveArray.sbtIndexOffsetBuffer         = sphereSbtIndex.d_pointer();
    aabb_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes    = sizeof( uint32_t );
    aabb_input.customPrimitiveArray.sbtIndexOffsetStrideInBytes  = 0; 
    aabb_input.customPrimitiveArray.primitiveIndexOffset         = 0;

    sphere_gas = buildGAS(aabb_input, sphereBuffer);
  }

  OptixTraversableHandle Scene::buildIAS() {
    OptixTraversableHandle asHandle {0};
    std::vector<OptixInstance> instances;

    {

        OptixInstance instance = {};
        float transform[12] = {1,0,0,0,
                               0,1,0,0,
                               0,0,1,0};
        instance.instanceId = 0;
        instance.visibilityMask = 255;
        instance.sbtOffset = 0;
        instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        memcpy(instance.transform, transform, sizeof(float)*12);
        instance.traversableHandle = sphere_gas;
        instances.push_back(instance);
    }
    {
        OptixInstance instance = {};
        float transform[12] = {1,0,0,0,
                               0,1,0,0,
                               0,0,1,0};
        instance.instanceId = 0;
        instance.visibilityMask = 255;
        instance.sbtOffset = 0;
        instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        memcpy(instance.transform, transform, sizeof(float)*12);
        instance.traversableHandle = triangle_gas;
        instances.push_back(instance);
    }

    cuBuffer instancesBuffer;
    instancesBuffer.alloc_and_copy_to_device(instances);

    // Instance build input.
    OptixBuildInput buildInput = {};

    buildInput.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    buildInput.instanceArray.instances    = instancesBuffer.d_pointer();
    buildInput.instanceArray.numInstances = static_cast<unsigned int>( instances.size() );

    OptixAccelBuildOptions accelBuildOptions = {};
    accelBuildOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE;
    accelBuildOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes bufferSizesIAS;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( optixContext, &accelBuildOptions, &buildInput,
                                               1,  // Number of build inputs
                                               &bufferSizesIAS ) );


    cuBuffer tempBuffer;
    tempBuffer.alloc(bufferSizesIAS.tempSizeInBytes);
    IASBuffer.alloc(bufferSizesIAS.outputSizeInBytes);

    OPTIX_CHECK( optixAccelBuild( optixContext,
                                  0,  // CUDA stream
                                  &accelBuildOptions,
                                  &buildInput,
                                  1,  // num build inputs
                                  tempBuffer.d_pointer(),
                                  tempBuffer.size_in_bytes,
                                  IASBuffer.d_pointer(),
                                  IASBuffer.size_in_bytes,
                                  &asHandle,
                                  nullptr,  // emitted property list
                                  0 ) );    // num emitted properties

    tempBuffer.free();
    instancesBuffer.free();

    return asHandle;
  }

  OptixTraversableHandle Scene::buildGAS(OptixBuildInput& build_input, cuBuffer& asBuffer) {
    OptixTraversableHandle asHandle {0};

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

    // ==================================================================
    // BLAS setup
    // ==================================================================
    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage
                (optixContext,
                 &accel_options,
                 &build_input,
                 1,  // num_build_inputs
                 &blasBufferSizes
                 ));
    
    // ==================================================================
    // prepare compaction
    // ==================================================================
    
    cuBuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();
    
    // ==================================================================
    // execute build (main stage)
    // ==================================================================
    
    cuBuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);
    
    cuBuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);
      
    OPTIX_CHECK(optixAccelBuild(optixContext,
                                /* stream */0,
                                &accel_options,
                                &build_input,
                                1,  
                                tempBuffer.d_pointer(),
                                tempBuffer.size_in_bytes,
                                outputBuffer.d_pointer(),
                                outputBuffer.size_in_bytes,
                                &asHandle,
                                &emitDesc,
                                1
                                ));
    CUDA_SYNC_CHECK();
    
    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compactedSize;
    compactedSizeBuffer.copy_from_device(&compactedSize,1);
    
    asBuffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(optixContext,
                                  /*stream:*/0,
                                  asHandle,
                                  asBuffer.d_pointer(),
                                  asBuffer.size_in_bytes,
                                  &asHandle));
    CUDA_SYNC_CHECK();
    
    // ==================================================================
    // aaaaaand .... clean up
    // ==================================================================
    outputBuffer.free(); // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();

    return asHandle;
  }
  



  /*! render one frame */
  void Scene::render() {
    // sanity check: make sure we launch only after first resize is
    // already done:
    if (launchParams.frame.size.x == 0) return;

    launchParamsBuffer.copy_to_device(&launchParams,1);
      
    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                            pipeline,stream,
                            /*! parameters and SBT */
                            launchParamsBuffer.d_pointer(),
                            launchParamsBuffer.size_in_bytes,
                            &sbt,
                            /*! dimensions of the launch: */
                            launchParams.frame.size.x,
                            launchParams.frame.size.y,
                            1
                            ));
    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    CUDA_SYNC_CHECK();
  }

  /*! set camera to render with */
  void Scene::setCamera(const Camera &camera)
  {
    lastSetCamera = camera;
    launchParams.camera.position  = camera.from;
    launchParams.camera.direction = normalize(camera.at-camera.from);
    const float cosFovy = 0.66f;
    const float aspect = launchParams.frame.size.x / float(launchParams.frame.size.y);
    launchParams.camera.horizontal
      = cosFovy * aspect * normalize(cross(launchParams.camera.direction,
                                           camera.up));
    launchParams.camera.vertical
      = cosFovy * normalize(cross(launchParams.camera.horizontal,
                                  launchParams.camera.direction));
  }
  

  /*! resize frame buffer to given resolution */
  void Scene::resize(const int2 &newSize) {
    // if window minimized
    if (newSize.x == 0 | newSize.y == 0) return;
    
    // resize our cuda frame buffer
    colorBuffer.resize(newSize.x*newSize.y*sizeof(uint32_t));

    // update the launch parameters that we'll pass to the optix
    // launch:
    launchParams.frame.size      = newSize;
    launchParams.frame.colorBuffer = (uint32_t*)colorBuffer.d_ptr;

    // and re-set the camera, since aspect may have changed
    setCamera(lastSetCamera);
  }

  /*! download the rendered color buffer */
  void Scene::download_pixels(uint32_t h_pixels[]) {
    colorBuffer.copy_from_device(h_pixels,
                         launchParams.frame.size.x*launchParams.frame.size.y);
  }
  
}
