#include "scene.h"
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

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
    launchParams.traversable = buildAccel();
    createPipeline();
    createSBT();

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
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
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
    // for this simple example, we set up a single hit group
    hitgroupPGs.resize(1);
      
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH            = module;           
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
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

    // we don't actually have any objects in this example, but let's
    // create a dummy one so the SBT doesn't have any null pointers
    // (which the sanity checks in compilation would complain about)
    int numObjects = 1;
    std::vector<HitGroupSbtRecord> hitgroupRecords;
    for (int i=0;i<numObjects;i++) {
      int objectType = 0;
      HitGroupSbtRecord rec;
      rec.data.radius = 1.5f;
      OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[objectType],&rec));
      hitgroupRecords.push_back(rec);
    }
    hitgroupRecordsBuffer.alloc_and_copy_to_device(hitgroupRecords);
    sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    sbt.hitgroupRecordCount         = 1;


  }

  OptixTraversableHandle Scene::buildAccel() {
    OptixTraversableHandle asHandle {0};

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;
    
    // AABB build input
    std::vector<float> aabb = {-1.5f, -1.5f, -1.5f, 1.5f, 1.5f, 1.5f};
    aabbBuffer.alloc_and_copy_to_device(aabb);
    CUdeviceptr d_aabb_buffer = aabbBuffer.d_pointer();
                
    OptixBuildInput aabb_input = {};
    aabb_input.type                               = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.customPrimitiveArray.aabbBuffers   = &d_aabb_buffer;
    aabb_input.customPrimitiveArray.numPrimitives = 1;

    uint32_t aabb_input_flags[1]                  = {OPTIX_GEOMETRY_FLAG_NONE};
    aabb_input.customPrimitiveArray.flags         = aabb_input_flags;
    aabb_input.customPrimitiveArray.numSbtRecords = 1;

    // ==================================================================
    // BLAS setup
    // ==================================================================
    
    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage
                (optixContext,
                 &accel_options,
                 &aabb_input,
                 1,  // num_build_inputs
                 &blasBufferSizes
                 ));
    
    // ==================================================================
    // prepare compaction
    // ==================================================================
    
    cuBuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    //CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    //size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    //CUDA_CHECK( cudaMalloc(
                //reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ),
                //compactedSizeOffset + 8
                //) );

    //OptixAccelEmitDesc emitProperty = {};
    //emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    //emitProperty.result             = ( CUdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );
    
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
                                &aabb_input,
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
