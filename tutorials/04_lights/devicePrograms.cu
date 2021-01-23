#include <optix_device.h>
#include "launchParams.h"
#include "scene.h"

using namespace optix7tutorial;

namespace optix7tutorial {

  extern "C" __constant__ LaunchParams optixLaunchParams;

  // for this simple example, we have a single ray type
  enum { SURFACE_RAY_TYPE=0, RAY_TYPE_COUNT };
  
  static __forceinline__ __device__
  void *unpackPointer( uint32_t i0, uint32_t i1 ) {
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr ); 
    return ptr;
  }

  static __forceinline__ __device__
  void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 ) {
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
  }

  template<typename T>
  static __forceinline__ __device__ T *getARD() { 
    const uint32_t u0 = optixGetAttribute_0();
    const uint32_t u1 = optixGetAttribute_1();
    return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
  }

  template<typename T>
  static __forceinline__ __device__ T *getPRD() { 
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
  }
  
  extern "C" __global__ void __anyhit__radiance() {}

  extern "C" __global__ void __raygen__renderFrame() {
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const auto &camera = optixLaunchParams.camera;

    // our per-ray data for this example. what we initialize it to
    // won't matter, since this value will be overwritten by either
    // the miss or hit program, anyway
    float3 color = make_float3(0.5f);

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer( &color, u0, u1 );

    // normalized screen plane position, in [0,1]^2
    const float2 screen = (make_float2(ix+.5f,iy+.5f)
                       / make_float2(optixLaunchParams.frame.size.x, optixLaunchParams.frame.size.y));
    
    // generate ray direction
    float3 rayDir = normalize(camera.direction
                             + (screen.x - 0.5f) * camera.horizontal
                             + (screen.y - 0.5f) * camera.vertical);

    optixTrace(optixLaunchParams.traversable,
               camera.position,
               rayDir,
               0.f,    // tmin
               1e20f,  // tmax
               0.0f,   // rayTime
               OptixVisibilityMask( 255 ),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,             // SBT offset
               RAY_TYPE_COUNT,               // SBT stride
               SURFACE_RAY_TYPE,             // missSBTIndex 
               u0, u1 );

    color = clamp(color, 0, 1);
    const int3 rgb = make_int3(color.x*255, color.y*255, color.z*255);
    //const int3 rgb = make_int3(color.x*255.99f, color.y*255.99f, color.z*255.99f);

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    const uint32_t rgba = 0xff000000
      | (rgb.x<<0) | (rgb.y<<8) | (rgb.z<<16);

    // and write to frame buffer ...
    const uint32_t fbIndex = ix + iy*optixLaunchParams.frame.size.x;
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
  }


  extern "C" __global__ void __miss__radiance() {
    float3 &prd = *getPRD<float3>();

    // set to constant white as background color
    const float3 rayDir = optixGetWorldRayDirection();
    //prd = rayDir;
    float z = rayDir.y;
    float rho = sqrt(rayDir.x*rayDir.x + rayDir.z*rayDir.z);
    float theta = atan2(z, rho);
    float t = (theta + M_PI/2)/M_PI;
    const float3 color = (1-t)*make_float3(1.0,1.0,1.0) + t*make_float3(.3,.5,1);
    prd = color;
  }


extern "C" __global__ void __intersection__is() {
    auto* hg_data  = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );
    const float3 orig = optixGetObjectRayOrigin();
    const float3 dir  = optixGetObjectRayDirection();

    const float3 center = {0.f, 0.f, 0.f};
    const float  radius = hg_data->geometry.sphere.radius;

    const float3 O      = orig - center;
    const float  l      = 1 / length( dir );
    const float3 D      = dir * l;

    const float b    = dot( O, D );
    const float c    = dot( O, O ) - radius * radius;
    const float disc = b * b - c;
    if( disc > 0.0f )
    {
        const float sdisc = sqrtf( disc );
        const float root1 = ( -b - sdisc );

        const float        root11        = 0.0f;
        float3       shading_normal = ( O + ( root1 + root11 ) * D ) / radius;

        uint32_t u0, u1;
        packPointer( &shading_normal, u0, u1 );

        optixReportIntersection(
                root1,      // t hit
                0,          // user hit kind
                u0, u1
                );
    }
}

extern "C" __global__ void __closesthit__sphere() {
    auto* hg_data  = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );
    const auto &camera = optixLaunchParams.camera;
    const float ambient = hg_data->material.ambient;
    const float diffuse = hg_data->material.diffuse;
    const float specular = hg_data->material.specular;
    const float shininess = hg_data->material.shininess;
    const float radius = hg_data->geometry.sphere.radius;

    float3 &prd = *getPRD<float3>();
    float3 &att = *getARD<float3>();
    float3 normal = normalize( optixTransformNormalFromObjectToWorldSpace( att) );
    float3 color = normal * 0.5f + 0.5f;

    float3 lightPos = optixLaunchParams.lightPos;
    float3 intersection = radius*normal;
    float3 lightDir = normalize(lightPos - intersection);
    float diffuseComp = diffuse*max(dot(normal, lightDir), 0.0f);

    float3 viewDir = normalize(camera.position - intersection);
    float3 reflectDir = reflect(-lightDir, normal);
    float specularComp = specular*pow(max(dot(viewDir, reflectDir), 0.0f), shininess);

    prd = (ambient + diffuseComp)*(color + specularComp);
}

extern "C" __global__ void __closesthit__triangle() {
    auto* hg_data  = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );
    const auto &camera = optixLaunchParams.camera;
    const float ambient = hg_data->material.ambient;
    const float diffuse = hg_data->material.diffuse;
    const float specular = hg_data->material.specular;
    const float shininess = hg_data->material.shininess;

    const int   primID = optixGetPrimitiveIndex();
    const int3  index  = hg_data->geometry.triangle_mesh.index[primID];
    const float3* vertex = hg_data->geometry.triangle_mesh.vertex;
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    float3 &prd = *getPRD<float3>();
    float3 color = make_float3(.1,.5,.1);

    float3 normal = make_float3(0,1,0);
    float3 lightPos = optixLaunchParams.lightPos;
    float3 intersection = (1 - u - v)*vertex[index.x] +
                          u*vertex[index.y] +
                          v*vertex[index.z];
    float3 lightDir = normalize(lightPos - intersection);
    float diffuseComp = diffuse*max(dot(normal, lightDir), 0.0f);

    float3 viewDir = normalize(camera.position - intersection);
    float3 reflectDir = reflect(-lightDir, normal);
    float specularComp = specular*pow(max(dot(viewDir, reflectDir), 0.0f), shininess);

    prd = (ambient + diffuseComp)*(color + specularComp);
}

}
