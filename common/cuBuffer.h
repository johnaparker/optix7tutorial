#pragma once

#include "optix7.h"
#include <vector>
#include <assert.h>

namespace optix7tutorial {

  struct cuBuffer {
    inline CUdeviceptr d_pointer() const { 
        return (CUdeviceptr) d_ptr;
    }

    void resize(size_t size) {
      if (d_ptr) 
          free();
      alloc(size);
    }
    
    void alloc(size_t size) {
      assert(d_ptr == nullptr);
      size_in_bytes = size;
      CUDA_CHECK(Malloc( (void**)&d_ptr, size_in_bytes));
    }

    void free() {
      CUDA_CHECK(Free(d_ptr));
      d_ptr = nullptr;
      size_in_bytes = 0;
    }

    template<typename T>
    void alloc_and_copy_to_device(const std::vector<T> &vec) {
      alloc(vec.size()*sizeof(T));
      copy_to_device((const T*)vec.data(), vec.size());
    }
    
    template<typename T>
    void copy_to_device(const T *t, size_t count) {
      assert(d_ptr != nullptr);
      assert(size_in_bytes == count*sizeof(T));
      CUDA_CHECK(Memcpy(d_ptr, (void *)t,
                        count*sizeof(T), cudaMemcpyHostToDevice));
    }
    
    template<typename T>
    void copy_from_device(T *t, size_t count) {
      assert(d_ptr != nullptr);
      assert(size_in_bytes == count*sizeof(T));
      CUDA_CHECK(Memcpy((void *)t, d_ptr,
                        count*sizeof(T), cudaMemcpyDeviceToHost));
    }
    
    size_t size_in_bytes {0};
    void *d_ptr {nullptr};
  };
}
