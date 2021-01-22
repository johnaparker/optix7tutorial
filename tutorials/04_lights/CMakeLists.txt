set(NAME 04_lights)
cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)

add_executable(${NAME}
  ${embedded_ptx_code}
  scene.h
  scene.cpp
  main.cpp
  )

target_link_libraries(${NAME} ${LIBS} glfWindow sutil_7_sdk)
