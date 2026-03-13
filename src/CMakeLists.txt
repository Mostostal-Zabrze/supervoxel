find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core supervoxel_pybind.cpp)

target_include_directories(_core PRIVATE ${CMAKE_SOURCE_DIR})

target_compile_features(_core PRIVATE cxx_std_11)

if(MSVC)
    target_compile_options(_core PRIVATE /W3)
else()
    target_compile_options(_core PRIVATE -Wall -Wextra)
endif()

install(TARGETS _core DESTINATION supervoxel)
