cmake_minimum_required(VERSION 3.20)

project(GEBP VERSION 0.1.0 LANGUAGES CXX)

add_executable(gebp
    src/main.cpp
)

target_include_directories(gebp PRIVATE include)
target_compile_features(gebp PRIVATE cxx_std_20)

set_target_properties(gebp PROPERTIES
    CXX_EXTENSIONS OFF
)
