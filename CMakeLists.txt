# CMakeLists template provided by https://gitlab.com/CLIUtils/modern-cmake/-/tree/master/examples/extended-project/

cmake_minimum_required(VERSION 3.14...3.27)
 
project(
  bnns_training
  VERSION 1.0
  DESCRIPTION "Wrapper library for Apple's Basic Neural Network Subprograms' deep learning primitives.")

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(CMAKE_CXX_EXTENSIONS OFF)

  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  include(CTest)

  # Docs only available if this is the main app
  find_package(Doxygen)
  if(Doxygen_FOUND)
    add_subdirectory(docs)
  else()
    message(STATUS "Doxygen not found, not building docs")
  endif()
endif()

include(FetchContent)

# The compiled library code is here
add_subdirectory(src)

# The executable code is here
# add_subdirectory(apps)

if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
   AND BUILD_TESTING)
  add_subdirectory(tests)
endif()
