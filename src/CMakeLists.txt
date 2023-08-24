# Note that headers are optional, and do not affect add_library, but they will not
# show up in IDEs unless they are listed in add_library.

# Optionally glob, but only for CMake 3.12 or later:
# file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${ModernCMakeExample_SOURCE_DIR}/include/modern/*.hpp")
set(HEADER_LIST "${bnns_training_SOURCE_DIR}/include/bnns_training/data_loader.h")
set(HEADER_LIST "${bnns_training_SOURCE_DIR}/include/bnns_training/dense_layer.h")
set(HEADER_LIST "${bnns_training_SOURCE_DIR}/include/bnns_training/loss_layer.h")
set(HEADER_LIST "${bnns_training_SOURCE_DIR}/include/bnns_training/mlp.h")
set(HEADER_LIST "${bnns_training_SOURCE_DIR}/include/bnns_training/output_layer.h")

# Make an automatic library - will be static or dynamic based on user setting
add_library(bnns_training lib.c)

# We need this directory, and users of our library will need it too
target_include_directories(bnns_training PUBLIC ../include)

# This depends on (header only) boost
# target_link_libraries(modern_library PRIVATE Boost::boost)

# All users of this library will need at least C++11
# target_compile_features(bnns_training PUBLIC cxx_std_11)

# IDEs should put the headers in a nice place
source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST})
