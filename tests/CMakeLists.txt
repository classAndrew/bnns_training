# Testing library
include(FetchContent)
FetchContent_Declare(
  googletest
  # Specify the commit you depend on and update it regularly.
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_library(APPLE_ACCELERATE Accelerate)
if (NOT APPLE_ACCELERATE)
    message(FATAL_ERROR "Apple Accelerate not found")
endif()

# Tests need to be added as executables first
add_executable(testlib testlib.cc)

# Should be linked to the main library
target_link_libraries(testlib PRIVATE bnns_training gtest_main ${APPLE_ACCELERATE})

# If you register a test, then ctest and make test will run it.
# You can also run examples and check the output, as well.
# add_test(NAME testlibtest COMMAND testlib) # Command can be a target
