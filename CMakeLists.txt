cmake_minimum_required(VERSION 3.15)
project(asc_python C CXX)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags for different build types
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -Wall -Wextra -pedantic")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g -Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra -pedantic")

# Find Python
find_package(Python3 REQUIRED COMPONENTS Development NumPy)

# NumPy include directory
set(NUMPY_INCLUDE_DIR "${Python3_NumPy_INCLUDE_DIRS}")

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIR}
)

# Add the Python extension module
Python3_add_library(asc_python MODULE
    src/asc_module.cpp
)

# Link against canasc (static) and vector_dbc (dynamic) libraries
target_link_libraries(asc_python PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/libcanasc.a
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/libVector_DBC.dll.a
    -Wl,-Bstatic
    -lstdc++
    -lwinpthread
    -Wl,-Bdynamic
    -static-libgcc
)

# Disable cast-function-type warning for Python C API compatibility
target_compile_options(asc_python PRIVATE -Wno-cast-function-type)

# Set output directory
set_target_properties(asc_python PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/asc_python
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/asc_python
)

# Remove the 'lib' prefix on Windows if present
set_target_properties(asc_python PROPERTIES PREFIX "")

# Ensure .pyd extension on Windows
if(WIN32)
    set_target_properties(asc_python PROPERTIES SUFFIX ".pyd")
endif()

# Copy required DLLs to asc_python directory for Python module to load
add_custom_command(TARGET asc_python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/libVector_DBC.dll
        ${CMAKE_CURRENT_SOURCE_DIR}/asc_python/libVector_DBC.dll
    COMMENT "Copying DLLs to asc_python directory"
)
