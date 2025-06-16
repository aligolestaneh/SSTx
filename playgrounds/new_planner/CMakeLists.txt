cmake_minimum_required(VERSION 3.10)

project(new_planner)

set(CMAKE_CXX_STANDARD 11)

find_package(ompl REQUIRED)

include_directories(${OMPL_INCLUDE_DIRS})

add_library(FUSION_LIBRARY fusion.cpp utils.cpp)
target_link_libraries(FUSION_LIBRARY ${OMPL_LIBRARIES})

set_target_properties(FUSION_LIBRARY PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Check if we're being called from setup.py (Python module build)
if(DEFINED PYTHON_EXECUTABLE)
    find_package(pybind11 REQUIRED)
    
    # Create Python module
    pybind11_add_module(FUSION python_bindings.cpp)
    
    target_link_libraries(FUSION PRIVATE FUSION_LIBRARY ${OMPL_LIBRARIES})
    
    target_link_libraries(FUSION PRIVATE ompl)
    
    target_include_directories(FUSION PRIVATE ${OMPL_INCLUDE_DIRS})
    
    # Compiler-specific options for the module
    target_compile_definitions(FUSION PRIVATE VERSION_INFO="0.1.0")
else()
    add_executable(new_planner main.cpp)
    target_link_libraries(new_planner FUSION_LIBRARY ${OMPL_LIBRARIES} ompl)
    
    # Test executable for logging
    add_executable(test_logging test_logging.cpp utils.cpp)
endif() 