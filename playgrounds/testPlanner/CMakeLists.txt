cmake_minimum_required(VERSION 3.10)

project(testPlanner)

set(CMAKE_CXX_STANDARD 11)

find_package(ompl REQUIRED)

include_directories(${OMPL_INCLUDE_DIRS})

add_library(MY_LIBRARY fusion.cpp)
target_link_libraries(MY_LIBRARY ${OMPL_LIBRARIES})

set_target_properties(MY_LIBRARY PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Check if we're being called from setup.py (Python module build)
if(DEFINED PYTHON_EXECUTABLE)
    find_package(pybind11 REQUIRED)
    
    # Create Python module
    pybind11_add_module(my_custom_planner_module python_bindings.cpp)
    
    target_link_libraries(my_custom_planner_module PRIVATE MY_LIBRARY ${OMPL_LIBRARIES})
    
    target_link_libraries(my_custom_planner_module PRIVATE ompl)
    
    target_include_directories(my_custom_planner_module PRIVATE ${OMPL_INCLUDE_DIRS})
    
    target_compile_definitions(my_custom_planner_module PRIVATE VERSION_INFO="0.1.0")
else()
    add_executable(testPlanner main.cpp)
    target_link_libraries(testPlanner MY_LIBRARY ${OMPL_LIBRARIES} ompl)
endif()