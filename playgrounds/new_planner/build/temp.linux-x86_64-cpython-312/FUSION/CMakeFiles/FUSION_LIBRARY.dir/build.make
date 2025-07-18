# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/aligoles/wpi/research/icra2025/playgrounds/new_planner

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/aligoles/wpi/research/icra2025/playgrounds/new_planner/build/temp.linux-x86_64-cpython-312/FUSION

# Include any dependencies generated for this target.
include CMakeFiles/FUSION_LIBRARY.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/FUSION_LIBRARY.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/FUSION_LIBRARY.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/FUSION_LIBRARY.dir/flags.make

CMakeFiles/FUSION_LIBRARY.dir/fusion.cpp.o: CMakeFiles/FUSION_LIBRARY.dir/flags.make
CMakeFiles/FUSION_LIBRARY.dir/fusion.cpp.o: ../../../fusion.cpp
CMakeFiles/FUSION_LIBRARY.dir/fusion.cpp.o: CMakeFiles/FUSION_LIBRARY.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aligoles/wpi/research/icra2025/playgrounds/new_planner/build/temp.linux-x86_64-cpython-312/FUSION/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/FUSION_LIBRARY.dir/fusion.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/FUSION_LIBRARY.dir/fusion.cpp.o -MF CMakeFiles/FUSION_LIBRARY.dir/fusion.cpp.o.d -o CMakeFiles/FUSION_LIBRARY.dir/fusion.cpp.o -c /home/aligoles/wpi/research/icra2025/playgrounds/new_planner/fusion.cpp

CMakeFiles/FUSION_LIBRARY.dir/fusion.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FUSION_LIBRARY.dir/fusion.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aligoles/wpi/research/icra2025/playgrounds/new_planner/fusion.cpp > CMakeFiles/FUSION_LIBRARY.dir/fusion.cpp.i

CMakeFiles/FUSION_LIBRARY.dir/fusion.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FUSION_LIBRARY.dir/fusion.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aligoles/wpi/research/icra2025/playgrounds/new_planner/fusion.cpp -o CMakeFiles/FUSION_LIBRARY.dir/fusion.cpp.s

CMakeFiles/FUSION_LIBRARY.dir/utils.cpp.o: CMakeFiles/FUSION_LIBRARY.dir/flags.make
CMakeFiles/FUSION_LIBRARY.dir/utils.cpp.o: ../../../utils.cpp
CMakeFiles/FUSION_LIBRARY.dir/utils.cpp.o: CMakeFiles/FUSION_LIBRARY.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aligoles/wpi/research/icra2025/playgrounds/new_planner/build/temp.linux-x86_64-cpython-312/FUSION/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/FUSION_LIBRARY.dir/utils.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/FUSION_LIBRARY.dir/utils.cpp.o -MF CMakeFiles/FUSION_LIBRARY.dir/utils.cpp.o.d -o CMakeFiles/FUSION_LIBRARY.dir/utils.cpp.o -c /home/aligoles/wpi/research/icra2025/playgrounds/new_planner/utils.cpp

CMakeFiles/FUSION_LIBRARY.dir/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FUSION_LIBRARY.dir/utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aligoles/wpi/research/icra2025/playgrounds/new_planner/utils.cpp > CMakeFiles/FUSION_LIBRARY.dir/utils.cpp.i

CMakeFiles/FUSION_LIBRARY.dir/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FUSION_LIBRARY.dir/utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aligoles/wpi/research/icra2025/playgrounds/new_planner/utils.cpp -o CMakeFiles/FUSION_LIBRARY.dir/utils.cpp.s

# Object files for target FUSION_LIBRARY
FUSION_LIBRARY_OBJECTS = \
"CMakeFiles/FUSION_LIBRARY.dir/fusion.cpp.o" \
"CMakeFiles/FUSION_LIBRARY.dir/utils.cpp.o"

# External object files for target FUSION_LIBRARY
FUSION_LIBRARY_EXTERNAL_OBJECTS =

libFUSION_LIBRARY.a: CMakeFiles/FUSION_LIBRARY.dir/fusion.cpp.o
libFUSION_LIBRARY.a: CMakeFiles/FUSION_LIBRARY.dir/utils.cpp.o
libFUSION_LIBRARY.a: CMakeFiles/FUSION_LIBRARY.dir/build.make
libFUSION_LIBRARY.a: CMakeFiles/FUSION_LIBRARY.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/aligoles/wpi/research/icra2025/playgrounds/new_planner/build/temp.linux-x86_64-cpython-312/FUSION/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libFUSION_LIBRARY.a"
	$(CMAKE_COMMAND) -P CMakeFiles/FUSION_LIBRARY.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FUSION_LIBRARY.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/FUSION_LIBRARY.dir/build: libFUSION_LIBRARY.a
.PHONY : CMakeFiles/FUSION_LIBRARY.dir/build

CMakeFiles/FUSION_LIBRARY.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/FUSION_LIBRARY.dir/cmake_clean.cmake
.PHONY : CMakeFiles/FUSION_LIBRARY.dir/clean

CMakeFiles/FUSION_LIBRARY.dir/depend:
	cd /home/aligoles/wpi/research/icra2025/playgrounds/new_planner/build/temp.linux-x86_64-cpython-312/FUSION && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/aligoles/wpi/research/icra2025/playgrounds/new_planner /home/aligoles/wpi/research/icra2025/playgrounds/new_planner /home/aligoles/wpi/research/icra2025/playgrounds/new_planner/build/temp.linux-x86_64-cpython-312/FUSION /home/aligoles/wpi/research/icra2025/playgrounds/new_planner/build/temp.linux-x86_64-cpython-312/FUSION /home/aligoles/wpi/research/icra2025/playgrounds/new_planner/build/temp.linux-x86_64-cpython-312/FUSION/CMakeFiles/FUSION_LIBRARY.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/FUSION_LIBRARY.dir/depend

