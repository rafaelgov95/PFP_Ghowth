# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/clion-2018.1.1/bin/cmake/bin/cmake

# The command to remove a file.
RM = /opt/clion-2018.1.1/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/rafael/Downloads/PFP_Ghowth-dc

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rafael/Downloads/PFP_Ghowth-dc/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/cpu_library_fp_growth.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cpu_library_fp_growth.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cpu_library_fp_growth.dir/flags.make

CMakeFiles/cpu_library_fp_growth.dir/src/PFPArray.cpp.o: CMakeFiles/cpu_library_fp_growth.dir/flags.make
CMakeFiles/cpu_library_fp_growth.dir/src/PFPArray.cpp.o: ../src/PFPArray.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rafael/Downloads/PFP_Ghowth-dc/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cpu_library_fp_growth.dir/src/PFPArray.cpp.o"
	/usr/bin/g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cpu_library_fp_growth.dir/src/PFPArray.cpp.o -c /home/rafael/Downloads/PFP_Ghowth-dc/src/PFPArray.cpp

CMakeFiles/cpu_library_fp_growth.dir/src/PFPArray.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpu_library_fp_growth.dir/src/PFPArray.cpp.i"
	/usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rafael/Downloads/PFP_Ghowth-dc/src/PFPArray.cpp > CMakeFiles/cpu_library_fp_growth.dir/src/PFPArray.cpp.i

CMakeFiles/cpu_library_fp_growth.dir/src/PFPArray.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpu_library_fp_growth.dir/src/PFPArray.cpp.s"
	/usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rafael/Downloads/PFP_Ghowth-dc/src/PFPArray.cpp -o CMakeFiles/cpu_library_fp_growth.dir/src/PFPArray.cpp.s

CMakeFiles/cpu_library_fp_growth.dir/src/PFPArray.cpp.o.requires:

.PHONY : CMakeFiles/cpu_library_fp_growth.dir/src/PFPArray.cpp.o.requires

CMakeFiles/cpu_library_fp_growth.dir/src/PFPArray.cpp.o.provides: CMakeFiles/cpu_library_fp_growth.dir/src/PFPArray.cpp.o.requires
	$(MAKE) -f CMakeFiles/cpu_library_fp_growth.dir/build.make CMakeFiles/cpu_library_fp_growth.dir/src/PFPArray.cpp.o.provides.build
.PHONY : CMakeFiles/cpu_library_fp_growth.dir/src/PFPArray.cpp.o.provides

CMakeFiles/cpu_library_fp_growth.dir/src/PFPArray.cpp.o.provides.build: CMakeFiles/cpu_library_fp_growth.dir/src/PFPArray.cpp.o


CMakeFiles/cpu_library_fp_growth.dir/src/PFPTree.cpp.o: CMakeFiles/cpu_library_fp_growth.dir/flags.make
CMakeFiles/cpu_library_fp_growth.dir/src/PFPTree.cpp.o: ../src/PFPTree.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rafael/Downloads/PFP_Ghowth-dc/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/cpu_library_fp_growth.dir/src/PFPTree.cpp.o"
	/usr/bin/g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cpu_library_fp_growth.dir/src/PFPTree.cpp.o -c /home/rafael/Downloads/PFP_Ghowth-dc/src/PFPTree.cpp

CMakeFiles/cpu_library_fp_growth.dir/src/PFPTree.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpu_library_fp_growth.dir/src/PFPTree.cpp.i"
	/usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rafael/Downloads/PFP_Ghowth-dc/src/PFPTree.cpp > CMakeFiles/cpu_library_fp_growth.dir/src/PFPTree.cpp.i

CMakeFiles/cpu_library_fp_growth.dir/src/PFPTree.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpu_library_fp_growth.dir/src/PFPTree.cpp.s"
	/usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rafael/Downloads/PFP_Ghowth-dc/src/PFPTree.cpp -o CMakeFiles/cpu_library_fp_growth.dir/src/PFPTree.cpp.s

CMakeFiles/cpu_library_fp_growth.dir/src/PFPTree.cpp.o.requires:

.PHONY : CMakeFiles/cpu_library_fp_growth.dir/src/PFPTree.cpp.o.requires

CMakeFiles/cpu_library_fp_growth.dir/src/PFPTree.cpp.o.provides: CMakeFiles/cpu_library_fp_growth.dir/src/PFPTree.cpp.o.requires
	$(MAKE) -f CMakeFiles/cpu_library_fp_growth.dir/build.make CMakeFiles/cpu_library_fp_growth.dir/src/PFPTree.cpp.o.provides.build
.PHONY : CMakeFiles/cpu_library_fp_growth.dir/src/PFPTree.cpp.o.provides

CMakeFiles/cpu_library_fp_growth.dir/src/PFPTree.cpp.o.provides.build: CMakeFiles/cpu_library_fp_growth.dir/src/PFPTree.cpp.o


# Object files for target cpu_library_fp_growth
cpu_library_fp_growth_OBJECTS = \
"CMakeFiles/cpu_library_fp_growth.dir/src/PFPArray.cpp.o" \
"CMakeFiles/cpu_library_fp_growth.dir/src/PFPTree.cpp.o"

# External object files for target cpu_library_fp_growth
cpu_library_fp_growth_EXTERNAL_OBJECTS =

libcpu_library_fp_growth.a: CMakeFiles/cpu_library_fp_growth.dir/src/PFPArray.cpp.o
libcpu_library_fp_growth.a: CMakeFiles/cpu_library_fp_growth.dir/src/PFPTree.cpp.o
libcpu_library_fp_growth.a: CMakeFiles/cpu_library_fp_growth.dir/build.make
libcpu_library_fp_growth.a: CMakeFiles/cpu_library_fp_growth.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/rafael/Downloads/PFP_Ghowth-dc/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libcpu_library_fp_growth.a"
	$(CMAKE_COMMAND) -P CMakeFiles/cpu_library_fp_growth.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cpu_library_fp_growth.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cpu_library_fp_growth.dir/build: libcpu_library_fp_growth.a

.PHONY : CMakeFiles/cpu_library_fp_growth.dir/build

CMakeFiles/cpu_library_fp_growth.dir/requires: CMakeFiles/cpu_library_fp_growth.dir/src/PFPArray.cpp.o.requires
CMakeFiles/cpu_library_fp_growth.dir/requires: CMakeFiles/cpu_library_fp_growth.dir/src/PFPTree.cpp.o.requires

.PHONY : CMakeFiles/cpu_library_fp_growth.dir/requires

CMakeFiles/cpu_library_fp_growth.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cpu_library_fp_growth.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cpu_library_fp_growth.dir/clean

CMakeFiles/cpu_library_fp_growth.dir/depend:
	cd /home/rafael/Downloads/PFP_Ghowth-dc/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rafael/Downloads/PFP_Ghowth-dc /home/rafael/Downloads/PFP_Ghowth-dc /home/rafael/Downloads/PFP_Ghowth-dc/cmake-build-debug /home/rafael/Downloads/PFP_Ghowth-dc/cmake-build-debug /home/rafael/Downloads/PFP_Ghowth-dc/cmake-build-debug/CMakeFiles/cpu_library_fp_growth.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cpu_library_fp_growth.dir/depend

