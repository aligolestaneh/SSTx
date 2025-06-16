import os
import sys
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # The directory where the compiled extension should be placed.
        extdir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()

        # Build configuration (Debug or Release)
        debug = (
            int(os.environ.get("DEBUG", 0))
            if self.debug is None
            else self.debug
        )
        cfg = "Debug" if debug else "Release"

        # ====================================================================
        # The Most Important Step for Your Project: Find OMPL
        # ====================================================================
        # Use the system OMPL installation
        ompl_prefix_path = Path("/usr/local")
        ompl_cmake_dir = ompl_prefix_path / "share" / "ompl" / "cmake"

        if not ompl_cmake_dir.exists():
            print(
                f"ERROR: Could not find OMPL CMake files at {ompl_cmake_dir}.",
                "Please ensure OMPL is installed correctly.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(
            f"--- Found OMPL, setting CMAKE_PREFIX_PATH to: {ompl_prefix_path}"
        )
        # ====================================================================

        # Arguments to pass to CMake
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            # Pass the path to the OMPL installation to CMake
            f"-DCMAKE_PREFIX_PATH={ompl_prefix_path}",
        ]

        # Arguments to pass to the build tool (e.g., 'make' or 'ninja')
        build_args = []
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            if hasattr(self, "parallel") and self.parallel:
                build_args += [f"-j{self.parallel}"]

        # Create the build directory if it doesn't exist
        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        # Run CMake to configure the project
        print(f"--- Running CMake with args: {cmake_args}")
        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )

        # Run the build tool to compile the extension
        print(f"--- Building extension with args: {build_args}")
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )


# The main setup() call
setup(
    name="my_custom_planner_module",  # The name of your pip package
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A custom planner for OMPL with Python bindings",
    long_description="This package provides a custom planner that integrates with OMPL data structures.",
    # Tell setuptools to use our custom CMake build step
    ext_modules=[
        # The name here MUST MATCH the first argument to pybind11_add_module()
        # and PYBIND11_MODULE() in your C++ code.
        CMakeExtension("my_custom_planner_module")
    ],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.8",
)
