# Host toolchain for ggml's vulkan-shaders-gen sub-build when cross-compiling
# for Android on a Windows host (see the android-arm64 preset). Run the build
# from a Visual Studio x64 developer environment (vcvars64.bat) so cl.exe and
# the Windows SDK are on PATH/INCLUDE/LIB; the Android targets themselves keep
# the NDK toolchain.
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_C_COMPILER cl)
set(CMAKE_CXX_COMPILER cl)
