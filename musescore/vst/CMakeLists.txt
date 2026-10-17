# Acappella Render VST3 effects (see player.cpp). Built by ../install.sh:
#   cmake -S . -B build -DVST3_SDK_DIR=/path/to/vst3sdk && cmake --build build --config Release
cmake_minimum_required(VERSION 3.25)

set(VST3_SDK_DIR "" CACHE PATH "Path to a checkout of https://github.com/steinbergmedia/vst3sdk")
if(NOT EXISTS "${VST3_SDK_DIR}/pluginterfaces")
    message(FATAL_ERROR "Pass -DVST3_SDK_DIR=/path/to/vst3sdk")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(SMTG_ENABLE_VST3_PLUGIN_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SMTG_ENABLE_VSTGUI_SUPPORT OFF CACHE BOOL "" FORCE)
set(SMTG_CREATE_PLUGIN_LINK OFF CACHE BOOL "" FORCE)
set(vst3sdk_SOURCE_DIR "${VST3_SDK_DIR}")

# The SDK asks xcodebuild for the Xcode version, which fails with only the
# Command Line Tools installed; their compilers are all this needs.
if(APPLE AND NOT DEFINED ENV{XCODE_VERSION})
    execute_process(COMMAND xcodebuild -version RESULT_VARIABLE _no_xcode OUTPUT_QUIET ERROR_QUIET)
    if(_no_xcode)
        set(ENV{XCODE_VERSION} 16.0)
        set(XCODE_VERSION 16.0)
    endif()
endif()

project(AcappellaRender VERSION 1.0.0)

add_subdirectory("${VST3_SDK_DIR}" "${PROJECT_BINARY_DIR}/vst3sdk")
smtg_enable_vst3_sdk()

# the master-bus player, and the effect for a part's channel (the same code,
# built with ACAPPELLA_PART): MuseScore only loads one effect from a bundle
foreach(target AcappellaRender AcappellaRenderPart)
    smtg_add_vst3plugin(${target}
        player.cpp
        "${VST3_SDK_DIR}/public.sdk/source/vst/vstsinglecomponenteffect.cpp")
    target_link_libraries(${target} PRIVATE sdk)
    target_compile_features(${target} PRIVATE cxx_std_17)
endforeach()
target_compile_definitions(AcappellaRenderPart PRIVATE ACAPPELLA_PART=1)
smtg_target_set_bundle(AcappellaRender
    BUNDLE_IDENTIFIER "com.github.acappella-render.player"
    COMPANY_NAME "acappella-render")
smtg_target_set_bundle(AcappellaRenderPart
    BUNDLE_IDENTIFIER "com.github.acappella-render.part"
    COMPANY_NAME "acappella-render")
