cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(MLTrackFinding LANGUAGES CXX)

set( ${PROJECT_NAME}_VERSION_MAJOR 0 )
set( ${PROJECT_NAME}_VERSION_MINOR 0 )
set( ${PROJECT_NAME}_VERSION_PATCH 99 )

set( ${PROJECT_NAME}_VERSION  "${${PROJECT_NAME}_VERSION_MAJOR}.${${PROJECT_NAME}_VERSION_MINOR}.${${PROJECT_NAME}_VERSION_PATCH}" )

# dependencies
# For some reason, the find_package with DD4hep must be the first one
find_package(DD4hep REQUIRED COMPONENTS DDRec DDG4 DDParsers)
# dd4hep_set_compiler_flags() expects DD4hep_SET_RPATH to be set to ON
# otherwise it will not set the rpath when installing
set(DD4HEP_SET_RPATH ON)
dd4hep_set_compiler_flags()
find_package(k4FWCore 1.3 REQUIRED)
find_package(EDM4HEP)
find_package(onnxruntime REQUIRED)

include(cmake/Key4hepConfig.cmake)
include(GNUInstallDirs)

add_subdirectory(TrackFinding)

include(CTest)
set(USE_EXTERNAL_CATCH2 AUTO CACHE STRING "Link against an external Catch2 v3 static library, otherwise build it locally")
set_property(CACHE USE_EXTERNAL_CATCH2 PROPERTY STRINGS AUTO ON OFF)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
