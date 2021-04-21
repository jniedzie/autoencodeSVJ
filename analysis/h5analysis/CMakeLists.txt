cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project (h5analysis)

list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS})

find_package(ROOT REQUIRED COMPONENTS Minuit Eve Geom)
include(${ROOT_USE_FILE})

find_package(HDF5)
if(HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIR})
    set(_hdf5_libs hdf5 hdf5_cpp)
else()
    # we did not find it so we fetch it from the web, configure, compile, and link
    include(ExternalProject)
    set(ExternalProjectCMakeArgs
        -DHDF5_BUILD_CPP_LIB=ON
        )
    set(_hdf5_version "1.8.15-patch1")
    file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/hdf5_local/src)
    ExternalProject_Add(hdf5_local
        DOWNLOAD_COMMAND curl http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-${_hdf5_version}.tar.gz | tar xvz -C ${PROJECT_BINARY_DIR}/hdf5_local/src
        CMAKE_ARGS ${ExternalProjectCMakeArgs}
        SOURCE_DIR ${PROJECT_BINARY_DIR}/hdf5_local/src/hdf5-${_hdf5_version}
        BINARY_DIR ${PROJECT_BINARY_DIR}/hdf5_local/build
        STAMP_DIR  ${PROJECT_BINARY_DIR}/hdf5_local/stamp
        TMP_DIR    ${PROJECT_BINARY_DIR}/hdf5_local/tmp
        INSTALL_COMMAND ""
        )
    include_directories(
        ${PROJECT_BINARY_DIR}/hdf5_local/src/hdf5-${_hdf5_version}/src
        ${PROJECT_BINARY_DIR}/hdf5_local/src/hdf5-${_hdf5_version}/c++/src
        ${PROJECT_BINARY_DIR}/hdf5_local/build
        )
    set(_hdf5_libs
        ${PROJECT_BINARY_DIR}/hdf5_local/build/bin/libhdf5_cpp.a
        ${PROJECT_BINARY_DIR}/hdf5_local/build/bin/libhdf5.a
        -ldl
        )
endif()

SET(CMAKE_SKIP_BUILD_RPATH FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE) 
SET(CMAKE_INSTALL_RPATH "/cvmfs/sft.cern.ch/lcg/releases/vdt/0.4.3-992df/x86_64-centos7-gcc8-opt/lib:/cvmfs/sft.cern.ch/lcg/releases/tbb/2019_U7-ba3eb/x86_64-centos7-gcc8-opt/lib:/cvmfs/sft.cern.ch/lcg/releases/Davix/0.7.3-d94fa/x86_64-centos7-gcc8-opt/lib64/:/cvmfs/sft.cern.ch/lcg/releases/png/1.6.37-9c2fe/x86_64-centos7-gcc8-opt/lib/:/cvmfs/sft.cern.ch/lcg/releases/GSL/2.5-32fc5/x86_64-centos7-gcc8-opt/lib/")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_executable(h5parser h5parser.cpp)

set_property(TARGET h5parser PROPERTY CXX_STANDARD 17)

include_directories("${PROJECT_SOURCE_DIR}/include")

add_library(Analysis  Event.cpp
                      EventProcessor.cpp
                      Jet.cpp
                      Constituent.cpp
)

set_property(TARGET Analysis PROPERTY CXX_STANDARD 17)

target_link_libraries(h5parser Analysis ${ROOT_LIBRARIES} ${_hdf5_libs})
