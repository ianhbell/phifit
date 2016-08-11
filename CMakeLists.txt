cmake_minimum_required(VERSION 3.0)

project(PHIFIT)

# Add CoolProp dependency
set(COOLPROP_STATIC_LIBRARY true CACHE BOOL "Use CoolProp")
add_subdirectory("${CMAKE_SOURCE_DIR}/externals/CoolProp" "CoolProp")

# Add NISTfit dependency
set(NISTFIT_STATIC_LIBRARY true CACHE BOOL "Use NISTfit")
add_subdirectory("${CMAKE_SOURCE_DIR}/externals/NISTfit" "NISTfit")

if (PHIFIT_MAIN)
    # Add fitter code
    add_executable(Main "${CMAKE_SOURCE_DIR}/src/fitter.cpp")

    # Link it to its dependent libraries
    target_link_libraries (Main CoolProp NISTfit)
else()
    message(FATAL_ERROR "Must select a module: PHIFIT_MAIN")
endif()

