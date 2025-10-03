cmake_minimum_required(VERSION 3.22)
project(Geant4Simulations LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Geant4 installation
find_package(Geant4 REQUIRED ui_all vis_all)

# Include Geant4 use file to set up compiler flags and definitions
include(${Geant4_USE_FILE})

# Add subdirectories
add_subdirectory(projects/example1)
