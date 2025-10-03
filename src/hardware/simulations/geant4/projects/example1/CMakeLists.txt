project(Example1)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hh)

# Create executable
add_executable(example1 main.cc ${sources} ${headers})

# Link Geant4 libraries
target_link_libraries(example1 ${Geant4_LIBRARIES})

# Copy macros to build directory
set(MACROS
    init.mac
    vis.mac
    run1.mac
)

foreach(macro ${MACROS})
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/macros/${macro}
        ${CMAKE_CURRENT_BINARY_DIR}/${macro}
        COPYONLY
    )
endforeach()

# Install
install(TARGETS example1 DESTINATION bin)
install(FILES ${MACROS} DESTINATION bin)
