file(GLOB_RECURSE examples_srcs "${PROJECT_SOURCE_DIR}/classification/*.cpp")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
foreach(source_file ${examples_srcs})
    
  # get file name
  get_filename_component(name ${source_file} NAME_WE)
  add_executable("${name}" ${source_file})
  target_link_libraries("${name}" ${OpenCV_LIBS} paddle_lib cnml_lib cnrt_lib)

endforeach()

