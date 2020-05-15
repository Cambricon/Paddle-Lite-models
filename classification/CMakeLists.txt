file(GLOB_RECURSE examples_srcs "${PROJECT_SOURCE_DIR}/classification/*.cpp")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/third-party/googletest/include/)
include_directories(${PROJECT_SOURCE_DIR}/core/include/)
foreach(source_file ${examples_srcs})
 
  # get file name
  get_filename_component(name ${source_file} NAME_WE)
  add_executable("${name}" ${source_file})
  target_link_libraries("${name}" gtest paddle_lib ${OpenCV_LIBS} cnml_lib cnrt_lib)
  target_compile_options(${name} PRIVATE -DLITE_WITH_MLU -DLITE_WITH_X86)
endforeach()

