find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
if(NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
  find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
endif()
message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
include_directories("${OpenCV_INCLUDE_DIRS}")

set(PADDLE_LINK_PATH $ENV{PADDLE_LINK_PATH})
set(PADDLE_INC_PATH $ENV{PADDLE_INC_PATH})
include_directories("${PADDLE_INC_PATH}")
find_library(PADDLE_LIB_FILE NAMES paddle_full_api_shared
	PATHS ${PADDLE_LINK_PATH})
add_library(paddle_lib SHARED IMPORTED GLOBAL)
set_property(TARGET paddle_lib PROPERTY IMPORTED_LOCATION ${PADDLE_LIB_FILE})
