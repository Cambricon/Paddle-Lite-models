# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

cmake_minimum_required(VERSION 3.0)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -DNDEBUG")
# set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++11 -g -DDEBUG")

include(mlu)
include(Dependencies)

add_subdirectory(third_party/googletest)
add_subdirectory(classification)
#add_subdirectory(detection)
