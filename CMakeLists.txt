CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
PROJECT(AutoEncoder)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#source directory
AUX_SOURCE_DIRECTORY(src DIR_SRCS)
#test directory
AUX_SOURCE_DIRECTORY(test DIR_TEST)

ADD_EXECUTABLE(AutoEncoder ${DIR_TEST} ${DIR_SRCS})

#add link library
TARGET_LINK_LIBRARIES(AutoEncoder gtest gtest_main -lpthread)