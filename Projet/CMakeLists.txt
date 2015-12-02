CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

FIND_PACKAGE(OpenCV REQUIRED)

PROJECT(Projet)

ADD_EXECUTABLE(GC graphCuts.cpp maxflow/graph.cpp image.cpp disparite.cpp image.h)
TARGET_LINK_LIBRARIES(GC ${OpenCV_LIBS})
