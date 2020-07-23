cmake_minimum_required (VERSION 2.8.11)
project (mcts-hanabi)

set(CMAKE_C_FLAGS "-O2 -std=c++11 -fPIC")
set(CMAKE_CXX_FLAGS "-O2 -std=c++11 -fPIC")

add_subdirectory (hanabi_lib)

add_library (pyhanabi SHARED pyhanabi.cc)
target_link_libraries (pyhanabi LINK_PUBLIC hanabi)

install(TARGETS pyhanabi LIBRARY DESTINATION mcts-hanabi)
install(FILES __init__.py DESTINATION mcts-hanabi)
install(FILES rl_env.py DESTINATION mcts-hanabi)
install(FILES pyhanabi.py DESTINATION mcts-hanabi)
install(FILES pyhanabi.h DESTINATION mcts-hanabi)
