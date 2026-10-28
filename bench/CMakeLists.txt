add_executable(dpd_bench bench_main.cpp)
target_link_libraries(dpd_bench PRIVATE dpd_dust dpd_options)
