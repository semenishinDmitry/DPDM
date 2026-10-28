add_executable(dpd_tests
  test_kernels.cpp test_linalg.cpp test_explicit.cpp test_bdf.cpp test_auto.cpp
  test_failures.cpp test_events.cpp test_dust.cpp test_paper.cpp test_config.cpp)
target_link_libraries(dpd_tests PRIVATE dpd_dust Catch2::Catch2WithMain dpd_options)
target_compile_definitions(dpd_tests PRIVATE DPD_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

# Allocation audit: replaces global operator new, so it is a separate binary.
add_executable(dpd_alloc_audit test_alloc.cpp)
target_link_libraries(dpd_alloc_audit PRIVATE dpd_dust dpd_options)

include(Catch)
catch_discover_tests(dpd_tests PROPERTIES LABELS "unit")
add_test(NAME alloc_audit COMMAND dpd_alloc_audit)
set_tests_properties(alloc_audit PROPERTIES LABELS "unit")
if(DPD_BUILD_APPS)
  add_test(NAME cli_reference_case
           COMMAND dustppd run ${PROJECT_SOURCE_DIR}/configs/reference/paper_1m_100au.toml
                   -o ${CMAKE_CURRENT_BINARY_DIR}/ref.csv -s ${CMAKE_CURRENT_BINARY_DIR}/ref.json)
  set_tests_properties(cli_reference_case PROPERTIES LABELS "integration")
  add_test(NAME cli_bad_config COMMAND dustppd run ${PROJECT_SOURCE_DIR}/configs/reference/paper_1m_100au.toml
                                       --set physics.nonexistent=1)
  set_tests_properties(cli_bad_config PROPERTIES WILL_FAIL TRUE LABELS "integration")
endif()
