# run_tests.py  (place next to the distnet_2d/ folder)
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from distnet_2d.model.window_spatial_attention import (
    WindowSpatialAttention,
    test_coordinate_handling,
    test_window_coverage_with_geometrical,
    test_multiquery_equivalence,
    test_edge_cases,
    test_attention_computation_details,
    test_window_processing_modes_equivalence,
    test_window_processing_memory_profile,
)

for mode in ['3d']: # , '2d'
    test_coordinate_handling(multi_query=True, mode=mode)
    test_window_coverage_with_geometrical(multi_query=True, mode=mode)
    test_multiquery_equivalence(mode=mode)
    test_edge_cases(mode=mode)
    test_attention_computation_details(mode=mode)
    test_window_processing_modes_equivalence(mode=mode)
    test_window_processing_memory_profile(mode=mode)