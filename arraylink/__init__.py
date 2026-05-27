"""ArrayLink near-field LoS MIMO simulator."""
from .channel import (
    compute_channel_matrix,
    compute_singular_values,
    singular_value_ratio,
    degrees_of_freedom,
    spectral_efficiency,
    mimo_region_bounds,
    theoretical_singular_value_ratio,
)
from .array_geometry import (
    upa_positions,
    center_dense_positions,
    uniform_grid_positions,
    place_subarrays,
    build_ground_station,
)
from .beamforming import (
    parabolic_gain_dbi,
    total_array_gain_dbi,
    marginal_gain_dbi,
    dc_weights,
    compute_distances_batched,
    compute_beam_pattern,
    compute_beam_pattern_numpy,
)
from .utils import (
    spherical2cartesian,
    cartesian2spherical,
    linear2db,
    db2linear,
    generate_grid_points,
)
