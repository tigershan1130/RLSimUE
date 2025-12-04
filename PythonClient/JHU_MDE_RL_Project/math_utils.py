"""
Mathematical utility functions for drone navigation and reward calculations.
Provides common mathematical operations used in RL reward shaping.
"""

import numpy as np
from typing import Tuple, Union
import math


# ========== DISTANCE FUNCTIONS ==========

def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points"""
    return float(np.linalg.norm(p1 - p2))


def manhattan_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate Manhattan (L1) distance between two points"""
    return float(np.sum(np.abs(p1 - p2)))


def squared_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate squared Euclidean distance (faster, avoids sqrt)"""
    diff = p1 - p2
    return float(np.dot(diff, diff))


def distance_2d(p1: np.ndarray, p2: np.ndarray, axis: int = 0) -> float:
    """Calculate 2D distance (ignoring one axis, e.g., for horizontal distance ignoring Z)"""
    if axis == 0:  # Ignore X
        return float(np.linalg.norm(p1[1:] - p2[1:]))
    elif axis == 1:  # Ignore Y
        return float(np.linalg.norm([p1[0], p1[2]] - [p2[0], p2[2]]))
    elif axis == 2:  # Ignore Z
        return float(np.linalg.norm(p1[:2] - p2[:2]))
    else:
        return euclidean_distance(p1, p2)


# ========== NORMALIZATION FUNCTIONS ==========

def normalize_vector(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize a vector to unit length"""
    norm = np.linalg.norm(v)
    if norm < eps:
        return np.zeros_like(v)
    return v / norm


def normalize_to_range(value: float, min_val: float, max_val: float, 
                       new_min: float = 0.0, new_max: float = 1.0) -> float:
    """Normalize a value from [min_val, max_val] to [new_min, new_max]"""
    if max_val == min_val:
        return new_min
    normalized = (value - min_val) / (max_val - min_val)
    return new_min + normalized * (new_max - new_min)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))


def smooth_clamp(value: float, min_val: float, max_val: float, 
                 smoothness: float = 0.1) -> float:
    """Smooth clamping using sigmoid-like function"""
    center = (min_val + max_val) / 2.0
    width = (max_val - min_val) / 2.0
    x = (value - center) / (width * smoothness)
    return center + width * np.tanh(x)


# ========== REWARD SHAPING FUNCTIONS ==========

def exponential_reward(distance: float, max_distance: float, 
                       scale: float = 1.0, exponent: float = 2.0) -> float:
    """
    Exponential reward function: reward = scale * (1 - distance/max_distance)^exponent
    
    Args:
        distance: Current distance to target
        max_distance: Maximum expected distance
        scale: Reward scaling factor
        exponent: Exponent for exponential shaping (higher = more reward near target)
    
    Returns:
        Reward value (higher when closer to target)
    """
    normalized_distance = clamp(distance / max_distance, 0.0, 1.0)
    closeness = 1.0 - normalized_distance
    return scale * (closeness ** exponent)


def linear_reward(distance: float, max_distance: float, scale: float = 1.0) -> float:
    """Linear reward: reward = scale * (1 - distance/max_distance)"""
    normalized_distance = clamp(distance / max_distance, 0.0, 1.0)
    return scale * (1.0 - normalized_distance)


def inverse_distance_reward(distance: float, scale: float = 1.0, 
                           offset: float = 1.0) -> float:
    """
    Inverse distance reward: reward = scale / (distance + offset)
    Provides strong reward when very close, decays quickly
    """
    return scale / (distance + offset)


def gaussian_reward(distance: float, sigma: float = 1.0, 
                   scale: float = 1.0, center: float = 0.0) -> float:
    """
    Gaussian reward: reward = scale * exp(-0.5 * ((distance - center) / sigma)^2)
    Peak reward at center, decays with distance
    """
    return scale * np.exp(-0.5 * ((distance - center) / sigma) ** 2)


def sigmoid_reward(distance: float, max_distance: float, 
                  scale: float = 1.0, steepness: float = 5.0) -> float:
    """
    Sigmoid reward: smooth transition from low to high reward
    reward = scale / (1 + exp(steepness * (distance/max_distance - 0.5)))
    """
    normalized_distance = clamp(distance / max_distance, 0.0, 1.0)
    return scale / (1.0 + np.exp(steepness * (normalized_distance - 0.5)))


# ========== PENALTY FUNCTIONS ==========

def quadratic_penalty(value: float, threshold: float, 
                      scale: float = 1.0) -> float:
    """
    Quadratic penalty: penalty = scale * (value / threshold)^2
    Penalty increases quadratically with value
    """
    if value <= threshold:
        return 0.0
    return -scale * ((value / threshold) ** 2)


def exponential_penalty(value: float, threshold: float, 
                        scale: float = 1.0, exponent: float = 2.0) -> float:
    """
    Exponential penalty: penalty = -scale * ((value / threshold)^exponent)
    """
    if value <= threshold:
        return 0.0
    return -scale * ((value / threshold) ** exponent)


def linear_penalty(value: float, threshold: float, 
                   scale: float = 1.0) -> float:
    """Linear penalty: penalty = -scale * (value / threshold)"""
    if value <= threshold:
        return 0.0
    return -scale * (value / threshold)


def boundary_penalty(distance_to_boundary: float, warning_distance: float,
                     max_penalty: float = 100.0) -> float:
    """
    Smooth boundary penalty that increases as drone approaches boundary
    
    Args:
        distance_to_boundary: Current distance to nearest boundary
        warning_distance: Distance at which penalty starts
        max_penalty: Maximum penalty at boundary
    
    Returns:
        Penalty value (negative, more negative closer to boundary)
    """
    if distance_to_boundary >= warning_distance:
        return 0.0
    if distance_to_boundary <= 0:
        return -max_penalty
    
    # Smooth transition: penalty increases as distance decreases
    safe_ratio = distance_to_boundary / warning_distance
    penalty = max_penalty * (1.0 - safe_ratio)
    return -penalty


# ========== ANGLE AND DIRECTION FUNCTIONS ==========

def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate angle (in radians) between two vectors"""
    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return math.acos(dot_product)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors (range: -1 to 1)"""
    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)
    return float(np.dot(v1_norm, v2_norm))


def direction_reward(velocity: np.ndarray, target_direction: np.ndarray,
                    scale: float = 1.0) -> float:
    """
    Reward for moving in the direction of the target
    
    Args:
        velocity: Current velocity vector
        target_direction: Direction to target (normalized)
        scale: Reward scaling factor
    
    Returns:
        Reward proportional to alignment with target direction
    """
    if np.linalg.norm(velocity) < 1e-6:
        return 0.0
    velocity_norm = normalize_vector(velocity)
    similarity = cosine_similarity(velocity_norm, target_direction)
    # Clamp to [0, 1] to only reward positive alignment
    return scale * max(0.0, similarity)


# ========== SMOOTHING FUNCTIONS ==========

def exponential_moving_average(current: float, previous: float, 
                               alpha: float = 0.1) -> float:
    """
    Exponential moving average: EMA = alpha * current + (1 - alpha) * previous
    
    Args:
        current: Current value
        previous: Previous EMA value
        alpha: Smoothing factor (0-1), higher = more responsive to current value
    """
    return alpha * current + (1.0 - alpha) * previous


def smooth_step(t: float) -> float:
    """
    Smooth step function: smooth transition from 0 to 1
    t should be in [0, 1]
    """
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smoother_step(t: float) -> float:
    """
    Smoother step function (Ken Perlin's improved version)
    t should be in [0, 1]
    """
    t = clamp(t, 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


# ========== VELOCITY FUNCTIONS ==========

def velocity_magnitude(velocity: np.ndarray) -> float:
    """Calculate velocity magnitude (speed)"""
    return float(np.linalg.norm(velocity))


def velocity_smoothness_penalty(current_velocity: np.ndarray,
                                previous_velocity: np.ndarray,
                                scale: float = 1.0) -> float:
    """
    Penalty for sudden velocity changes (encourages smooth motion)
    
    Args:
        current_velocity: Current velocity vector
        previous_velocity: Previous velocity vector
        scale: Penalty scaling factor
    
    Returns:
        Negative penalty (more negative = larger change)
    """
    if previous_velocity is None or np.linalg.norm(previous_velocity) < 1e-6:
        return 0.0
    velocity_change = np.linalg.norm(current_velocity - previous_velocity)
    return -scale * (velocity_change ** 2)


# ========== PROGRESS FUNCTIONS ==========

def progress_ratio(current_distance: float, initial_distance: float) -> float:
    """
    Calculate progress ratio: how much closer we are to target
    Returns 1.0 if at target, 0.0 if at start, negative if moved away
    """
    if initial_distance < 1e-6:
        return 1.0 if current_distance < 1e-6 else 0.0
    return (initial_distance - current_distance) / initial_distance


def progress_reward(current_distance: float, previous_distance: float,
                    scale: float = 1.0) -> float:
    """
    Reward for making progress (getting closer to target)
    
    Args:
        current_distance: Current distance to target
        previous_distance: Previous distance to target
        scale: Reward scaling factor
    
    Returns:
        Positive reward if closer, 0 if same or farther
    """
    if previous_distance is None:
        return 0.0
    distance_change = previous_distance - current_distance
    if distance_change > 0:  # Got closer
        return scale * distance_change
    return 0.0


# ========== UTILITY FUNCTIONS ==========

def safe_divide(numerator: float, denominator: float, 
                default: float = 0.0, eps: float = 1e-8) -> float:
    """Safe division that handles division by zero"""
    if abs(denominator) < eps:
        return default
    return numerator / denominator


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b"""
    t = clamp(t, 0.0, 1.0)
    return a + t * (b - a)


def inverse_lerp(value: float, a: float, b: float) -> float:
    """Inverse linear interpolation: returns t such that lerp(a, b, t) = value"""
    if abs(b - a) < 1e-8:
        return 0.0
    return clamp((value - a) / (b - a), 0.0, 1.0)


def remap(value: float, old_min: float, old_max: float,
         new_min: float, new_max: float) -> float:
    """Remap value from [old_min, old_max] to [new_min, new_max]"""
    t = inverse_lerp(value, old_min, old_max)
    return lerp(new_min, new_max, t)


# ========== STATISTICAL FUNCTIONS ==========

def mean_squared_error(values: np.ndarray, target: float) -> float:
    """Calculate mean squared error"""
    return float(np.mean((values - target) ** 2))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Calculate weighted mean"""
    if np.sum(weights) < 1e-8:
        return float(np.mean(values))
    return float(np.sum(values * weights) / np.sum(weights))


# ========== EXAMPLE USAGE ==========
"""
Example usage in reward functions:

from math_utils import (
    euclidean_distance, exponential_reward, boundary_penalty,
    direction_reward, progress_reward, cosine_similarity
)

# Calculate distance-based reward
distance = euclidean_distance(drone_pos, target_pos)
reward = exponential_reward(distance, max_distance=50.0, scale=10.0, exponent=1.5)

# Boundary penalty
boundary_dist = min_distance_to_boundary(drone_pos, bounds)
penalty = boundary_penalty(boundary_dist, warning_distance=3.0, max_penalty=100.0)

# Direction alignment reward
target_dir = normalize_vector(target_pos - drone_pos)
dir_reward = direction_reward(velocity, target_dir, scale=2.0)

# Progress reward
progress = progress_reward(current_dist, previous_dist, scale=0.5)
"""


