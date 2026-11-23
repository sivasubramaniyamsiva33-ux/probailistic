"""
Module 2: Linear Algebra package.
"""
from .demand_matrix import DemandMatrixBuilder, DemandMatrixAnalysis
from .least_squares_predictor import (
    LeastSquaresPredictor,
    modified_gram_schmidt_qr,
    back_substitution,
    solve_normal_equations,
    solve_qr,
    solve_svd_pinv,
    solve_ridge,
)
from .svd_pca_patterns import SVDDemandPatterns, perform_svd_analysis
from .markov_matrix_analysis import MarkovSpectralAnalysis, manual_power_iteration
from .recommender import CanteenVisitRecommender, recommend_best_time_to_visit

__all__ = [
    "DemandMatrixBuilder",
    "DemandMatrixAnalysis",
    "LeastSquaresPredictor",
    "modified_gram_schmidt_qr",
    "back_substitution",
    "solve_normal_equations",
    "solve_qr",
    "solve_svd_pinv",
    "solve_ridge",
    "SVDDemandPatterns",
    "perform_svd_analysis",
    "MarkovSpectralAnalysis",
    "manual_power_iteration",
    "CanteenVisitRecommender",
    "recommend_best_time_to_visit",
]
