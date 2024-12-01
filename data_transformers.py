from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import (KBinsDiscretizer, MinMaxScaler,
                                   StandardScaler)


class DataTransformer:
    """
    A comprehensive data transformation class that handles multiple types of transformations
    on a dataset.
    """

    def __init__(self, data: Union[pd.DataFrame, np.ndarray], columns: Optional[List[str]] = None):
        """
        Initialize the transformer with a dataset.

        Args:
            data: Input dataset as DataFrame or numpy array
            columns: Column names if data is numpy array
        """
        self.original_data = self._validate_input(data, columns)
        self.transformed_data = self.original_data.copy()

        # Initialize transformers
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.discretizer = None
        self.transformed_columns = {}  # Track which columns have been transformed

    def _validate_input(self, data: Union[pd.DataFrame, np.ndarray],
                        columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Validate and convert input data to DataFrame"""
        if isinstance(data, np.ndarray):
            if columns is None:
                columns = [f'feature_{i}' for i in range(data.shape[1])]
            return pd.DataFrame(data, columns=columns)
        elif isinstance(data, pd.DataFrame):
            return data.copy()
        else:
            raise ValueError(
                "Data must be either pandas DataFrame or numpy array")

    def standardize(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Standardize specified columns (z-score normalization)
        """
        columns = columns or self.transformed_data.columns
        self.transformed_data[columns] = self.standard_scaler.fit_transform(
            self.transformed_data[columns]
        )
        self.transformed_columns['standardized'] = columns
        return self.transformed_data

    def normalize(self, columns: Optional[List[str]] = None,
                  feature_range: tuple = (0, 1)) -> pd.DataFrame:
        """
        Normalize specified columns to a given range
        """
        columns = columns or self.transformed_data.columns
        self.minmax_scaler = MinMaxScaler(feature_range=feature_range)
        self.transformed_data[columns] = self.minmax_scaler.fit_transform(
            self.transformed_data[columns]
        )
        self.transformed_columns['normalized'] = columns
        return self.transformed_data

    def difference(self, columns: Optional[List[str]] = None,
                   periods: int = 1) -> pd.DataFrame:
        """
        Calculate differences for specified columns
        """
        columns = columns or self.transformed_data.columns
        for col in columns:
            self.transformed_data[f'{col}_diff'] = self.transformed_data[col].diff(
                periods=periods).fillna(0)
        self.transformed_columns['differenced'] = columns
        return self.transformed_data

    def discretize(self, columns: Optional[List[str]] = None, n_bins: int = 5,
                   encode: str = 'ordinal', strategy: str = 'uniform') -> pd.DataFrame:
        """
        Discretize specified columns into bins
        """
        columns = columns or self.transformed_data.columns
        self.discretizer = KBinsDiscretizer(
            n_bins=n_bins, encode=encode, strategy=strategy)
        self.transformed_data[columns] = self.discretizer.fit_transform(
            self.transformed_data[columns]
        )
        self.transformed_columns['discretized'] = columns
        return self.transformed_data

    def binarize(self, columns: Optional[List[str]] = None,
                 threshold: float = 0.0) -> pd.DataFrame:
        """
        Binarize specified columns based on threshold
        """
        columns = columns or self.transformed_data.columns
        for col in columns:
            self.transformed_data[col] = (
                self.transformed_data[col] > threshold).astype(float)
        self.transformed_columns['binarized'] = columns
        return self.transformed_data

    def inverse_transform(self, transformation_type: str,
                          columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Reverse a specific transformation for given columns

        Args:
            transformation_type: One of 'standardized', 'normalized', 'discretized'
            columns: Specific columns to inverse transform
        """
        if transformation_type not in self.transformed_columns:
            raise ValueError(f"No {transformation_type} transformation found")

        columns = columns or self.transformed_columns[transformation_type]

        if transformation_type == 'standardized':
            self.transformed_data[columns] = self.standard_scaler.inverse_transform(
                self.transformed_data[columns]
            )
        elif transformation_type == 'normalized':
            self.transformed_data[columns] = self.minmax_scaler.inverse_transform(
                self.transformed_data[columns]
            )
        elif transformation_type == 'discretized' and self.discretizer:
            self.transformed_data[columns] = self.discretizer.inverse_transform(
                self.transformed_data[columns]
            )
        else:
            raise ValueError(f"Cannot inverse transform {transformation_type}")

        return self.transformed_data

    def reset(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Reset transformations for specified columns or all if none specified
        """
        columns = columns or self.transformed_data.columns
        self.transformed_data[columns] = self.original_data[columns]
        return self.transformed_data

    def get_data(self) -> pd.DataFrame:
        """
        Get the current state of the transformed data
        """
        return self.transformed_data.copy()
