from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects or drops specified columns."""

    def __init__(self, columns: List[str], selector_type: str = "keep"):
        """
        Initializes a new instance of the `ColumnSelector` class.

        Args:
            columns : list of str
                List of column names to select or drop.
            selector_type : str, optional (default='keep')
                Type of selection. Must be either 'keep' or 'drop'.
        """
        self.columns = columns
        assert selector_type in ["keep", "drop"]
        self.selector_type = selector_type

    def fit(self, X: pd.DataFrame, y=None):
        """
        No-op

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame):
        """
        Applies the column selection.

        Args:
            X : pandas.DataFrame - The input data.
        Returns:
            pandas.DataFrame: The transformed data.
        """
        if self.selector_type == "keep":
            retained_cols = [col for col in self.columns if col in X.columns]
            X = X[retained_cols].copy()
        elif self.selector_type == "drop":
            dropped_cols = [col for col in X.columns if col in self.columns]
            X = X.drop(dropped_cols, axis=1)
        return X


class TypeCaster(BaseEstimator, TransformerMixin):
    """
    A custom transformer that casts the specified variables in the input data
    to a specified data type.
    """

    def __init__(self, vars: List[str], cast_type: str):
        """
        Initializes a new instance of the `TypeCaster` class.

        Args:
            vars : list of str
                List of variable names to be transformed.
            cast_type : data type
                Data type to which the specified variables will be cast.
        """
        super().__init__()
        self.vars = vars
        self.cast_type = cast_type

    def fit(self, X: pd.DataFrame, y=None):
        """
        No-op.

        Returns:
            self
        """
        return self

    def transform(self, data: pd.DataFrame):
        """
        Applies the casting to given features in input dataframe.

        Args:
            data : pandas DataFrame
                Input data to be transformed.
        Returns:
            data : pandas DataFrame
                Transformed data.
        """
        data = data.copy()
        applied_cols = [col for col in self.vars if col in data.columns]
        for var in applied_cols:
            if data[var].notnull().any():  # check if the column has any non-null values
                data[var] = data[var].astype(self.cast_type)
            else:
                # all values are null. so no-op
                pass
        return data


class TimeColCaster(BaseEstimator, TransformerMixin):
    """
    A custom transformer that casts the time col in the input data
    to either a datetime type or the integer type, given its type
    in the schema.
    """

    def __init__(self, time_col: str, data_type: str):
        """
        Initializes a new instance of the `TimeColCaster` class.

        Args:
            time_col (str): Name of the time field.
            cast_type (str): Data type to which the specified variables
                             will be cast.
        """
        super().__init__()
        self.time_col = time_col
        self.data_type = data_type

    def fit(self, X: pd.DataFrame, y=None):
        """
        No-op.

        Returns:
            self
        """
        return self

    def transform(self, data: pd.DataFrame):
        """
        Applies the casting to given features in input dataframe.

        Args:
            data : pandas DataFrame
                Input data to be transformed.
        Returns:
            data : pandas DataFrame
                Transformed data.
        """
        data = data.copy()
        if self.data_type == "INT":
            data[self.time_col] = data[self.time_col].astype(int)
        elif self.data_type in ["DATETIME", "DATE"]:
            data[self.time_col] = pd.to_datetime(data[self.time_col])
        else:
            raise ValueError(f"Invalid data type for time column: {self.data_type}")
        return data


class DataFrameSorter(BaseEstimator, TransformerMixin):
    """
    Sorts a pandas DataFrame based on specified columns and their corresponding sort orders.
    """

    def __init__(self, sort_columns: List[str], ascending: List[bool]):
        """
        Initializes a new instance of the `DataFrameSorter` class.

        Args:
            sort_columns : list of str
                List of column names to sort by.
            ascending : list of bool
                List of boolean values corresponding to each column in `sort_columns`.
                Each value indicates whether to sort the corresponding column in ascending order.
        """
        assert len(sort_columns) == len(
            ascending
        ), "sort_columns and ascending must be of the same length"
        self.sort_columns = sort_columns
        self.ascending = ascending

    def fit(self, X: pd.DataFrame, y=None):
        """
        No-op

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame):
        """
        Sorts the DataFrame based on specified columns and order.

        Args:
            X : pandas.DataFrame - The input data.
        Returns:
            pandas.DataFrame: The sorted DataFrame.
        """
        X = X.sort_values(by=self.sort_columns, ascending=self.ascending)
        return X


class ScalerByFeatureDim(BaseEstimator, TransformerMixin):
    """
    Scales the history and forecast parts of a time-series based on history data.

    The scaler is fitted using only the history part of the time-series and is
    then used to transform both the history and forecast parts. Values are scaled
    to a range and capped to an upper bound.

    Attributes:
        encode_len (int): The length of the history (encoding) window in the time-series.
        upper_bound (float): The upper bound to which values are capped after scaling.
    """

    def __init__(self, columns: List[str], scaler_type: str = "standard"):
        """
        Initializes the ScalerByFeatureDim.

        Args:
            columns (List): The columns to scale.
        """
        self.columns = columns
        self.scaler_type = scaler_type
        if scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(
                f"Invalid scaler type: {scaler_type}"
                "Allowed types are [`standard`, `minmax`, `robust`]"
            )
        self.fitted = False

    def fit(self, X: pd.DataFrame, y=None) -> "ScalerByFeatureDim":
        """
        No-op

        Args:
            X (pd.DataFrame): The input dataframe.
            y: Ignored. Exists for compatibility with the sklearn transformer interface.

        Returns:
            ScalerByFeatureDim: The fitted scaler.
        """
        if self.fitted:
            return self
        X_scaled = X.copy()[self.columns]
        self.scaler.fit(X_scaled)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the scaling transformation to the input data.

        Args:
            X (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: The transformed dataframe.
        """
        X_scaled = X.copy()
        X_scaled.loc[:, self.columns] = self.scaler.transform(
            X_scaled.loc[:, self.columns]
        )
        return X_scaled


class LabelEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical columns using label encoding."""

    def __init__(self, columns: List[str]):
        """
        Initializes a new instance of the `LabelEncoder` class.

        Args:
            columns : list of str
                List of column names to encode.
        """
        self.columns = columns
        self.encoders = {}

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the label encoders for the specified columns.

        Args:
            X : pandas.DataFrame - The input data.
        Returns:
            self
        """
        for col in self.columns:
            if col in X.columns:
                self.encoders[col] = {
                    label: idx for idx, label in enumerate(sorted(X[col].unique()))
                }
        return self

    def transform(self, X: pd.DataFrame):
        """
        Applies the label encoding to the specified columns.

        Args:
            X : pandas.DataFrame - The input data.
        Returns:
            pandas.DataFrame: The transformed data.
        """

        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = X[col].map(self.encoders[col])
        return X

    def inverse_transform(
        self,
        X: pd.DataFrame,
    ):
        """
        Applies the inverse of the label encoding to the specified columns.

        Args:
            X : pandas.DataFrame - The input data.
        Returns:
            pandas.DataFrame: The inverse transformed data.
        """
        X = X.copy()
        for col in self.columns:
            inv_encoders = {v: k for k, v in self.encoders[col].items()}
            X[col] = X[col].map(inv_encoders)
        return X
