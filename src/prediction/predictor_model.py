import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score
from multiprocessing import cpu_count
from tqdm import tqdm

from logger import get_logger
from schema.data_schema import TimeStepClassificationSchema

warnings.filterwarnings("ignore")
PREDICTOR_FILE_NAME = "predictor.joblib"

logger = get_logger(task_name=__file__)

# Determine the number of CPUs available
n_cpus = cpu_count()

# Set n_jobs to be one less than the number of CPUs, with a minimum of 1
n_jobs = max(1, n_cpus - 1)
logger.info(f"Using n_jobs = {n_jobs}")

class TimeStepClassifier:
    """Random Forest TimeStepClassifier.

    This class provides a consistent interface that can be used with other
    TimeStepClassifier models.
    """

    MODEL_NAME = "Random_Forest_TimeStepClassifier"

    def __init__(
        self,
        data_schema: TimeStepClassificationSchema,
        window_length_factor: float = 4.0,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_split: int = 2,
        **kwargs,
    ):
        """
        Construct a new Random Forest TimeStepClassifier.

        Args:
            data_schema (TimeStepClassificationSchema): The schema of the data.
            window_length_factor (float): Factor used to adjust the base window length
                                         derived from the logarithm of the minimum row
                                         count per group.
            n_estimators (int): Number of trees in the forest.
            max_depth (int): Maximum depth of the tree.
            min_samples_split (int): Minimum number of samples required to split an internal node.
        """
        self.data_schema = data_schema
        self.window_length_factor = window_length_factor
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.stride = 2
        self.model = self.build_model()
        self.window_length = None # set using training data
        self._is_trained = False

    def build_model(self) -> RandomForestClassifier:
        """Build a new Random Forest classifier."""
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_jobs=n_jobs,
        )
        return model
    
    def create_windows_for_prediction(self, X_arr, y_arr) -> np.ndarray:
        """
        Create windows for prediction.

        Args:
            X_arr (np.ndarray): The data to create windows for.
                               Shape is (L, D)
                                where L is the number of rows and
                                D is the number of dimensions.
            
            y_arr (Union[np.ndarray, None]): The data to create windows for.
                               If not None Shape is (L)
                                where L is the number of rows.

        Returns:
            Union[
                Tuple[np.ndarray, np.ndarray],
                Tuple[np.ndarray, None]
            ]: The windowed X and y data.
        """
        X_windows_list = []
        y_windows_list = []
        start_idx_list = []
        seen_start_indices = set()
        data_len = len(X_arr)
        for i in range(0, len(X_arr), self.stride):
            start_idx = i
            if start_idx + self.window_length > data_len - 1:
                # last window runs out of space, so slide it in to fit
                start_idx = data_len - self.window_length
            if start_idx in seen_start_indices:
                continue  # Skip if the start_idx has already been processed
            seen_start_indices.add(start_idx)
            start_idx_list.append(start_idx)
            X_windows_list.append(X_arr[start_idx : start_idx + self.window_length].flatten())
            if y_arr is not None:
                y_windows_list.append(y_arr[start_idx : start_idx + self.window_length])
        stacked_X = np.stack(X_windows_list, axis=0)
        start_idx_list = np.array(start_idx_list)
        if y_arr is not None:
            stacked_y = np.stack(y_windows_list, axis=0)
        else:
            stacked_y = None
        return stacked_X, stacked_y, start_idx_list


    def fit(self, train_data):
        self.train_data = train_data.sort_values(
            by=[self.data_schema.id_col, self.data_schema.time_col]
        )
        grouped = train_data.groupby(self.data_schema.id_col)
        min_row_count = grouped.size().min()
        log_min_count = np.log2(min_row_count)
        self.window_length = int(log_min_count * self.window_length_factor)
        logger.info(f"Calculated window length = {self.window_length}")
        
        all_X, all_y = [], []
        for _, group in grouped:
            X_vals = group[self.data_schema.features].values
            y_vals = group[self.data_schema.target].values
            windowed_X, windowed_y, _ = self.create_windows_for_prediction(
                X_vals, y_vals
            )
            all_X.append(windowed_X)
            all_y.append(windowed_y)
        
        train_X = np.concatenate(all_X, axis=0)
        train_y = np.concatenate(all_y, axis=0)
        self.model.fit(train_X, train_y)
        self._is_trained = True
        return self.model

    def predict(self, test_data):

        if self.data_schema.target in test_data.columns:
            test_data = test_data.drop(self.data_schema.target)

        id_cols = [self.data_schema.id_col, self.data_schema.time_col]
        encoded_target_cols = [
            int(i) for i in range(len(self.data_schema.target_classes))
        ]        
        grouped = test_data.groupby(self.data_schema.id_col)

        all_X = []
        all_ids = []
        for id_, group in grouped:
            X_vals = group[self.data_schema.features].values
            windowed_X, _, start_idx_list = self.create_windows_for_prediction(
                X_vals, None
            )
            all_X.append(windowed_X)
            for start_idx in start_idx_list:
                end_idx = start_idx + self.window_length
                all_ids.append(
                    group.iloc[start_idx:end_idx][id_cols].values
                )
        
        test_X = np.concatenate(all_X, axis=0)
        all_ids = np.concatenate(all_ids, axis=0)
        pred_probs = self.model.predict_proba(test_X)
        pred_probs = np.stack(pred_probs, axis=0)
        pred_probs = np.transpose(pred_probs, (1, 0, 2))
        pred_probs = pred_probs.reshape(-1, len(encoded_target_cols))
        all_preds_df = pd.DataFrame(np.concat([
            all_ids,
            pred_probs
        ], axis=1))
        all_preds_df.columns = id_cols + encoded_target_cols

        # Average by id and time columns since the same time idx can be repeated over
        # many overlapping windows
        averaged_preds = (
            all_preds_df.groupby(id_cols)[encoded_target_cols].mean().reset_index()
        )
        return averaged_preds[encoded_target_cols].values

    def evaluate(self, test_data, truth_labels):
        """Evaluate the model and return the loss and metrics"""
        predictions = self.predict(test_data)
        predictions = np.argmax(predictions, axis=1)
        f1 = f1_score(truth_labels, predictions, average="weighted")
        return f1

    def save(self, model_dir_path: str) -> None:
        """Save the Random Forest TimeStepClassifier to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "TimeStepClassifier":
        """Load the Random Forest TimeStepClassifier from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            TimeStepClassifier: A new instance of the loaded Random Forest
                                TimeStepClassifier.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model


def train_predictor_model(
    train_data: np.ndarray,
    data_schema: TimeStepClassificationSchema,
    hyperparameters: dict,
) -> TimeStepClassifier:
    """
    Instantiate and train the TimeStepClassifier model.

    Args:
        train_data (np.ndarray): The train split from training data.
        data_schema (TimeStepClassificationSchema): The data schema.
        hyperparameters (dict): Hyperparameters for the TimeStepClassifier.

    Returns:
        'TimeStepClassifier': The TimeStepClassifier model
    """
    model = TimeStepClassifier(
        data_schema=data_schema,
        **hyperparameters,
    )
    model.fit(train_data=train_data)
    return model


def predict_with_model(model: TimeStepClassifier, test_data: np.ndarray) -> np.ndarray:
    """
    Make forecast.

    Args:
        model (TimeStepClassifier): The TimeStepClassifier model.
        test_data (np.ndarray): The test input data for classification.

    Returns:
        np.ndarray: The predictions.
    """
    return model.predict(test_data)


def save_predictor_model(model: TimeStepClassifier, predictor_dir_path: str) -> None:
    """
    Save the TimeStepClassifier model to disk.

    Args:
        model (TimeStepClassifier): The TimeStepClassifier model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> TimeStepClassifier:
    """
    Load the TimeStepClassifier model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        TimeStepClassifier: A new instance of the loaded TimeStepClassifier model.
    """
    return TimeStepClassifier.load(predictor_dir_path)


def evaluate_predictor_model(
    model: TimeStepClassifier, test_split: np.ndarray, truth_labels: np.ndarray
) -> float:
    """
    Evaluate the TimeStepClassifier model and return the r-squared value.

    Args:
        model (TimeStepClassifier): The TimeStepClassifier model.
        test_split (np.ndarray): Test data.
        truth_labels (np.ndarray): The true labels.

    Returns:
        float: The r-squared value of the TimeStepClassifier model.
    """
    return model.evaluate(test_split, truth_labels)


class InsufficientDataError(Exception):
    """Raised when the data length is less that encode length"""

