import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def pre_process_data(train_split, test_split, num_cols, cat_cols):
    """
    Preprocessing training and testing dataframes for a PyTorch model.

    Parameters:
    -----------
    train_df : pd.Dataframe
      The training dataset split.
    test_df : pd.Dataframe
      The testing dataset split.
    num_cols : list of str
      The names of numerical columns in original dataframe.
    cat_cols : list of str
      The names of categorical columns in original dataframe.

    Returns:
    --------
    X_train_processed : np.ndarray
      Processed feature array for training.
    X_test_processed : np.ndarray
      Processed feature array for testing.
    cat_cardinalities : list of int
      Number of unique categories per categorical feature.
    preprocessor : ColumnTransformer
      The fitted scikit-learn preprocessor object for future inference.Shoul
    """

    # We first define the pipelines for the numerical and categorical features.
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('encoder', OrdinalEncoder(
          handle_unknown='use_encoded_value', 
          unknown_value=-1,
          dtype=np.int64))
    ])

    # Here we combine both piplines into a ColumnTransformer.
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    # We then only fit onto the training data and then transform both the training
    # and the testing data.
    X_train_processed = preprocessor.fit_transform(train_split)
    X_test_processed = preprocessor.transform(test_split)

    # We finally extract category counts for PyTorch nn.Embedding layers later on
    # when we actually create the model.
    #
    # Because unknown values are set to -1, adding +1 ensure our embedding table
    # accomodates the index -1 securely.
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_cardinalities = [len(cats) + 1 for cats in cat_encoder.categories_]

    return X_train_processed, X_test_processed, cat_cardinalities, preprocessor
