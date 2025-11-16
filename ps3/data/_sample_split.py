import hashlib
from sklearn.model_selection import GroupShuffleSplit

import numpy as np

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def create_sample_split(df, id_column, training_frac=0.8):
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.8

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """

    unique_ids = df[id_column].unique()
    id_map = {original_id : numerical_id for numerical_id, original_id in enumerate(unique_ids)}
    df["numerical_id"] = df[id_column].map(id_map)

    split_generator = GroupShuffleSplit(test_size=(1-training_frac), n_splits=1, random_state=99)
    train_ids, test_ids = next(split_generator.split(df, groups=df["numerical_id"]))
    df.loc[train_ids, "sample"] = "train"
    df["sample"].fillna("test", inplace=True)
    df.drop(columns=["numerical_id"], inplace=True)

    return df
