==========================================
MLPipeline: Philosophy and Structure
==========================================

The **MLPipeline** is designed to build a modular and flexible machine learning pipeline for data wrangling, preprocessing, training estimators, and validating results. Its primary goals include:

- **One pipeline trains one model**: This ensures the pipeline maintains a single focus, avoiding unnecessary complexity.
- **Each transformer and estimator performs one function**: Keeping each step's purpose singular simplifies debugging and improves maintainability.
- **Reusability of transformers and estimators**: Components of the pipeline (such as transformers or estimators) are designed to be reusable across different pipelines and datasets.
- **Consistency of input/output formats**: All pipeline steps should maintain consistent input/output formats, simplifying integration between steps.

The pipeline is broken down into several key phases: **data wrangling**, **splitting**, **fitting** (including **data processing** and **estimator** steps), **training**, **predicting**, and **exporting**. These phases ensure proper data preparation, training, and evaluation.

.. contents:: Table of Contents
   :local:
   :depth: 3

######################
Pipeline Structure
######################

The `MLPipeline` class is structured around several important components that guide the flow of data from raw form to predictions and validation.

---------------------
Data Wrangling
---------------------

**Data wrangling** refers to transformations that change the values of observations based on information specific to those observations. These transformations can be applied to the entire dataset, regardless of whether the data is intended for training or testing, since each transformation affects only the observation in question.

This step is critical to clean and prepare the raw dataset, ensuring that it is free from noise or inconsistencies. Typical transformations might include:

- Imputing missing values
- Removing outliers
- Converting categorical values to numerical form
- Feature engineering that depends only on the observation

Wrangling transformations should not depend on the global data distribution, making it safe to apply them before splitting the data into training and test sets.

.. code-block:: python

    def wrangle(self, data):
        """Apply the wrangling steps to the entire dataset."""
        data_wrangling_pipeline = Pipeline(self.data_wrangling_steps)

        # Apply wrangling pipeline to the full dataset
        data_wrangled = data_wrangling_pipeline.transform(data)

        return data_wrangled

---------------------
Data Splitting
---------------------

**Splitting** is performed after the data wrangling phase. The reason for this sequencing is that any data wrangling transformations operate independently on each observation. After wrangling, the dataset can be split into training and test sets, as subsequent data processing steps might depend on global features of the training data.

The `MLPipeline` supports both default splitting methods (such as `train_test_split`) and custom splitting functions for more complex scenarios (e.g., stratified splitting or time series splitting).

.. code-block:: python

    def set_splitter(self, splitter):
        self.splitter = splitter

    def train(self, X, y_column, y_value=None, split=True, **split_args):
        """Run the full pipeline: wrangle, split, fit, predict
        and validate on test data."""
        # Wrangle the data
        X_wrangled = self.wrangle(X)

        y = self.infer_y(X_wrangled, y_column, y_value)

        if split:
            # Split the data (with optional arguments for custom splits)
            X_train, X_test, y_train, y_test = self.splitter(
                X_wrangled, y, **split_args
            )
        else:
            X_train = X_wrangled
            y_train = y
            X_test = X_wrangled
            y_test = y

        # Fit the pipeline on training data
        self.fit(X_train, y_train)

---------------------
Fitting
---------------------

The **fitting** process consists of two steps: **Data Processing** and **Estimator**.

+++++++++++++++++++++
Data Processing
+++++++++++++++++++++

**Data processing** involves transformations on features that require fitting on training data (e.g., scaling). These transformations are retained and applied to test data during prediction, avoiding data leakage. This phase ensures that the model is trained with proper preprocessing while preventing overfitting.

Transformations included in data processing could be:

- Standardization (e.g., scaling)
- One-hot encoding
- Principal Component Analysis (PCA)
- Feature selection techniques

.. code-block:: python

    def fit(self, X, y):
        """Initialize and fit the preprocessing and estimator pipeline."""
        # Initialize the pipeline of preprocessing steps and estimator
        pipeline = Pipeline(self.preprocessing_steps + self.estimator)

        # Fit the pipeline
        pipeline.fit(X, y)

        # Store the fitted pipeline
        self.trained_pipeline = pipeline

        return pipeline


+++++++++++++++++++++
Estimator
+++++++++++++++++++++

The **estimator** is the machine learning model that is trained on the preprocessed data. This step follows preprocessing and forms the final stage of the pipeline. The estimator can be any model from `scikit-learn`, such as logistic regression, decision trees, or ensemble methods.

After the preprocessing steps are applied to the training data, the estimator is fitted to the processed data.

.. code-block:: python

    def fit(self, X, y):
        """Initialize and fit the preprocessing and estimator pipeline."""
        # Initialize the pipeline of preprocessing steps and estimator
        pipeline = Pipeline(self.preprocessing_steps + self.estimator)

        # Fit the pipeline
        pipeline.fit(X, y)

        # Store the fitted pipeline
        self.trained_pipeline = pipeline

        return pipeline


############################################
Working with MLPipeline
############################################

------
Train
------

The `train` function is responsible for running the full pipeline, from wrangling to fitting and validation. It performs the following steps:

1. **Wrangle** the raw dataset using the defined wrangling transformations.
2. **Infer the y values** from the cleaned dataset based on the `y_column` and optional `y_value`. This step helps to isolate the target variable from the feature set, which may involve filtering rows or extracting a specific column.
    - *Side note*: The `infer_y` function allows flexibility in how the target variable is derived from the dataset.
3. **Split** the data into training and testing subsets using the chosen splitting strategy.
4. **Fit** the pipeline (preprocessing and estimator) on the training data.
5. **Validate** the model performance on the test data.

.. code-block:: python

    def train(self, X, y_column, y_value=None, split=True, **split_args):
        """Run the full pipeline: wrangle, split, fit, predict
        and validate on test data."""
        # Wrangle the data
        X_wrangled = self.wrangle(X)

        y = self.infer_y(X_wrangled, y_column, y_value)

        if split:
            # Split the data (with optional arguments for custom splits)
            X_train, X_test, y_train, y_test = self.splitter(
                X_wrangled, y, **split_args
            )
        else:
            X_train = X_wrangled
            y_train = y
            X_test = X_wrangled
            y_test = y

        # Fit the pipeline on training data
        self.fit(X_train, y_train)

        y_pred = self.trained_pipeline.predict(X_test)
        y_score = self.trained_pipeline.predict_proba(X_test)[:, 1]

        # Validate the training results
        self.validate(y_test, y_pred, y_score)
        _, _, _, self.optimal_threshold = calculate_optimal_threshold(
            y_test, y_score
        )

--------
Predict
--------

The `predict` function utilizes the trained pipeline to make predictions on new or test data. If specified, the function can also wrangle the input data before making predictions.

Steps include:

1. Ensures the pipeline has been fitted before prediction.
2. Optionally **wrangles** the input data if `wrangle=True`.
3. Uses the trained pipeline to **predict** the target values.

.. code-block:: python

    def predict(self, X, wrangle=False):
        """Predict using the trained pipeline."""
        if not self.trained_pipeline:
            raise RuntimeError("Pipeline has not been fitted yet.")

        if wrangle:
            self.wrangle(X)
        # Use the trained pipeline for prediction (preprocessing + estimator)
        return self.trained_pipeline.predict(X)

####################
Validation
####################

The `validate` function evaluates the trained model's performance on a test dataset using specified evaluation metrics. Key steps include:

1. **Calculating accuracy**: If the `"accuracy"` metric is selected, the accuracy score is computed using `accuracy_score`.
2. **Generating ROC curve and calculating AUC**: For the `"roc_auc"` metric, the function generates a ROC curve and calculates the area under the curve (AUC) using `generate_roc_curve` and `roc_auc_score`.
3. **Calculating precision**: If the `"precision"` metric is chosen, the precision score is computed using `precision_score`.

The results for the chosen metrics are returned in a dictionary.

.. code-block:: python

    def validate(self, y_true, y_pred, y_score, metrics=["accuracy", "roc_auc"]):
        """Validate the trained estimator on test data using
        specified metrics."""
        # Calculate and return the desired metrics
        results = {}
        if "accuracy" in metrics:
            results["accuracy"] = accuracy_score(y_true, y_pred)

        if "roc_auc" in metrics:
            generate_roc_curve(y_true, y_score)
            results["roc_auc"] = roc_auc_score(y_true, y_score)

        if "precision" in metrics:
            results["precision"] = precision_score(y_true, y_pred)


############################################
Export
############################################

--------------------
Exporting Pipeline
--------------------

The `export_pipeline` function allows users to export the fitted pipeline, either including the wrangling steps or just the trained parts (preprocessing + estimator). The pipeline can be serialized and saved to a file for later use.

Steps include:

1. Ensuring the pipeline has been fitted.
2. Wrangling the input data if needed.
3. Serializing the full pipeline (or parts of it) using Python’s `pickle` module.

.. code-block:: python

    def export_pipeline(self, wrangle=False, save_path=None):
        "Export the pipeline"
        if not self.trained_pipeline:
            raise RuntimeError("Pipeline has not been fitted yet.")

        if wrangle:
            full_pipeline = Pipeline(
                steps=[
                    *self.data_wrangling_steps,
                    *self.trained_pipeline.steps,
                ]
            )
        else:
            full_pipeline = self.trained_pipeline

        full_pipeline.optimal_threshold = self.optimal_threshold

        if save_path:
            dump(full_pipeline, open(save_path, "wb"))

        return full_pipeline

---------------------------------------
Exporting Predictions
---------------------------------------

The `export_predictions` function exports the model's predictions for a given dataset to a CSV file. The process involves:

1. **Ensuring the pipeline has been trained**: It first checks whether the pipeline has been fitted, raising an error if it has not.
2. **Optionally wrangling the input data**: If `wrangle=True`, the data wrangling steps are applied before prediction.
3. **Generating predictions**: The function computes the probability scores for the input data using the pipeline's `predict_proba` method, then applies a threshold to convert these scores into binary predictions (e.g., `cancer_diagnosis`).
4. **Exporting to CSV**: The predictions are saved as a CSV file at the specified `save_path`.

.. code-block:: python

    def export_predictions(self, data, save_path, wrangle=True):
        if not self.trained_pipeline:
            raise RuntimeError("Pipeline has not been fitted yet.")

        model = self.export_pipeline(wrangle)

        y_score = model.predict_proba(data)[:, 1]
        y_pred = y_score > model.optimal_threshold
        df_saxs_pred = pd.DataFrame(
            data=y_pred, index=data.index, columns=["cancer_diagnosis"]
        )

        df_saxs_pred.to_csv(save_path)
