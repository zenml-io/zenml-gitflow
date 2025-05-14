from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from utils.utils import generate_model_report, mock_data
import pandas as pd
import numpy as np
import datetime
from typing import Annotated, Tuple
from zenml import ArtifactConfig, step
from zenml import log_metadata, step
from zenml.enums import ArtifactType
from zenml.types import HTMLString

@step(enable_cache=False)
def train_model(
    data: pd.DataFrame, 
    epochs: int
) -> Tuple[
    Annotated[
        Pipeline, ArtifactConfig(name="price_prediction_model", artifact_type=ArtifactType.MODEL,)], 
    Annotated[HTMLString, "model_report"]
]:
    """Train a model to predict product prices."""
    
    # Define features and target
    # Note: We exclude product_id since it's just an identifier
    categorical_features = ["category", "discount_offered"]
    numeric_features = ["brand_rating", "num_reviews", "days_since_release", 
                        "shipping_weight", "competitors_price", "manufacturing_cost"]
    
    features = categorical_features + numeric_features
    X = data[features]
    y = data["price"]
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=epochs * 10,  # Use epochs to control complexity
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        ))
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Get feature importance from the model
    gbr = model.named_steps['regressor']
    
    # Get feature names after preprocessing
    feature_names = []
    # Add numeric feature names directly
    feature_names.extend(numeric_features)
    # Get one-hot encoded feature names
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    categorical_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
    feature_names.extend(categorical_feature_names)
    
    # Ensure feature_importance matches the actual features
    # If the gradient boosting model has fewer features than we expect,
    # we'll just use the first N importance values
    importances = gbr.feature_importances_[:len(feature_names)] if len(gbr.feature_importances_) > len(feature_names) else gbr.feature_importances_
    
    # Map feature names to importance values
    raw_feature_importance = dict(zip(feature_names, importances))
    
    # We need to combine one-hot encoded importances back to original features
    feature_importance = {}
    for feature, importance in raw_feature_importance.items():
        # For numeric features, just copy the importance
        if feature in numeric_features:
            feature_importance[feature] = round(importance, 4)
        else:
            # For categorical features, extract the original feature name
            # Format is typically feature_value
            original_feature = feature.split('_')[0]
            feature_importance[original_feature] = round(feature_importance.get(original_feature, 0) + importance, 4)
    
    # Create learning curves by using cross-validation
    cv_scores = []
    train_sizes = [int(epochs * i/5) for i in range(1, 6)]  # 5 points on the learning curve
    
    for n in train_sizes:
        gbr_cv = GradientBoostingRegressor(n_estimators=n * 10, learning_rate=0.1, max_depth=4, random_state=42)
        model_cv = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', gbr_cv)])
        model_cv.fit(X_train, y_train)
        train_score = -mean_squared_error(y_train, model_cv.predict(X_train))
        test_score = -mean_squared_error(y_test, model_cv.predict(X_test))
        cv_scores.append((train_score, test_score))
    
    # Extract training and validation loss
    train_loss = [-score[0] for score in cv_scores]
    val_loss = [-score[1] for score in cv_scores]
    
    # Create learning curves (real)
    learning_curve = {
        "epochs": train_sizes,
        "train_loss": train_loss,
        "val_loss": val_loss
    }
    
    # Create model output
    model_metrics = {
        "metrics": {
            "r2_score": round(float(r2), 4),
            "mse": round(float(mse), 4),
            "rmse": round(float(rmse), 4),
            "mae": round(float(mae), 4),
        },
        "feature_importance": feature_importance,
        "learning_curve": learning_curve,
        "model_params": {
            "epochs": epochs,
            "model_type": "GradientBoostingRegressor",
            "features": features,
            "timestamp": datetime.datetime.now().isoformat()
        }
    }
    
    metadata = {
        "metrics": {
            "r2_score": round(float(r2), 4),
            "mse": round(float(mse), 4),
            "rmse": round(float(rmse), 4),
            "mae": round(float(mae), 4),
        },
        "epochs": epochs,
        "feature_importance": feature_importance,
        "timestamp": datetime.datetime.now().isoformat()
    }

    # Log detailed metrics about the model
    log_metadata(
        artifact_name="price_prediction_model",
        infer_artifact=True,
        metadata=metadata
    )

    # Log detailed metrics about the model
    log_metadata(
        metadata=metadata
    )

    return model, HTMLString(generate_model_report(data=data, model=model_metrics))