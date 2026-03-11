import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# Import the preprocessor class from its own module (required for pickle to work)
from preprocessor import HousePricePreprocessor  # noqa: F401 — needed for joblib.load


# ── Feature engineering — must match training notebook exactly ────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['TotalSF'] = (
        df['TotalBsmtSF'].fillna(0) +
        df['1stFlrSF'].fillna(0) +
        df['2ndFlrSF'].fillna(0)
    )
    df['TotalBath'] = (
        df['FullBath'].fillna(0) +
        0.5 * df['HalfBath'].fillna(0) +
        df['BsmtFullBath'].fillna(0) +
        0.5 * df['BsmtHalfBath'].fillna(0)
    )
    df['HouseAge']    = df['YrSold'] - df['YearBuilt']
    df['RemodAge']    = df['YrSold'] - df['YearRemodAdd']
    df['IsRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    porch_cols = ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    df['TotalPorch']   = df[[c for c in porch_cols if c in df.columns]].fillna(0).sum(axis=1)
    df['HasPool']      = (df['PoolArea'].fillna(0) > 0).astype(int)
    df['HasGarage']    = (df['GarageArea'].fillna(0) > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'].fillna(0) > 0).astype(int)
    df['HasBasement']  = (df['TotalBsmtSF'].fillna(0) > 0).astype(int)
    return df


# ── Load artifacts at startup ─────────────────────────────────────────────────
model        = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

app = FastAPI(
    title="House Price Prediction API",
    description="Predicts US residential house sale prices using XGBoost. "
                "Send house features as JSON, get predicted price in USD.",
    version="1.0.0"
)


# ── Input schema ──────────────────────────────────────────────────────────────
# Note: fields with special characters (1stFlrSF, 2ndFlrSF, 3SsnPorch) are
# renamed to FirstFlrSF, SecondFlrSF, ThreeSsnPorch for JSON compatibility.
class HouseFeatures(BaseModel):
    MSSubClass:    Optional[float] = None
    MSZoning:      Optional[str]   = None
    LotFrontage:   Optional[float] = None
    LotArea:       Optional[float] = None
    Street:        Optional[str]   = None
    Alley:         Optional[str]   = None
    LotShape:      Optional[str]   = None
    LandContour:   Optional[str]   = None
    Utilities:     Optional[str]   = None
    LotConfig:     Optional[str]   = None
    LandSlope:     Optional[str]   = None
    Neighborhood:  Optional[str]   = None
    Condition1:    Optional[str]   = None
    Condition2:    Optional[str]   = None
    BldgType:      Optional[str]   = None
    HouseStyle:    Optional[str]   = None
    OverallQual:   Optional[float] = None
    OverallCond:   Optional[float] = None
    YearBuilt:     Optional[float] = None
    YearRemodAdd:  Optional[float] = None
    RoofStyle:     Optional[str]   = None
    RoofMatl:      Optional[str]   = None
    Exterior1st:   Optional[str]   = None
    Exterior2nd:   Optional[str]   = None
    MasVnrType:    Optional[str]   = None
    MasVnrArea:    Optional[float] = None
    ExterQual:     Optional[str]   = None
    ExterCond:     Optional[str]   = None
    Foundation:    Optional[str]   = None
    BsmtQual:      Optional[str]   = None
    BsmtCond:      Optional[str]   = None
    BsmtExposure:  Optional[str]   = None
    BsmtFinType1:  Optional[str]   = None
    BsmtFinSF1:    Optional[float] = None
    BsmtFinType2:  Optional[str]   = None
    BsmtFinSF2:    Optional[float] = None
    BsmtUnfSF:     Optional[float] = None
    TotalBsmtSF:   Optional[float] = None
    Heating:       Optional[str]   = None
    HeatingQC:     Optional[str]   = None
    CentralAir:    Optional[str]   = None
    Electrical:    Optional[str]   = None
    FirstFlrSF:    Optional[float] = None
    SecondFlrSF:   Optional[float] = None
    LowQualFinSF:  Optional[float] = None
    GrLivArea:     Optional[float] = None
    BsmtFullBath:  Optional[float] = None
    BsmtHalfBath:  Optional[float] = None
    FullBath:      Optional[float] = None
    HalfBath:      Optional[float] = None
    BedroomAbvGr:  Optional[float] = None
    KitchenAbvGr:  Optional[float] = None
    KitchenQual:   Optional[str]   = None
    TotRmsAbvGrd:  Optional[float] = None
    Functional:    Optional[str]   = None
    Fireplaces:    Optional[float] = None
    FireplaceQu:   Optional[str]   = None
    GarageType:    Optional[str]   = None
    GarageYrBlt:   Optional[float] = None
    GarageFinish:  Optional[str]   = None
    GarageCars:    Optional[float] = None
    GarageArea:    Optional[float] = None
    GarageQual:    Optional[str]   = None
    GarageCond:    Optional[str]   = None
    PavedDrive:    Optional[str]   = None
    WoodDeckSF:    Optional[float] = None
    OpenPorchSF:   Optional[float] = None
    EnclosedPorch: Optional[float] = None
    ThreeSsnPorch: Optional[float] = None
    ScreenPorch:   Optional[float] = None
    PoolArea:      Optional[float] = None
    PoolQC:        Optional[str]   = None
    Fence:         Optional[str]   = None
    MiscFeature:   Optional[str]   = None
    MiscVal:       Optional[float] = None
    MoSold:        Optional[float] = None
    YrSold:        Optional[float] = None
    SaleType:      Optional[str]   = None
    SaleCondition: Optional[str]   = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "OverallQual": 7,
                "GrLivArea": 1710,
                "GarageCars": 2,
                "TotalBsmtSF": 856,
                "FirstFlrSF": 856,
                "SecondFlrSF": 854,
                "FullBath": 2,
                "YearBuilt": 2003,
                "YearRemodAdd": 2003,
                "YrSold": 2010,
                "MoSold": 2,
                "Neighborhood": "CollgCr",
                "MSZoning": "RL",
                "LotArea": 8450
            }
        }
    }


@app.get("/")
def root():
    return {
        "message": "House Price Prediction API is running",
        "docs": "/docs",
        "predict_endpoint": "POST /predict"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: HouseFeatures):
    data = features.model_dump()

    # Map renamed fields back to original column names
    row = {
        "MSSubClass":    data["MSSubClass"],
        "MSZoning":      data["MSZoning"],
        "LotFrontage":   data["LotFrontage"],
        "LotArea":       data["LotArea"],
        "Street":        data["Street"],
        "Alley":         data["Alley"],
        "LotShape":      data["LotShape"],
        "LandContour":   data["LandContour"],
        "Utilities":     data["Utilities"],
        "LotConfig":     data["LotConfig"],
        "LandSlope":     data["LandSlope"],
        "Neighborhood":  data["Neighborhood"],
        "Condition1":    data["Condition1"],
        "Condition2":    data["Condition2"],
        "BldgType":      data["BldgType"],
        "HouseStyle":    data["HouseStyle"],
        "OverallQual":   data["OverallQual"],
        "OverallCond":   data["OverallCond"],
        "YearBuilt":     data["YearBuilt"],
        "YearRemodAdd":  data["YearRemodAdd"],
        "RoofStyle":     data["RoofStyle"],
        "RoofMatl":      data["RoofMatl"],
        "Exterior1st":   data["Exterior1st"],
        "Exterior2nd":   data["Exterior2nd"],
        "MasVnrType":    data["MasVnrType"],
        "MasVnrArea":    data["MasVnrArea"],
        "ExterQual":     data["ExterQual"],
        "ExterCond":     data["ExterCond"],
        "Foundation":    data["Foundation"],
        "BsmtQual":      data["BsmtQual"],
        "BsmtCond":      data["BsmtCond"],
        "BsmtExposure":  data["BsmtExposure"],
        "BsmtFinType1":  data["BsmtFinType1"],
        "BsmtFinSF1":    data["BsmtFinSF1"],
        "BsmtFinType2":  data["BsmtFinType2"],
        "BsmtFinSF2":    data["BsmtFinSF2"],
        "BsmtUnfSF":     data["BsmtUnfSF"],
        "TotalBsmtSF":   data["TotalBsmtSF"],
        "Heating":       data["Heating"],
        "HeatingQC":     data["HeatingQC"],
        "CentralAir":    data["CentralAir"],
        "Electrical":    data["Electrical"],
        "1stFlrSF":      data["FirstFlrSF"],
        "2ndFlrSF":      data["SecondFlrSF"],
        "LowQualFinSF":  data["LowQualFinSF"],
        "GrLivArea":     data["GrLivArea"],
        "BsmtFullBath":  data["BsmtFullBath"],
        "BsmtHalfBath":  data["BsmtHalfBath"],
        "FullBath":      data["FullBath"],
        "HalfBath":      data["HalfBath"],
        "BedroomAbvGr":  data["BedroomAbvGr"],
        "KitchenAbvGr":  data["KitchenAbvGr"],
        "KitchenQual":   data["KitchenQual"],
        "TotRmsAbvGrd":  data["TotRmsAbvGrd"],
        "Functional":    data["Functional"],
        "Fireplaces":    data["Fireplaces"],
        "FireplaceQu":   data["FireplaceQu"],
        "GarageType":    data["GarageType"],
        "GarageYrBlt":   data["GarageYrBlt"],
        "GarageFinish":  data["GarageFinish"],
        "GarageCars":    data["GarageCars"],
        "GarageArea":    data["GarageArea"],
        "GarageQual":    data["GarageQual"],
        "GarageCond":    data["GarageCond"],
        "PavedDrive":    data["PavedDrive"],
        "WoodDeckSF":    data["WoodDeckSF"],
        "OpenPorchSF":   data["OpenPorchSF"],
        "EnclosedPorch": data["EnclosedPorch"],
        "3SsnPorch":     data["ThreeSsnPorch"],
        "ScreenPorch":   data["ScreenPorch"],
        "PoolArea":      data["PoolArea"],
        "PoolQC":        data["PoolQC"],
        "Fence":         data["Fence"],
        "MiscFeature":   data["MiscFeature"],
        "MiscVal":       data["MiscVal"],
        "MoSold":        data["MoSold"],
        "YrSold":        data["YrSold"],
        "SaleType":      data["SaleType"],
        "SaleCondition": data["SaleCondition"],
    }

    df = pd.DataFrame([row])
    df = engineer_features(df)
    X_processed = preprocessor.transform(df)
    pred_log    = model.predict(X_processed)[0]
    pred_price  = float(np.expm1(pred_log))

    return {
        "predicted_price_usd": round(pred_price, 2),
        "predicted_price_formatted": f"${pred_price:,.0f}"
    }
