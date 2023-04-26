#!/usr/bin/env sh

export MLFLOW_TRACKING_URI=sqlite:///mlruns.db

mlflow models serve -m "models:/sklearn-5-neighbors-regressor-model/Production"