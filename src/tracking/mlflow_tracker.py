# mlflow_tracker.py
# Clase centralizada para track & trace de MLflow en evaluación/entrenamiento
# ===========================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import os
import json
import hashlib
import numpy as np
import pandas as pd

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    import mlflow.sklearn
    import mlflow.pyfunc
    from mlflow.pyfunc import PythonModel
except Exception:  # pragma: no cover
    mlflow = None
    MlflowClient = None


@dataclass
class MLflowTracker:
    """Gestor centralizado de tracking con MLflow.

    - Registra parámetros, métricas y artefactos de evaluación
    - Registra/versiona modelos en el Model Registry
    - Determina y etiqueta campeones por zona y global
    - Puede re-evaluar corridas existentes del experimento
    """
    tracking_uri: Optional[str] = None
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    enabled: bool = True
    champion_metric_name: str = "RMSE"
    champion_higher_is_better: bool = False

    def _extract_model_params(self, model: Any) -> Dict[str, Any]:
        """Extrae parámetros simples de un estimador/wrapper.

        Intenta `get_params()` (sklearn) y algunos atributos comunes
        usados en este proyecto (VAR/LSTM/RF/XGB).
        """
        params: Dict[str, Any] = {}
        try:
            if hasattr(model, "get_params"):
                wrapped = model.get_params() or {}
                for k, v in wrapped.items():
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        params[k] = v
        except Exception:
            pass

        # VAR specifics
        for attr in ("best_lag",):
            if hasattr(model, attr):
                try:
                    params[attr] = getattr(model, attr)
                except Exception:
                    pass

        # RF/XGB pipeline params
        pipeline_obj = None
        for attr in ("rf_model", "xgb_model", "model", "pipeline"):
            if hasattr(model, attr):
                cand = getattr(model, attr)
                if hasattr(cand, "get_params"):
                    pipeline_obj = cand
                    break
        try:
            if pipeline_obj is not None and hasattr(pipeline_obj, "get_params"):
                est_params = pipeline_obj.get_params()
                keys = {
                    "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features",
                    "learning_rate", "subsample", "colsample_bytree", "gamma", "reg_lambda", "reg_alpha",
                }
                for k in keys:
                    if k in est_params:
                        v = est_params[k]
                        if isinstance(v, (str, int, float, bool)) or v is None:
                            params[k] = v
        except Exception:
            pass

        # LSTM specifics
        for attr in ("n_steps_in", "n_features", "target_col"):
            if hasattr(model, attr):
                try:
                    params[attr] = getattr(model, attr)
                except Exception:
                    pass
        return params

    def _log_predictions_artifact(
        self,
        zone: str,
        model_name: str,
        test_df: pd.DataFrame,
        predictions: Any,
        run_dir: Path,
        n_steps: int,
    ) -> Optional[Path]:
        """Guarda CSV con y_true vs y_pred como artefacto."""
        try:
            y_true = test_df.head(n_steps)[zone]
            if isinstance(predictions, pd.Series):
                y_pred = predictions
            elif isinstance(predictions, pd.DataFrame):
                y_pred = predictions[zone] if zone in predictions.columns else predictions.iloc[:, 0]
            else:
                y_pred = pd.Series(predictions, index=y_true.index)

            out_dir = run_dir / "predictions"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"pred_{model_name}_{zone}.csv"

            pd.DataFrame({
                "y_true": y_true.values,
                "y_pred": np.array(y_pred)[: len(y_true)],
            }, index=y_true.index).to_csv(out_path, index=True)

            return out_path
        except Exception:
            return None

    def _reg_name_from_key(self, base_model_name: str, key: str) -> str:
        if key == "VAR":
            return f"{base_model_name}_VAR"
        if key.startswith("RF_"):
            return f"{base_model_name}_RF_{key.split('_',1)[1]}"
        if key.startswith("XGB_"):
            return f"{base_model_name}_XGB_{key.split('_',1)[1]}"
        if key.startswith("LSTM_"):
            return f"{base_model_name}_LSTM_{key.split('_',1)[1]}"
        return f"{base_model_name}_{key}"

    def _register_single_model(self, *, key: str, obj: Any) -> None:
        """Registra/Versiona un modelo en el Model Registry usando el run activo."""
        if mlflow is None:
            return
        base_model_name = os.getenv("MODEL_NAME", "powerTetouan")
        reg_name = self._reg_name_from_key(base_model_name, key)

        class _WrapperPyfunc(PythonModel):
            def __init__(self, wrapper):
                self.wrapper = wrapper
            def predict(self, context, model_input):  # pragma: no cover
                raise NotImplementedError("Use project-specific inference pipeline.")

        try:
            mlflow.set_tags({"model_register": "evaluation", "model_key": key, "registered_name": reg_name})
        except Exception:
            pass

        # Evitar duplicar versiones: si ya existe una versión registrada para este run, omitir
        try:
            active = mlflow.active_run()
            run_id = active.info.run_id if active else None
        except Exception:
            run_id = None
        if run_id:
            try:
                client = MlflowClient()
                versions = client.search_model_versions(f"run_id = '{run_id}'")
                for mv in versions:
                    if mv.name == reg_name:
                        return
            except Exception:
                pass

        est = None
        for attr in ("rf_model", "xgb_model", "model", "pipeline"):
            if hasattr(obj, attr):
                cand = getattr(obj, attr)
                if hasattr(cand, "get_params") and hasattr(cand, "predict"):
                    est = cand
                    break
        if est is not None:
            try:
                mlflow.sklearn.log_model(est, artifact_path="model", registered_model_name=reg_name)
                return
            except Exception:
                pass
        try:
            mlflow.pyfunc.log_model(artifact_path="model", python_model=_WrapperPyfunc(obj), registered_model_name=reg_name)
        except Exception:
            try:
                tmp_params = {}
                if hasattr(obj, "get_params"):
                    p = obj.get_params() or {}
                    for k, v in p.items():
                        if isinstance(v, (str, int, float, bool)) or v is None:
                            tmp_params[k] = v
                tmp_path = Path("reports") / "mlflow_runs" / "models_meta"
                tmp_path.mkdir(parents=True, exist_ok=True)
                meta_file = tmp_path / f"{key}_params.json"
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump({"model_key": key, "params": tmp_params}, f, indent=2)
                mlflow.log_artifact(str(meta_file), artifact_path="models_meta")
            except Exception:
                pass

    def _register_models(self, models: Dict[str, Any]) -> None:
        """Registra/Versiona modelos en el Model Registry.

        Preferentemente usa `mlflow.sklearn.log_model`; si no es posible,
        registra un `pyfunc` contenedor como fallback y adjunta metadatos.
        """
        if mlflow is None:
            return
        base_model_name = os.getenv("MODEL_NAME", "powerTetouan")

        class _WrapperPyfunc(PythonModel):
            def __init__(self, wrapper):
                self.wrapper = wrapper
            def predict(self, context, model_input):  # pragma: no cover
                raise NotImplementedError("Use project-specific inference pipeline.")

        for key, obj in models.items():
            reg_name = self._reg_name_from_key(base_model_name, key)
            run_name_reg = f"Register-{key}"
            with mlflow.start_run(run_name=run_name_reg, nested=True):
                mlflow.set_tags({"model_register": "evaluation", "model_key": key, "registered_name": reg_name})
                est = None
                for attr in ("rf_model", "xgb_model", "model", "pipeline"):
                    if hasattr(obj, attr):
                        cand = getattr(obj, attr)
                        if hasattr(cand, "get_params") and hasattr(cand, "predict"):
                            est = cand
                            break
                if est is not None:
                    try:
                        mlflow.sklearn.log_model(est, artifact_path="model", registered_model_name=reg_name)
                        continue
                    except Exception:
                        pass
                try:
                    mlflow.pyfunc.log_model(artifact_path="model", python_model=_WrapperPyfunc(obj), registered_model_name=reg_name)
                except Exception:
                    # Meta como fallback
                    try:
                        tmp_params = {}
                        if hasattr(obj, "get_params"):
                            p = obj.get_params() or {}
                            for k, v in p.items():
                                if isinstance(v, (str, int, float, bool)) or v is None:
                                    tmp_params[k] = v
                        tmp_path = Path("reports") / "mlflow_runs" / "models_meta"
                        tmp_path.mkdir(parents=True, exist_ok=True)
                        meta_file = tmp_path / f"{key}_params.json"
                        with open(meta_file, "w", encoding="utf-8") as f:
                            json.dump({"model_key": key, "params": tmp_params}, f, indent=2)
                        mlflow.log_artifact(str(meta_file), artifact_path="models_meta")
                    except Exception:
                        pass

    def _dataset_fingerprint(self, df: pd.DataFrame, sample_rows: int = 1000) -> Dict[str, Any]:
        """Crea una firma ligera del dataset para trazabilidad."""
        try:
            cols = list(df.columns)
            dtypes = {c: str(df[c].dtype) for c in cols}
            nulls = {c: int(df[c].isna().sum()) for c in cols}
            sample_csv = df.head(sample_rows).to_csv(index=True)
            sha = hashlib.sha256(sample_csv.encode("utf-8")).hexdigest()
            sig: Dict[str, Any] = {
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "columns": cols,
                "dtypes": dtypes,
                "nulls": nulls,
                "sample_hash_sha256": sha,
            }
            try:
                sig["index_name"] = df.index.name
                sig["index_dtype"] = str(df.index.dtype)
            except Exception:
                pass
            return sig
        except Exception:
            return {"rows": int(df.shape[0]), "cols": int(df.shape[1])}

    def _log_datasets(self, *, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Loguea artefactos y metadatos de datasets de train y test en el run activo."""
        try:
            active_run = mlflow.active_run()
            run_id = active_run.info.run_id if active_run else None
        except Exception:
            run_id = None

        train_sig = self._dataset_fingerprint(train_df)
        test_sig = self._dataset_fingerprint(test_df)

        # Params y tags resumidos
        try:
            mlflow.log_params({
                "data.train_rows": int(train_df.shape[0]),
                "data.test_rows": int(test_df.shape[0]),
                "data.train_cols": int(train_df.shape[1]),
                "data.test_cols": int(test_df.shape[1]),
            })
        except Exception:
            pass

    def _log_summary_metrics(self, *, metrics: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        """Loguea en el run padre un resumen de métricas por zona y modelo."""
        try:
            for zone, models_metrics in (metrics or {}).items():
                if not isinstance(models_metrics, dict):
                    continue
                for model_name, mvals in (models_metrics or {}).items():
                    if not isinstance(mvals, dict):
                        continue
                    for mkey, mval in mvals.items():
                        if mval is None:
                            continue
                        try:
                            mlflow.log_metric(f"{zone}.{model_name}.{mkey}", float(mval))
                        except Exception:
                            continue
        except Exception:
            pass
        try:
            mlflow.set_tags({
                "dataset.train.hash": train_sig.get("sample_hash_sha256", ""),
                "dataset.test.hash": test_sig.get("sample_hash_sha256", ""),
            })
        except Exception:
            pass

        # Artefactos
        try:
            base = Path("reports") / "mlflow_runs"
            if run_id:
                base = base / run_id
            data_dir = base / "datasets"
            (data_dir / "train").mkdir(parents=True, exist_ok=True)
            (data_dir / "test").mkdir(parents=True, exist_ok=True)

            with open(data_dir / "train" / "signature.json", "w", encoding="utf-8") as f:
                json.dump(train_sig, f, indent=2)
            with open(data_dir / "test" / "signature.json", "w", encoding="utf-8") as f:
                json.dump(test_sig, f, indent=2)
            train_df.head(200).to_csv(data_dir / "train" / "head.csv", index=True)
            test_df.head(200).to_csv(data_dir / "test" / "head.csv", index=True)

            mlflow.log_artifacts(str(data_dir / "train"), artifact_path="datasets/train")
            mlflow.log_artifacts(str(data_dir / "test"), artifact_path="datasets/test")
        except Exception:
            pass

    def _log_global_artifacts(self, *, metrics_output: Path, figures_output: Path) -> None:
        """Sube artefactos globales (métricas JSON y figuras) al run activo."""
        try:
            if metrics_output.exists():
                mlflow.log_artifact(str(metrics_output), artifact_path="metrics")
        except Exception:
            pass
        try:
            if figures_output.exists():
                for img in figures_output.glob("*.png"):
                    mlflow.log_artifact(str(img), artifact_path="figures")
        except Exception:
            pass

    def _log_nested_model_runs(
        self,
        *,
        models: Dict[str, Any],
        metrics: Dict[str, Dict[str, Dict[str, float]]],
        predictions: Dict[str, Any],
        test_df: pd.DataFrame,
        n_steps: int,
    ) -> None:
        """Crea runs anidados por zona/modelo, loguea métricas y versiona modelos."""
        try:
            zones = list(metrics.keys())
            for zone in zones:
                zone_metrics = metrics.get(zone, {}) or {}

                # VAR: usar métricas por zona
                if "VAR" in zone_metrics and "VAR" in models and "VAR" in predictions:
                    m = zone_metrics["VAR"]
                    with mlflow.start_run(run_name=f"Evaluate-VAR-{zone}", nested=True):
                        try:
                            mlflow.set_tags({"stage": "evaluate", "zone": zone, "model_name": "VAR"})
                        except Exception:
                            pass
                        try:
                            mlflow.log_params(self._extract_model_params(models["VAR"]))
                        except Exception:
                            pass
                        try:
                            mlflow.log_metrics({k: float(v) for k, v in m.items() if v is not None})
                        except Exception:
                            pass
                        try:
                            run_id = mlflow.active_run().info.run_id
                            run_dir = Path("reports") / "mlflow_runs" / run_id
                            art = self._log_predictions_artifact(zone, "VAR", test_df, predictions["VAR"], run_dir, n_steps)
                            if art is not None:
                                mlflow.log_artifact(str(art), artifact_path="predictions")
                        except Exception:
                            pass
                        try:
                            self._register_single_model(key="VAR", obj=models["VAR"])
                        except Exception:
                            pass

                # Modelos por zona
                for short_name in ("RF", "XGB", "LSTM"):
                    if short_name in zone_metrics:
                        model_key = f"{short_name}_{zone}"
                        if model_key not in models:
                            continue
                        m = zone_metrics[short_name]
                        with mlflow.start_run(run_name=f"Evaluate-{short_name}-{zone}", nested=True):
                            try:
                                mlflow.set_tags({"stage": "evaluate", "zone": zone, "model_name": short_name})
                            except Exception:
                                pass
                            try:
                                mlflow.log_params(self._extract_model_params(models[model_key]))
                            except Exception:
                                pass
                            try:
                                mlflow.log_metrics({k: float(v) for k, v in m.items() if v is not None})
                            except Exception:
                                pass
                            try:
                                if zone in predictions and short_name in predictions.get(zone, {}):
                                    run_id = mlflow.active_run().info.run_id
                                    run_dir = Path("reports") / "mlflow_runs" / run_id
                                    art = self._log_predictions_artifact(zone, short_name, test_df, predictions[zone][short_name], run_dir, n_steps)
                                    if art is not None:
                                        mlflow.log_artifact(str(art), artifact_path="predictions")
                            except Exception:
                                pass
                            try:
                                self._register_single_model(key=model_key, obj=models[model_key])
                            except Exception:
                                pass
        except Exception:
            # No interrumpe si falla algo en runs anidados
            pass

    def promote_best_run_to_champion(
        self,
        *,
        metric_name: str = "RMSE",
        higher_is_better: bool = False,
        registered_model_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Busca la mejor corrida del experimento y asigna alias 'champion'."""
        if not self.enabled or mlflow is None:
            return None

        exp_name = self.experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", "TetouanCityPower")
        try:
            experiment = mlflow.get_experiment_by_name(exp_name)
            if experiment is None:
                return None
        except Exception:
            return None

        try:
            df = mlflow.search_runs(experiment_ids=[experiment.experiment_id], output_format="pandas")
        except Exception:
            return None

        col = f"metrics.{metric_name}"
        if col not in df.columns:
            return None
        if "tags.stage" in df.columns:
            df = df[df["tags.stage"] == "evaluate"]
        df = df[df[col].notna()]
        if df.empty:
            return None

        ascending = not higher_is_better
        df = df.sort_values(col, ascending=ascending)
        best_row = df.iloc[0]
        best_run_id = str(best_row["run_id"])
        best_val = float(best_row[col])

        try:
            client = MlflowClient()
        except Exception:
            return None

        # Nombre preferido: tag 'registered_name' del run
        reg_name = None
        try:
            best_run = client.get_run(best_run_id)
            reg_name = best_run.data.tags.get("registered_name")
        except Exception:
            reg_name = None
        if not reg_name:
            reg_name = registered_model_name or os.getenv("MODEL_NAME", "powerTetouan")

        # Buscar si ya existe una versión para ese run
        version_to_promote = None
        try:
            versions = client.search_model_versions(f"run_id = '{best_run_id}'")
            if versions:
                version_to_promote = versions[0].version
                reg_name = versions[0].name or reg_name
        except Exception:
            versions = []

        # Si no hay versión, intenta registrar el modelo del run
        if version_to_promote is None:
            try:
                result = mlflow.register_model(model_uri=f"runs:/{best_run_id}/model", name=reg_name)
                version_to_promote = result.version
            except Exception:
                return None

        # Asignar alias champion
        try:
            client.set_registered_model_alias(name=reg_name, alias="champion", version=version_to_promote)
        except Exception:
            return None

        return {
            "experiment": exp_name,
            "run_id": best_run_id,
            "metric": metric_name,
            "value": best_val,
            "model_name": reg_name,
            "version": str(version_to_promote),
            "alias": "champion",
        }

    def log_evaluation(
        self,
        *,
        models: Dict[str, Any],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        predictions: Dict[str, Any],
        metrics: Dict[str, Dict[str, Dict[str, float]]],
        n_steps: int,
        models_dir: Path,
        metrics_output: Path,
        figures_output: Path,
    ) -> None:
        """Registra evaluación completa y etiqueta campeones de esta corrida."""
        if not self.enabled or mlflow is None:
            return

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", "TetouanCityPower"))

        run_name = self.run_name or os.getenv("MLFLOW_RUN_NAME", "ModelEvaluation")
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "stage": "evaluate",
                "framework": "custom+sklearn+statsmodels+tf",
                "n_steps": str(n_steps),
                "models_dir": str(models_dir),
            })

            # Params globales
            mlflow.log_params({
                "eval.n_steps": n_steps,
                "data.train_rows": int(train_df.shape[0]),
                "data.test_rows": int(test_df.shape[0]),
            })

            # Track & trace de datasets (artefactos y firmas)
            self._log_datasets(train_df=train_df, test_df=test_df)

            # Resumen de métricas en el run padre
            self._log_summary_metrics(metrics=metrics)

            # Artefactos globales
            self._log_global_artifacts(metrics_output=metrics_output, figures_output=figures_output)

            # Runs anidados por modelo
            self._log_nested_model_runs(
                models=models,
                metrics=metrics,
                predictions=predictions,
                test_df=test_df,
                n_steps=n_steps,
            )

            # Registrar/Versionar también usando _register_models
            self._register_models(models)

            # Seleccionar champion global dentro del experimento
            try:
                self.promote_best_run_to_champion(
                    metric_name=self.champion_metric_name,
                    higher_is_better=self.champion_higher_is_better,
                )
            except Exception:
                pass

    def evaluate_existing_runs_and_set_champions(
        self,
        *,
        metric_name: str = "RMSE",
        higher_is_better: bool = False,
    ) -> Dict[str, Any]:
        """Evalúa corridas existentes del experimento y define campeones.

        - Busca corridas finalizadas con `tags.stage = evaluate` y `tags.zone`.
        - Selecciona, por zona, el run con mejor métrica.
        - Selecciona un campeón global considerando todas las zonas.
        - Etiqueta estos runs y, si es posible, asigna alias en el Registry:
          `champion` (por zona) y `overall_champion` (global).

        Retorna un resumen con los campeones seleccionados.
        """
        summary: Dict[str, Any] = {"best_by_zone": {}, "best_overall": None}
        if not self.enabled or mlflow is None:
            return summary

        try:
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            exp_name = self.experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", "TetouanCityPower")
            experiment = mlflow.get_experiment_by_name(exp_name)
            if experiment is None:
                return summary
        except Exception:
            return summary

        try:
            client = MlflowClient() if MlflowClient is not None else None
        except Exception:
            client = None
        if client is None:
            return summary

        order = "DESC" if higher_is_better else "ASC"
        try:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="attributes.status = 'FINISHED' and tags.stage = 'evaluate' and tags.zone IS NOT NULL",
                order_by=[f"metrics.{metric_name} {order}"],
                max_results=50000,
            )
        except Exception:
            runs = []

        best_by_zone: Dict[str, Any] = {}
        best_overall: Optional[Dict[str, Any]] = None
        for r in runs:
            zone = r.data.tags.get("zone")
            model_name = r.data.tags.get("model_name")
            val = r.data.metrics.get(metric_name)
            if zone is None or val is None:
                continue
            current = best_by_zone.get(zone)
            better = (val > current["metric"]) if (higher_is_better and current) else (
                val < current["metric"] if (not higher_is_better and current) else True
            )
            if current is None or better:
                best_by_zone[zone] = {"run_id": r.info.run_id, "metric": val, "model_name": model_name}
            if best_overall is None:
                best_overall = {"run_id": r.info.run_id, "metric": val, "model_name": model_name, "zone": zone}
            else:
                if (higher_is_better and val > best_overall["metric"]) or ((not higher_is_better) and val < best_overall["metric"]):
                    best_overall = {"run_id": r.info.run_id, "metric": val, "model_name": model_name, "zone": zone}

        # Etiquetado y alias por zona
        for zone, info in best_by_zone.items():
            run_id = info["run_id"]
            try:
                client.set_tag(run_id, "champion", "true")
                client.set_tag(run_id, "champion.zone", zone)
                client.set_tag(run_id, "champion.metric", metric_name)
                client.set_tag(run_id, "champion.value", str(info["metric"]))
            except Exception:
                pass
            try:
                versions = client.search_model_versions(f"run_id = '{run_id}'")
                for mv in versions:
                    try:
                        client.set_model_version_tag(mv.name, mv.version, "champion", "true")
                        client.set_registered_model_alias(mv.name, alias="champion", version=mv.version)
                        client.transition_model_version_stage(mv.name, mv.version, stage="Staging", archive_existing_versions=False)
                    except Exception:
                        continue
            except Exception:
                pass

        # Alias global overall_champion
        if best_overall is not None:
            run_id = best_overall["run_id"]
            try:
                client.set_tag(run_id, "overall_champion", "true")
                client.set_tag(run_id, "overall.metric", metric_name)
                client.set_tag(run_id, "overall.value", str(best_overall["metric"]))
                client.set_tag(run_id, "overall.zone", str(best_overall.get("zone")))
            except Exception:
                pass
            try:
                versions = client.search_model_versions(f"run_id = '{run_id}'")
                for mv in versions:
                    try:
                        client.set_registered_model_alias(mv.name, alias="overall_champion", version=mv.version)
                    except Exception:
                        continue
            except Exception:
                pass

        summary["best_by_zone"] = best_by_zone
        summary["best_overall"] = best_overall
        return summary

