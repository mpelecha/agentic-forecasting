"""TimesFM 3.0 predictor — Google Research's zero-shot foundation model.

``TimesFM3Predictor`` wraps `google-research/timesfm
<https://github.com/google-research/timesfm>`_ version 3.0 behind the shared
:class:`~aieng.forecasting.evaluation.predictor.Predictor` interface. Unlike
every other predictor in :mod:`aieng.forecasting.methods.numerical`, it is not
fit at each origin — the checkpoint is pretrained once and loaded lazily on
first use, and every ``predict()`` call is a forward pass conditioned on
whatever history ``context`` exposes at that origin.

Requires the optional ``timesfm`` extra (``pip install timesfm[torch]``); the
import is deferred into :meth:`predict` so importing this module does not
require the dependency to be installed.

**License.** TimesFM 3.0's pretrained weights are distributed under
``timesfm-non-commercial-license-v1.0``, not Apache-2.0 like the source code
and earlier checkpoints — restricted to non-commercial, non-production use.
Fine for a research backtest; check the license before using this predictor
anywhere near production.

**Pretraining-leakage caveat.** TimesFM 3.0 is pretrained on GiftEvalPretrain
plus Wikipedia Pageviews (disclosed cutoff Nov 2023) and Google Trends top
queries (disclosed cutoff EoY 2022), alongside synthetic/augmented data. A
zero-shot forecast is only genuinely out-of-sample if the backtest origin
falls after whatever cutoff applies to the series being forecast — and that
per-series cutoff is not fully disclosed. Treat any backtest origin before
~2024 as a weaker signal than the other predictors in this package, which are
fit fresh at every origin from data ``context`` alone and carry no such risk.

**No log-return variant.** Every other predictor in this package that offers
one (:class:`~aieng.forecasting.methods.numerical.darts_arima.DartsAutoARIMAPredictor`,
:class:`~aieng.forecasting.methods.numerical.error_correction_regression.ErrorCorrectionRegressionPredictor`)
reconstructs a price-space forecast by cumulating *sample paths* of daily log
returns, which is valid regardless of how those steps are correlated because
it operates on realized draws, not marginal quantiles. TimesFM 3.0's
``predict_batch`` returns only a point forecast and marginal per-step
quantiles, no sample paths — cumulating marginal quantiles across horizon
steps would require assuming the daily quantile ranks are perfectly
correlated (comonotonic), which is not verified here and would silently
mis-state interval width. This predictor forecasts price level directly and
leaves a log-return variant for whoever verifies that assumption first.

Usage::

    from aieng.forecasting.methods.numerical.timesfm3 import TimesFM3Predictor
    from aieng.forecasting.evaluation import backtest, BacktestSpec

    predictor = TimesFM3Predictor()
    result = backtest(predictor=predictor, spec=spec, data_service=svc)
    print(f"Mean CRPS: {result.mean_score:.4f}")
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask

# TimesFM 3.0's native quantile head emits these 9 levels (0.1 to 0.9). This
# is *not* aieng.forecasting.evaluation.prediction.STANDARD_QUANTILES (which
# also has 0.05 and 0.95) -- the model does not expose those tails, and
# extrapolating them from a parametric tail assumption would be inventing
# data the model never produced. Payload quantile dicts from this predictor
# are therefore missing the two outermost STANDARD_QUANTILES levels; CRPS and
# 80% coverage (keyed on 0.1/0.9) are unaffected.
_TIMESFM3_QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

_TARGET_COLUMN = "__target__"


class TimesFM3Predictor(Predictor):
    """Zero-shot probabilistic predictor wrapping Google's TimesFM 3.0.

    No fitting: the pretrained checkpoint is loaded once (lazily, on first
    ``predict()`` call) and every origin is a single forward pass over the
    trailing ``max_context`` observations available at that origin.

    Parameters
    ----------
    covariate_series_ids : list[str] | None, default=None
        Optional covariate panel, passed to TimesFM 3.0 as past-and-future
        dynamic covariates. ``None`` runs the univariate model.
    level_valued_covariate_series_ids : list[str] | None, default=None
        Subset of ``covariate_series_ids`` that are price/index levels rather
        than returns. Required when ``covariate_series_ids`` is set, so the
        future covariate path (beyond the forecast origin, which the model
        needs but the caller cannot observe) can be built correctly: a level
        series is held at its last cutoff-safe value, a return series is held
        at 0 (no further change) -- the same "future covariates held flat"
        convention used elsewhere in this project
        (:class:`~aieng.forecasting.methods.numerical.error_correction_regression.ErrorCorrectionRegressionPredictor`),
        applied per-series rather than uniformly. Every id in
        ``covariate_series_ids`` not listed here is treated as return-valued.
    max_context : int, default=2048
        Number of most-recent target observations fed to the model. TimesFM
        3.0 supports up to a 16k-token context; 2048 (~8 years of business
        days for a daily series) is a default chosen for inference speed, not
        a model limit -- raise it if more history should inform the forecast.
    checkpoint : str, default="google/timesfm-3.0-pytorch"
        Hugging Face checkpoint id passed to ``ModelConfig``.
    device : str, default="cpu"
        Passed straight to ``ModelConfig``. Set ``"cuda"`` if a GPU is
        available; CPU inference is slow enough at large ``max_context`` that
        a dense multi-year backtest grid can take a long time on CPU alone.
    per_core_batch_size : int, default=32
        Passed straight to ``ModelConfig``.
    variant_tag : str | None, default=None
        Optional suffix appended to ``predictor_id``, for running more than
        one configuration (e.g. different ``max_context`` values) without
        cache collisions.

    Notes
    -----
    Requires ``pip install timesfm[torch]`` (the ``timesfm`` optional
    dependency group in this project's ``pyproject.toml``). Not installed by
    default because the pretrained weights carry a non-commercial license and
    the package pulls in a full PyTorch stack.
    """

    def __init__(
        self,
        covariate_series_ids: list[str] | None = None,
        *,
        level_valued_covariate_series_ids: list[str] | None = None,
        max_context: int = 2048,
        checkpoint: str = "google/timesfm-3.0-pytorch",
        device: str = "cpu",
        per_core_batch_size: int = 32,
        variant_tag: str | None = None,
    ) -> None:
        if covariate_series_ids and level_valued_covariate_series_ids is None:
            raise ValueError(
                "level_valued_covariate_series_ids is required when covariate_series_ids is set -- "
                "TimesFM3Predictor needs to know which covariates are levels (held flat at their last "
                "value beyond the origin) versus returns (held at 0) to build the future covariate path."
            )
        unknown = set(level_valued_covariate_series_ids or []) - set(covariate_series_ids or [])
        if unknown:
            raise ValueError(f"level_valued_covariate_series_ids contains ids not present in covariate_series_ids: {sorted(unknown)}")

        self._covariate_series_ids = list(covariate_series_ids or [])
        self._level_valued = set(level_valued_covariate_series_ids or [])
        self._max_context = max_context
        self._checkpoint = checkpoint
        self._device = device
        self._per_core_batch_size = per_core_batch_size
        self._variant_tag = variant_tag
        self._forecaster: Any = None  # lazily constructed; see _get_forecaster

    @property
    def predictor_id(self) -> str:
        """Return a stable identifier, suffixed ``_cov`` when covariates are used."""
        suffix = "_cov" if self._covariate_series_ids else ""
        tag = f"_{self._variant_tag}" if self._variant_tag else ""
        return f"timesfm3{suffix}{tag}"

    def _get_forecaster(self) -> Any:
        """Load the checkpoint once and reuse it across every origin."""
        if self._forecaster is None:
            from timesfm3 import ModelConfig, TimesFM3Evaluator  # noqa: PLC0415

            config = ModelConfig(
                checkpoint_path=self._checkpoint,
                per_core_batch_size=self._per_core_batch_size,
                device=self._device,
            )
            self._forecaster = TimesFM3Evaluator(config)
        return self._forecaster

    def _aligned_frame(self, task: ForecastingTask, context: ForecastContext) -> pd.DataFrame:
        """Inner-join target and covariates on ``timestamp``, trailing ``max_context`` rows."""
        target = context.get_series(task.target_series_id)[["timestamp", "value"]].rename(columns={"value": _TARGET_COLUMN})
        merged = target
        for series_id in self._covariate_series_ids:
            cov = context.get_series(series_id)[["timestamp", "value"]].rename(columns={"value": series_id})
            merged = pd.merge(merged, cov, on="timestamp", how="inner")
        merged = merged.sort_values("timestamp").dropna().reset_index(drop=True)
        return merged.iloc[-self._max_context :].reset_index(drop=True)

    def _future_covariate_path(self, frame: pd.DataFrame, horizon: int) -> np.ndarray:
        """Return the ``(n_covariates, horizon)`` future covariate path, held flat per-series."""
        rows = []
        for series_id in self._covariate_series_ids:
            last_value = float(frame[series_id].iloc[-1])
            hold_value = last_value if series_id in self._level_valued else 0.0
            rows.append(np.full(horizon, hold_value, dtype=np.float32))
        return np.stack(rows, axis=0)

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Produce a TimesFM 3.0 forecast for every horizon in the task.

        Parameters
        ----------
        task : ForecastingTask
            Defines the target series, horizons, and frequency.
        context : ForecastContext
            Cutoff-scoped data view.

        Returns
        -------
        list[Prediction]
            One ``ContinuousForecast`` per horizon step in ``task.horizons``,
            carrying TimesFM 3.0's native 9 quantile levels (0.1-0.9).
        """
        forecaster = self._get_forecaster()
        frame = self._aligned_frame(task, context)
        target_context = frame[_TARGET_COLUMN].to_numpy(dtype=np.float32)
        max_horizon = max(task.horizons)

        kwargs: dict[str, Any] = {}
        if self._covariate_series_ids:
            past_cov = frame[self._covariate_series_ids].to_numpy(dtype=np.float32).T  # (n_cov, max_context)
            future_cov = self._future_covariate_path(frame, max_horizon)  # (n_cov, max_horizon)
            past_future_cov = np.concatenate([past_cov, future_cov], axis=1)
            kwargs["past_future_covariates"] = [past_future_cov]

        outputs = list(
            forecaster.predict_batch(
                contexts=[target_context],
                horizon=max_horizon,
                return_quantiles=True,
                use_symmetric_averaging=False,
                **kwargs,
            )
        )
        output = outputs[0]
        point_forecast: np.ndarray = np.asarray(output.forecast).reshape(-1)
        quantiles: np.ndarray = np.asarray(output.quantiles).reshape(max_horizon, len(_TIMESFM3_QUANTILES))

        offset = pd.tseries.frequencies.to_offset(task.frequency)
        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        predictions: list[Prediction] = []
        for h in task.horizons:
            payload = ContinuousForecast(
                point_forecast=float(point_forecast[h - 1]),
                quantiles={q: float(quantiles[h - 1, i]) for i, q in enumerate(_TIMESFM3_QUANTILES)},
            )
            forecast_date: datetime = (pd.Timestamp(context.as_of) + offset * h).to_pydatetime()
            predictions.append(
                Prediction(
                    predictor_id=self.predictor_id,
                    task_id=task.task_id,
                    issued_at=issued_at,
                    as_of=context.as_of,
                    forecast_date=forecast_date,
                    payload=payload,
                )
            )

        return predictions
