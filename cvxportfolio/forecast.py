# Copyright (C) 2023-2024 Enzo Busseti
#
# This file is part of Cvxportfolio.
#
# Cvxportfolio is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Cvxportfolio is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# Cvxportfolio. If not, see <https://www.gnu.org/licenses/>.
r"""This module implements the simple forecasting models used by Cvxportfolio.

These are standard ones like historical mean, variance, and covariance. In
most cases the models implemented here are equivalent to the relevant Pandas
DataFrame methods, including (most importantly) the logic used to skip over any
``np.nan``. There are some subtle differences explained below.

Our forecasters are optimized to be evaluated sequentially in time: at each
point in time in a back-test the forecast computed at the previous time step
is updated with the most recent observation. This is in some cases (e.g.,
covariances) much more efficient than computing from scratch.

Most of our forecasters implement both a rolling window and exponential moving
average logic. These are specified by the ``rolling`` and ``half_life``
parameters respectively, which are either Pandas Timedeltas or ``np.inf``.
The latter is the default, and means that the whole past is used, with no
exponential smoothing. Note that it's possible to use both, *e.g.*,
estimate covariance matrices ignoring past returns older than 5 years and
smoothing the recent ones using an exponential kernel with half-life of 1 year.

Finally, we note that the covariance, variance and standard deviation
forecasters implement the ``kelly`` parameter, which is True by default.
This is a simple trick explained in
:paper:`section 4.2 (page 28) <section.4.2>` of the paper, simplifies the
computation and provides in general (slightly) higher performance.
For example, using the notation of the paper, the classical definition of
covariance is

.. math::

    \Sigma = \mathbf{E}(r_t - \mu)(r_t - \mu)^T,

this is what you get by setting ``kelly=False``. The default, ``kelly=True``,
gives instead

.. math::

    \Sigma^\text{kelly} = \mathbf{E}r_t r_t^T = \Sigma + \mu \mu^T,

so that the resulting Markowitz-style optimization problem corresponds to
the second order Taylor approximation of a (risk-constrained) Kelly objective,
as is explained briefly :paper:`at page 28 of the paper <section.4.2>`, or with
more detail (and hard-to-read math) in `section 6 of the Risk-Constrained Kelly
Gambling paper
<https://web.stanford.edu/~boyd/papers/pdf/kelly.pdf#section.6>`_.

Lastly, some forecasters implement a basic caching mechanism.
This is used in two ways. First, online (e.g., in back-test): if multiple
copies of the same forecaster need access to the estimated value, as is the
case in :class:`cvxportfolio.MultiPeriodOptimization` policies, the expensive
evaluation is only done once. Then, offline, provided that the
:class:`cvxportfolio.data.MarketData` server used implements the
:meth:`cvxportfolio.data.MarketData.partial_universe_signature` method
(so that we can certify which market data the cached values are computed on).
This type of caching simply saves on disk the forecasted values, and makes it
available automatically next time the user runs a back-test on the same market
data (and same universe). This is especially useful when doing hyper-parameter
optimization, so that expensive computations like evaluating large covariance
matrices are only done once.

How to use them
~~~~~~~~~~~~~~~

These forecasters are each the default option of some Cvxportfolio optimization
term, for example :class:`HistoricalMeanReturn` is the default used by
:class:`cvxportfolio.ReturnsForecast`. In this way each is used with its
default options. If you want to change the options you can simply pass
the relevant forecaster class, instantiated with the options of your choice,
to the Cvxportfolio object. For example

.. code-block::

    import cvxportfolio as cvx
    from cvxportfolio.forecast import HistoricalMeanReturn
    import pandas as pd

    returns_forecast = cvx.ReturnsForecast(
        r_hat = HistoricalMeanReturn(
            half_life=pd.Timedelta(days=365),
            rolling=pd.Timedelta(days=365*5)))

if you want to apply exponential smoothing to the mean returns forecaster with
half-life of 1 year, and skip over all observations older than 5 years. Both
are relative to each point in time at which the policy is evaluated.
"""

import logging
import warnings

import numpy as np
import pandas as pd

from .errors import DataError, ForecastError
from .estimator import DataEstimator, Estimator, SimulatorEstimator
from .hyperparameters import _resolve_hyperpar

logger = logging.getLogger(__name__)

class BaseForecast(Estimator):
    """Base class for forecasters."""

    # Will be exposed to the user, for now it's a class-level constant
    _CACHED = False

    def values_in_time_recursive( # pylint: disable=arguments-differ
            self, t, cache=None, **kwargs):
        """Override default method to handle caching.

        :param t: Current timestamp in execution or back-test.
        :type t: pd.Timestamp
        :param cache: Cache dictionary, if available. Default None.
        :type cache: dict or None
        :param kwargs: Various arguments to :meth:`values_in_time_recursive`.
        :type kwargs: dict

        :returns: Forecasted value for this period, possibly retrieved from
            cache.
        :rtype: pd.Series, pd.DataFrame, pd.Series
        """

        # TODO: implement workspace logic

        #     # initialize workspace
        #     if 'workspace' not in cache:
        #         cache['workspace'] = {}
        #         cache['workspace']['t'] = t

        #     # move things from workspace into long term
        #     if cache['workspace']['t'] != t:
        #         for stored_key in cache['workspace']

        if self._CACHED and cache is not None: # in execute() cache is disabled

            if str(self) not in cache:
                cache[str(self)] = {}

            if t in cache[str(self)]:
                logger.info(
                    '%s.values_in_time at time %s is retrieved from cache.',
                    self, t)
                self._current_value = cache[str(self)][t]
            else:
                self._current_value = super().values_in_time_recursive(
                    t=t, cache=cache, **kwargs)
                logger.info('%s.values_in_time at time %s is stored in cache.',
                    self, t)
                cache[str(self)][t] = self.current_value

            return self.current_value

        return super().values_in_time_recursive(t=t, cache=cache, **kwargs)

    def estimate(self, market_data, t):
        """Estimate the forecaster at given time on given market data.

        This uses the same logic used by a trading policy to evaluate the
        forecaster at a given point in time.

        :param market_data: Market data server, used to provide data to the
            forecaster. If you wish to forecast the value for the last
            period available (``market_data.trading_calendar()[-1]``), you
            typically need to set ``online_usage=True`` in its constructor.
        :type market_data: cvx.MarketData instance
        :param t: Trading period at which to make the forecast. **Only data
            available at that time or before is used**, like ``past_returns``.
            It must be among the timestamps provided by the
            :meth:`cvxportfolio.data.MarketData.trading_calendar` method of the
            market data server.
        :type t: str, pd.Timestamp

        :Example:

        >>> md = cvx.DownloadedMarketData(['AAPL', 'MSFT', 'GOOG'])
        >>> cvx.forecast.HistoricalCovariance().estimate(
            market_data=md, t=md.trading_calendar()[-3])

        :raises ValueError: If the provided time t is not in the trading
            calendar.

        :returns: Forecasted value.
        :rtype: np.array, pd.DataFrame
        """

        trading_calendar = market_data.trading_calendar()
        t = pd.Timestamp(t)

        if not t in trading_calendar:
            if (t.tz is None and (trading_calendar.tz is not None)) or \
                (t.tz is not None and (trading_calendar.tz is None)):
                raise ValueError(
                    f"Provided time {t} does not have timezone and the "
                    "market data does, or the other way round.")
            before = trading_calendar[trading_calendar < t]
            after = trading_calendar[trading_calendar > t]
            raise ValueError(f'Provided time {t} must be in the '
            + 'trading calendar implied by the market data server; '
            + (f'last valid timestamp before t is {before[-1]}; '
                if len(before) > 0 else '')
            + (f'first valid timestamp after t is {after[0]}; '
                if len(after) > 0 else '')
            )

        past_returns, _, past_volumes, _, current_prices = market_data.serve(t)

        self.initialize_estimator_recursive(
            universe=past_returns.columns,
            trading_calendar=trading_calendar[trading_calendar >= t])

        forecast = self.values_in_time_recursive(
            t=t, past_returns=past_returns, past_volumes=past_volumes,
            current_weights=None, current_portfolio_value=None,
            current_prices=current_prices)

        self.finalize_estimator_recursive()

        return forecast


def _is_timedelta_or_inf(value):
    """Check that a value is pd.Timedelta, or np.inf; else raise exception."""
    if isinstance(value, pd.Timedelta):
        if value <= pd.Timedelta('0d'):
            raise ValueError(
                '(Exponential) moving average window must be positive')
        return True
    if isinstance(value, float) and np.isposinf(value):
        return False
    raise ValueError(
        '(Exponential) moving average window can only be'
        ' pandas Timedeltas or np.inf.')

class UpdatingForecaster(BaseForecast):
    """Forecaster that updates internal forecast at each period."""

    _last_time = None

    # Gets populated with current universe; should probably be done upstream
    _universe = None

    def initialize_estimator( # pylint: disable=arguments-differ
            self, universe, **kwargs):
        """Initialize internal variables.

        :param universe: Current trading universe, including cash.
        :type universe: pd.Index
        :param kwargs: Other unused arguments to :meth:`initialize_estimator`.
        :type kwargs: dict
        """
        self._last_time = None
        self._universe = universe

    def finalize_estimator(self, **kwargs):
        """Dereference internal variables.

        :param kwargs: Unused arguments to :meth:`finalize_estimator`.
        :type kwargs: dict
        """
        self._last_time = None
        self._universe = None

    def values_in_time( # pylint: disable=arguments-differ
            self, t, past_returns, **kwargs):
        """Choose whether to make forecast from scratch or update last one.

        :param t: Current trading time.
        :type t: pd.Timestamp
        :param past_returns: Past market returns.
        :type past_returns: pd.DataFrame
        :param kwargs: Other unused arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Forecasted value for the period, either computed from scratch
            or updated with last observation.
        :rtype: pd.Series, pd.DataFrame, np.array
        """
        if (self._last_time is None) or (
            self._last_time != past_returns.index[-1]):
            logger.debug(
                '%s.values_in_time at time %s is computed from scratch.',
                self, t)
            return self._initial_compute(
                t=t, past_returns=past_returns, **kwargs)
        logger.debug(
            '%s.values_in_time at time %s is updated from previous value.',
            self, t)
        return self._online_update(
            t=t, past_returns=past_returns, **kwargs)

    def _initial_compute(self, **kwargs):
        """Make forecast from scratch."""
        raise NotImplementedError # pragma: no cover

    def _online_update(self, **kwargs):
        """Update forecast from period before."""
        raise NotImplementedError # pragma: no cover

class DataFrameSelector:
    """Companion class used to specify the DataFrame for the forecast."""

    def past_history(self, **kwargs):
        """Return all past history of the dataframe we work with.

        :param kwargs: All arguments passed to
            :meth:`estimator.Estimator.values_in_time`.
        :type kwargs: dict

        :returns: All past history (relative to the point in time in the
            back-test) of the DataFrame.
        :rtype: pd.DataFrame
        """
        raise NotImplementedError # pragma: no cover

    def last_row(self, **kwargs):
        """Return last row of the dataframe we work with.

        You may redefine it if obtaining the full dataframe is expensive,
        during online update (in most cases) only this method is required.

        :param kwargs: All arguments passed to
            :meth:`estimator.Estimator.values_in_time`.
        :type kwargs: dict

        :returns: Most recent row, relative to the point in time in the
            back-test, of the DataFrame returned by :meth:`past_history`.
        :rtype: pd.Series
        """
        return self.past_history(**kwargs).iloc[-1]

class PastReturns(DataFrameSelector):
    """Operate on past returns."""

    def past_history( # pylint: disable=arguments-differ
            self, past_returns, **kwargs):
        """Past returns, skipping cash.

        :param past_returns: Past market returns, including cash.
        :type past_returns: pd.DataFrame
        :param kwargs: All other arguments passed to
            :meth:`estimator.Estimator.values_in_time`.
        :type kwargs: dict

        :raises DataError: If the forecaster is used in a policy without market
            data server.

        :returns: Past market returns.
        :rtype: pd.DataFrame
        """
        if past_returns is None:
            raise DataError(
                "Forecaster needs past returns; can only be used if"
                + " MarketData is not None.")
        return past_returns.iloc[:, :-1]

class PastReturnsSquared(PastReturns):
    """Operate on past returns squared."""

    def past_history( # pylint: disable=arguments-differ
            self, **kwargs):
        """Past returns squared, skipping cash.

        :param kwargs: All arguments passed to
            :meth:`estimator.Estimator.values_in_time`.
        :type kwargs: dict

        :returns: Past market returns, squared.
        :rtype: pd.DataFrame
        """
        return super().past_history(**kwargs)**2

    def last_row(self, **kwargs):
        """Most recent past returns.

        :param kwargs: All arguments passed to
            :meth:`estimator.Estimator.values_in_time`.
        :type kwargs: dict

        :returns: Most recent market returns, squared.
        :rtype: pd.Series
        """
        return super().past_history(**kwargs).iloc[-1]**2

class PastVolumes(DataFrameSelector):
    """Operate on past volumes."""

    def past_history( # pylint: disable=arguments-differ
            self, past_volumes, **kwargs):
        """Full history of past market volumes.

        :param past_volumes: Past market volumes.
        :type past_volumes: pd.DataFrame
        :param kwargs: All other arguments passed to
            :meth:`estimator.Estimator.values_in_time`.
        :type kwargs: dict

        :raises DataError: If the forecaster is used in a policy whose market
            data server does not provides past volumes.

        :returns: Past market volumes.
        :rtype: pd.DataFrame
        """
        if past_volumes is None:
            raise DataError(
                "Forecaster needs past volumes; can only be used if MarketData"
                + " provides market volumes.")
        return past_volumes

class SumForecaster(UpdatingForecaster):
    """Base forecaster that implements a sum operation.

    We use this to implement the logic for rolling sum and exponential
    smoothing. Actual forecasters typically are the composition of two of
    these: they are means, so both their numerator and denominator are
    subclasses of this.

    :param dataframe_selector: Instance of DataFrame selector for this
        forecaster.
    :type dataframe_selector: DataFrameSelector
    :param half_life: Length of the exponential smoothing half-life.
    :type half_life: pd.Timedelta or np.inf
    :param rolling: Length of the rolling window.
    :type rolling: pd.Timedelta or np.inf
    """

    def __init__(self, dataframe_selector, half_life=np.inf, rolling=np.inf):
        self._dataframe_selector = dataframe_selector
        self.half_life = half_life
        self.rolling = rolling

    def _batch_compute(self, df, emw_weights):
        """Compute the value for a batch of observations (vectorization)."""
        raise NotImplementedError # pragma: no cover

    def _single_compute(self, last_row):
        """Compute the value for a single observation."""
        raise NotImplementedError # pragma: no cover

    # def _dataframe_selector(self, **kwargs):
    #     """Return dataframe we work with.

    #     This method receives the **kwargs passed to :meth:`values_in_time`.
    #     """
    #     raise NotImplementedError # pragma: no cover

    # def _get_last_row(self, **kwargs):
    #     """Return last row of the dataframe we work with.

    #     This method receives the **kwargs passed to :meth:`values_in_time`.

    #     You may redefine it if obtaining the full dataframe is expensive,
    #     during online update (in most cases) only this method is required.
    #     """
    #     return self._dataframe_selector(**kwargs).iloc[-1]

    def _emw_weights(self, index, t):
        """Get weights to apply to the past observations for EMW."""
        index_in_halflifes = (index - t) / _resolve_hyperpar(self.half_life)
        return np.exp(index_in_halflifes * np.log(2))

    def _initial_compute(self, t, **kwargs): # pylint: disable=arguments-differ
        """Make forecast from scratch.

        This method receives the **kwargs passed to :meth:`values_in_time`.
        """
        df = self._dataframe_selector.past_history(t=t, **kwargs)

        # Moving average window logic
        if _is_timedelta_or_inf(_resolve_hyperpar(self.rolling)):
            df = df.loc[df.index >= t-_resolve_hyperpar(self.rolling)]

        # If EMW, compute weights here
        if _is_timedelta_or_inf(_resolve_hyperpar(self.half_life)):
            emw_weights = self._emw_weights(df.index, t)
        else:
            emw_weights = None

        result = self._batch_compute(df, emw_weights)

        # update internal timestamp
        self._last_time = t

        return result

    def _online_update(self, t, **kwargs): # pylint: disable=arguments-differ
        """Update forecast from period before.

        This method receives the **kwargs passed to :meth:`values_in_time`.
        """
        last_row = self._dataframe_selector.last_row(t=t, **kwargs)

        # if emw discount past
        if _is_timedelta_or_inf(_resolve_hyperpar(self.half_life)):
            time_passed_in_halflifes = (
                self._last_time - t)/_resolve_hyperpar(self.half_life)
            discount_factor = np.exp(time_passed_in_halflifes * np.log(2))
        else:
            discount_factor = 1.

        # for emw we also need to discount last element
        result = self.current_value + self._single_compute(last_row)

        # emw discounting
        result *= discount_factor

        # Moving average window logic: subtract elements that have gone out
        if _is_timedelta_or_inf(_resolve_hyperpar(self.rolling)):
            df = self._dataframe_selector.past_history(t=t, **kwargs)
            result = self._remove_part_gone_out_of_ma(result, df, t)

        # update internal timestamp
        self._last_time = t

        return result

    def _remove_part_gone_out_of_ma(self, result, df, t):
        """Subtract value from observations that are too old, if rolling."""

        observations_to_subtract = df.loc[
            (df.index >= (self._last_time - _resolve_hyperpar(self.rolling)))
            & (df.index < (t - _resolve_hyperpar(self.rolling)))]

        # If EMW, compute weights here
        if _is_timedelta_or_inf(_resolve_hyperpar(self.half_life)):
            emw_weights = self._emw_weights(observations_to_subtract.index, t)
        else:
            emw_weights = None

        result -= self._batch_compute(
            observations_to_subtract, emw_weights)

        return result


class VectorCount(SumForecaster): # pylint: disable=abstract-method
    """Intermediate class, count of non-NaN values of vectors."""

    def _batch_compute(self, df, emw_weights):
        """Compute for a batch at once."""
        if emw_weights is None:
            return df.count()
        ones = (~df.isnull()) * 1.
        return ones.multiply(emw_weights, axis=0).sum()

    def _single_compute(self, last_row):
        """Update with one observation."""
        return ~(last_row.isnull())

    def values_in_time( # pylint: disable=arguments-differ
            self, t, **kwargs):
        """Check that we have enough observations, call super() method.

        :param t: Current trading time.
        :type t: pd.Timestamp
        :param kwargs: Other arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :raises ForecastError: If there are not enough observations to compute
            derived quantities.

        :returns: Current count of non-NaN values.
        :rtype: pd.Series or np.array
        """
        result = super().values_in_time(t=t, **kwargs)

        mindenom = np.min(result.values, axis=None)

        if mindenom == 0:
            raise ForecastError(
                f'{self.__class__.__name__} can not compute the forecast at'
                + f' time {t} because there are no observation for either some'
                ' asset or some pair of assets (in the case of covariance).')
        if mindenom < 5:
            logger.warning(
                '%s at time %s is given 5 or less observations for either some'
                + ' asset or some pair of assets (in the case of covariance).',
                self.__class__.__name__, t)

        return result

# class CountPastReturns(VectorCount, OnPastReturns):
#     """Count non-nan past returns, excluding cash."""

# class CountPastVolumes(VectorCount, OnPastVolumes):
#     """Count non-nan past volumes."""

class VectorSum(SumForecaster): # pylint: disable=abstract-method
    """Intermediate class, sum of non-NaN values of vectors."""

    def _batch_compute(self, df, emw_weights):
        """Compute from scratch."""
        if emw_weights is None:
            return df.sum()
        return df.multiply(emw_weights, axis=0).sum()

    def _single_compute(self, last_row):
        """Update with last observation."""
        return last_row.fillna(0.)

# class SumPastReturns(VectorSum, OnPastReturns):
#     """Sum non-nan past returns, excluding cash."""

# class SumPastReturnsSquared(VectorSum, OnPastReturnsSquared):
#     """Sum non-nan past returns squared, excluding cash."""

# class SumPastVolumes(VectorSum, OnPastVolumes):
#     """Sum non-nan past volumes."""

class HistoricalMeanReturn(BaseForecast):
    r"""Historical means of non-cash returns.

    .. versionadded:: 1.2.0

        Added the ``half_life`` and ``rolling`` parameters.

    When both ``half_life`` and ``rolling`` are infinity, this is equivalent to

    .. code-block::

        past_returns.iloc[:,:-1].mean()

    where ``past_returns`` is a time-indexed dataframe containing the past
    returns (if in back-test that's relative to each point in time, ), and its
    last column, which we skip over, are the cash returns. We use the same
    logic as Pandas to handle ``np.nan`` values.

    :param half_life: Half-life of exponential smoothing, expressed as
        Pandas Timedelta. If in back-test, that is with respect to each point
        in time. Default ``np.inf``, meaning no exponential smoothing.
    :type half_life: pandas.Timedelta or np.inf
    :param rolling: Rolling window used: observations older than this Pandas
        Timedelta are skipped over. If in back-test, that is with respect to
        each point in time. Default ``np.inf``, meaning that all past is used.
    :type rolling: pandas.Timedelta or np.inf
    """
    def __init__(self, half_life=np.inf, rolling=np.inf):
        self._dataframe_selector = PastReturns()
        self.half_life = half_life
        self.rolling = rolling
        self._numerator = VectorSum(
            dataframe_selector = self._dataframe_selector,
            half_life=half_life, rolling=rolling)
        self._denominator = VectorCount(
            dataframe_selector = self._dataframe_selector,
            half_life=half_life, rolling=rolling)

    def values_in_time( # pylint: disable=arguments-differ
            self, **kwargs):
        """Mean of the past returns, excluding cash.

        :param kwargs: Unused arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Mean of the past returns, excluding cash.
        :rtype: pd.Series
        """
        return self._numerator.current_value / self._denominator.current_value

class HistoricalVariance(BaseForecast):
    r"""Historical variances of non-cash returns.

    .. versionadded:: 1.2.0

        Added the ``half_life`` and ``rolling`` parameters.

    When both ``half_life`` and ``rolling`` are infinity, this is equivalent to

    .. code-block::

        past_returns.iloc[:,:-1].var(ddof=0)

    if you set ``kelly=False`` and

    .. code-block::

        (past_returns**2).iloc[:,:-1].mean()

    otherwise (we use the same logic to handle ``np.nan`` values).

    :param half_life: Half-life of exponential smoothing, expressed as
        Pandas Timedelta. If in back-test, that is with respect to each point
        in time. Default ``np.inf``, meaning no exponential smoothing.
    :type half_life: pandas.Timedelta or np.inf
    :param rolling: Rolling window used: observations older than this Pandas
        Timedelta are skipped over. If in back-test, that is with respect to
        each point in time. Default ``np.inf``, meaning that all past is used.
    :type rolling: pandas.Timedelta or np.inf
    :param kelly: if ``True`` compute :math:`\mathbf{E}[r^2]`, else
        :math:`\mathbf{E}[r^2] - {\mathbf{E}[r]}^2`. The second corresponds
        to the classic definition of variance, while the first is what is
        obtained by Taylor approximation of the Kelly gambling objective.
        See discussion above. Default True.
    :type kelly: bool
    """
    def __init__(self, half_life=np.inf, rolling=np.inf, kelly=True):
        self._dataframe_selector = PastReturnsSquared()
        self.half_life = half_life
        self.rolling = rolling
        self.kelly = kelly
        self._denominator = VectorCount(
            dataframe_selector = self._dataframe_selector,
            half_life=half_life, rolling=rolling)
        self._numerator = VectorSum(
            dataframe_selector = self._dataframe_selector,
            half_life=half_life, rolling=rolling)
        if not self.kelly:
            self._correction = HistoricalMeanReturn(
                half_life=half_life, rolling=rolling)

    def values_in_time( # pylint: disable=arguments-differ
            self, **kwargs):
        """Variance of the past returns, excluding cash.

        :param kwargs: Unused arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Variance of the past returns, excluding cash.
        :rtype: pd.Series
        """
        result = (
            self._numerator.current_value / self._denominator.current_value)
        if not self.kelly:
            result -= self._correction.current_value ** 2

        # when using rolling numerical errors may cause small negative values
        return np.maximum(result, 0.)

class HistoricalMeanError(HistoricalVariance):
    r"""Historical standard deviations of the mean of non-cash returns.

    .. versionadded:: 1.2.0

        Added the ``half_life`` and ``rolling`` parameters.

    For a given time series of past returns :math:`r_{t-1}, r_{t-2},
    \ldots, r_0` this is :math:`\sqrt{\text{Var}[r]/t}`. When there are
    missing values we ignore them, both to compute the variance and the
    count.

    :param half_life: Half-life of exponential smoothing, expressed as
        Pandas Timedelta. If in back-test, that is with respect to each point
        in time. Default ``np.inf``, meaning no exponential smoothing.
    :type half_life: pandas.Timedelta or np.inf
    :param rolling: Rolling window used: observations older than this Pandas
        Timedelta are skipped over. If in back-test, that is with respect to
        each point in time. Default ``np.inf``, meaning that all past is used.
    :type rolling: pandas.Timedelta or np.inf
    :param kelly: Same as in :class:`cvxportfolio.forecast.HistoricalVariance`.
        Default False.
    :type kelly: bool
    """

    def __init__(self, rolling=np.inf, half_life=np.inf, kelly=False):
        super().__init__(rolling=rolling, half_life=half_life, kelly=kelly)

    def values_in_time(self, **kwargs):
        """Obtain current value either by update or from scratch.

        :param kwargs: All arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Standard deviation of the mean of past returns (excluding
            cash).
        :rtype: pd.Series
        """
        variance = super().values_in_time(**kwargs)
        return np.sqrt(variance / self._denominator.current_value)

class HistoricalStandardDeviation(HistoricalVariance, SimulatorEstimator):
    """Historical standard deviation of non-cash returns.

    .. versionadded:: 1.2.0

        Added the ``half_life`` and ``rolling`` parameters.

    When both ``half_life`` and ``rolling`` are infinity, this is equivalent to

    .. code-block::

        past_returns.iloc[:,:-1].std(ddof=0)

    if you set ``kelly=False`` and

    .. code-block::

        np.sqrt((past_returns**2).iloc[:,:-1].mean())

    otherwise (we use the same logic to handle ``np.nan`` values).

    :param half_life: Half-life of exponential smoothing, expressed as
        Pandas Timedelta. If in back-test, that is with respect to each point
        in time. Default ``np.inf``, meaning no exponential smoothing.
    :type half_life: pandas.Timedelta or np.inf
    :param rolling: Rolling window used: observations older than this Pandas
        Timedelta are skipped over. If in back-test, that is with respect to
        each point in time. Default ``np.inf``, meaning that all past is used.
    :type rolling: pandas.Timedelta or np.inf
    :param kelly: Same as in :class:`cvxportfolio.forecast.HistoricalVariance`.
        Default True.
    :type kelly: bool
    """

    def values_in_time(self, **kwargs):
        """Obtain current value either by update or from scratch.

        :param kwargs: All arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Standard deviations of past returns (excluding cash).
        :rtype: pd.Series
        """
        return np.sqrt(super().values_in_time(**kwargs))

    def simulate( # pylint: disable=arguments-differ
            self, **kwargs):
        """Obtain current value for use in simulation (transaction cost).

        :param kwargs: All arguments to :meth:`simulate`.
        :type kwargs: dict

        :returns: Standard deviations of past returns (excluding cash).
        :rtype: pd.Series
        """
        # TODO could take last return as well

        # with new design of forecasters we need to launch recursive loop
        return self.values_in_time_recursive(
            t=kwargs['t'],
            current_weights=kwargs['current_weights'],
            current_portfolio_value=kwargs['current_portfolio_value'],
            past_returns=kwargs['past_returns'],
            past_volumes=kwargs['past_volumes'],
            current_prices=kwargs['current_prices']
        )

class HistoricalMeanVolume(BaseForecast):
    r"""Historical means of traded volume in units of value (e.g., dollars).

    .. versionadded:: 1.2.0

    :param half_life: Half-life of exponential smoothing, expressed as
        Pandas Timedelta. If in back-test, that is with respect to each point
        in time. Default ``np.inf``, meaning no exponential smoothing.
    :type half_life: pandas.Timedelta or np.inf
    :param rolling: Rolling window used: observations older than this Pandas
        Timedelta are skipped over. If in back-test, that is with respect to
        each point in time. Default ``np.inf``, meaning that all past is used.
    :type rolling: pandas.Timedelta or np.inf
    """
    def __init__(self, half_life=np.inf, rolling=np.inf):
        self._dataframe_selector = PastVolumes()
        self.half_life = half_life
        self.rolling = rolling
        self._numerator = VectorSum(
            dataframe_selector = self._dataframe_selector,
            half_life=half_life, rolling=rolling)
        self._denominator = VectorCount(
            dataframe_selector = self._dataframe_selector,
            half_life=half_life, rolling=rolling)

    def values_in_time( # pylint: disable=arguments-differ
            self, **kwargs):
        """Mean of the past volumes.

        :param kwargs: Unused arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Mean of the past volumes.
        :rtype: pd.Series
        """
        return self._numerator.current_value / self._denominator.current_value

class JointCount(VectorCount): # pylint: disable=abstract-method
    """Intermediate class: joint count for the denominator of covariances.
    
    We inherit from :class:`VectorCount` which implements a check for
    too few observations.
    """

    def _batch_compute(self, df, emw_weights):
        """Compute for a batch at once."""
        ones = (~df.isnull()) * 1.
        if emw_weights is None:
            return ones.T @ ones
        tmp = ones.multiply(emw_weights, axis=0)
        return tmp.T @ ones

    def _single_compute(self, last_row):
        """Update with one observation."""
        nonnull = ~(last_row.isnull())
        return np.outer(nonnull, nonnull)


# class JointCountPastReturns(JointCount, OnPastReturns):
#     """Compute denominator of (Kelly) covariance of past returns."""

class JointSum(SumForecaster): # pylint: disable=abstract-method
    """Intermediate class: joint sum for the numerator of covariances."""

    def _batch_compute(self, df, emw_weights):
        """Compute for a batch at once."""
        filled = df.fillna(0.)
        if emw_weights is None:
            return filled.T @ filled
        tmp = filled.multiply(emw_weights, axis=0)
        return tmp.T @ filled

    def _single_compute(self, last_row):
        """Update with one observation."""
        filled = last_row.fillna(0.)
        return np.outer(filled, filled)

# class JointSumPastReturns(JointSum, OnPastReturns):
#     """Compute numerator of (Kelly) covariance of past returns."""

class JointMean(SumForecaster): # pylint: disable=abstract-method
    """Intermediate class: corrector for non-Kelly covariance."""

    def _batch_compute(self, df, emw_weights):
        r"""Compute precursor of :math:`\Sigma_{i,j} =
        \mathbf{E}[r^{i}]\mathbf{E}[r^{j}]`."""
        nonnull = (~df.isnull()) * 1.
        if emw_weights is None:
            return nonnull.T @ df.fillna(0.)
        return nonnull.T @ df.fillna(0.).multiply(emw_weights, axis=0)

    def _single_compute(self, last_row):
        r"""Update precursor of :math:`\Sigma_{i,j} =
        \mathbf{E}[r^{i}]\mathbf{E}[r^{j}]`."""
        return last_row.fillna(0.)

# class JointMeanReturns(JointMean, OnPastReturns):
#     """Compute corrector for non-Kelly covariance."""

class HistoricalCovariance(BaseForecast):
    r"""Historical covariance matrix.

    .. versionadded:: 1.2.0

        Added the ``half_life`` and ``rolling`` parameters.

    :param half_life: Half-life of exponential smoothing, expressed as
        Pandas Timedelta. If in back-test, that is with respect to each point
        in time. Default ``np.inf``, meaning no exponential smoothing.
    :type half_life: pandas.Timedelta or np.inf
    :param rolling: Rolling window used: observations older than this Pandas
        Timedelta are skipped over. If in back-test, that is with respect to
        each point in time. Default ``np.inf``, meaning that all past is used.
    :type rolling: pandas.Timedelta or np.inf
    :param kelly: Same as in :class:`cvxportfolio.forecast.HistoricalVariance`.
        Default False.
    :type kelly: bool
    """

    def __init__(self, half_life=np.inf, rolling=np.inf, kelly=True):
        self._dataframe_selector = PastReturns()
        self.half_life = half_life
        self.rolling = rolling
        self.kelly = kelly
        self._denominator = JointCount(
            dataframe_selector = self._dataframe_selector,
            half_life=half_life, rolling=rolling)
        self._numerator = JointSum(
            dataframe_selector = self._dataframe_selector,
            half_life=half_life, rolling=rolling)
        if not self.kelly:
            self._correction = JointMean(
                dataframe_selector = self._dataframe_selector,
                half_life=half_life, rolling=rolling)

    def values_in_time( # pylint: disable=arguments-differ
            self, **kwargs):
        """Current historical covariance matrix.

        :param kwargs: Unused arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Historical covariance.
        :rtype: pd.DataFrame
        """
        result = (
            self._numerator.current_value / self._denominator.current_value)
        if not self.kelly:
            tmp = (
                self._correction.current_value
                    / self._denominator.current_value)
            result -= tmp.T * tmp
        return result

###
# Linear regression
###

class UserProvidedRegressor(DataEstimator):
    """User provided regressor series."""

    def __init__(self, data, min_obs=10):
        assert isinstance(data, pd.Series)
        assert isinstance(data.index, pd.DatetimeIndex)
        assert data.name is not None
        assert isinstance(data.name, str)
        super().__init__(data, use_last_available_time=True)
        self._min_obs = min_obs

    def get_all_history(self, pandas_obj_idx):
        """Get history of this regressor indexed on pandas obj."""
        result = self.data.reindex(
            pandas_obj_idx, method='ffill').dropna()
        if len(result) < self._min_obs:
            raise DataError(
                f'Regressor {self.name} at time {pandas_obj_idx[-1]} '
                f' has less history than min_obs={self._min_obs},'
                ' changing regressor in time is not (currently) supported.')
        return result

    def get_one_time(self, timestamp):
        """Get value of regressor at specific timestamp."""
        # breakpoint()
        # TODO this is not correct I'm afraid
        # return float(self.current_value)
        return self.data[self.data.index <= timestamp].iloc[-1]

    @property
    def name(self):
        """Name of the regressor.

        :returns: Name
        :rtype: str
        """
        return self.data.name

# probably can restate this as another intermediate class, not inheriting
# from OnPastReturns, and define specialized ones for other raw dataframes


class WeightedPastReturns(PastReturns): # pylint: disable=abstract-method
    """Intermediate class, operate on past returns weighted by regressor."""

    # could put the __init__ we use in derived classes here, but then
    # would have to be careful to use correct inheritance order for MRO

    # this needs to be populated by __init__ of derived class
    # regressor = None

    def __init__(self, regressor):
        self.regressor = regressor

    def past_history( # pylint: disable=arguments-differ
            self, **kwargs):
        """Past returns, skipping cash, weighted by regressor.

        This method receives the full arguments to :meth:`values_in_time`.
        """
        raw_past_df = super().past_history(**kwargs)
        regressor_history = self.regressor.get_all_history(
            raw_past_df.index)
        # with the dropna we remove (old) observations for which regressor had
        # no data
        return raw_past_df.multiply(
            regressor_history, axis=0).dropna(how='all')

    # TODO: the below breaks on test for some reason, figure out why
    # def _get_last_row(self, **kwargs):
    #     """Return last row of the dataframe we work with.

    #     This method receives the **kwargs passed to :meth:`values_in_time`.

    #     You may redefine it if obtaining the full dataframe is expensive,
    #     during online update (in most cases) only this method is required.
    #     """
    #     raw_last_row = super()._get_last_row(**kwargs)
    #     regressor_on_last_row = self.regressor.get_one_time(
    #         raw_last_row.name) # check that this is robust enough?
    #     breakpoint()
    #     return raw_last_row * regressor_on_last_row

# class CountWeightedPastReturns(VectorCount, OnWeightedPastReturns):
#     """Count non-nan past returns, excluding cash, weighted by regressor."""

#     def __init__(self, regressor, **kwargs):
#         self.regressor = regressor
#         super().__init__(**kwargs) # this goes to SumForecaster

# class SumWeightedPastReturns(VectorSum, OnWeightedPastReturns):
#     """Sum non-nan past returns, excluding cash, weighted by regressor."""

#     def __init__(self, regressor, **kwargs):
#         self.regressor = regressor
#         super().__init__(**kwargs) # this goes to SumForecaster

# We can reproduce this design pattern for other base forecasters


class OnPastRegressors(SumForecaster):
    """Select history of regressors."""

    def __init__(self, regressors, **kwargs):
        self.regressors = regressors
        super().__init__(**kwargs)

    def _dataframe_selector( # pylint: disable=arguments-differ
            self, **kwargs):
        """History of past regressors, on index of given asset."""

class XtXMatrix(SumForecaster):
    ...


class RegressionXtYReturns(HistoricalMeanReturn):
    """Class for the XtY matrix of returns regression forecaster."""

    def __init__(self, regressor, **kwargs):
        assert isinstance(regressor, UserProvidedRegressor)

        # call super().__init__ first, then overwrite num and denom
        super().__init__(**kwargs)

        # regression part
        self._regressor = regressor
        self._dataframe_selector = WeightedPastReturns(
            regressor=self._regressor)
        self._numerator = VectorSum(
            dataframe_selector=self._dataframe_selector, **kwargs)
        self._denominator = VectorCount(
            dataframe_selector=self._dataframe_selector, **kwargs)

    # def _work_with(self, past_returns, **kwargs):
    #     """Base DataFrame we work with."""
    #     return past_returns.iloc[:, :-1]

    # # pylint: disable=arguments-differ
    # def _dataframe_selector(self, **kwargs):
    #     """Return dataframe to compute the historical means of."""
    #     regr_on_df = self.regressor._get_all_history(
    #         self._work_with(**kwargs).index)
    #     return self._work_with(
    #         **kwargs).multiply(regr_on_df, axis=0).dropna(how='all')

    # def _get_last_row(self, **kwargs):
    #     """Return last row of the dataframe we work with.

    #     This method receives the **kwargs passed to :meth:`values_in_time`.

    #     You may redefine it if obtaining the full dataframe is expensive,
    #     during online update (in most cases) only this method is required.
    #     """
    #     return self._work_with(
    #         **kwargs).iloc[-1] * self.regressor.current_value


class RegressionMeanReturn(BaseForecast): # pragma: no cover
    """Test class."""

    def __init__(self, regressors, **kwargs):
        # super().__init__(**kwargs)
        self.regressors = [
            UserProvidedRegressor(regressor) for regressor in regressors]

        self.XtY_forecasters = {
            regressor.name: RegressionXtY(regressor)
                for regressor in self.regressors}

        self.__subestimators__ = tuple(
            [HistoricalMeanReturn()] + self.regressors
            + list(self.XtY_forecasters.values()))

        self.XtX_matrices = None

    def initialize_estimator(self, universe, **kwargs):
        """Initialize, create XtX matrices knowing the current universe.

        :param universe: Current trading universe.
        :type universe: pd.Index
        :param **kwargs: Other arguments to :meth:`initialize_estimator`.
        :type **kwargs: dict
        """

        self.XtX_matrices = {
            asset: RegressorsXtXMatrix(
                col_name=asset, regressors=self.regressors)
                    for asset in universe[:-1]}

        for XtX in self.XtX_matrices.values():
            XtX.initialize_estimator_recursive(universe=universe, **kwargs)

        # this method is called *after* having iterated on __subestimators__,
        # so it's safe to do this
        self.__subestimators__ = tuple(
            list(self.__subestimators__ ) + list(self.XtX_matrices.values()))

    def finalize_estimator(self, **kwargs):
        """Remove XtX matrices at change of universe.

        :param **kwargs: Unused arguments to :meth:`finalize_estimator`.
        :type **kwargs: dict
        """

        self.__subestimators__ = tuple(
            [HistoricalMeanReturn()] + self.regressors
            + list(self.XtY_forecasters.values()))

        self.XtX_matrices = None

    def values_in_time(self, t, past_returns, **kwargs):
        """Do it from scratch."""
        assets = past_returns.columns[:-1]

        # print('all_X_matrices')
        # print(self.all_X_matrices)

        # self._all_XtY_means = {
        #     regressor.name: self.multiply_df_by_regressor(
        #         past_returns.iloc[:, :-1], regressor).mean()
        #             for regressor in self.regressors}

        test = {
            regressor.name: pd.Series(self.XtY_forecasters[regressor.name].current_value, past_returns.columns[:-1])
                    for regressor in self.regressors}

        # print(test)
        # print(self._all_XtY_means)
        # assert np.allclose(pd.Series(test['VIX']), pd.Series(self._all_XtY_means['VIX']))
        self._all_XtY_means = test
        # raise Exception

        self._all_XtY_means[
            'intercept'] = pd.Series(
                self.__subestimators__[0].current_value,
                past_returns.columns[:-1])

        # print('all_XtY_means')
        # print(self.all_XtY_means)

        X_last = pd.Series(1., index=['intercept'])
        for regressor in self.regressors:
            X_last[regressor.name] = regressor.current_value

        # print('X_last')
        # print(X_last)

        all_solves = {
            asset: self.solve_for_single_X(
                asset, X_last, quad_reg=0.) for asset in assets}

        # print('all_solves')
        # print(all_solves)

        # result should be an array
        result = pd.Series(index = assets, dtype=float)
        for asset in assets:
            result[asset] = np.dot(
                all_solves[asset],
                    [self._all_XtY_means[regressor][asset]
                        for regressor in all_solves[asset].index])

        return result

    def solve_for_single_X(self, asset, X_last, quad_reg):
        """Solve with X_last."""
        XtX_mean = self.XtX_matrices[asset].current_value

        tikho_diag = np.array(np.diag(XtX_mean))
        tikho_diag[0] = 0. # intercept
        return pd.Series(np.linalg.solve(
            XtX_mean + np.diag(tikho_diag * quad_reg), X_last), X_last.index)

    # @staticmethod
    # def multiply_df_by_regressor(df, regressor):
    #     """Multiply time-indexed dataframe by time-indexed regressor.

    #     At each point in time, use last available observation of the regressor.
    #     """
    #     regr_on_df = regressor._get_all_history(df)
    #     return df.multiply(regr_on_df, axis=0).dropna(how='all')


class RegressorsXtXMatrix(HistoricalCovariance): # pragma: no cover
    """XtX matrix used for linear regression.

    The user doesn't interact with this class directly, it is managed by the
    regression forecasters.
    """

    def __init__(self, regressors, col_name, **kwargs):
        self.regressors = regressors
        self.col_name = col_name
        super().__init__(kelly=True, **kwargs)

    def _work_with(self, past_returns, **kwargs):
        """Which base dataframe do we work with.

        This receives the arguments to :meth:`values_in_time`.

        In this class we only use its missing values structure, not its values!

        :param past_returns: Past market returns.
        :type past_returns: pd.DataFrame
        """
        return past_returns

    def _dataframe_selector(self, **kwargs):
        """Return dataframe we work with."""
        # TODO: might take just the index here
        col_of_df = self._work_with(**kwargs)[self.col_name].dropna()
        regr_on_col = []
        for regressor in self.regressors:
            regr_on_col.append(regressor._get_all_history(col_of_df.index))
        ones = pd.Series(1., col_of_df.index, name='intercept')
        res = pd.concat([ones] + regr_on_col, axis=1, sort=True)
        # print(res)
        return res

    def _get_last_row(self, **kwargs):
        """Return last row of the dataframe we work with.

        This method receives the **kwargs passed to :meth:`values_in_time`.

        During online update (in most cases) only this method is required,
        so :meth:`_dataframe_selector` is not called.
        """
        #result = self._dataframe_selector(**kwargs).iloc[-1]
        result1 = pd.Series(1., ['intercept'])
        for regressor in self.regressors:
            result1[regressor.name] = regressor.current_value
        #assert np.all(result == result1)
        # raise Exception
        return result1

###
# More covariance classes
###

def project_on_psd_cone_and_factorize(covariance):
    """Factorize matrix and remove negative eigenvalues.

    :param covariance: Square (symmetric) approximate covariance matrix, can
         have negative eigenvalues.
    :type covariance: numpy.array

    :returns: Square root factorization with negative eigenvalues removed.
    :rtype: numpy.array
    """
    eigval, eigvec = np.linalg.eigh(covariance)
    eigval = np.maximum(eigval, 0.)
    return eigvec @ np.diag(np.sqrt(eigval))

class HistoricalFactorizedCovariance(HistoricalCovariance):
    r"""Historical covariance matrix of non-cash returns, factorized.

    .. versionadded:: 1.2.0

        Added the ``half_life`` and ``rolling`` parameters.

    When both ``half_life`` and ``rolling`` are infinity, this is equivalent
    to, before factorization

    .. code-block::

        past_returns.iloc[:,:-1].cov(ddof=0)

    if you set ``kelly=False``. We use the same logic to handle ``np.nan``
    values. For ``kelly=True`` it is not possible to reproduce with one single
    Pandas method (but we do test against Pandas in the unit tests).

    :param half_life: Half-life of exponential smoothing, expressed as
        Pandas Timedelta. Default ``np.inf``, meaning no exponential smoothing.
    :type half_life: pandas.Timedelta or np.inf
    :param rolling: Rolling window used: observations older than this Pandas
        Timedelta are skipped over. If in back-test, that is with respect to
        each point in time. Default ``np.inf``, meaning that all past is used.
    :type rolling: pandas.Timedelta or np.inf
    :param kelly: if ``True`` each element of the covariance matrix
        :math:`\Sigma_{i,j}` is equal to :math:`\mathbf{E} r^{i} r^{j}`,
        otherwise it is
        :math:`\mathbf{E} r^{i} r^{j} - \mathbf{E} r^{i} \mathbf{E} r^{j}`.
        The second case corresponds to the classic definition of covariance,
        while the first is what is obtained by Taylor approximation of
        the Kelly gambling objective. (See discussion above.)
        In the second case, the estimated covariance is the same
        as what is returned by ``pandas.DataFrame.cov(ddof=0)``, *i.e.*,
        we use the same logic to handle missing data.
    :type kelly: bool
    """

    # this is used by FullCovariance
    FACTORIZED = True

    # Will be exposed to the user, for now it's a class level constant
    _CACHED = True

    def values_in_time( # pylint: disable=arguments-differ
            self, t, **kwargs):
        """Obtain current value of the covariance estimate.

        :param t: Current time period (possibly of simulation).
        :type t: pandas.Timestamp
        :param kwargs: All arguments passed to :meth:`values_in_time`.
        :type kwargs: dict

        :raises cvxportfolio.errors.ForecastError: The procedure failed,
            typically because there are too many missing values (*e.g.*, some
            asset has only missing values).

        :returns: Square root factorized covariance matrix (excludes cash).
        :rtype: numpy.array
        """

        covariance = super().values_in_time(t=t, **kwargs)

        try:
            return project_on_psd_cone_and_factorize(covariance)
        except np.linalg.LinAlgError as exc: # pragma: no cover
            raise ForecastError(f'Covariance estimation at time {t} failed;'
                + ' there are (probably) too many missing values in the'
                + ' past returns.') from exc


class HistoricalLowRankCovarianceSVD(BaseForecast):
    """Build factor model covariance using truncated SVD.

    .. note::

        This forecaster is experimental and not covered by semantic versioning,
        we may change it without warning.

    :param num_factors: How many factors in the low rank model.
    :type num_factors: int
    :param svd_iters: How many iteration of truncated SVD to apply. If you
        get a badly conditioned covariance you may to lower this.
    :type svd_iters: int
    :param svd: Which SVD routine to use, currently only dense (LAPACK) via
        Numpy.
    :type svd: str
    """

    def __init__(self, num_factors, svd_iters=10, svd='numpy'):
        self.num_factors = num_factors
        self.svd_iters = svd_iters
        self.svd = svd

    # num_factors: int
    # svd_iters: int = 10
    # svd: str = 'numpy'

    # brought back from old commit;
    #
    #  https://github.com/cvxgrp/cvxportfolio/commit
    #     /aa3d2150d12d85a6fb1befdf22cb7967fcc27f30
    #
    # matches original 2016 method from example
    # notebooks with new heuristic for NaNs. can probably be improved
    # by terminating early if idiosyncratic becomes negative

    @staticmethod
    def build_low_rank_model(rets, num_factors=10, iters=10, svd='numpy'):
        r"""Build a low rank risk model from past returns that include NaNs.

        This is an experimental procedure that may work well on past
        returns matrices with few NaN values (say, below 20% of the
        total entries). If there are (many) NaNs, one should probably
        also use a rather large risk forecast error.

        :param rets: Past returns, excluding cash.
        :type rets: pandas.DataFrame
        :param num_factors: How many factors in the fitted model.
        :type num_factors: int
        :param iters: How many iterations of SVD are performed.
        :type iters: int
        :param svd: Which singular value decomposition routine is used,
            default (and currently the only one supported) is ``'numpy'``.
        :type svd: str

        :raises SyntaxError: If wrong ``svd`` parameter is supplied.
        :raises cvxportfolio.errors.ForecastError: If the procedure fails;
             you may try with lower ``num_factors`` or ``iters``.

        :returns: (F, d)
        :rtype: (numpy.array, numpy.array)
        """
        nan_fraction = rets.isnull().sum().sum() / np.prod(rets.shape)
        normalizer = np.sqrt((rets**2).mean())
        normalized = rets
        if nan_fraction:
            nan_implicit_imputation = pd.DataFrame(0.,
                columns=normalized.columns, index = normalized.index)
            for _ in range(iters):
                if svd == 'numpy':
                    u, s, v = np.linalg.svd(
                        normalized.fillna(nan_implicit_imputation),
                        full_matrices=False)
                else:
                    raise SyntaxError(
                        'Currently only numpy svd is implemented')
                nan_implicit_imputation = pd.DataFrame(
                    (u[:, :num_factors] * (s[:num_factors]
                        )) @ v[:num_factors],
                    columns = normalized.columns, index = normalized.index)
        else:
            if svd == 'numpy':
                u, s, v = np.linalg.svd(normalized, full_matrices=False)
            else:
                raise SyntaxError(
                    'Currently only numpy svd is implemented')
        F = v[:num_factors].T * s[:num_factors] / np.sqrt(len(rets))
        F = pd.DataFrame(F.T, columns=normalizer.index)
        idyosyncratic = normalizer**2 - (F**2).sum(0)
        if not np.all(idyosyncratic >= 0.):
            raise ForecastError(
                "Low rank risk estimation with iterative SVD did not work."
                + " You probably have too many missing values in the past"
                + " returns. You may try with HistoricalFactorizedCovariance,"
                + " or change your universe.")
        return F.values, idyosyncratic.values

    # Will be exposed to the user, for now it's a class level constant
    _CACHED = True

    def values_in_time( # pylint: disable=arguments-differ
            self, past_returns, **kwargs):
        """Current low-rank model, also cached.

        :param past_returns: Past market returns (including cash).
        :type past_returns: pandas.DataFrame
        :param kwargs: Extra arguments passed to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Low-rank plus diagonal covariance model: (F, d); excludes
             cash.
        :rtype: (numpy.array, numpy.array)
        """
        return self.build_low_rank_model(past_returns.iloc[:, :-1],
            num_factors=self.num_factors,
            iters=self.svd_iters, svd=self.svd)
