


def detect_non_monotonicity(cdf_vals: List[float]) -> tuple[bool, int]:
    """
    Detect if a CDF is non-monotonic (decreasing at any point).

    Args:
        cdf_vals: list of CDF values that should be monotonically increasing

    Returns:
        Tuple of (is_non_monotonic, count_of_violations)
    """
    violations = 0
    for i in range(1, len(cdf_vals)):
        if cdf_vals[i] < cdf_vals[i - 1]:
            violations += 1
    return violations > 0, violations


def apply_isotonic_regression(cdf_vals: List[float]) -> List[float]:
    """
    Apply isotonic regression to enforce monotonicity using Pool Adjacent Violators Algorithm (PAVA).

    Args:
        cdf_vals: list of CDF values that may be non-monotonic

    Returns:
        Monotonically increasing CDF values
    """
    if len(cdf_vals) == 0:
        return []

    # Pool Adjacent Violators Algorithm (PAVA)
    # This is a simple, efficient algorithm for isotonic regression
    result = list(cdf_vals)
    n = len(result)

    i = 0
    while i < n - 1:
        if result[i] > result[i + 1]:
            # Found a violation, pool adjacent violators
            j = i
            sum_val = result[i]
            count = 1

            # Extend the pool while there are violations
            while j < n - 1 and result[j] > result[j + 1]:
                j += 1
                sum_val += result[j]
                count += 1

            # Replace all values in the pool with their average
            avg = sum_val / count
            for k in range(i, j + 1):
                result[k] = avg

            # Check if we need to merge with previous pools
            while i > 0 and result[i - 1] > result[i]:
                i -= 1
        else:
            i += 1

    return result


def debiased_percentiles(
    dataset: List[Dict],
    title: str,
    apply_monotonic_correction: bool = False
) -> None:
    """
    Compute debiased percentiles for the true-high and true-low groups using
    randomized-response debiasing.

    WARNING: This individual-record approach can produce non-monotonic CDFs when:
      - Records have mixed trigger rates (different randomized_trigger_rate values)
      - The debiasing weights become negative (which is mathematically correct but
        causes cumulative sums to be non-monotonic)

    For more stable percentile estimates with mixed trigger rates, use the
    histogram-based approach: split_by_trigger_rate() + run_multi_rum_dp_pipeline()

    Args:
      dataset: list of dicts, each must contain:
        - 'duration_ms' (numeric)
        - 'randomized_trigger_rate' (float in [0,1])
        - 'randomized_confidence' ('high' or 'low')
      title: the title used to describe the data being displayed
      apply_monotonic_correction: if True, apply isotonic regression to enforce
        monotonicity when CDF violations are detected. Useful for small sample
        sizes (< 400 records) but introduces slight bias. Default False.
    """
    count_all = len(dataset)
    if count_all == 0:
        raise ValueError("dataset is empty")

    # accumulators for debiasing true-high and true-low
    durations = []
    cumulative_weights_high = []
    cumulative_weights_low = []
    sum_weights_high = 0.0
    sum_weights_low = 0.0

    # Sort by duration
    sorted_dataset = sorted(dataset, key=lambda x: x['duration_ms'])
    for item in sorted_dataset:
        # Load duration and increment total_duration to compute mean.
        duration = float(item.get('duration_ms', 0.0))

        # Validate duration is non-negative
        if duration < 0:
            raise ValueError(f"Invalid duration: {duration}. Durations must be non-negative.")

        # Let p be the record's randomizedTriggerRate, clipped to stabilize extreme p values.
        p = clip(float(item.get('randomized_trigger_rate', 0.0)))

        # Let c be the record's randomized_confidence.
        c = str(item.get('randomized_confidence', '')).lower()

        # Let R be 1 when c is high, otherwise 0.
        R = 1 if c == 'high' else 0

        # Compute per-record weight w based on c:
        #   For estimating the high mean: w = (R - (p / 2)) / (1 - p).
        #   For estimating the low mean: w = ((1 - R) - (p / 2)) / (1 - p).
        w_high = (R - (p / 2.0)) / (1.0 - p)
        w_low = ((1 - R) - (p / 2.0)) / (1.0 - p)

        # Let sum_weights be the sum of w values across all records.
        sum_weights_high += w_high
        sum_weights_low += w_low

        durations.append(duration)
        cumulative_weights_high.append(sum_weights_high)
        cumulative_weights_low.append(sum_weights_low)

    # Let debiased_mean = total_weighted_duration / sum_weights, provided sum_weights is not near zero.
    unstable_high = abs(sum_weights_high) <= SMALL_THRESHOLD
    unstable_low  = abs(sum_weights_low)  <= SMALL_THRESHOLD

    # Additional stability check: warn if sum is negative
    if sum_weights_high < 0:
        print(f"WARNING: Sum of weights for high confidence is negative ({sum_weights_high:.4f}). Percentiles may be unreliable.")
        unstable_high = True
    if sum_weights_low < 0:
        print(f"WARNING: Sum of weights for low confidence is negative ({sum_weights_low:.4f}). Percentiles may be unreliable.")
        unstable_low = True

    # Compute cumulative distribution function (CDF)
    cdf_high = []
    cdf_low = []

    if not unstable_high:
        for index, _ in enumerate(durations):
            cdf_high.append(cumulative_weights_high[index] / sum_weights_high)

        # Check that CDF ends near 1.0
        if abs(cdf_high[-1] - 1.0) > 0.01:
            print(f"WARNING: Final CDF value for high confidence is {cdf_high[-1]:.4f}, expected ~1.0. Results may be unreliable.")

        # Check for non-monotonicity
        is_non_monotonic, violation_count = detect_non_monotonicity(cdf_high)
        if is_non_monotonic:
            print(f"WARNING: CDF for high confidence is non-monotonic ({violation_count} violations detected).")
            print("  This is expected when records have mixed trigger rates, as debiasing weights can be negative.")
            print("  For stable percentiles with mixed trigger rates, use: split_by_trigger_rate() + run_multi_rum_dp_pipeline()")
            if apply_monotonic_correction:
                print("  Applying isotonic regression to enforce monotonicity...")
                cdf_high = apply_isotonic_regression(cdf_high)
                print("  Monotonic correction applied.")
            else:
                print("  Consider using apply_monotonic_correction=True if percentiles are unreliable.")

    if not unstable_low:
        for index, _ in enumerate(durations):
            cdf_low.append(cumulative_weights_low[index] / sum_weights_low)

        # Check that CDF ends near 1.0
        if abs(cdf_low[-1] - 1.0) > 0.01:
            print(f"WARNING: Final CDF value for low confidence is {cdf_low[-1]:.4f}, expected ~1.0. Results may be unreliable.")

        # Check for non-monotonicity
        is_non_monotonic, violation_count = detect_non_monotonicity(cdf_low)
        if is_non_monotonic:
            print(f"WARNING: CDF for low confidence is non-monotonic ({violation_count} violations detected).")
            print("  This is expected when records have mixed trigger rates, as debiasing weights can be negative.")
            print("  For stable percentiles with mixed trigger rates, use: split_by_trigger_rate() + run_multi_rum_dp_pipeline()")
            if apply_monotonic_correction:
                print("  Applying isotonic regression to enforce monotonicity...")
                cdf_low = apply_isotonic_regression(cdf_low)
                print("  Monotonic correction applied.")
            else:
                print("  Consider using apply_monotonic_correction=True if percentiles are unreliable.")

    def invert_cdf(cdf_vals, quantile):
        """
        Invert a (sorted-duration) debiased CDF to find the quantile.

        Args:
          cdf_vals: list of debiased CDF values evaluated at the sorted durations.
                    Must be in the same order as the global `durations` list (ascending).
          quantile: target quantile in [0,1].

        Returns:
          A duration value corresponding to the q-th percentile (with linear interpolation).
          Note: this function expects a `durations` sequence to be available and aligned with
          `cdf_vals` (i.e., durations[i] corresponds to cdf_vals[i]).
        """
        # Walk the CDF values in ascending order looking for the first point
        # where the CDF reaches or exceeds the target quantile.
        for i, v in enumerate(cdf_vals):
            # If this CDF value already meets or exceeds quantile, we've found the bracket.
            if v >= quantile:
                # If it's the very first element, return the smallest duration.
                if i == 0:
                    return durations[0]

                # Otherwise, interpolate between the previous and current durations.
                lower_duration, upper_duration = durations[i - 1], durations[i]       # durations bracketing quantile
                lower_cdf_value, upper_cdf_value = cdf_vals[i - 1], cdf_vals[i]       # corresponding CDF values

                # If the two CDF values are identical (flat region), return the upper duration.
                if upper_cdf_value == lower_cdf_value:
                    return upper_duration

                # Compute interpolation fraction on the CDF axis, then map to duration axis.
                interpolation_fraction = (quantile - lower_cdf_value) / (upper_cdf_value - lower_cdf_value)
                return lower_duration + (upper_duration - lower_duration) * interpolation_fraction

        # If q is larger than all CDF values, return the max duration.
        return durations[-1]

    def compute_debiased_percentiles(cdf_vals, unstable):
        if len(cdf_vals) == 0 or unstable:
            return {"p25": float('nan'), "p50": float('nan'),
                    "p75": float('nan'), "p90": float('nan'), "p99": float('nan')}
        return {
            "p25": invert_cdf(cdf_vals, 0.25),
            "p50": invert_cdf(cdf_vals, 0.50),
            "p75": invert_cdf(cdf_vals, 0.75),
            "p90": invert_cdf(cdf_vals, 0.90),
            "p99": invert_cdf(cdf_vals, 0.99),
        }

    percentiles_high = compute_debiased_percentiles(cdf_high, unstable_high)
    percentiles_low = compute_debiased_percentiles(cdf_low, unstable_low)
    percentiles_all = {
        "p25": np.percentile(durations, 25),
        "p50": np.percentile(durations, 50),
        "p75": np.percentile(durations, 75),
        "p90": np.percentile(durations, 90),
        "p99": np.percentile(durations, 99),
    }

    # Print results
    print(title)
    print("Total items:", count_all)
    print("Estimated High confidence count:", sum_weights_high)
    print("Estimated Low confidence count:", sum_weights_low)
    print("Percentiles (all):")
    for k, v in percentiles_all.items():
        print(f"    {k}: {v}")

    if unstable_high:
        print("Debiased Percentiles (high): UNSTABLE (insufficient data or extreme weights)")
    else:
        print("Debiased Percentiles (high):")
        for k, v in percentiles_high.items():
            print(f"    {k}: {v}")

    if unstable_low:
        print("Debiased Percentiles (low): UNSTABLE (insufficient data or extreme weights)")
    else:
        print("Debiased Percentiles (low):")
        for k, v in percentiles_low.items():
            print(f"    {k}: {v}")
    print("-" * 40)



