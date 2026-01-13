"""
This script illustrates the application of randomized response to performance data
and demonstrates how debiasing techniques can recover meaningful aggregates such as
means and percentiles.

For transparency, the code used to generate the randomized dataset is included but
commented out, allowing others to review and scrutinize the full process.
"""

import numpy as np
import math
from typing import List, Dict, Tuple
from collections import defaultdict


def print_summarized_dataset(
    dataset: List[Dict],
    title: str,
    confidence_key: str = "confidence") -> None:
    """
    Compute and display comprehensive statistics for duration data grouped by confidence level.

    This function analyzes a dataset of performance measurements, computing means and percentiles
    for the overall dataset as well as separately for high and low confidence observations. It
    works with both original (non-randomized) data and data that has undergone randomized response,
    depending on the confidence_key parameter.

    Statistics computed:
        - Counts: Total items, high confidence count, low confidence count
        - Means: Overall mean, high confidence mean, low confidence mean
        - Percentiles (for each group): 25th, 50th, 75th, 90th, 95th, 99th

    Args:
        dataset: List of records, each containing:
                 - 'duration_ms' (numeric): Performance duration in milliseconds
                 - confidence_key (str): Confidence level, either 'high' or 'low'
        title: Descriptive title printed at the start of the output
        confidence_key: The dictionary key to use for confidence values. Use "confidence"
                       for original data or "randomized_confidence" for data after
                       randomized response has been applied. Default: "confidence"

    Prints:
        - Title
        - Total item count and counts by confidence level
        - Mean durations (overall, high, low)
        - Percentiles for all observations
        - Percentiles for high confidence observations
        - Percentiles for low confidence observations
        - Separator line

    Example Output:
        ```
        Original Data Statistics
        Total items: 500
        High confidence count: 350
        Low confidence count: 150
        Mean (all): 486.32
        Mean (high): 245.67
        Mean (low): 1823.45
        Percentiles (all):
            p25: 180
            p50: 250
            ...
        ```

    Use Cases:
        - Analyzing original data before randomization
        - Examining randomized data (without debiasing)
        - Comparing original vs randomized distributions
    """
    # Extract durations
    durations = np.array([item["duration_ms"] for item in dataset])
    is_high = np.array([item[confidence_key] == "high" for item in dataset])
    high_durations = durations[is_high]
    low_durations = durations[~is_high]

    # Counts
    count_high = len(high_durations)
    count_low = len(low_durations)
    count_all = len(durations)

    # Compute means
    mean_all = np.mean(durations)
    mean_high = np.mean(high_durations) if count_high > 0 else float('nan')
    mean_low = np.mean(low_durations) if count_low > 0 else float('nan')

    # Helper for percentiles
    def compute_percentiles(values):
        if len(values) == 0:
            return {"p25": float('nan'), "p50": float('nan'),
                    "p75": float('nan'), "p90": float('nan'), "p99": float('nan')}
        return {
            "p25": np.percentile(values, 25),
            "p50": np.percentile(values, 50),
            "p75": np.percentile(values, 75),
            "p90": np.percentile(values, 90),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99),
        }

    # Percentiles
    percentiles_all = compute_percentiles(durations)
    percentiles_high = compute_percentiles(high_durations)
    percentiles_low = compute_percentiles(low_durations)

    # Print results
    print(title)
    print(f"Total items: {count_all}")
    print(f"High confidence count: {count_high}")
    print(f"Low confidence count: {count_low}")
    print(f"Mean (all): {mean_all}")
    print(f"Mean (high): {mean_high}")
    print(f"Mean (low): {mean_low}")

    print("Percentiles (all):")
    for k, v in percentiles_all.items():
        print(f"    {k}: {v}")
    print("Percentiles (high):")
    for k, v in percentiles_high.items():
        print(f"    {k}: {v}")
    print("Percentiles (low):")
    for k, v in percentiles_low.items():
        print(f"    {k}: {v}")
    print("-" * 40)


# Threshold to flag unstable denominators
SMALL_THRESHOLD = 1e-9

def clip(p: float) -> float:
    """
    Clip a probability value to a safe numeric range for use in debiasing calculations.

    This function constrains trigger rates to [0.001, 0.999] to prevent numerical
    instability in randomized response debiasing formulas. When p appears in
    denominators like (1 - p), extreme values (0 or 1) can cause division by zero
    or produce extremely large weights that destabilize estimators.

    The bounds (0.001, 0.999) are conservative defaults that maintain numerical
    stability while minimally distorting the true probability. Adjust these bounds
    if your application requires a different stability/accuracy tradeoff.

    Args:
        p: Input probability value, typically a randomized response trigger rate.
           Expected to be in [0, 1], but will be clipped if outside [0.001, 0.999].

    Returns:
        The clipped probability value in the interval [0.001, 0.999].

    Examples:
        >>> clip(0.5)
        0.5
        >>> clip(0.0)
        0.001
        >>> clip(1.0)
        0.999
        >>> clip(0.25)
        0.25

    Use Cases:
        - Pre-processing trigger rates before per-record debiasing
        - Stabilizing weight calculations in debias_randomized_data_and_print_summarized_dataset
        - Protecting against edge cases where reported trigger rates are exactly 0 or 1
    """
    return max(0.001, min(0.999, p))

def debias_randomized_data_and_print_summarized_dataset(
    dataset: List[Dict],
    title: str
) -> None:
    """
    Debias randomized response data using per-record weighting and compute mean statistics.

    This function implements per-record debiasing for data that has undergone randomized
    response (RR) randomization. Unlike histogram-level debiasing (recover_dp_histogram),
    this approach computes a weight for each individual record based on its observed
    confidence and trigger rate, then aggregates weighted durations to estimate true means.

    Debiasing Algorithm:
        For each record with observed confidence R (1 if 'high', 0 if 'low') and trigger
        rate p, compute weights:
            w_high = (R - p/2) / (1 - p)
            w_low = ((1-R) - p/2) / (1 - p)

        The estimated mean for high confidence is:
            mean_high = Σ(duration × w_high) / Σ(w_high)

        These weights invert the randomization process:
        - Records with R=1 (observed high) contribute positively to high estimate
        - Records with R=0 (observed low) contribute negatively to high estimate
        - The scale factor 1/(1-p) amplifies the truthful signal

    Estimated Counts:
        The sum of weights (Σw_high, Σw_low) estimates the true count of high/low
        confidence records before randomization. These may not be integers and can
        differ from observed counts due to the debiasing correction.

    Args:
        dataset: List of records after randomized response has been applied. Each
                 record must contain:
                 - 'duration_ms' (numeric): Performance duration in milliseconds
                 - 'randomized_confidence' (str): Observed confidence after RR ('high' or 'low')
                 - 'randomized_trigger_rate' (float): Trigger rate used for this record,
                   in range [0, 1]. Will be clipped to [0.001, 0.999] for stability.
        title: Descriptive title printed at the start of the output

    Prints:
        - Title
        - Total item count
        - Estimated true counts for high and low confidence (sum of weights)
        - Overall mean (no debiasing applied)
        - Debiased mean for high confidence records
        - Debiased mean for low confidence records
        - Warning messages if weight sums are negative or near zero
        - Separator line

    Raises:
        ValueError: If dataset is empty or if any duration is negative

    Stability Warnings:
        - If sum of weights is near zero (< 1e-9), results marked as UNSTABLE
        - If sum of weights is negative, warning printed (can occur with extreme
          randomization or small sample sizes)

    Comparison to Histogram-Level Debiasing:
        Per-record debiasing (this function) is suitable for computing means but cannot
        directly recover percentiles. For percentile estimation, use the histogram-level
        approach in debias_randomized_data_and_print_summarized_percentiles().
    """
    count_all = len(dataset)
    if count_all == 0:
        raise ValueError("dataset is empty")

    # accumulators for debiasing true-high and true-low
    total_weighted_duration_high = 0.0
    sum_weights_high = 0.0
    total_weighted_duration_low = 0.0
    sum_weights_low = 0.0
    total_duration = 0.0

    for item in dataset:
        # Load duration and increment total_duration to compute mean.
        duration = float(item.get('duration_ms', 0.0))

        # Validate duration is non-negative
        if duration < 0:
            raise ValueError(f"Invalid duration: {duration}. Durations must be non-negative.")

        total_duration += duration

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

        # Let weighted_duration = duration * w.
        weighted_duration_high = duration * w_high
        weighted_duration_low = duration * w_low

        # Let total_weighted_duration be the sum of weighted_duration values across all records.
        total_weighted_duration_high += weighted_duration_high
        total_weighted_duration_low += weighted_duration_low

        # Let sum_weights be the sum of w values across all records.
        sum_weights_high += w_high
        sum_weights_low += w_low

    # Let debiased_mean = total_weighted_duration / sum_weights, provided sum_weights is not near zero.
    unstable_high = abs(sum_weights_high) <= SMALL_THRESHOLD
    unstable_low  = abs(sum_weights_low)  <= SMALL_THRESHOLD

    # Additional stability check: warn if sum is negative (can happen with extreme randomization)
    if sum_weights_high < 0:
        print(f"WARNING: Sum of weights for high confidence is negative ({sum_weights_high:.4f}). Results may be unreliable.")
    if sum_weights_low < 0:
        print(f"WARNING: Sum of weights for low confidence is negative ({sum_weights_low:.4f}). Results may be unreliable.")

    debiased_mean_high = None if unstable_high else (total_weighted_duration_high / sum_weights_high)
    debiased_mean_low  = None if unstable_low  else (total_weighted_duration_low  / sum_weights_low)

    # Compute mean for all items
    mean_all = total_duration / count_all

    # Print results
    print(title)
    print("Total items:", count_all)
    print("Estimated High confidence count:", sum_weights_high)
    print("Estimated Low confidence count:", sum_weights_low)
    print("Mean (all):", mean_all)
    if unstable_high:
        print("Debiased Mean (high): UNSTABLE (sum of weights near zero)")
    else:
        print("Debiased Mean (high):", debiased_mean_high)
    if unstable_low:
        print("Debiased Mean (low): UNSTABLE (sum of weights near zero)")
    else:
        print("Debiased Mean (low):", debiased_mean_low)
    print("-" * 40)


def histogram_percentiles(
    histogram: Dict[int, int]
) -> Dict[int, int]:
    """
    Compute percentiles from a value->count histogram using a single-pass algorithm.

    This function efficiently computes the 25th, 50th, 75th, 90th, 95th, and 99th
    percentiles from a histogram representation of data, where keys are values and
    values are occurrence counts. The algorithm makes a single pass through the
    sorted histogram buckets, identifying percentile boundaries as cumulative
    counts reach each target threshold.

    Args:
        histogram: Dictionary mapping duration values to their occurrence counts.
                  For example: {100: 5, 200: 10, 300: 3} means value 100 appears
                  5 times, value 200 appears 10 times, etc.

    Returns:
        Dictionary mapping percentile number to the corresponding duration value.
        For example: {25: 150, 50: 225, 75: 290, 90: 340, 95: 380, 99: 450}
        Returns NaN for all percentiles if histogram is empty.

    Algorithm:
        1. Sort histogram buckets by value
        2. Compute total count across all buckets
        3. Calculate target positions for each percentile (e.g., P50 = position total_count/2)
        4. Iterate through sorted buckets, accumulating counts
        5. When cumulative count reaches a percentile target, record that bucket's value
        6. Early exit once all percentiles are found
    """
    # Hardcoded percentiles to compute
    PERCENTILES = [25, 50, 75, 90, 95, 99]

    # Step 1: Sort buckets by value (ascending order)
    # This allows us to process values from smallest to largest
    sorted_items = sorted(histogram.items())

    # Step 2: Compute total count across all buckets
    # This represents the total number of observations in the dataset
    total_count = 0
    for _, count in sorted_items:
        total_count += count

    # Handle empty histogram edge case
    if total_count == 0:
        return {p: float('nan') for p in PERCENTILES}

    # Step 3: Pre-compute percentile target positions
    # For each percentile p, calculate the position in the sorted data where that
    # percentile occurs: position = ceil(p * total_count / 100)
    # Example: If total_count=500, P50 target position = ceil(50*500/100) = 250
    # Sort targets by position for efficient scanning
    percentile_targets = sorted(
        [(p, math.ceil(p * total_count / 100.0)) for p in PERCENTILES],
        key=lambda x: x[1]
    )

    # Step 4: Single pass through sorted histogram to find all percentiles
    results = {}
    cumulative = 0  # Running sum of counts seen so far
    target_idx = 0  # Current percentile target we're looking for
    n_targets = len(percentile_targets)

    # Iterate through each bucket in ascending value order
    for value, count in sorted_items:
        # Add this bucket's count to our running total
        cumulative += count

        # Step 5: Check if we've reached or passed any percentile targets
        # We may satisfy multiple percentiles with a single bucket if it has a large count
        while target_idx < n_targets and cumulative >= percentile_targets[target_idx][1]:
            p, _ = percentile_targets[target_idx]
            # Record this bucket's value as the value for percentile p
            results[p] = value
            target_idx += 1

        # Step 6: Early exit optimization - stop once all percentiles found
        if target_idx >= n_targets:
            break

    return results


def recover_dp_histogram(
    hist_high: Dict[int, int],
    hist_low: Dict[int, int],
    noise_rate: float,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Recover true histograms from randomized response data using histogram-level debiasing.

    This function inverts the Randomized Response (RR) mechanism to estimate the original
    distribution of high and low confidence values before randomization was applied. Unlike
    per-record debiasing, this operates directly on histogram buckets, making it efficient
    for aggregated data.

    Randomized Response Mechanism:
        For each record with true confidence C (high or low), the RR algorithm:
        1. With probability (1 - noise_rate): reports C truthfully
        2. With probability noise_rate: flips a fair coin and reports high/low randomly

    Mathematical Inversion:
        Given observed counts (O_high, O_low) in a bucket, we solve for true counts (T_high, T_low):
            O_high = T_high * (1 - noise_rate/2) + T_low * (noise_rate/2)
            O_low = T_low * (1 - noise_rate/2) + T_high * (noise_rate/2)

        Solving for T_high:
            T_high = (O_high - (O_high + O_low) * noise_rate/2) / (1 - noise_rate)

    Args:
        hist_high: Histogram of observed "high" confidence durations after randomization.
                   Maps duration -> count. Example: {100: 15, 200: 23}
        hist_low: Histogram of observed "low" confidence durations after randomization.
                  Maps duration -> count. Example: {100: 5, 200: 12}
        noise_rate: The trigger rate (flip probability) used in randomized response.
                    Typically 0.25, 0.5, or 0.75. Must be in (0, 1).

    Returns:
        Tuple of (corrected_high, corrected_low) histograms representing the estimated
        true distributions before randomization was applied. Both use the same bucket
        structure with corrected counts.

    Algorithm:
        1. Pre-compute scaling constants from noise_rate
        2. Identify all unique buckets across both histograms
        3. For each bucket, apply matrix inversion formula to recover true counts
        4. Clamp results to valid range [0, total_count]
        5. Ensure high + low = total_count for consistency
    """
    # Step 1: Pre-compute constants for the inversion formula
    # noise_factor = noise_rate / 2 represents the probability of randomly selecting
    # either "high" or "low" when the RR mechanism triggers (k=2 outcomes)
    noise_factor = noise_rate / 2

    # scale_factor = 1 / (1 - noise_rate) amplifies the debiased signal
    # since only (1 - noise_rate) fraction of records report truthfully
    scale_factor = 1.0 / (1.0 - noise_rate)

    # Step 2: Get all unique duration buckets from both histograms
    # We need to process buckets that appear in either histogram to avoid
    # missing any duration values that may only have high or only low counts
    all_buckets = set(hist_high.keys()) | set(hist_low.keys())

    # Initialize output histograms
    corrected_high = {}
    corrected_low = {}

    # Step 3: Process each bucket independently
    # Since RR is applied per-record, histogram buckets can be corrected separately
    for bucket in all_buckets:
        # Get observed counts for this bucket (default to 0 if bucket not present)
        observed_high = hist_high.get(bucket, 0)
        observed_low = hist_low.get(bucket, 0)
        total_count = observed_high + observed_low

        # Handle edge case: empty bucket
        if total_count == 0:
            corrected_high[bucket] = 0
            corrected_low[bucket] = 0
            continue

        # Step 4: Apply the inversion formula
        # Formula: corrected_high = (observed_high - total * noise_factor) * scale_factor
        # This removes the noise contribution and scales up the truthful responses
        corrected_val = (observed_high - (total_count * noise_factor)) * scale_factor

        # Step 5: Clamp corrected value to valid range [0, total_count]
        # Clamping prevents negative counts (from sampling variance) and ensures
        # corrected_high doesn't exceed total observations in this bucket
        corrected_high_val = int(round(max(0, min(corrected_val, total_count))))
        corrected_high[bucket] = corrected_high_val

        # Step 6: Ensure consistency - corrected_low is the remainder
        # This guarantees corrected_high + corrected_low = total_count
        corrected_low[bucket] = int(total_count - corrected_high_val)

    return corrected_high, corrected_low


def debias_randomized_data_and_print_summarized_percentiles(
    dataset: List[Dict],
    title: str) -> None:
    """
    This function processes a dataset that has undergone randomized response (RR)
    randomization, potentially with different trigger rates across records. It groups
    records by their trigger rate, constructs histograms, applies histogram-level
    debiasing to recover the true distributions, merges results across all trigger
    rates, and computes percentiles separately for high and low confidence groups.

    Workflow:
        1. Group records by trigger rate (handles heterogeneous RR parameters)
        2. Build duration histograms for high/low confidence within each group
        3. Debias each group's histograms using recover_dp_histogram()
        4. Merge debiased histograms across all trigger rates
        5. Compute percentiles (25th, 50th, 75th, 90th, 95th, 99th) for both groups
        6. Display results

    Why group by trigger rate?
        Different trigger rates require different debiasing corrections. By grouping
        records with the same trigger rate, we can apply the appropriate debiasing
        formula to each subset before merging. This is essential when data comes from
        multiple sources (e.g., different browsers, collection endpoints) using
        different randomization parameters.

    Args:
        dataset: List of records after randomized response has been applied. Each
                 record must contain:
                 - 'duration_ms' (int): Performance duration in milliseconds
                 - 'randomized_confidence' (str): Observed confidence after RR ('high' or 'low')
                 - 'randomized_trigger_rate' (float): The trigger rate used for this record
        title: Descriptive title printed at the start of the output

    Prints:
        - Total item count
        - Debiased percentiles for high confidence durations
        - Debiased percentiles for low confidence durations

    Raises:
        ValueError: If dataset is empty

    Example Output:
        ```
        Debiased Results
        Total items: 500
        Debiased Percentiles (high):
          P25: 180
          P50: 240
          P75: 300
          P90: 380
          P95: 450
          P99: 620
        Debiased Percentiles (low):
          P25: 1200
          P50: 1850
          P75: 2500
          P90: 3400
          P95: 4100
          P99: 5800
        ```
    """
    if not dataset:
        raise ValueError("dataset is empty")

    # Group by trigger rate and build histograms (single pass)
    grouped_histograms = defaultdict(lambda: {
        "hist_high": defaultdict(int),
        "hist_low": defaultdict(int)
    })

    for item in dataset:
        rate = item["randomized_trigger_rate"]
        duration = item["duration_ms"]
        confidence = item["randomized_confidence"]

        group = grouped_histograms[rate]
        group["hist_high" if confidence == "high" else "hist_low"][duration] += 1

    # Multiple trigger rates: accumulate directly into merged histogram
    merged_high = defaultdict(int)
    merged_low = defaultdict(int)

    for rate, group in grouped_histograms.items():
        recovered_high, recovered_low = recover_dp_histogram(
            hist_high=group["hist_high"],
            hist_low=group["hist_low"],
            noise_rate=rate
        )
        for bucket, count in recovered_high.items():
            merged_high[bucket] += count
        for bucket, count in recovered_low.items():
            merged_low[bucket] += count

    # Print results
    print(title)
    print("Total items:", len(dataset))
    dp_high_percentiles = histogram_percentiles(dict(merged_high))
    dp_low_percentiles = histogram_percentiles(dict(merged_low))
    print("Debiased Percentiles (high):")
    for p, v in sorted(dp_high_percentiles.items()):
        print(f"  P{p}: {v}")
    print("Debiased Percentiles (low):")
    for p, v in sorted(dp_low_percentiles.items()):
        print(f"  P{p}: {v}")



#
# How to use:
#   - Run the script directly with Python.
#   - It will print out a Python list called `data` that you can copy and paste
#     into another script for analysis or experimentation.
#
# What to expect:
#   - The overall mean duration will be close to ~500 ms.
#   - "High" confidence items will cluster around ~250 ms.
#   - "Low" confidence items will skew above ~750 ms, with a long tail.
#   - The 99th percentile (P99) will exceed 1000 ms, ensuring some long-duration outliers.
#
# This makes the dataset useful for simulating performance timing distributions
# and testing debiasing or randomized response algorithms in RUM workflows.

"""
import random
import numpy as np

# Parameters
N = 500
min_duration = 100
max_duration = 10000  # 10 seconds in ms

data = []

for _ in range(N):
    # Generate durations with skew: mostly short, some long tail
    if random.random() < 0.7:
        duration = int(random.gauss(250, 80))   # cluster near 250ms
    else:
        duration = int(random.gauss(2000, 1500))  # long tail

    # Clamp to valid range
    duration = max(min_duration, min(duration, max_duration))

    # Pseudo-random confidence: probability depends on duration
    prob_high = max(0.05, 1.0 - (duration / 2000))  # decays with duration
    confidence = "high" if random.random() < prob_high else "low"

    data.append({"duration_ms": duration, "confidence": confidence})

# Print dataset in copy-pasteable Python format
print("data = [")
for item in data:
    print(f"    {item},")
print("]")
"""

data = [
    {'duration_ms': 437, 'confidence': 'high'},
    {'duration_ms': 294, 'confidence': 'high'},
    {'duration_ms': 129, 'confidence': 'high'},
    {'duration_ms': 229, 'confidence': 'high'},
    {'duration_ms': 100, 'confidence': 'high'},
    {'duration_ms': 338, 'confidence': 'low'},
    {'duration_ms': 1828, 'confidence': 'low'},
    {'duration_ms': 1311, 'confidence': 'high'},
    {'duration_ms': 100, 'confidence': 'high'},
    {'duration_ms': 265, 'confidence': 'high'},
    {'duration_ms': 274, 'confidence': 'high'},
    {'duration_ms': 202, 'confidence': 'high'},
    {'duration_ms': 3128, 'confidence': 'high'},
    {'duration_ms': 358, 'confidence': 'low'},
    {'duration_ms': 237, 'confidence': 'high'},
    {'duration_ms': 4694, 'confidence': 'low'},
    {'duration_ms': 2869, 'confidence': 'low'},
    {'duration_ms': 318, 'confidence': 'low'},
    {'duration_ms': 2648, 'confidence': 'low'},
    {'duration_ms': 3950, 'confidence': 'low'},
    {'duration_ms': 315, 'confidence': 'high'},
    {'duration_ms': 268, 'confidence': 'high'},
    {'duration_ms': 234, 'confidence': 'high'},
    {'duration_ms': 252, 'confidence': 'high'},
    {'duration_ms': 2648, 'confidence': 'low'},
    {'duration_ms': 268, 'confidence': 'high'},
    {'duration_ms': 339, 'confidence': 'high'},
    {'duration_ms': 163, 'confidence': 'high'},
    {'duration_ms': 281, 'confidence': 'high'},
    {'duration_ms': 344, 'confidence': 'high'},
    {'duration_ms': 251, 'confidence': 'high'},
    {'duration_ms': 1860, 'confidence': 'low'},
    {'duration_ms': 269, 'confidence': 'high'},
    {'duration_ms': 239, 'confidence': 'high'},
    {'duration_ms': 239, 'confidence': 'high'},
    {'duration_ms': 258, 'confidence': 'high'},
    {'duration_ms': 159, 'confidence': 'high'},
    {'duration_ms': 7094, 'confidence': 'low'},
    {'duration_ms': 298, 'confidence': 'high'},
    {'duration_ms': 4172, 'confidence': 'low'},
    {'duration_ms': 194, 'confidence': 'high'},
    {'duration_ms': 3707, 'confidence': 'low'},
    {'duration_ms': 124, 'confidence': 'high'},
    {'duration_ms': 100, 'confidence': 'high'},
    {'duration_ms': 213, 'confidence': 'high'},
    {'duration_ms': 241, 'confidence': 'high'},
    {'duration_ms': 1626, 'confidence': 'low'},
    {'duration_ms': 233, 'confidence': 'high'},
    {'duration_ms': 2572, 'confidence': 'low'},
    {'duration_ms': 266, 'confidence': 'high'},
    {'duration_ms': 1414, 'confidence': 'high'},
    {'duration_ms': 193, 'confidence': 'high'},
    {'duration_ms': 272, 'confidence': 'high'},
    {'duration_ms': 255, 'confidence': 'high'},
    {'duration_ms': 343, 'confidence': 'high'},
    {'duration_ms': 100, 'confidence': 'low'},
    {'duration_ms': 187, 'confidence': 'high'},
    {'duration_ms': 119, 'confidence': 'high'},
    {'duration_ms': 321, 'confidence': 'high'},
    {'duration_ms': 282, 'confidence': 'high'},
    {'duration_ms': 283, 'confidence': 'high'},
    {'duration_ms': 186, 'confidence': 'high'},
    {'duration_ms': 262, 'confidence': 'low'},
    {'duration_ms': 262, 'confidence': 'low'},
    {'duration_ms': 272, 'confidence': 'high'},
    {'duration_ms': 906, 'confidence': 'high'},
    {'duration_ms': 2094, 'confidence': 'low'},
    {'duration_ms': 100, 'confidence': 'high'},
    {'duration_ms': 321, 'confidence': 'high'},
    {'duration_ms': 252, 'confidence': 'high'},
    {'duration_ms': 124, 'confidence': 'low'},
    {'duration_ms': 244, 'confidence': 'high'},
    {'duration_ms': 334, 'confidence': 'high'},
    {'duration_ms': 1828, 'confidence': 'low'},
    {'duration_ms': 210, 'confidence': 'high'},
    {'duration_ms': 400, 'confidence': 'high'},
    {'duration_ms': 1847, 'confidence': 'low'},
    {'duration_ms': 214, 'confidence': 'high'},
    {'duration_ms': 336, 'confidence': 'high'},
    {'duration_ms': 2323, 'confidence': 'low'},
    {'duration_ms': 310, 'confidence': 'high'},
    {'duration_ms': 949, 'confidence': 'low'},
    {'duration_ms': 304, 'confidence': 'high'},
    {'duration_ms': 2138, 'confidence': 'low'},
    {'duration_ms': 435, 'confidence': 'high'},
    {'duration_ms': 263, 'confidence': 'high'},
    {'duration_ms': 486, 'confidence': 'low'},
    {'duration_ms': 167, 'confidence': 'high'},
    {'duration_ms': 210, 'confidence': 'high'},
    {'duration_ms': 166, 'confidence': 'high'},
    {'duration_ms': 180, 'confidence': 'high'},
    {'duration_ms': 372, 'confidence': 'high'},
    {'duration_ms': 329, 'confidence': 'high'},
    {'duration_ms': 3541, 'confidence': 'low'},
    {'duration_ms': 134, 'confidence': 'high'},
    {'duration_ms': 334, 'confidence': 'high'},
    {'duration_ms': 242, 'confidence': 'high'},
    {'duration_ms': 2405, 'confidence': 'low'},
    {'duration_ms': 263, 'confidence': 'high'},
    {'duration_ms': 1831, 'confidence': 'low'},
    {'duration_ms': 100, 'confidence': 'high'},
    {'duration_ms': 100, 'confidence': 'high'},
    {'duration_ms': 133, 'confidence': 'high'},
    {'duration_ms': 245, 'confidence': 'high'},
    {'duration_ms': 182, 'confidence': 'high'},
    {'duration_ms': 148, 'confidence': 'high'},
    {'duration_ms': 100, 'confidence': 'high'},
    {'duration_ms': 408, 'confidence': 'low'},
    {'duration_ms': 1391, 'confidence': 'low'},
    {'duration_ms': 534, 'confidence': 'high'},
    {'duration_ms': 108, 'confidence': 'high'},
    {'duration_ms': 5130, 'confidence': 'low'},
    {'duration_ms': 205, 'confidence': 'high'},
    {'duration_ms': 333, 'confidence': 'high'},
    {'duration_ms': 440, 'confidence': 'high'},
    {'duration_ms': 189, 'confidence': 'high'},
    {'duration_ms': 218, 'confidence': 'high'},
    {'duration_ms': 3768, 'confidence': 'low'},
    {'duration_ms': 108, 'confidence': 'high'},
    {'duration_ms': 389, 'confidence': 'high'},
    {'duration_ms': 243, 'confidence': 'high'},
    {'duration_ms': 4803, 'confidence': 'low'},
    {'duration_ms': 231, 'confidence': 'high'},
    {'duration_ms': 2439, 'confidence': 'low'},
    {'duration_ms': 178, 'confidence': 'high'},
    {'duration_ms': 235, 'confidence': 'high'},
    {'duration_ms': 1345, 'confidence': 'high'},
    {'duration_ms': 267, 'confidence': 'high'},
    {'duration_ms': 372, 'confidence': 'high'},
    {'duration_ms': 197, 'confidence': 'high'},
    {'duration_ms': 358, 'confidence': 'high'},
    {'duration_ms': 4629, 'confidence': 'low'},
    {'duration_ms': 233, 'confidence': 'high'},
    {'duration_ms': 100, 'confidence': 'high'},
    {'duration_ms': 888, 'confidence': 'low'},
    {'duration_ms': 898, 'confidence': 'low'},
    {'duration_ms': 100, 'confidence': 'high'},
    {'duration_ms': 376, 'confidence': 'high'},
    {'duration_ms': 1075, 'confidence': 'high'},
    {'duration_ms': 328, 'confidence': 'low'},
    {'duration_ms': 2380, 'confidence': 'low'},
    {'duration_ms': 309, 'confidence': 'low'},
    {'duration_ms': 156, 'confidence': 'high'},
    {'duration_ms': 209, 'confidence': 'high'},
    {'duration_ms': 419, 'confidence': 'high'},
    {'duration_ms': 192, 'confidence': 'high'},
    {'duration_ms': 100, 'confidence': 'high'},
    {'duration_ms': 264, 'confidence': 'low'},
    {'duration_ms': 324, 'confidence': 'high'},
    {'duration_ms': 309, 'confidence': 'high'},
    {'duration_ms': 536, 'confidence': 'high'},
    {'duration_ms': 228, 'confidence': 'high'},
    {'duration_ms': 245, 'confidence': 'high'},
    {'duration_ms': 197, 'confidence': 'high'},
    {'duration_ms': 207, 'confidence': 'high'},
    {'duration_ms': 100, 'confidence': 'high'},
    {'duration_ms': 137, 'confidence': 'high'},
    {'duration_ms': 787, 'confidence': 'high'},
    {'duration_ms': 163, 'confidence': 'high'},
    {'duration_ms': 100, 'confidence': 'high'},
    {'duration_ms': 345, 'confidence': 'high'},
    {'duration_ms': 278, 'confidence': 'high'},
    {'duration_ms': 247, 'confidence': 'high'},
    {'duration_ms': 292, 'confidence': 'low'},
    {'duration_ms': 167, 'confidence': 'high'},
    {'duration_ms': 265, 'confidence': 'high'},
    {'duration_ms': 339, 'confidence': 'high'},
    {'duration_ms': 231, 'confidence': 'high'},
    {'duration_ms': 271, 'confidence': 'high'},
    {'duration_ms': 274, 'confidence': 'low'},
    {'duration_ms': 277, 'confidence': 'high'},
    {'duration_ms': 280, 'confidence': 'high'},
    {'duration_ms': 100, 'confidence': 'high'},
    {'duration_ms': 176, 'confidence': 'high'},
    {'duration_ms': 213, 'confidence': 'high'},
    {'duration_ms': 164, 'confidence': 'high'},
    {'duration_ms': 185, 'confidence': 'high'},
    {'duration_ms': 207, 'confidence': 'high'},
    {'duration_ms': 127, 'confidence': 'high'},
    {'duration_ms': 240, 'confidence': 'high'},
    {'duration_ms': 2189, 'confidence': 'low'},
    {'duration_ms': 182, 'confidence': 'high'},
    {'duration_ms': 180, 'confidence': 'high'},
    {'duration_ms': 1546, 'confidence': 'low'},
    {'duration_ms': 320, 'confidence': 'high'},
    {'duration_ms': 311, 'confidence': 'high'},
    {'duration_ms': 1828, 'confidence': 'low'},
    {'duration_ms': 310, 'confidence': 'high'},
    {'duration_ms': 177, 'confidence': 'high'},
    {'duration_ms': 1761, 'confidence': 'low'},
    {'duration_ms': 341, 'confidence': 'high'},
    {'duration_ms': 4114, 'confidence': 'low'},
    {'duration_ms': 344, 'confidence': 'low'},
    {'duration_ms': 3145, 'confidence': 'low'},
    {'duration_ms': 268, 'confidence': 'high'},
    {'duration_ms': 3495, 'confidence': 'high'},
    {'duration_ms': 230, 'confidence': 'high'},
    {'duration_ms': 2323, 'confidence': 'low'},
    {'duration_ms': 253, 'confidence': 'high'},
    {'duration_ms': 264, 'confidence': 'low'},
    {'duration_ms': 198, 'confidence': 'high'},
    {'duration_ms': 470, 'confidence': 'high'},
    {'duration_ms': 441, 'confidence': 'high'},
    {'duration_ms': 2920, 'confidence': 'low'},
    {'duration_ms': 342, 'confidence': 'high'},
    {'duration_ms': 142, 'confidence': 'high'},
    {'duration_ms': 211, 'confidence': 'high'},
    {'duration_ms': 144, 'confidence': 'high'},
    {'duration_ms': 294, 'confidence': 'high'},
    {'duration_ms': 254, 'confidence': 'high'},
    {'duration_ms': 5568, 'confidence': 'low'},
    {'duration_ms': 286, 'confidence': 'high'},
    {'duration_ms': 382, 'confidence': 'high'},
    {'duration_ms': 308, 'confidence': 'low'},
    {'duration_ms': 317, 'confidence': 'high'},
    {'duration_ms': 340, 'confidence': 'high'},
    {'duration_ms': 2110, 'confidence': 'low'},
    {'duration_ms': 242, 'confidence': 'high'},
    {'duration_ms': 268, 'confidence': 'high'},
    {'duration_ms': 357, 'confidence': 'low'},
    {'duration_ms': 299, 'confidence': 'high'},
    {'duration_ms': 312, 'confidence': 'low'},
    {'duration_ms': 216, 'confidence': 'low'},
    {'duration_ms': 272, 'confidence': 'high'},
    {'duration_ms': 155, 'confidence': 'high'},
    {'duration_ms': 395, 'confidence': 'high'},
    {'duration_ms': 223, 'confidence': 'high'},
    {'duration_ms': 354, 'confidence': 'low'},
    {'duration_ms': 235, 'confidence': 'low'},
    {'duration_ms': 286, 'confidence': 'high'},
    {'duration_ms': 2645, 'confidence': 'low'},
    {'duration_ms': 181, 'confidence': 'high'},
    {'duration_ms': 2367, 'confidence': 'low'},
    {'duration_ms': 1309, 'confidence': 'high'},
    {'duration_ms': 290, 'confidence': 'high'},
    {'duration_ms': 164, 'confidence': 'high'},
    {'duration_ms': 190, 'confidence': 'high'},
    {'duration_ms': 5172, 'confidence': 'low'},
    {'duration_ms': 295, 'confidence': 'high'},
    {'duration_ms': 247, 'confidence': 'low'},
    {'duration_ms': 128, 'confidence': 'high'},
    {'duration_ms': 291, 'confidence': 'high'},
    {'duration_ms': 2486, 'confidence': 'low'},
    {'duration_ms': 201, 'confidence': 'high'},
    {'duration_ms': 312, 'confidence': 'high'},
    {'duration_ms': 248, 'confidence': 'high'},
    {'duration_ms': 226, 'confidence': 'high'},
    {'duration_ms': 203, 'confidence': 'high'},
    {'duration_ms': 5440, 'confidence': 'low'},
    {'duration_ms': 3021, 'confidence': 'low'},
    {'duration_ms': 204, 'confidence': 'high'},
    {'duration_ms': 323, 'confidence': 'high'},
    {'duration_ms': 271, 'confidence': 'high'},
    {'duration_ms': 337, 'confidence': 'high'},
    {'duration_ms': 372, 'confidence': 'high'},
    {'duration_ms': 459, 'confidence': 'low'},
    {'duration_ms': 319, 'confidence': 'high'},
    {'duration_ms': 2307, 'confidence': 'low'},
    {'duration_ms': 304, 'confidence': 'high'},
    {'duration_ms': 234, 'confidence': 'high'},
    {'duration_ms': 258, 'confidence': 'high'},
    {'duration_ms': 158, 'confidence': 'high'},
    {'duration_ms': 1503, 'confidence': 'high'},
    {'duration_ms': 252, 'confidence': 'high'},
    {'duration_ms': 291, 'confidence': 'high'},
    {'duration_ms': 172, 'confidence': 'high'},
    {'duration_ms': 244, 'confidence': 'high'},
    {'duration_ms': 256, 'confidence': 'high'},
    {'duration_ms': 208, 'confidence': 'high'},
    {'duration_ms': 414, 'confidence': 'high'},
    {'duration_ms': 2569, 'confidence': 'low'},
    {'duration_ms': 299, 'confidence': 'low'},
    {'duration_ms': 164, 'confidence': 'high'},
    {'duration_ms': 256, 'confidence': 'high'},
    {'duration_ms': 2407, 'confidence': 'low'},
    {'duration_ms': 370, 'confidence': 'high'},
    {'duration_ms': 237, 'confidence': 'high'},
    {'duration_ms': 451, 'confidence': 'low'},
    {'duration_ms': 100, 'confidence': 'high'},
    {'duration_ms': 315, 'confidence': 'high'},
    {'duration_ms': 188, 'confidence': 'high'},
    {'duration_ms': 262, 'confidence': 'low'},
    {'duration_ms': 304, 'confidence': 'high'},
    {'duration_ms': 100, 'confidence': 'high'},
    {'duration_ms': 206, 'confidence': 'high'},
    {'duration_ms': 235, 'confidence': 'high'},
    {'duration_ms': 2786, 'confidence': 'low'},
    {'duration_ms': 198, 'confidence': 'low'},
    {'duration_ms': 196, 'confidence': 'low'},
    {'duration_ms': 2654, 'confidence': 'low'},
    {'duration_ms': 222, 'confidence': 'low'},
    {'duration_ms': 280, 'confidence': 'high'},
    {'duration_ms': 3471, 'confidence': 'low'},
    {'duration_ms': 174, 'confidence': 'high'},
    {'duration_ms': 306, 'confidence': 'high'},
    {'duration_ms': 169, 'confidence': 'high'},
    {'duration_ms': 585, 'confidence': 'high'},
    {'duration_ms': 275, 'confidence': 'high'},
    {'duration_ms': 100, 'confidence': 'high'},
    {'duration_ms': 3557, 'confidence': 'low'},
    {'duration_ms': 188, 'confidence': 'high'},
    {'duration_ms': 2078, 'confidence': 'low'},
    {'duration_ms': 284, 'confidence': 'high'},
    {'duration_ms': 135, 'confidence': 'high'},
    {'duration_ms': 246, 'confidence': 'high'},
    {'duration_ms': 100, 'confidence': 'high'},
    {'duration_ms': 158, 'confidence': 'high'},
    {'duration_ms': 284, 'confidence': 'high'},
    {'duration_ms': 2645, 'confidence': 'low'},
    {'duration_ms': 4218, 'confidence': 'low'},
    {'duration_ms': 199, 'confidence': 'high'},
    {'duration_ms': 1819, 'confidence': 'low'},
    {'duration_ms': 123, 'confidence': 'high'},
    {'duration_ms': 1415, 'confidence': 'low'},
    {'duration_ms': 100, 'confidence': 'high'},
    {'duration_ms': 428, 'confidence': 'high'},
    {'duration_ms': 290, 'confidence': 'high'},
    {'duration_ms': 251, 'confidence': 'high'},
    {'duration_ms': 311, 'confidence': 'high'},
    {'duration_ms': 4696, 'confidence': 'low'},
    {'duration_ms': 290, 'confidence': 'high'},
    {'duration_ms': 355, 'confidence': 'high'},
    {'duration_ms': 218, 'confidence': 'high'},
    {'duration_ms': 368, 'confidence': 'high'},
    {'duration_ms': 175, 'confidence': 'low'},
    {'duration_ms': 228, 'confidence': 'low'},
    {'duration_ms': 1671, 'confidence': 'low'},
    {'duration_ms': 335, 'confidence': 'high'},
    {'duration_ms': 202, 'confidence': 'high'},
    {'duration_ms': 372, 'confidence': 'high'},
    {'duration_ms': 3118, 'confidence': 'low'},
    {'duration_ms': 1232, 'confidence': 'low'},
    {'duration_ms': 254, 'confidence': 'high'},
    {'duration_ms': 144, 'confidence': 'high'},
    {'duration_ms': 226, 'confidence': 'high'},
    {'duration_ms': 181, 'confidence': 'high'},
    {'duration_ms': 250, 'confidence': 'high'},
    {'duration_ms': 295, 'confidence': 'high'},
    {'duration_ms': 1188, 'confidence': 'high'},
    {'duration_ms': 4657, 'confidence': 'low'},
    {'duration_ms': 3419, 'confidence': 'low'},
    {'duration_ms': 324, 'confidence': 'low'},
    {'duration_ms': 217, 'confidence': 'high'},
    {'duration_ms': 414, 'confidence': 'high'},
    {'duration_ms': 3520, 'confidence': 'low'},
    {'duration_ms': 272, 'confidence': 'high'},
    {'duration_ms': 5551, 'confidence': 'low'},
    {'duration_ms': 218, 'confidence': 'high'},
    {'duration_ms': 329, 'confidence': 'high'},
    {'duration_ms': 2325, 'confidence': 'low'},
    {'duration_ms': 858, 'confidence': 'low'},
    {'duration_ms': 279, 'confidence': 'high'},
    {'duration_ms': 349, 'confidence': 'high'},
    {'duration_ms': 389, 'confidence': 'high'},
    {'duration_ms': 279, 'confidence': 'high'},
    {'duration_ms': 145, 'confidence': 'high'},
    {'duration_ms': 2793, 'confidence': 'low'},
    {'duration_ms': 1536, 'confidence': 'high'},
    {'duration_ms': 334, 'confidence': 'high'},
    {'duration_ms': 255, 'confidence': 'high'},
    {'duration_ms': 100, 'confidence': 'high'},
    {'duration_ms': 408, 'confidence': 'high'},
    {'duration_ms': 195, 'confidence': 'high'},
    {'duration_ms': 225, 'confidence': 'high'},
    {'duration_ms': 132, 'confidence': 'high'},
    {'duration_ms': 226, 'confidence': 'low'},
    {'duration_ms': 413, 'confidence': 'high'},
    {'duration_ms': 182, 'confidence': 'high'},
    {'duration_ms': 105, 'confidence': 'high'},
    {'duration_ms': 136, 'confidence': 'high'},
    {'duration_ms': 151, 'confidence': 'high'},
    {'duration_ms': 217, 'confidence': 'high'},
    {'duration_ms': 191, 'confidence': 'high'},
    {'duration_ms': 1390, 'confidence': 'low'},
    {'duration_ms': 380, 'confidence': 'high'},
    {'duration_ms': 944, 'confidence': 'high'},
    {'duration_ms': 3227, 'confidence': 'low'},
    {'duration_ms': 111, 'confidence': 'low'},
    {'duration_ms': 183, 'confidence': 'high'},
    {'duration_ms': 362, 'confidence': 'low'},
    {'duration_ms': 232, 'confidence': 'high'},
    {'duration_ms': 393, 'confidence': 'high'},
    {'duration_ms': 223, 'confidence': 'high'},
    {'duration_ms': 139, 'confidence': 'low'},
    {'duration_ms': 303, 'confidence': 'high'},
    {'duration_ms': 191, 'confidence': 'high'},
    {'duration_ms': 199, 'confidence': 'low'},
    {'duration_ms': 171, 'confidence': 'high'},
    {'duration_ms': 485, 'confidence': 'high'},
    {'duration_ms': 2106, 'confidence': 'low'},
    {'duration_ms': 2040, 'confidence': 'low'},
    {'duration_ms': 2234, 'confidence': 'low'},
    {'duration_ms': 406, 'confidence': 'high'},
    {'duration_ms': 102, 'confidence': 'high'},
    {'duration_ms': 205, 'confidence': 'high'},
    {'duration_ms': 226, 'confidence': 'high'},
    {'duration_ms': 308, 'confidence': 'high'},
    {'duration_ms': 347, 'confidence': 'low'},
    {'duration_ms': 192, 'confidence': 'high'},
    {'duration_ms': 238, 'confidence': 'high'},
    {'duration_ms': 1374, 'confidence': 'low'},
    {'duration_ms': 2202, 'confidence': 'low'},
    {'duration_ms': 309, 'confidence': 'low'},
    {'duration_ms': 100, 'confidence': 'high'},
    {'duration_ms': 323, 'confidence': 'high'},
    {'duration_ms': 304, 'confidence': 'high'},
    {'duration_ms': 318, 'confidence': 'high'},
    {'duration_ms': 165, 'confidence': 'high'},
    {'duration_ms': 1434, 'confidence': 'low'},
    {'duration_ms': 252, 'confidence': 'high'},
    {'duration_ms': 160, 'confidence': 'high'},
    {'duration_ms': 376, 'confidence': 'high'},
    {'duration_ms': 411, 'confidence': 'high'},
    {'duration_ms': 201, 'confidence': 'high'},
    {'duration_ms': 194, 'confidence': 'high'},
    {'duration_ms': 283, 'confidence': 'low'},
    {'duration_ms': 283, 'confidence': 'high'},
    {'duration_ms': 205, 'confidence': 'high'},
    {'duration_ms': 270, 'confidence': 'high'},
    {'duration_ms': 165, 'confidence': 'high'},
    {'duration_ms': 3103, 'confidence': 'low'},
    {'duration_ms': 349, 'confidence': 'high'},
    {'duration_ms': 1575, 'confidence': 'low'},
    {'duration_ms': 283, 'confidence': 'high'},
    {'duration_ms': 319, 'confidence': 'high'},
    {'duration_ms': 412, 'confidence': 'high'},
    {'duration_ms': 267, 'confidence': 'high'},
    {'duration_ms': 2716, 'confidence': 'low'},
    {'duration_ms': 183, 'confidence': 'high'},
    {'duration_ms': 201, 'confidence': 'high'},
    {'duration_ms': 304, 'confidence': 'low'},
    {'duration_ms': 2115, 'confidence': 'low'},
    {'duration_ms': 345, 'confidence': 'low'},
    {'duration_ms': 335, 'confidence': 'high'},
    {'duration_ms': 108, 'confidence': 'high'},
    {'duration_ms': 327, 'confidence': 'high'},
    {'duration_ms': 319, 'confidence': 'high'},
    {'duration_ms': 2361, 'confidence': 'low'},
    {'duration_ms': 186, 'confidence': 'high'},
    {'duration_ms': 420, 'confidence': 'high'},
    {'duration_ms': 143, 'confidence': 'high'},
    {'duration_ms': 191, 'confidence': 'high'},
    {'duration_ms': 199, 'confidence': 'low'},
    {'duration_ms': 254, 'confidence': 'high'},
    {'duration_ms': 277, 'confidence': 'high'},
    {'duration_ms': 158, 'confidence': 'high'},
    {'duration_ms': 329, 'confidence': 'low'},
    {'duration_ms': 371, 'confidence': 'high'},
    {'duration_ms': 179, 'confidence': 'high'},
    {'duration_ms': 170, 'confidence': 'high'},
    {'duration_ms': 295, 'confidence': 'high'},
    {'duration_ms': 285, 'confidence': 'high'},
    {'duration_ms': 393, 'confidence': 'high'},
    {'duration_ms': 191, 'confidence': 'high'},
    {'duration_ms': 282, 'confidence': 'high'},
    {'duration_ms': 215, 'confidence': 'high'},
    {'duration_ms': 218, 'confidence': 'high'},
    {'duration_ms': 148, 'confidence': 'high'},
    {'duration_ms': 330, 'confidence': 'high'},
    {'duration_ms': 432, 'confidence': 'high'},
    {'duration_ms': 187, 'confidence': 'high'},
    {'duration_ms': 161, 'confidence': 'high'},
    {'duration_ms': 211, 'confidence': 'high'},
    {'duration_ms': 279, 'confidence': 'high'},
    {'duration_ms': 2976, 'confidence': 'high'},
    {'duration_ms': 189, 'confidence': 'high'},
    {'duration_ms': 307, 'confidence': 'high'},
    {'duration_ms': 2381, 'confidence': 'low'},
    {'duration_ms': 2819, 'confidence': 'low'},
    {'duration_ms': 183, 'confidence': 'low'},
    {'duration_ms': 203, 'confidence': 'high'},
    {'duration_ms': 1955, 'confidence': 'high'},
    {'duration_ms': 3796, 'confidence': 'low'},
    {'duration_ms': 191, 'confidence': 'high'},
    {'duration_ms': 2625, 'confidence': 'low'},
    {'duration_ms': 100, 'confidence': 'high'},
    {'duration_ms': 271, 'confidence': 'low'},
    {'duration_ms': 100, 'confidence': 'high'},
    {'duration_ms': 295, 'confidence': 'high'},
    {'duration_ms': 342, 'confidence': 'high'},
    {'duration_ms': 249, 'confidence': 'high'},
    {'duration_ms': 1147, 'confidence': 'low'},
    {'duration_ms': 195, 'confidence': 'high'},
    {'duration_ms': 276, 'confidence': 'low'},
    {'duration_ms': 442, 'confidence': 'high'},
    {'duration_ms': 281, 'confidence': 'high'},
    {'duration_ms': 103, 'confidence': 'high'},
    {'duration_ms': 266, 'confidence': 'high'},
    {'duration_ms': 211, 'confidence': 'high'},
    {'duration_ms': 228, 'confidence': 'high'},
    {'duration_ms': 213, 'confidence': 'high'},
    {'duration_ms': 270, 'confidence': 'high'},
    {'duration_ms': 2527, 'confidence': 'low'},
    {'duration_ms': 278, 'confidence': 'high'},
    {'duration_ms': 554, 'confidence': 'low'},
    {'duration_ms': 1927, 'confidence': 'low'},
    {'duration_ms': 100, 'confidence': 'high'},
    {'duration_ms': 2751, 'confidence': 'low'},
    {'duration_ms': 266, 'confidence': 'high'},
    {'duration_ms': 100, 'confidence': 'high'},
]

# This script applies a randomized response algorithm to an existing dataset of
# performance timings with confidence labels. Each record is re-labeled using
# randomized coin flips, with a trigger rate chosen from {0.25, 0.4994798, 0.75}.
# The value 0.4994798 is weighted to appear twice as often as the others.
#
# How to use:
#   - Ensure you have a `data` variable defined (a list of dicts with
#     "duration_ms" and "confidence").
#   - Run this script to produce a new list called `randomized_data`.
#   - The output will be printed in Python list format so you can copy and paste
#     it into other scripts for analysis.
#
# What to expect:
#   - Each item in `randomized_data` contains:
#       • "duration_ms": the original duration
#       • "randomized_confidence": the confidence after randomized response
#       • "randomized_trigger_rate": the trigger rate used for that record
#   - The randomized confidences will differ from the originals, reflecting the
#     privacy‑preserving coin‑flip process.
#   - This makes the dataset suitable for testing debiasing methods and verifying
#     how randomized response affects aggregate statistics in RUM workflows.

"""
import numpy as np
import random

# Weighted choice of trigger rates: 0.4994798 twice as likely
trigger_rates = [0.25, 0.4994798, 0.4994798, 0.75]

def generate_randomized_confidence(confidence: str):
    #
    # Apply randomized response algorithm to a confidence value.
    # Returns both randomized confidence and the trigger rate used.
    #
    randomizedTriggerRate = random.choice(trigger_rates)

    # Step 1: Toss a coin (random double between 0 and 1)
    first_coin_flip = random.random()

    # Step 2: If heads (>= flip probability), answer honestly
    if first_coin_flip >= randomizedTriggerRate:
        randomized_confidence = confidence
    else:
        # Step 3: If tails (< flip probability), toss again
        second_coin_flip = random.randint(0, 1)
        randomized_confidence = "high" if second_coin_flip == 0 else "low"

    return randomized_confidence, randomizedTriggerRate

# Apply randomized response to your dataset
randomized_data = []
for item in data:
    randomized_confidence, randomizedTriggerRate = generate_randomized_confidence(item["confidence"])
    randomized_data.append({
        "duration_ms": item["duration_ms"],
        "randomized_confidence": randomized_confidence,
        "randomized_trigger_rate": randomizedTriggerRate
    })

print("randomized_data = [")
for item in randomized_data:
    print(f"    {item},")
print("]")
"""

# Data has randomized response applied
randomized_data = [
    {'duration_ms': 437, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 294, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 129, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 229, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 100, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 338, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 1828, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 1311, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 100, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 265, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 274, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 202, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 3128, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 358, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 237, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 4694, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 2869, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 318, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 2648, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 3950, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 315, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 268, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 234, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 252, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 2648, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 268, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 339, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 163, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 281, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 344, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 251, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 1860, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 269, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 239, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 239, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 258, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 159, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 7094, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 298, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 4172, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 194, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 3707, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 124, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 100, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 213, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 241, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 1626, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 233, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 2572, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 266, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 1414, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 193, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 272, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 255, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 343, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 100, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 187, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 119, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 321, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 282, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 283, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 186, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 262, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 262, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 272, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 906, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 2094, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 100, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 321, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 252, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 124, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 244, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 334, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 1828, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 210, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 400, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 1847, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 214, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 336, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 2323, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 310, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 949, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 304, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 2138, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 435, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 263, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 486, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 167, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 210, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 166, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 180, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 372, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 329, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 3541, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 134, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 334, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 242, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 2405, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 263, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 1831, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 100, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 100, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 133, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 245, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 182, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 148, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 100, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 408, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 1391, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 534, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 108, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 5130, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 205, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 333, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 440, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 189, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 218, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 3768, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 108, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 389, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 243, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 4803, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 231, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 2439, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 178, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 235, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 1345, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 267, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 372, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 197, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 358, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 4629, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 233, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 100, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 888, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 898, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 100, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 376, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 1075, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 328, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 2380, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 309, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 156, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 209, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 419, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 192, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 100, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 264, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 324, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 309, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 536, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 228, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 245, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 197, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 207, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 100, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 137, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 787, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 163, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 100, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 345, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 278, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 247, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 292, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 167, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 265, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 339, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 231, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 271, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 274, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 277, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 280, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 100, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 176, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 213, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 164, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 185, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 207, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 127, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 240, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 2189, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 182, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 180, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 1546, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 320, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 311, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 1828, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 310, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 177, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 1761, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 341, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 4114, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 344, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 3145, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 268, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 3495, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 230, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 2323, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 253, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 264, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 198, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 470, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 441, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 2920, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 342, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 142, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 211, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 144, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 294, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 254, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 5568, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 286, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 382, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 308, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 317, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 340, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 2110, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 242, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 268, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 357, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 299, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 312, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 216, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 272, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 155, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 395, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 223, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 354, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 235, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 286, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 2645, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 181, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 2367, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 1309, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 290, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 164, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 190, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 5172, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 295, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 247, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 128, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 291, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 2486, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 201, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 312, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 248, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 226, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 203, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 5440, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 3021, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 204, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 323, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 271, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 337, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 372, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 459, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 319, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 2307, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 304, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 234, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 258, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 158, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 1503, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 252, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 291, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 172, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 244, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 256, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 208, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 414, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 2569, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 299, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 164, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 256, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 2407, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 370, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 237, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 451, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 100, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 315, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 188, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 262, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 304, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 100, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 206, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 235, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 2786, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 198, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 196, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 2654, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 222, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 280, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 3471, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 174, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 306, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 169, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 585, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 275, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 100, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 3557, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 188, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 2078, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 284, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 135, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 246, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 100, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 158, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 284, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 2645, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 4218, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 199, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 1819, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 123, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 1415, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 100, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 428, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 290, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 251, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 311, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 4696, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 290, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 355, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 218, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 368, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 175, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 228, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 1671, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 335, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 202, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 372, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 3118, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 1232, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 254, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 144, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 226, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 181, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 250, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 295, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 1188, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 4657, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 3419, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 324, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 217, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 414, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 3520, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 272, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 5551, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 218, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 329, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 2325, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 858, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 279, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 349, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 389, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 279, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 145, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 2793, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 1536, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 334, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 255, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 100, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 408, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 195, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 225, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 132, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 226, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 413, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 182, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 105, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 136, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 151, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 217, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 191, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 1390, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 380, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 944, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 3227, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 111, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 183, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 362, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 232, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 393, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 223, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 139, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 303, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 191, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 199, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 171, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 485, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 2106, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 2040, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 2234, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 406, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 102, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 205, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 226, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 308, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 347, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 192, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 238, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 1374, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 2202, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 309, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 100, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 323, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 304, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 318, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 165, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 1434, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 252, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 160, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 376, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 411, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 201, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 194, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 283, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 283, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 205, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 270, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 165, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 3103, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 349, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 1575, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 283, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 319, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 412, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 267, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 2716, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 183, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 201, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 304, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 2115, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 345, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 335, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 108, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 327, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 319, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 2361, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 186, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 420, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 143, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 191, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 199, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 254, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 277, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 158, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 329, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 371, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 179, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 170, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 295, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 285, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 393, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 191, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 282, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 215, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 218, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 148, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 330, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 432, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 187, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 161, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 211, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 279, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 2976, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 189, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 307, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 2381, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 2819, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 183, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 203, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 1955, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 3796, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 191, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 2625, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 100, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 271, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 100, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 295, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 342, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 249, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 1147, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 195, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 276, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 442, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 281, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 103, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 266, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 211, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 228, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 213, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 270, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.25},
    {'duration_ms': 2527, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 278, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
    {'duration_ms': 554, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 1927, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 100, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 2751, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 266, 'randomized_confidence': 'low', 'randomized_trigger_rate': 0.4994798},
    {'duration_ms': 100, 'randomized_confidence': 'high', 'randomized_trigger_rate': 0.75},
]


if __name__ == "__main__":
    # Print out statistics for the original data for comparison.
    print_summarized_dataset(data, "Original Data", confidence_key="confidence")

    # Print out statistics for the randomized response data.
    print_summarized_dataset(randomized_data, "Randomized Response", confidence_key="randomized_confidence")

    # Compute and print debiased means.
    debias_randomized_data_and_print_summarized_dataset(randomized_data, "Debiased Means")

    # Compute and print debiased percentiles
    debias_randomized_data_and_print_summarized_percentiles(randomized_data, "Debiased Percentiles")
