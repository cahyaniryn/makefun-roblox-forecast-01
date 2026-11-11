#!/usr/bin/env python3
"""Forecast Roblox Up-and-Coming chart ranks using only the supplied history."""

import json
import math
import os
from collections import defaultdict
from datetime import datetime


INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

BANDS = ((1, 1), (2, 3), (4, 10), (11, 30), (31, 200))
LAGS = (3.0, 6.0, 12.0, 24.0, 48.0, 72.0, 96.0, 144.0)

# Archive-level log-rank change means and standard deviations by starting band
# and lag. Runtime observations blend into these priors when enough pairs exist.
PRIOR_MEAN = {
    (1, 1): (0.10, 0.20, 0.34, 0.66, 1.44, 1.94, 2.33, 2.60),
    (2, 3): (0.03, 0.05, 0.12, 0.24, 0.77, 1.32, 1.85, 2.15),
    (4, 10): (0.02, 0.04, 0.08, 0.25, 0.60, 0.91, 1.14, 1.30),
    (11, 30): (0.01, 0.02, 0.04, 0.13, 0.26, 0.42, 0.55, 0.65),
    (31, 200): (-0.01, -0.02, -0.04, -0.09, -0.17, -0.23, -0.27, -0.30),
}
PRIOR_SD = {
    (1, 1): (0.25, 0.36, 0.51, 0.66, 0.79, 0.66, 0.57, 0.60),
    (2, 3): (0.32, 0.46, 0.53, 0.69, 0.85, 0.89, 0.88, 0.90),
    (4, 10): (0.22, 0.31, 0.46, 0.64, 0.84, 0.89, 0.90, 0.90),
    (11, 30): (0.19, 0.27, 0.38, 0.58, 0.81, 0.84, 0.83, 0.85),
    (31, 200): (0.14, 0.20, 0.28, 0.47, 0.69, 0.86, 0.94, 0.95),
}
PRIOR_WEIGHT = 40.0
MIN_PAIRS = 20
RHO = 0.19
BASE_EXIT_PER_DAY = 0.06
T_DF = 3.0
T_SCALE = math.sqrt((T_DF - 2.0) / T_DF)
PLACES = 10


def parse_time(value):
    """Parse the UTC timestamps used by the recorder and API details."""
    if not isinstance(value, str):
        raise ValueError("timestamp must be a string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed


def load_history(input_dir):
    with open(os.path.join(input_dir, "task.json"), encoding="utf-8") as source:
        task = json.load(source)

    reads = []
    with open(os.path.join(input_dir, "history.ndjson"), encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            chart = {}
            for row in record.get("up_and_coming") or ():
                universe_id = int(row["universe_id"])
                rank = int(row["rank"])
                if rank > 0:
                    playing = row.get("playing")
                    chart[universe_id] = (
                        rank,
                        float(playing) if playing is not None else None,
                    )
            if chart:
                reads.append(
                    {
                        "at": parse_time(record["at"]),
                        "chart": chart,
                        "details": record.get("details") or {},
                    }
                )
    reads.sort(key=lambda read: read["at"])
    if not reads:
        raise ValueError("history contains no usable Up-and-Coming chart reads")
    return task, reads


def band_for(rank):
    for band in BANDS:
        if band[0] <= rank <= band[1]:
            return band
    return BANDS[-1]


def interpolate_log_lag(values, hours):
    """Interpolate a lag curve linearly in log-hours."""
    if hours <= LAGS[0]:
        return values[0]
    if hours >= LAGS[-1]:
        return values[-1]
    log_hours = math.log(hours)
    for index in range(len(LAGS) - 1):
        lower = math.log(LAGS[index])
        upper = math.log(LAGS[index + 1])
        if lower <= log_hours <= upper:
            fraction = (log_hours - lower) / (upper - lower)
            return values[index] + fraction * (values[index + 1] - values[index])
    return values[-1]


def fit_rank_moments(reads):
    """Estimate log-rank drift and spread, blended with archive priors."""
    buckets = defaultdict(list)
    times = [read["at"] for read in reads]
    for left_index, left in enumerate(reads):
        for right_index in range(left_index + 1, len(reads)):
            elapsed = (times[right_index] - times[left_index]).total_seconds() / 3600.0
            if elapsed > 150.0:
                break
            safe_elapsed = max(elapsed, 0.5)
            lag_index = min(
                range(len(LAGS)),
                key=lambda index: abs(math.log(LAGS[index]) - math.log(safe_elapsed)),
            )
            if abs(elapsed - LAGS[lag_index]) > max(1.5, 0.25 * LAGS[lag_index]):
                continue
            later = reads[right_index]["chart"]
            for universe_id, (old_rank, _) in left["chart"].items():
                if universe_id in later:
                    new_rank = later[universe_id][0]
                    buckets[(lag_index, band_for(old_rank))].append(
                        math.log(new_rank) - math.log(old_rank)
                    )

    means = {}
    deviations = {}
    for lag_index in range(len(LAGS)):
        for band in BANDS:
            observations = buckets[(lag_index, band)]
            prior_mean = PRIOR_MEAN[band][lag_index]
            prior_sd = PRIOR_SD[band][lag_index]
            if len(observations) < MIN_PAIRS:
                means[(lag_index, band)] = prior_mean
                deviations[(lag_index, band)] = prior_sd
                continue
            observed_mean = sum(observations) / len(observations)
            variance = sum((value - observed_mean) ** 2 for value in observations) / len(observations)
            observed_sd = math.sqrt(max(variance, 1e-6))
            weight = len(observations) / (len(observations) + PRIOR_WEIGHT)
            means[(lag_index, band)] = (
                weight * observed_mean + (1.0 - weight) * prior_mean
            )
            deviations[(lag_index, band)] = weight * observed_sd + (1.0 - weight) * prior_sd
    return means, deviations


def fit_newcomer_share(reads, horizon_hours=72.0):
    """Estimate the fraction of top-place outcomes due to off-chart games."""
    counts = [[0, 0] for _ in range(PLACES)]
    times = [read["at"] for read in reads]
    for start_index, start in enumerate(reads):
        present = set(start["chart"])
        for later_index in range(start_index + 1, len(reads)):
            elapsed = (times[later_index] - times[start_index]).total_seconds() / 3600.0
            if elapsed > horizon_hours:
                break
            ordered = sorted(
                reads[later_index]["chart"].items(),
                key=lambda item: (item[1][0], item[0]),
            )
            for place in range(min(PLACES, len(ordered))):
                counts[place][1] += 1
                if ordered[place][0] not in present:
                    counts[place][0] += 1

    result = []
    for newcomers, total in counts:
        estimate = (newcomers + 20.0 * 0.06) / (total + 20.0) if total else 0.06
        result.append(min(0.20, max(0.01, estimate)))
    return result


def fit_exit_rate(reads):
    """Estimate the chart's per-day game exit rate from adjacent reads."""
    exits = 0
    exposure = 0.0
    for current, following in zip(reads, reads[1:]):
        elapsed = (following["at"] - current["at"]).total_seconds() / 3600.0
        if elapsed <= 0.0 or elapsed > 24.0:
            continue
        exits += len(set(current["chart"]) - set(following["chart"]))
        exposure += len(current["chart"]) * elapsed / 24.0
    if exposure == 0.0:
        return BASE_EXIT_PER_DAY
    return min(0.20, max(0.02, exits / exposure))


def latest_created_dates(reads):
    """Return each game's newest available creation timestamp."""
    created = {}
    for read in reads:
        for key, detail in read["details"].items():
            if not isinstance(detail, dict) or not detail.get("created"):
                continue
            try:
                created[int(key)] = parse_time(detail["created"])
            except (TypeError, ValueError):
                continue
    return created


def game_features(reads, created_dates):
    """Build end-of-history rank, momentum, age, and player-growth features."""
    latest = reads[-1]["chart"]
    times = [read["at"] for read in reads]
    first_seen = {}
    for index, read in enumerate(reads):
        for universe_id in read["chart"]:
            first_seen.setdefault(universe_id, times[index])

    end_time = times[-1]
    features = {}
    for universe_id, (rank, playing) in latest.items():
        # EWMA of recent log-rank movement, expressed per six hours.
        velocity_sum = 0.0
        weight_sum = 0.0
        weight = 1.0
        for index in range(len(reads) - 1, 0, -1):
            current, previous = reads[index], reads[index - 1]
            if universe_id not in current["chart"] or universe_id not in previous["chart"]:
                break
            elapsed = (times[index] - times[index - 1]).total_seconds() / 3600.0
            if elapsed <= 0.0 or elapsed > 24.0:
                break
            change = math.log(current["chart"][universe_id][0]) - math.log(
                previous["chart"][universe_id][0]
            )
            velocity_sum += weight * change * (6.0 / elapsed)
            weight_sum += weight
            weight *= 0.5
            if weight < 0.05:
                break
        velocity = velocity_sum / weight_sum if weight_sum else 0.0

        created = created_dates.get(universe_id)
        age_origin = created if created is not None else first_seen[universe_id]
        age_days = max(0.0, (end_time - age_origin).total_seconds() / 86400.0)

        player_counts = []
        for index in range(len(reads) - 1, -1, -1):
            if universe_id not in reads[index]["chart"]:
                break
            age_hours = (end_time - times[index]).total_seconds() / 3600.0
            if age_hours > 96.0:
                break
            count = reads[index]["chart"][universe_id][1]
            if count is not None and math.isfinite(count):
                player_counts.append(count)
            if index == 0 or len(player_counts) == 6:
                break

        playing_growth = 0.0
        if len(player_counts) >= 4:
            counts = list(reversed(player_counts))
            early = (counts[0] + counts[1]) / 2.0
            late = (counts[-1] + counts[-2]) / 2.0
            if early is not None and late is not None and early > 0.0 and late > 0.0:
                playing_growth = math.log(late / early)

        features[universe_id] = {
            "rank": rank,
            "velocity": velocity,
            "age_days": age_days,
            "playing_growth": playing_growth,
        }
    return features


def beta_fraction(a, b, x):
    """Continued fraction used by the regularized incomplete beta function."""
    max_iterations = 200
    epsilon = 3e-14
    minimum = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < minimum:
        d = minimum
    d = 1.0 / d
    result = d
    for iteration in range(1, max_iterations + 1):
        twice = 2 * iteration
        coefficient = iteration * (b - iteration) * x / (
            (qam + twice) * (a + twice)
        )
        d = 1.0 + coefficient * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + coefficient / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        result *= d * c

        coefficient = (
            -(a + iteration) * (qab + iteration) * x
            / ((a + twice) * (qap + twice))
        )
        d = 1.0 + coefficient * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + coefficient / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return result


def regularized_beta(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    factor = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * beta_fraction(a, b, x) / a
    return 1.0 - factor * beta_fraction(b, a, 1.0 - x) / b


def student_t_cdf(value):
    x = T_DF / (T_DF + value * value)
    tail = 0.5 * regularized_beta(T_DF / 2.0, 0.5, x)
    return 1.0 - tail if value >= 0.0 else tail


def rank_probabilities(log_rank_mean, sigma):
    """Discretize Student-t log-rank diffusion into probabilities for ranks 1–10."""
    probabilities = []
    for rank in range(1, PLACES + 1):
        lower = -math.inf if rank == 1 else math.log(rank - 0.5)
        upper = math.log(rank + 0.5)
        high = student_t_cdf((upper - log_rank_mean) / (sigma * T_SCALE))
        low = 0.0 if lower == -math.inf else student_t_cdf(
            (lower - log_rank_mean) / (sigma * T_SCALE)
        )
        probabilities.append(max(0.0, high - low))
    return probabilities


def forecast(reads, task, means, deviations, newcomer_share, exit_rate):
    slots = int(task.get("window", {}).get("slots", 72))
    places = min(PLACES, int(task.get("places", PLACES)))
    features = game_features(reads, latest_created_dates(reads))
    chart_size = len(features)
    scores = {}

    for universe_id, feature in features.items():
        rank = feature["rank"]
        band = band_for(rank)
        initial_log_rank = math.log(rank)
        drift_curve = [means[(index, band)] for index in range(len(LAGS))]
        spread_curve = [
            max(0.12, deviations[(index, band)]) for index in range(len(LAGS))
        ]
        age = feature["age_days"]
        hazard = exit_rate
        if age > 18.0:
            hazard += 0.10 * min(1.0, (age - 18.0) / 4.0)
        if age > 25.0:
            hazard += 0.80
        hazard = min(hazard, 1.5)

        totals = [0.0] * places
        for hour in range(1, slots + 1):
            mean = initial_log_rank + interpolate_log_lag(drift_curve, float(hour))
            steps = hour / 6.0
            momentum = feature["velocity"] * (RHO / (1.0 - RHO)) * (
                1.0 - RHO ** steps
            )
            mean += max(-0.5, min(0.5, momentum))
            spread = max(0.15, interpolate_log_lag(spread_curve, float(hour)))
            survival = math.exp(-hazard * hour / 24.0)
            rank_mass = rank_probabilities(mean, spread)

            # A modest mixture captures unusual player-count breakouts without
            # allowing a single volatile count to dominate the forecast.
            growth = feature["playing_growth"]
            if growth > 1.0:
                mixture = min(0.18, 0.05 * (growth - 1.0))
                surge = rank_probabilities(mean - 1.0, spread * 1.2)
                rank_mass = [
                    (1.0 - mixture) * ordinary + mixture * elevated
                    for ordinary, elevated in zip(rank_mass, surge)
                ]
            for place in range(places):
                totals[place] += survival * rank_mass[place]
        scores[universe_id] = [value / slots for value in totals]

    rows = []
    for place in range(places):
        candidates = sorted(
            (
                (values[place], universe_id)
                for universe_id, values in scores.items()
                if values[place] > 0.0
            ),
            key=lambda item: (-item[0], item[1]),
        )
        unnamed = newcomer_share[place] if place < len(newcomer_share) else 0.06
        selected = []
        for value, universe_id in candidates:
            remaining = chart_size - len(selected)
            if value <= unnamed / max(1, remaining) and len(selected) >= 5:
                break
            selected.append((value, universe_id))
            if len(selected) >= 100:
                break
        denominator = sum(value for value, _ in selected)
        if denominator <= 0.0:
            continue
        for value, universe_id in selected:
            probability = round(value / denominator * 0.80, 6)
            if probability >= 0.0004:
                rows.append(
                    {"universe_id": universe_id, "place": place + 1, "p": probability}
                )

    # Rounding can add a few ulps; scale any place that exceeds the contract.
    by_place = defaultdict(float)
    for row in rows:
        by_place[row["place"]] += row["p"]
    for row in rows:
        total = by_place[row["place"]]
        if total > 1.0:
            row["p"] = round(row["p"] / total, 6)

    ranked = []
    for universe_id, values in scores.items():
        total = sum(values)
        if total <= 0.0:
            continue
        expected_rank = sum((index + 1) * value for index, value in enumerate(values)) / total
        ranked.append((-total, expected_rank, universe_id))
    ranked.sort()
    top10 = [universe_id for _, _, universe_id in ranked[:places]]
    if len(top10) < places:
        last_chart = sorted(features.items(), key=lambda item: (item[1]["rank"], item[0]))
        for universe_id, _ in last_chart:
            if universe_id not in top10:
                top10.append(universe_id)
            if len(top10) == places:
                break
    return {"probabilities": rows, "top10": top10[:places]}


def main():
    task, reads = load_history(INPUT_DIR)
    means, deviations = fit_rank_moments(reads)
    newcomer_share = fit_newcomer_share(reads)
    exit_rate = fit_exit_rate(reads)
    result = forecast(reads, task, means, deviations, newcomer_share, exit_rate)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = task.get("output", "forecast.json")
    output_path = os.path.join(OUTPUT_DIR, filename)
    encoded = json.dumps(result, separators=(",", ":"), ensure_ascii=True) + "\n"
    if len(encoded.encode("utf-8")) > 1_048_576:
        raise ValueError("forecast exceeds the 1 MiB output limit")
    with open(output_path, "w", encoding="utf-8", newline="\n") as destination:
        destination.write(encoded)


if __name__ == "__main__":
    main()
