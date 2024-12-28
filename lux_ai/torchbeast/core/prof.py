# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Performance profiling utilities for MonoBeast distributed training.

This module provides a simple but efficient profiling system that tracks execution
times of different code sections during training. It uses online algorithms to
compute running means and variances of timing measurements, allowing for memory-efficient
tracking of performance statistics.

The implementation is based on Sutton-Barto's incremental mean/variance computation
algorithm, which allows O(1) updates for both mean and variance statistics without
storing the full history of measurements.

Key Features:
- Memory-efficient online statistics computation
- Tracks mean, variance, and standard deviation of execution times
- Provides percentage breakdowns of time spent in different sections
- Thread-unsafe but lightweight for single-threaded profiling
"""

import collections
import timeit


class Timings:
    """Tracks execution time statistics for different code sections.

    This class maintains running statistics (mean, variance) of execution times
    for named code sections. It uses an online algorithm to update statistics
    without storing the full history of measurements, making it memory-efficient
    for long-running training sessions.

    The implementation uses the incremental mean/variance computation from
    Sutton-Barto's Reinforcement Learning book, allowing O(1) updates for both
    mean and variance statistics.

    Note:
        This class is not thread-safe and should only be used in single-threaded
        contexts or with appropriate synchronization.

    Attributes:
        _means (defaultdict[str, float]): Running means for each named section
        _vars (defaultdict[str, float]): Running variances for each section
        _counts (defaultdict[str, int]): Number of measurements per section
        last_time (float): Timestamp of last measurement
    """

    def __init__(self):
        """Initialize timing statistics trackers.

        Creates empty defaultdict containers for means, variances, and counts,
        and initializes the last_time timestamp. All statistics start at zero.
        """
        self._means = collections.defaultdict(int)   # Running means
        self._vars = collections.defaultdict(int)    # Running variances
        self._counts = collections.defaultdict(int)  # Measurement counts
        self.last_time = 0.                         # Last timestamp
        self.reset()

    def reset(self):
        """Reset the last timestamp to current time.

        This method is typically called at the start of a new timing sequence
        or when resuming timing after a pause.
        """
        self.last_time = timeit.default_timer()

    def time(self, name: str):
        """Record and update statistics for a named timing event.

        Uses Sutton-Barto's online update algorithm to compute running statistics
        for the time elapsed since the last call to time() or reset(). Updates
        are performed in O(1) time and space, making it efficient for long-running
        training sessions.

        The algorithm updates mean and variance estimates incrementally using:
            mean_{n+1} = mean_n + (x - mean_n)/(n + 1)
            var_{n+1} = (n * var_n + n * (mean_n - mean_{n+1})^2 + (x - mean_{n+1})^2)/(n + 1)

        References:
            - Sutton-Barto: http://www.incompleteideas.net/book/first/ebook/node19.html
            - Math derivation: https://math.stackexchange.com/a/103025/5051

        Args:
            name: Identifier for the code section being timed

        Note:
            This method modifies internal state (_means, _vars, _counts) and
            updates last_time. The elapsed time is measured from the previous
            call to time() or reset().
        """
        now = timeit.default_timer()
        x = now - self.last_time
        self.last_time = now

        n = self._counts[name]

        mean = self._means[name] + (x - self._means[name]) / (n + 1)
        var = (
            n * self._vars[name] + n * (self._means[name] - mean) ** 2 + (x - mean) ** 2
        ) / (n + 1)

        self._means[name] = mean
        self._vars[name] = var
        self._counts[name] += 1

    def means(self) -> dict:
        """Get the mean execution times for all tracked sections.

        Returns:
            dict: Mapping of section names to their mean execution times in seconds
        """
        return self._means

    def vars(self) -> dict:
        """Get the execution time variances for all tracked sections.

        Returns:
            dict: Mapping of section names to their timing variances in seconds²
        """
        return self._vars

    def stds(self) -> dict:
        """Get the standard deviations of execution times.

        Returns:
            dict: Mapping of section names to timing standard deviations in seconds
        """
        return {k: v ** 0.5 for k, v in self._vars.items()}

    def summary(self, prefix: str = "") -> str:
        """Generate a human-readable summary of timing statistics.

        Creates a formatted string showing execution times for all tracked sections,
        sorted by duration (longest first). For each section, shows:
        - Mean execution time in milliseconds
        - Standard deviation in milliseconds
        - Percentage of total execution time

        Args:
            prefix: Optional string to prepend to the summary

        Returns:
            str: Formatted timing summary with per-section statistics and total time

        Example output:
            Section A: 150.23ms +- 12.34ms (45.32%)
            Section B: 100.45ms +- 8.91ms (30.21%)
            Section C: 81.12ms +- 5.67ms (24.47%)
            Total: 331.80ms
        """
        means = self.means()
        stds = self.stds()
        total = sum(means.values())

        result = prefix
        # Sort sections by mean time (descending) and format statistics
        for k in sorted(means, key=means.get, reverse=True):
            result += f"\n    {k}: {1000 * means[k]:.2f}ms +- {1000 * stds[k]:.2f}ms ({100 * means[k] / total:.2f}%)"
        result += f"\nTotal: {1000 * total:.2f}ms"
        return result
