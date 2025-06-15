"""
Pattern recognition model for chart patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import cv2
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
import logging

logger = logging.getLogger(__name__)


class PatternRecognizer:
    """Recognize chart patterns using computer vision and ML"""
    
    def __init__(self):
        self.patterns = {
            'head_and_shoulders': self._detect_head_and_shoulders,
            'double_top': self._detect_double_top,
            'double_bottom': self._detect_double_bottom,
            'triangle': self._detect_triangle,
            'flag': self._detect_flag,
            'wedge': self._detect_wedge,
            'channel': self._detect_channel
        }
        
        self.min_pattern_bars = 20
        self.lookback_window = 100
        
    def detect_patterns(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Detect all patterns in the data"""
        
        detected_patterns = {}
        
        for pattern_name, detect_func in self.patterns.items():
            patterns = detect_func(df)
            if patterns:
                detected_patterns[pattern_name] = patterns
        
        return detected_patterns
    
    def _detect_head_and_shoulders(self, df: pd.DataFrame) -> List[Dict]:
        """Detect head and shoulders pattern"""
        
        patterns = []
        prices = df['close'].values
        
        if len(prices) < self.min_pattern_bars * 3:
            return patterns
        
        # Find peaks and valleys
        peaks, _ = find_peaks(prices, distance=5, prominence=prices.std() * 0.5)
        valleys, _ = find_peaks(-prices, distance=5, prominence=prices.std() * 0.5)
        
        # Look for pattern: valley -> peak -> valley -> peak -> valley -> peak -> valley
        for i in range(len(peaks) - 2):
            if i < 2 or i + 1 >= len(valleys):
                continue
            
            # Get three peaks (left shoulder, head, right shoulder)
            left_shoulder_idx = peaks[i]
            head_idx = peaks[i + 1]
            right_shoulder_idx = peaks[i + 2]
            
            # Check if head is higher than shoulders
            if (prices[head_idx] > prices[left_shoulder_idx] and 
                prices[head_idx] > prices[right_shoulder_idx]):
                
                # Check if shoulders are approximately equal
                shoulder_diff = abs(prices[left_shoulder_idx] - prices[right_shoulder_idx])
                avg_shoulder = (prices[left_shoulder_idx] + prices[right_shoulder_idx]) / 2
                
                if shoulder_diff / avg_shoulder < 0.05:  # 5% tolerance
                    # Find neckline
                    neckline_points = valleys[(valleys > left_shoulder_idx) & 
                                            (valleys < right_shoulder_idx)]
                    
                    if len(neckline_points) >= 2:
                        neckline_level = np.mean(prices[neckline_points])
                        
                        pattern = {
                            'type': 'head_and_shoulders',
                            'start_idx': left_shoulder_idx,
                            'end_idx': right_shoulder_idx,
                            'head_price': prices[head_idx],
                            'neckline': neckline_level,
                            'pattern_height': prices[head_idx] - neckline_level,
                            'target': neckline_level - (prices[head_idx] - neckline_level),
                            'confidence': 0.8
                        }
                        
                        patterns.append(pattern)
        
        return patterns
    
    def _detect_double_top(self, df: pd.DataFrame) -> List[Dict]:
        """Detect double top pattern"""
        
        patterns = []
        prices = df['close'].values
        
        # Find peaks
        peaks, properties = find_peaks(prices, distance=10, prominence=prices.std() * 0.3)
        
        # Look for two peaks at similar levels
        for i in range(len(peaks) - 1):
            peak1_idx = peaks[i]
            peak2_idx = peaks[i + 1]
            
            # Check if peaks are at similar levels
            peak_diff = abs(prices[peak1_idx] - prices[peak2_idx])
            avg_peak = (prices[peak1_idx] + prices[peak2_idx]) / 2
            
            if peak_diff / avg_peak < 0.03:  # 3% tolerance
                # Find valley between peaks
                valley_idx = np.argmin(prices[peak1_idx:peak2_idx]) + peak1_idx
                valley_price = prices[valley_idx]
                
                # Ensure sufficient depth
                if (avg_peak - valley_price) / avg_peak > 0.03:
                    pattern = {
                        'type': 'double_top',
                        'peak1_idx': peak1_idx,
                        'peak2_idx': peak2_idx,
                        'valley_idx': valley_idx,
                        'resistance_level': avg_peak,
                        'support_level': valley_price,
                        'pattern_height': avg_peak - valley_price,
                        'target': valley_price - (avg_peak - valley_price),
                        'confidence': 0.75
                    }
                    
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_double_bottom(self, df: pd.DataFrame) -> List[Dict]:
        """Detect double bottom pattern"""
        
        patterns = []
        prices = df['close'].values
        
        # Find valleys (invert prices to find bottoms)
        valleys, properties = find_peaks(-prices, distance=10, prominence=prices.std() * 0.3)
        
        # Look for two valleys at similar levels
        for i in range(len(valleys) - 1):
            valley1_idx = valleys[i]
            valley2_idx = valleys[i + 1]
            
            # Check if valleys are at similar levels
            valley_diff = abs(prices[valley1_idx] - prices[valley2_idx])
            avg_valley = (prices[valley1_idx] + prices[valley2_idx]) / 2
            
            if valley_diff / avg_valley < 0.03:  # 3% tolerance
                # Find peak between valleys
                peak_idx = np.argmax(prices[valley1_idx:valley2_idx]) + valley1_idx
                peak_price = prices[peak_idx]
                
                # Ensure sufficient height
                if (peak_price - avg_valley) / avg_valley > 0.03:
                    pattern = {
                        'type': 'double_bottom',
                        'valley1_idx': valley1_idx,
                        'valley2_idx': valley2_idx,
                        'peak_idx': peak_idx,
                        'support_level': avg_valley,
                        'resistance_level': peak_price,
                        'pattern_height': peak_price - avg_valley,
                        'target': peak_price + (peak_price - avg_valley),
                        'confidence': 0.75
                    }
                    
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_triangle(self, df: pd.DataFrame) -> List[Dict]:
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        
        patterns = []
        prices = df['close'].values
        
        # Use rolling window to detect converging trendlines
        window_size = 30
        
        for i in range(len(prices) - window_size):
            window_prices = prices[i:i + window_size]
            x = np.arange(len(window_prices))
            
            # Find local peaks and valleys
            peaks, _ = find_peaks(window_prices, distance=3)
            valleys, _ = find_peaks(-window_prices, distance=3)
            
            if len(peaks) >= 2 and len(valleys) >= 2:
                # Fit lines to peaks and valleys
                peak_fit = np.polyfit(peaks, window_prices[peaks], 1)
                valley_fit = np.polyfit(valleys, window_prices[valleys], 1)
                
                # Check for convergence
                upper_slope = peak_fit[0]
                lower_slope = valley_fit[0]
                
                # Classify triangle type
                if upper_slope < -0.001 and abs(lower_slope) < 0.001:
                    triangle_type = 'descending'
                elif abs(upper_slope) < 0.001 and lower_slope > 0.001:
                    triangle_type = 'ascending'
                elif upper_slope < -0.001 and lower_slope > 0.001:
                    triangle_type = 'symmetrical'
                else:
                    continue
                
                # Calculate apex
                if upper_slope != lower_slope:
                    apex_x = (valley_fit[1] - peak_fit[1]) / (peak_fit[0] - valley_fit[0])
                    
                    if apex_x > len(window_prices) and apex_x < len(window_prices) * 2:
                        pattern = {
                            'type': f'{triangle_type}_triangle',
                            'start_idx': i,
                            'end_idx': i + window_size,
                            'upper_slope': upper_slope,
                            'lower_slope': lower_slope,
                            'apex_distance': apex_x - len(window_prices),
                            'confidence': 0.7
                        }
                        
                        patterns.append(pattern)
        
        return patterns
    
    def _detect_flag(self, df: pd.DataFrame) -> List[Dict]:
        """Detect flag patterns"""
        
        patterns = []
        prices = df['close'].values
        
        # Look for strong move followed by consolidation
        for i in range(20, len(prices) - 20):
            # Check for pole (strong move)
            pole_start = i - 20
            pole_end = i
            pole_move = prices[pole_end] - prices[pole_start]
            pole_return = pole_move / prices[pole_start]
            
            # Significant move (>5%)
            if abs(pole_return) > 0.05:
                # Check for flag (consolidation)
                flag_prices = prices[pole_end:pole_end + 15]
                flag_std = np.std(flag_prices)
                flag_mean = np.mean(flag_prices)
                
                # Low volatility consolidation
                if flag_std / flag_mean < 0.02:
                    # Fit trendline to flag
                    x = np.arange(len(flag_prices))
                    flag_slope, flag_intercept = np.polyfit(x, flag_prices, 1)
                    
                    # Flag should be counter-trend
                    if (pole_move > 0 and flag_slope < 0) or (pole_move < 0 and flag_slope > 0):
                        pattern = {
                            'type': 'bull_flag' if pole_move > 0 else 'bear_flag',
                            'pole_start': pole_start,
                            'pole_end': pole_end,
                            'flag_end': pole_end + 15,
                            'pole_height': abs(pole_move),
                            'target': prices[pole_end + 15] + pole_move,
                            'confidence': 0.65
                        }
                        
                        patterns.append(pattern)
        
        return patterns
    
    def _detect_wedge(self, df: pd.DataFrame) -> List[Dict]:
        """Detect wedge patterns"""
        
        patterns = []
        prices = df['close'].values
        
        window_size = 30
        
        for i in range(len(prices) - window_size):
            window_prices = prices[i:i + window_size]
            
            # Find peaks and valleys
            peaks, _ = find_peaks(window_prices, distance=3)
            valleys, _ = find_peaks(-window_prices, distance=3)
            
            if len(peaks) >= 3 and len(valleys) >= 3:
                # Fit trendlines
                peak_fit = np.polyfit(peaks, window_prices[peaks], 1)
                valley_fit = np.polyfit(valleys, window_prices[valleys], 1)
                
                upper_slope = peak_fit[0]
                lower_slope = valley_fit[0]
                
                # Both lines should slope in same direction
                if upper_slope * lower_slope > 0:
                    # Check for convergence
                    upper_start = peak_fit[0] * 0 + peak_fit[1]
                    upper_end = peak_fit[0] * window_size + peak_fit[1]
                    lower_start = valley_fit[0] * 0 + valley_fit[1]
                    lower_end = valley_fit[0] * window_size + valley_fit[1]
                    
                    start_width = upper_start - lower_start
                    end_width = upper_end - lower_end
                    
                    if abs(end_width) < abs(start_width) * 0.7:
                        wedge_type = 'rising' if upper_slope > 0 else 'falling'
                        
                        pattern = {
                            'type': f'{wedge_type}_wedge',
                            'start_idx': i,
                            'end_idx': i + window_size,
                            'upper_slope': upper_slope,
                            'lower_slope': lower_slope,
                            'convergence_rate': (start_width - end_width) / start_width,
                            'confidence': 0.6
                        }
                        
                        patterns.append(pattern)
        
        return patterns
    
    def _detect_channel(self, df: pd.DataFrame) -> List[Dict]:
        """Detect channel patterns"""
        
        patterns = []
        prices = df['close'].values
        
        window_size = 50
        
        for i in range(len(prices) - window_size):
            window_prices = prices[i:i + window_size]
            
            # Calculate linear regression
            x = np.arange(len(window_prices))
            slope, intercept = np.polyfit(x, window_prices, 1)
            
            # Calculate channel boundaries
            regression_line = slope * x + intercept
            deviations = window_prices - regression_line
            
            upper_channel = regression_line + np.std(deviations) * 2
            lower_channel = regression_line - np.std(deviations) * 2
            
            # Check how well prices respect the channel
            touches_upper = np.sum(np.abs(window_prices - upper_channel) < window_prices.std() * 0.1)
            touches_lower = np.sum(np.abs(window_prices - lower_channel) < window_prices.std() * 0.1)
            
            if touches_upper >= 2 and touches_lower >= 2:
                # Determine channel type
                if abs(slope) < 0.0001:
                    channel_type = 'horizontal'
                elif slope > 0:
                    channel_type = 'ascending'
                else:
                    channel_type = 'descending'
                
                pattern = {
                    'type': f'{channel_type}_channel',
                    'start_idx': i,
                    'end_idx': i + window_size,
                    'slope': slope,
                    'channel_width': np.std(deviations) * 4,
                    'upper_touches': touches_upper,
                    'lower_touches': touches_lower,
                    'confidence': min(0.9, (touches_upper + touches_lower) / 10)
                }
                
                patterns.append(pattern)
        
        return patterns
    
    def pattern_to_signal(self, patterns: Dict[str, List[Dict]], current_idx: int) -> int:
        """Convert detected patterns to trading signal"""
        
        signal_weight = 0
        
        for pattern_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                # Check if pattern is recent and relevant
                if pattern.get('end_idx', 0) >= current_idx - 5:
                    confidence = pattern.get('confidence', 0.5)
                    
                    # Bullish patterns
                    if pattern_type in ['double_bottom', 'ascending_triangle', 
                                       'bull_flag', 'falling_wedge']:
                        signal_weight += confidence
                    
                    # Bearish patterns
                    elif pattern_type in ['head_and_shoulders', 'double_top', 
                                         'descending_triangle', 'bear_flag', 'rising_wedge']:
                        signal_weight -= confidence
        
        # Convert weight to signal
        if signal_weight > 0.5:
            return 1
        elif signal_weight < -0.5:
            return -1
        else:
            return 0