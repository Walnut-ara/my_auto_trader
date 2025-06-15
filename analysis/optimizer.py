"""
Advanced optimization engine using Optuna
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import optuna
from datetime import datetime
import json

from core.data_manager import DataManager
from trading.backtest import BacktestEngine
from analysis.performance import PerformanceAnalyzer
from utils.logger import get_logger

logger = get_logger(__name__)


class AdvancedOptimizer:
    """Advanced parameter optimization with multiple objectives"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.backtest_engine = BacktestEngine()
        self.performance_analyzer = PerformanceAnalyzer()
        
    def optimize(
        self,
        symbol: str,
        data: pd.DataFrame,
        target_return: float = 0.02,
        n_trials: int = 100,
        param_ranges: Dict = None,
        multi_objective: bool = False
    ) -> Dict:
        """Run optimization"""
        
        logger.info(f"Starting optimization for {symbol}")
        logger.info(f"Target return: {target_return:.1%}, Trials: {n_trials}")
        
        # Set default parameter ranges if not provided
        if param_ranges is None:
            from config.settings import OptimizationConfig
            param_ranges = OptimizationConfig.PARAMETER_RANGES['moderate']
        
        # Create study
        if multi_objective:
            study = optuna.create_study(
                directions=['maximize', 'minimize', 'maximize'],  # return, drawdown, sharpe
                study_name=f"{symbol}_multi_objective"
            )
        else:
            study = optuna.create_study(
                direction='maximize',
                study_name=f"{symbol}_single_objective"
            )
        
        # Optimization objective
        def objective(trial):
            # Sample parameters
            params = self._sample_parameters(trial, param_ranges)
            
            # Run backtest
            results = self.backtest_engine.run(data, params)
            
            if results['trades'].empty:
                return -1.0 if not multi_objective else [-1.0, -1.0, -1.0]
            
            # Calculate performance
            performance = self.performance_analyzer.analyze_performance(
                results['trades'],
                results['equity_curve']
            )
            
            monthly_return = performance['summary']['monthly_return'] / 100
            max_drawdown = abs(performance['risk']['max_drawdown'] / 100)
            sharpe_ratio = performance['risk']['sharpe_ratio']
            
            if multi_objective:
                return monthly_return, max_drawdown, sharpe_ratio
            else:
                # Single objective: weighted score
                score = self._calculate_optimization_score(
                    monthly_return, target_return, max_drawdown, sharpe_ratio
                )
                return score
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        
        # Get best parameters
        if multi_objective:
            best_params = self._select_best_multi_objective(study)
        else:
            best_params = study.best_params
        
        # Final backtest with best parameters
        final_params = self._convert_trial_params(best_params, param_ranges)
        final_results = self.backtest_engine.run(data, final_params)
        
        # Final performance
        final_performance = self.performance_analyzer.analyze_performance(
            final_results['trades'],
            final_results['equity_curve']
        )
        
        # Prepare results
        optimization_result = {
            'symbol': symbol,
            'success': True,
            'target_return': target_return,
            'best_params': final_params,
            'performance': final_performance['summary'],
            'risk_metrics': final_performance['risk'],
            'monthly_breakdown': final_performance['monthly'],
            'optimization_details': {
                'n_trials': len(study.trials),
                'best_value': study.best_value if not multi_objective else None,
                'optimization_time': datetime.now().isoformat()
            }
        }
        
        # Check if target achieved
        achieved_return = final_performance['summary']['monthly_return'] / 100
        optimization_result['target_achieved'] = achieved_return >= target_return * 0.9
        
        logger.info(f"Optimization complete for {symbol}")
        logger.info(f"Achieved return: {achieved_return:.2%}, Target: {target_return:.2%}")
        
        return optimization_result
    
    def walk_forward_optimization(
        self,
        symbol: str,
        data: pd.DataFrame,
        n_splits: int = 5,
        train_ratio: float = 0.8,
        param_ranges: Dict = None
    ) -> Dict:
        """Walk-forward optimization for robustness"""
        
        logger.info(f"Starting walk-forward optimization for {symbol}")
        
        results = []
        data_length = len(data)
        split_size = data_length // n_splits
        
        for i in range(n_splits):
            # Define train/test split
            test_end = data_length - (i * split_size)
            test_start = test_end - split_size
            train_end = test_start
            train_start = max(0, train_end - int(split_size * train_ratio / (1 - train_ratio)))
            
            if train_start >= train_end:
                continue
            
            # Split data
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            logger.info(f"Split {i+1}/{n_splits}: Train {len(train_data)} bars, Test {len(test_data)} bars")
            
            # Optimize on training data
            optimization_result = self.optimize(
                symbol, train_data,
                n_trials=50,  # Fewer trials for walk-forward
                param_ranges=param_ranges
            )
            
            # Test on out-of-sample data
            test_results = self.backtest_engine.run(
                test_data,
                optimization_result['best_params']
            )
            
            test_performance = self.performance_analyzer.analyze_performance(
                test_results['trades'],
                test_results['equity_curve']
            )
            
            results.append({
                'split': i + 1,
                'train_performance': optimization_result['performance'],
                'test_performance': test_performance['summary'],
                'parameters': optimization_result['best_params']
            })
        
        # Analyze walk-forward results
        wf_analysis = self._analyze_walk_forward_results(results)
        
        return {
            'symbol': symbol,
            'method': 'walk_forward',
            'n_splits': n_splits,
            'results': results,
            'analysis': wf_analysis,
            'recommended_params': wf_analysis['best_stable_params']
        }
    
    def _sample_parameters(self, trial: optuna.Trial, param_ranges: Dict) -> Dict:
        """Sample parameters from ranges"""
        
        params = {}
        
        for param_name, param_values in param_ranges.items():
            if isinstance(param_values, list):
                # Categorical parameter
                params[param_name] = trial.suggest_categorical(param_name, param_values)
            elif isinstance(param_values, tuple) and len(param_values) == 2:
                # Continuous parameter
                if isinstance(param_values[0], float):
                    params[param_name] = trial.suggest_float(
                        param_name, param_values[0], param_values[1]
                    )
                else:
                    params[param_name] = trial.suggest_int(
                        param_name, param_values[0], param_values[1]
                    )
            else:
                # Default to first value if format unclear
                params[param_name] = param_values[0] if isinstance(param_values, list) else param_values
        
        return params
    
    def _convert_trial_params(self, trial_params: Dict, param_ranges: Dict) -> Dict:
        """Convert trial parameters to final format"""
        
        final_params = {}
        
        for param_name, value in trial_params.items():
            if param_name in param_ranges:
                final_params[param_name] = value
            else:
                # Handle nested parameters
                if '_' in param_name:
                    category, sub_param = param_name.split('_', 1)
                    if category not in final_params:
                        final_params[category] = {}
                    final_params[category][sub_param] = value
                else:
                    final_params[param_name] = value
        
        return final_params
    
    def _calculate_optimization_score(
        self,
        monthly_return: float,
        target_return: float,
        max_drawdown: float,
        sharpe_ratio: float
    ) -> float:
        """Calculate optimization score"""
        
        # Return score (40% weight)
        return_score = min(monthly_return / target_return, 2.0) * 0.4
        
        # Risk score (30% weight) - lower drawdown is better
        risk_score = (1 - min(max_drawdown / 0.10, 1.0)) * 0.3
        
        # Sharpe ratio score (30% weight)
        sharpe_score = min(sharpe_ratio / 2.0, 1.0) * 0.3
        
        return return_score + risk_score + sharpe_score
    
    def _select_best_multi_objective(self, study: optuna.Study) -> Dict:
        """Select best parameters from multi-objective study"""
        
        # Get Pareto front
        pareto_trials = [
            t for t in study.trials
            if t.values and all(v is not None for v in t.values)
        ]
        
        if not pareto_trials:
            return {}
        
        # Score each trial
        scored_trials = []
        for trial in pareto_trials:
            monthly_return, max_drawdown, sharpe_ratio = trial.values
            
            # Combined score
            score = (
                monthly_return * 0.4 +
                (1 - max_drawdown) * 0.3 +
                sharpe_ratio * 0.3
            )
            
            scored_trials.append((score, trial))
        
        # Select best
        best_trial = max(scored_trials, key=lambda x: x[0])[1]
        
        return best_trial.params
    
    def _analyze_walk_forward_results(self, results: List[Dict]) -> Dict:
        """Analyze walk-forward optimization results"""
        
        # Extract performance metrics
        train_returns = [r['train_performance']['monthly_return'] for r in results]
        test_returns = [r['test_performance']['monthly_return'] for r in results]
        
        # Calculate consistency
        train_mean = np.mean(train_returns)
        test_mean = np.mean(test_returns)
        
        # Performance degradation
        degradation = (train_mean - test_mean) / train_mean if train_mean > 0 else 0
        
        # Parameter stability
        param_stability = self._calculate_parameter_stability(results)
        
        # Find most stable parameters
        best_stable_params = self._find_stable_parameters(results, param_stability)
        
        return {
            'train_avg_return': train_mean,
            'test_avg_return': test_mean,
            'performance_degradation': degradation,
            'consistency_score': 1 - degradation if degradation < 1 else 0,
            'parameter_stability': param_stability,
            'best_stable_params': best_stable_params,
            'robust': degradation < 0.3  # Less than 30% degradation
        }
    
    def _calculate_parameter_stability(self, results: List[Dict]) -> Dict:
        """Calculate how stable each parameter is across splits"""
        
        param_values = {}
        
        # Collect all parameter values
        for result in results:
            for param, value in result['parameters'].items():
                if param not in param_values:
                    param_values[param] = []
                param_values[param].append(value)
        
        # Calculate stability metrics
        stability = {}
        
        for param, values in param_values.items():
            if all(isinstance(v, (int, float)) for v in values):
                # Numerical parameter
                stability[param] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                }
            else:
                # Categorical parameter
                unique_values = list(set(values))
                mode_count = max(values.count(v) for v in unique_values)
                stability[param] = {
                    'mode': max(set(values), key=values.count),
                    'consistency': mode_count / len(values)
                }
        
        return stability
    
    def _find_stable_parameters(self, results: List[Dict], stability: Dict) -> Dict:
        """Find the most stable parameter set"""
        
        stable_params = {}
        
        for param, metrics in stability.items():
            if 'mean' in metrics:
                # Use mean for numerical parameters
                stable_params[param] = metrics['mean']
            else:
                # Use mode for categorical parameters
                stable_params[param] = metrics['mode']
        
        return stable_params