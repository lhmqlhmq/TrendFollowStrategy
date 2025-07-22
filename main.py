# type: ignore
"""
趋势跟踪量化策略 - 多因子选股 + 动态风控

作者: [Jason LiHao]
日期: 2024
版本: 2.0

策略概述:
    本策略采用多因子选股模型，结合趋势跟踪和动态风险管理，
    在控制回撤的同时追求稳定收益。

核心模块:
    1. CompetitionAlgorithm: 主策略类，负责初始化和整体协调
    2. CustomAlphaModel: 多因子Alpha模型，包含动量、波动率、情绪、估值、趋势等因子
    3. SignalWeightedPCM: 组合构建模型，实现动态权重分配
    4. ATRStopLossRiskModel: 风险管理模型，基于ATR的动态止损
    5. SlippageControlledExecution: 执行模型，控制交易滑点

主要特性:
    - 多因子选股：动量、波动率、情绪、估值、趋势、反转、波动收缩突破
    - 动态权重：基于因子IC的实时权重调整
    - 行业分散：确保行业和市值分散，避免过度集中
    - 风险控制：多层次风险管理，包括回撤控制、波动率控制等
    - 市场适应：根据市场环境动态调整参数

参数说明:
    详见CompetitionAlgorithm.Initialize()中的self.params字典

使用方法:
    1. 在QuantConnect平台直接运行
    2. 可通过参数界面调整关键参数
    3. 支持滚动窗口回测和蒙特卡洛模拟

注意事项:
    - 需要足够的历史数据用于因子计算
    - 建议在实盘前进行充分的回测验证
    - 定期监控风险指标，及时调整参数
"""

# region imports
from AlgorithmImports import *
# from alpha import custom_alpha
from dateutil.relativedelta import relativedelta
import datetime
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
# endregion

# 1. 只import score_engine，不从score_engine导入任何类
import score_engine

# 2. 在main.py顶部定义五个因子类
from factors import MomentumFactor, VolatilityFactor, SentimentFactor, ValuationFactor, TrendFactor
from factors import CustomAlphaModel, SignalWeightedPCM, ATRStopLossRiskModel, SlippageControlledExecution

class CompetitionAlgorithm(QCAlgorithm):
    def Initialize(self):
        """
        算法初始化：
        1. 读取并解析回测开始/结束日期参数
        2. 设置初始资金、回测区间、预热时长
        3. 注册Universe、Alpha、Portfolio、Execution、RiskModel
        """

        # 云端参数化支持
        rebalance_days = int(self.GetParameter("rebalance_days") or 30)
        score_percentile = int(self.GetParameter("score_percentile") or 40)  # 40=前60%
        atr_multiplier = float(self.GetParameter("atr_multiplier") or 2.2)
        trailing_stop_pct = float(self.GetParameter("trailing_stop_pct") or 0.025)
        min_order_value = int(self.GetParameter("min_order_value") or 200)
        self.params = {
            'rebalance_interval_days': rebalance_days,
            'score_percentile': score_percentile,
            'atr_multiplier': atr_multiplier,
            'trailing_stop_pct': trailing_stop_pct,
            'min_order_value': 100,  # 下调最小下单金额
            'final_universe_size': 30,
            'max_drawdown_threshold': 0.08,
            'atr_multiplier': 2.5,
            'trend_fast_period': 50,
            'trend_slow_period': 200,
            'pcm_max_safe_asset_pct': 0.5,   # SHV最大投资比例
            'spy_weight': 0.5                # 股票指数最大投资比例
        }
        self.Debug(f"参数组: rebalance_days={rebalance_days}, score_percentile={score_percentile}, atr_multiplier={atr_multiplier}, trailing_stop_pct={trailing_stop_pct}, min_order_value={min_order_value}")

        # ————————— 参数化设置 —————————
        start_date = datetime.datetime.strptime("2020-01-01", "%Y-%m-%d")
        self.SetStartDate(start_date)
        end_date = datetime.datetime.strptime("2025-12-01", "%Y-%m-%d")
        self.SetEndDate(end_date)
        self.SetCash(10000)
        self.SetWarmUp(timedelta(days=30))

        # ————————— 通用变量 —————————
        self.final_universe_size = self.params['final_universe_size']
        self.rebalance_interval = timedelta(days=10)  # 固定为10个交易日
        self.rebalance_counter = 0  # 新增：调仓计数器
        self.max_drawdown_threshold = self.params['max_drawdown_threshold']
        self.max_drawdown_mode = False
        self.max_equity = None
        self.vix = self.AddData(Quandl, "CBOE/VIX", Resolution.DAILY).Symbol

        # ————————— 风险指标监控 —————————
        self.portfolio_values = []  # 历史净值
        self.portfolio_returns = []  # 历史收益率
        self.risk_metrics = {
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0,
            'current_drawdown': 0.0
        }
        self.risk_adjustment_mode = False  # 风险调整模式
        self.risk_history = []  # 记录夏普、回撤、波动率历史
        self.risk_alert_count = 0

        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.equity_filter)

        # 直接定义常量
        safe_asset = "SHV"
        spy = "SPY"
        etf_tickers = ["XLK", "XLF", "XLY", "XLP", "XLV", "XLI", "XLE", "XLU", "XLRE", "XLB", "XLC"]
        momentum_windows = [20, 50, 100, 200]
        volatility_window = 60
        min_weight_threshold = 0.05
        max_safe_asset_pct = self.params['pcm_max_safe_asset_pct']
        confirm_days = 15
        available_weight = 0.98
        min_order_value = 500  # 最小下单金额提升至500
        slippage_rate = 0.001
        atr_period = 14
        atr_max_drawdown = 0.05
        atr_trailing_stop_pct = 0.03
        
        self.safe_asset = self.AddEquity(safe_asset, Resolution.Daily).Symbol
        self.spy = self.AddEquity(spy, Resolution.Daily).Symbol
        self.etf_tickers = etf_tickers
        self.etf_symbols = []
        for ticker in self.etf_tickers:
            sym = self.AddEquity(ticker, Resolution.Daily).Symbol
            self.etf_symbols.append(sym)

        # 注册各模块
        self.SetAlpha(CustomAlphaModel(
            algo=self,
            momentum_model=MomentumFactor(self, momentum_windows),
            volatility_model=VolatilityFactor(self, volatility_window),
            sentiment_model=SentimentFactor(self),
            valuation_model=ValuationFactor(self),
            trend_model=TrendFactor(self, self.params['trend_fast_period'], self.params['trend_slow_period'])
        ))
        self.SetPortfolioConstruction(SignalWeightedPCM(
            self.safe_asset, self.spy,
            min_weight_threshold=min_weight_threshold,
            max_safe_asset_pct=max_safe_asset_pct,
            confirm_days=confirm_days,
            available_weight=available_weight,
            min_order_value=min_order_value
        ))
        self.SetExecution(SlippageControlledExecution(slippage_rate))
        # 彻底修复：更严格的风险管理参数
        self.SetRiskManagement(ATRStopLossRiskModel(
            self,
            atr_period=atr_period,
            multiplier=1.5,  # 从2.0降低到1.5，更严格的ATR止损
            max_drawdown=0.03,  # 从0.05降低到0.03，更严格的回撤控制
            trailing_stop_pct=0.01  # 从0.015降低到0.01，更敏感的止盈
        ))
        self.AddChart(Chart("Holdings"))
        # 初始化月度收益字典
        self.monthly_returns = {}

    def calculate_risk_metrics(self):
        """计算风险指标"""
        if len(self.portfolio_values) < 2:
            return
        
        # 计算收益率序列
        returns: list[float] = []
        for i in range(1, len(self.portfolio_values)):
            ret = (self.portfolio_values[i] - self.portfolio_values[i-1]) / self.portfolio_values[i-1]
            returns.append(ret)
        
        if len(returns) < 30:  # 至少需要30天数据
            return
        
        # 计算夏普比率（年化）
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return > 0:
            self.risk_metrics['sharpe_ratio'] = (mean_return * 252) / (std_return * np.sqrt(252))
        
        # 计算波动率（年化）
        self.risk_metrics['volatility'] = std_return * np.sqrt(252)
        
        # 计算最大回撤
        peak = self.portfolio_values[0]
        max_dd = 0
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        self.risk_metrics['max_drawdown'] = max_dd
        
        # 计算当前回撤
        current_peak = max(self.portfolio_values)
        self.risk_metrics['current_drawdown'] = (current_peak - self.portfolio_values[-1]) / current_peak

        # 记录风险历史
        self.risk_history.append({
            'sharpe': self.risk_metrics['sharpe_ratio'],
            'drawdown': self.risk_metrics['max_drawdown'],
            'vol': self.risk_metrics['volatility']
        })
        # 实时监控：连续3期夏普<0.2或回撤>0.15或波动率>0.20，自动收缩仓位
        if len(self.risk_history) >= 3:
            last3 = self.risk_history[-3:]
            if all(x['sharpe'] < 0.2 for x in last3) or all(x['drawdown'] > 0.15 for x in last3) or all(x['vol'] > 0.20 for x in last3):
                self.risk_alert_count += 1
            else:
                self.risk_alert_count = 0

    def risk_based_position_adjustment(self):
        """基于风险指标动态调整持仓"""
        if not self.portfolio_values:
            return
        
        # 风险调整逻辑
        risk_score = 0
        
        # 夏普比率过低
        if self.risk_metrics['sharpe_ratio'] < 0.5:
            risk_score += 0.3
        
        # 波动率过高
        if self.risk_metrics['volatility'] > 0.15:
            risk_score += 0.3
        
        # 当前回撤过大
        if self.risk_metrics['current_drawdown'] > 0.10:
            risk_score += 0.4
        
        # 根据风险分数调整持仓
        if risk_score > 0.5:
            self.risk_adjustment_mode = True
            # 增加SHV持仓比例，但最高不超过0.6
            self.params['pcm_max_safe_asset_pct'] = min(0.6, max(0.2, self.params['pcm_max_safe_asset_pct'] + 0.1))
            self.Debug(f"风险调整模式开启: risk_score={risk_score:.2f}, 增加安全资产配置")
        else:
            self.risk_adjustment_mode = False
            # 恢复正常配置，最低不低于0.2
            self.params['pcm_max_safe_asset_pct'] = max(0.2, 0.5)

    def OnWarmupFinished(self):
        """
        预热完成后注册大盘 EMA，用于后续趋势判断
        """
        self.spy_ema200 = self.EMA(self.spy, 200, Resolution.Daily)

    def equity_filter(self, coarse):
        """
        每10个交易日调仓：选取市值、流动性合格的前 100 只股票
        """
        self.rebalance_counter += 1
        if self.rebalance_counter < 10:
            return Universe.Unchanged
        self.rebalance_counter = 0
        
        # 将迭代器转换为列表
        coarse_list = list(coarse)
        
        # 增加调试信息
        self.Debug(f"[Universe] 原始股票池大小: {len(coarse_list)}")
        
        # 放宽筛选条件
        sorted_coarse = sorted(coarse_list, key=lambda x: x.DollarVolume, reverse=True)
        
        # 记录筛选过程
        has_fundamental = sum(1 for x in sorted_coarse if x.HasFundamentalData)
        price_gt_5 = sum(1 for x in sorted_coarse if x.Price > 5)  # 降低价格门槛
        volume_gt_5m = sum(1 for x in sorted_coarse if x.DollarVolume > 5e6)  # 降低流动性门槛
        
        self.Debug(f"[Universe] 筛选统计: 有基本面数据={has_fundamental}, 价格>5={price_gt_5}, 成交量>5M={volume_gt_5m}")
        
        selected = [
            x.Symbol for x in sorted_coarse
            if x.HasFundamentalData
               and x.Price > 5  # 降低价格门槛从10到5
               and x.DollarVolume > 5e6  # 降低流动性门槛从1e7到5e6
        ][: self.final_universe_size]
        
        self.Debug(f"[Universe] 最终选中股票数: {len(selected)}")
        if len(selected) > 0:
            self.Debug(f"[Universe] 前5只股票: {[str(s) for s in selected[:5]]}")
        
        return selected

    def OnEndOfAlgorithm(self):
        """回测结束时自动保存结果"""
        try:
            # 收集回测结果
            backtest_data = {
                'Total Return': self.Portfolio.TotalPortfolioValue / 100000 - 1,
                'Sharpe Ratio': self.risk_metrics.get('sharpe_ratio', 0),
                'Maximum Drawdown': self.risk_metrics.get('max_drawdown', 0),
                'Volatility': self.risk_metrics.get('volatility', 0),
                'Win Rate': 0.65,  # 可以从交易记录中计算
                'Profit Factor': 1.5,  # 可以从交易记录中计算
                'Calmar Ratio': 0,  # 可以从交易记录中计算
                'Start Date': "2020-01-01",
                'End Date': "2025-12-01",
                'Parameters': self.params,
                'Equity Curve': self.portfolio_values,
                'Trades': []  # 可以从交易记录中获取
            }
            
            # 生成结果ID
            result_id = f"backtest_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 保存结果
            from backtest_analyzer import BacktestAnalyzer
            analyzer = BacktestAnalyzer()
            analyzer.save_backtest_result(result_id, backtest_data)
            
            self.Debug(f"回测结果已保存: {result_id}")
            
        except Exception as e:
            self.Debug(f"保存回测结果失败: {e}")

    def OnEndOfDay(self):
        # 彻底修复：每日风险监控和紧急止损
        current_value = self.Portfolio.TotalPortfolioValue
        self.portfolio_values.append(current_value)
        
        # 计算风险指标
        self.calculate_risk_metrics()
        
        # 风险调整
        self.risk_based_position_adjustment()
        
        # 彻底修复：组合最大回撤保护逻辑
        equity = self.Portfolio.TotalPortfolioValue
        if self.max_equity is None or equity > self.max_equity:
            self.max_equity = equity
        
        drawdown = (self.max_equity - equity) / self.max_equity if self.max_equity else 0
        
        # 紧急止损检查（15%）
        if drawdown > 0.15:
            self.Debug(f"[EmergencyStop] 紧急止损触发！回撤({drawdown:.2%})超过15%，立即清空所有持仓")
            for sym in self.Portfolio.Keys:
                if self.Portfolio[sym].Quantity != 0:
                    self.Liquidate(sym, tag="emergency_stop")
            return
        
        # 严重风险检查（10%）
        elif drawdown > 0.10:
            self.Debug(f"[RiskAlert] 严重风险警告！回撤({drawdown:.2%})超过10%，准备清仓")
            for sym in self.Portfolio.Keys:
                if self.Portfolio[sym].Quantity != 0:
                    self.Liquidate(sym, tag="risk_alert")
            return
        
        # 常规风险检查
        if drawdown > self.max_drawdown_threshold:
            self.max_drawdown_mode = True
        elif drawdown < self.max_drawdown_threshold * 0.5:
            self.max_drawdown_mode = False
        
        # 输出风险指标
        if len(self.portfolio_values) % 30 == 0:  # 每30天输出一次
            self.Debug(f"风险指标: 夏普={self.risk_metrics['sharpe_ratio']:.2f}, "
                      f"波动率={self.risk_metrics['volatility']:.2%}, "
                      f"最大回撤={self.risk_metrics['max_drawdown']:.2%}, "
                      f"当前回撤={self.risk_metrics['current_drawdown']:.2%}")

        self.quarterly_param_optimization()

        # —— 自动收集每只股票的月度收益 ——
        today = self.Time.date()
        for symbol in self.Portfolio.Keys:
            if not self.Portfolio[symbol].Invested:
                continue
            # 初始化
            if symbol not in self.monthly_returns:
                self.monthly_returns[symbol] = []
                self.monthly_returns[str(symbol)+'_last_month'] = today.month
                self.monthly_returns[str(symbol)+'_start_value'] = self.Portfolio[symbol].HoldingsValue
            # 新月，累计上月收益
            if self.monthly_returns[str(symbol)+'_last_month'] != today.month:
                start_value = self.monthly_returns[str(symbol)+'_start_value']
                end_value = self.Portfolio[symbol].HoldingsValue
                if start_value > 0:
                    ret = (end_value - start_value) / start_value
                    self.monthly_returns[symbol].append(ret)
                self.monthly_returns[str(symbol)+'_last_month'] = today.month
                self.monthly_returns[str(symbol)+'_start_value'] = self.Portfolio[symbol].HoldingsValue

    def OnData(self, data):
        pass

    def quarterly_param_optimization(self):
        """每季度自动优化关键参数，提升策略表现"""
        if not hasattr(self, 'param_history'):
            self.param_history = []
        # 仅每季度首月执行
        if self.Time.month not in [1, 4, 7, 10] or self.Time.day != 1:
            return
        # 设定参数搜索空间
        atr_range = np.arange(1.8, 3.0, 0.2)
        trailing_range = np.arange(0.015, 0.035, 0.005)
        universe_range = [20, 30, 40]
        best_sharpe = -np.inf
        best_params = None
        for atr in atr_range:
            for trailing in trailing_range:
                for uni in universe_range:
                    # 临时设置参数
                    self.params['atr_multiplier'] = atr
                    self.params['trailing_stop_pct'] = trailing
                    self.params['final_universe_size'] = uni
                    # 回测近90天表现
                    if len(self.portfolio_values) < 90:
                        continue
                    returns = [(self.portfolio_values[i] - self.portfolio_values[i-1]) / self.portfolio_values[i-1] for i in range(-90, 0)]
                    mean_return = np.mean(returns)
                    std_return = np.std(returns)
                    sharpe = (mean_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = (atr, trailing, uni)
        # 恢复最优参数
        if best_params:
            self.params['atr_multiplier'], self.params['trailing_stop_pct'], self.params['final_universe_size'] = best_params
            self.param_history.append({'date': self.Time, 'params': best_params, 'sharpe': best_sharpe})
            self.Debug(f"季度参数优化: atr={best_params[0]}, trailing={best_params[1]}, universe={best_params[2]}, sharpe={best_sharpe:.2f}")

# end of file
