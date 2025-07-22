from AlgorithmImports import Resolution
import numpy as np
import pandas as pd

def quantile_normalize(scores_dict):
    # 分位数归一化，返回每个symbol的分位数得分
    items = list(scores_dict.items())
    sorted_items = sorted(items, key=lambda x: x[1])
    n = len(items)
    rank_dict = {k: i/(n-1) if n > 1 else 0.5 for i, (k, v) in enumerate(sorted_items)}
    return rank_dict

def _get_closes_from_history(history, symbol):
    if history is None or len(history) == 0:
        return None
    if isinstance(history, pd.Series):
        return history
    if 'close' in history.columns:
        closes = history['close']
        if isinstance(closes, pd.DataFrame) and closes.columns.nlevels > 1:
            # 多symbol多字段
            try:
                closes = closes.xs(symbol, level=0, axis=1)
            except Exception:
                return None
        return closes
    try:
        closes = history.xs(symbol, level=0)['close']
        return closes
    except Exception:
        return None

def get_momentum_score(symbol, algo):
    history = algo.History([symbol], 21, Resolution.DAILY)
    closes = _get_closes_from_history(history, symbol)
    if closes is None or len(closes) < 2:
        return 0
    return float(closes.iloc[-1]) / float(closes.iloc[0]) - 1

def get_multi_momentum_score(symbol, algo, windows=[20, 50, 100]):
    scores: list[float] = []
    for win in windows:
        history = algo.History([symbol], win+1, Resolution.DAILY)
        closes = _get_closes_from_history(history, symbol)
        if closes is None or len(closes) < 2:
            scores.append(0)
        else:
            scores.append(float(closes.iloc[-1]) / float(closes.iloc[0]) - 1)
    mean = np.mean(scores)
    std = np.std(scores)
    z_scores = [(s - mean) / std if std > 0 else 0 for s in scores]
    filtered = [s if abs(z) < 3 else 0 for s, z in zip(scores, z_scores)]
    return np.mean(filtered)

def get_volatility_score(symbol, algo):
    history = algo.History([symbol], 21, Resolution.DAILY)
    closes = _get_closes_from_history(history, symbol)
    if closes is None or len(closes) < 2:
        return 0
    returns = np.diff(np.log(closes))
    std = np.std(returns)
    return 1 / (std + 1e-6)  # 防止除零

def get_sentiment_score(symbol, algo):
    """改进的情感因子：结合短期动量、成交量变化和RSI"""
    try:
        # 获取历史数据
        history = algo.History([symbol], 21, Resolution.DAILY)
        closes = _get_closes_from_history(history, symbol)
        if closes is None or len(closes) < 21:
            return 0
        
        # 1. 短期动量 (5日)
        short_momentum = float(closes.iloc[-1]) / float(closes.iloc[-6]) - 1
        
        # 2. 中期动量 (21日)
        medium_momentum = float(closes.iloc[-1]) / float(closes.iloc[0]) - 1
        
        # 3. 计算RSI
        rsi = algo.RSI(symbol, 14, MovingAverageType.WILDERS, Resolution.DAILY)
        rsi_val = rsi.Current.Value if rsi.IsReady else 50
        
        # 4. 成交量变化 (如果有成交量数据)
        volume_score = 0
        try:
            if 'volume' in history.columns:
                volumes = history['volume']
                if len(volumes) >= 5:
                    recent_vol_avg = np.mean(volumes.iloc[-5:])
                    past_vol_avg = np.mean(volumes.iloc[-21:-5])
                    if past_vol_avg > 0:
                        volume_change = (recent_vol_avg - past_vol_avg) / past_vol_avg
                        volume_score = np.clip(volume_change, -0.5, 0.5)
        except Exception:
            pass
        
        # 综合评分
        sentiment_score = (
            0.4 * short_momentum +      # 短期动量权重40%
            0.3 * medium_momentum +     # 中期动量权重30%
            0.2 * ((rsi_val - 50) / 50) +  # RSI偏离权重20%
            0.1 * volume_score          # 成交量变化权重10%
        )
        
        return sentiment_score
        
    except Exception:
        return 0

def get_valuation_score(symbol, algo):
    """改进的估值因子：结合P/E、P/B、ROE等基本面指标"""
    try:
        sec = algo.Securities[symbol]
        
        # 1. 价格动量反转 (技术面)
        history = algo.History([symbol], 63, Resolution.DAILY)  # 3个月
        closes = _get_closes_from_history(history, symbol)
        if closes is None or len(closes) < 63:
            return 0
        
        # 计算不同时间窗口的动量
        momentum_21d = float(closes.iloc[-1]) / float(closes.iloc[-22]) - 1
        momentum_63d = float(closes.iloc[-1]) / float(closes.iloc[0]) - 1
        
        # 反转信号：短期动量与中期动量背离
        reversal_signal = -momentum_21d if abs(momentum_21d) > 0.1 else 0
        
        # 2. 基本面估值 (如果有基本面数据)
        fundamental_score = 0
        try:
            if hasattr(sec, 'Fundamentals') and sec.Fundamentals is not None:
                # P/E比率
                pe_ratio = getattr(sec.Fundamentals.ValuationRatios, 'PERatio', None)
                if pe_ratio and pe_ratio > 0:
                    pe_score = 1.0 / (pe_ratio + 1)  # 越低越好
                    fundamental_score += 0.4 * pe_score
                
                # P/B比率
                pb_ratio = getattr(sec.Fundamentals.ValuationRatios, 'PBRatio', None)
                if pb_ratio and pb_ratio > 0:
                    pb_score = 1.0 / (pb_ratio + 1)  # 越低越好
                    fundamental_score += 0.3 * pb_score
                
                # ROE
                roe = getattr(sec.Fundamentals.OperationRatios.ROE.Value, 'Value', None)
                if roe is not None:
                    roe_score = np.clip(roe / 100, 0, 1)  # 标准化到0-1
                    fundamental_score += 0.3 * roe_score
                    
        except Exception:
            pass
        
        # 3. 综合估值分数
        valuation_score = (
            0.6 * reversal_signal +     # 技术反转权重60%
            0.4 * fundamental_score     # 基本面权重40%
        )
        
        return valuation_score
        
    except Exception:
        return 0

def get_roe_score(symbol, algo):
    # ROE: Return on Equity, 年度财报
    sec = algo.Securities[symbol]
    try:
        if hasattr(sec, 'Fundamentals') and sec.Fundamentals is not None:
            roe = getattr(sec.Fundamentals.OperationRatios.ROE.Value, 'Value', None)
            if roe is None:
                roe = sec.Fundamentals.OperationRatios.ROE.Value if hasattr(sec.Fundamentals.OperationRatios.ROE, 'Value') else None
            if roe is not None:
                return float(roe)
    except Exception:
        pass
    return 0

def get_profit_growth_score(symbol, algo):
    # 净利润增长率，年度财报
    sec = algo.Securities[symbol]
    try:
        if hasattr(sec, 'Fundamentals') and sec.Fundamentals is not None:
            growth = getattr(sec.Fundamentals.EarningReports.BasicEPS.ThreeMonthsGrowth, 'Value', None)
            if growth is None:
                growth = sec.Fundamentals.EarningReports.BasicEPS.ThreeMonthsGrowth if hasattr(sec.Fundamentals.EarningReports.BasicEPS, 'ThreeMonthsGrowth') else None
            if growth is not None:
                return float(growth)
    except Exception:
        pass
    return 0 