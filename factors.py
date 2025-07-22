import score_engine
import numpy as np
from AlgorithmImports import *
import pandas as pd
from datetime import timedelta
from typing import Dict, List

class MomentumFactor:
    def __init__(self, algo, windows):
        self.algo = algo
        self.windows = windows
    def get_score(self, symbol):
        scores = []
        for win in self.windows:
            try:
                history = self.algo.History([symbol], win+1, Resolution.DAILY)
                if history is None or len(history) == 0 or ('close' not in history):
                    closes = None
                elif hasattr(history, 'loc') and symbol in history.index and 'close' in history.columns:
                    closes = history.loc[symbol]['close']
                else:
                    closes = history['close']
                if closes is None or len(closes) < 2:
                    scores.append(0.0)
                else:
                    scores.append(float(closes[-1]) / float(closes[0]) - 1)
            except Exception:
                scores.append(0.0)
        arr = np.array(scores)
        mean, std = np.mean(arr), np.std(arr)
        arr = np.clip(arr, mean-2*std, mean+2*std)
        return float(np.mean(arr))

class VolatilityFactor:
    def __init__(self, algo, window):
        self.algo = algo
        self.window = window
    def get_score(self, symbol):
        try:
            history = self.algo.History([symbol], self.window+1, Resolution.DAILY)
            if history is None or len(history) == 0 or ('close' not in history):
                closes = None
            elif hasattr(history, 'loc') and symbol in history.index and 'close' in history.columns:
                closes = history.loc[symbol]['close']
            else:
                closes = history['close']
            if closes is None or len(closes) < 2:
                return 0
            returns = np.diff(np.log(closes))
            std = np.std(returns)
            return std * np.sqrt(252)
        except Exception:
            return 0

class SentimentFactor:
    def __init__(self, algo):
        self.algo = algo
    def get_score(self, symbol):
        return score_engine.get_sentiment_score(symbol, self.algo)

class ValuationFactor:
    def __init__(self, algo):
        self.algo = algo
    def get_score(self, symbol):
        return score_engine.get_valuation_score(symbol, self.algo)

class TrendFactor:
    def __init__(self, algo, fast_period=50, slow_period=200):
        self.algo = algo
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.emas = {} # symbol -> (fast_ema, slow_ema)
    def is_up(self, symbol):
        if symbol not in self.emas:
            fast_ema = self.algo.EMA(symbol, self.fast_period, Resolution.DAILY)
            slow_ema = self.algo.EMA(symbol, self.slow_period, Resolution.DAILY)
            self.emas[symbol] = (fast_ema, slow_ema)
        fast_ema, slow_ema = self.emas[symbol]
        
        # 检查均线对象是否创建
        if not (fast_ema and slow_ema):
            self.algo.Debug(f"[TrendFilter] {symbol} 均线对象未创建")
            return False
        
        # 如果均线未ready，允许通过（防止冷启动全部被过滤）
        if not (fast_ema.IsReady and slow_ema.IsReady):
            self.algo.Debug(f"[TrendFilter] {symbol} 均线未ready，允许通过")
            return True
        
        # 真正的趋势过滤：快线在慢线之上
        fast_value = fast_ema.Current.Value
        slow_value = slow_ema.Current.Value
        
        # 放宽趋势条件：快线在慢线的95%以上即可
        trend_threshold = 0.95
        is_uptrend = fast_value >= slow_value * trend_threshold
        
        if not is_uptrend:
            self.algo.Debug(f"[TrendFilter] {symbol} 趋势过滤: 快线={fast_value:.2f}, 慢线={slow_value:.2f}, 比例={fast_value/slow_value:.3f}")
        
        return is_uptrend

# ========== 以下为从main.py迁移的模型和工具 ==========

def quantile_normalize(scores):
    """
    分位数归一化：将每个符号的原始分数 -> [0,1] 区间
    """
    items = sorted(scores.items(), key=lambda kv: kv[1])
    n = len(items)
    if n == 0:
        return {}
    if all(v == 0 for _, v in items):
        return {sym: 0.5 for sym, _ in items}
    return {sym: idx/(n-1) if n>1 else 0.5 for idx,(sym,_) in enumerate(items)}

class CustomAlphaModel(AlphaModel):
    def __init__(self,
                 algo,
                 momentum_model,
                 volatility_model,
                 sentiment_model,
                 valuation_model,
                 trend_model):
        self.algo = algo
        self.mom = momentum_model
        self.vol = volatility_model
        self.sent = sentiment_model
        self.val = valuation_model
        self.trend = trend_model
        self.last_weight_update_month = None
        self.ic_history = {k: [] for k in ['mom','vol','sent','val','rev','vcb']}
        self.factor_weights = {'mom':0.2,'vol':0.2,'sent':0.2,'val':0.2,'rev':0.1,'vcb':0.1}

    def OnSecuritiesChanged(self, algorithm, changes):
        # 初始化securities列表（如果不存在）
        if not hasattr(self, 'securities'):
            self.securities = []
        
        # 添加新的证券
        for security in changes.AddedSecurities:
            if security.Symbol not in self.securities:
                self.securities.append(security.Symbol)
        
        # 移除被删除的证券
        for security in changes.RemovedSecurities:
            if security.Symbol in self.securities:
                self.securities.remove(security.Symbol)
        
        algorithm.Debug(f"[AlphaModel] 证券变更: 添加={len(changes.AddedSecurities)}, 移除={len(changes.RemovedSecurities)}, 当前总数={len(self.securities)}")
    def get_reversal_score(self, symbol):
        try:
            history = self.algo.History([symbol], 15, Resolution.DAILY)
            if history is None or len(history) == 0 or ('close' not in history):
                closes = None
            elif hasattr(history, 'loc') and symbol in history.index and 'close' in history.columns:
                closes = history.loc[symbol]['close']
            else:
                closes = history['close']
            if closes is None or len(closes) < 2:
                return 0
            short_return = (closes[-1] / closes[0]) - 1
            rsi = self.algo.RSI(symbol, 14, MovingAverageType.WILDERS, Resolution.DAILY)
            rsi_val = rsi.Current.Value if rsi.IsReady else 50
            score = 0
            if short_return < 0:
                score += abs(short_return)
            if rsi_val < 30:
                score += (30 - rsi_val) / 30
            return score
        except Exception:
            return 0

    def get_volatility_contraction_breakout_score(self, symbol):
        try:
            history = self.algo.History([symbol], 252, Resolution.DAILY)
            if history is None or len(history) == 0 or ('close' not in history):
                closes = None
            elif hasattr(history, 'loc') and symbol in history.index and 'close' in history.columns:
                closes = history.loc[symbol]['close']
            else:
                closes = history['close']
            if closes is None or len(closes) < 21:
                return 0
            recent = closes[-21:]
            past = closes[:-20]
            recent_vol = np.std(np.diff(np.log(recent)))
            all_vols = [np.std(np.diff(np.log(closes[i-20:i+1]))) for i in range(20, len(closes))]
            if not all_vols:
                return 0
            vol_percentile = sum(v < recent_vol for v in all_vols) / len(all_vols)
            ma20 = np.mean(recent)
            std20 = np.std(recent)
            boll_width = 4 * std20 / ma20 if ma20 > 0 else 0
            all_boll_widths = [4 * np.std(closes[i-20:i+1]) / np.mean(closes[i-20:i+1]) for i in range(20, len(closes)) if np.mean(closes[i-20:i+1]) > 0]
            if not all_boll_widths:
                return 0
            boll_percentile = sum(w < boll_width for w in all_boll_widths) / len(all_boll_widths)
            breakout = 1 if recent[-1] > max(recent[:-1]) else 0
            if vol_percentile < 0.2 and boll_percentile < 0.2 and breakout:
                return 1.0
            elif vol_percentile < 0.2 and breakout:
                return 0.5
            else:
                return 0
        except Exception:
            return 0

    def _update_factor_weights(self, algorithm, active, returns_history):
        import scipy.stats
        if not hasattr(algorithm, 'ic_history'):
            algorithm.ic_history = {k: [] for k in self.ic_history.keys()}
        if not hasattr(algorithm, 'factor_weights'):
            algorithm.factor_weights = {k: 0.2 for k in self.ic_history.keys()}
        ic_dict = {}
        for k, raw in returns_history['factor_scores'].items():
            ics = []
            for t in range(1, len(returns_history['rets'])):
                scores = [raw[s][t-1] for s in active if s in raw and len(raw[s])==len(returns_history['rets'])]
                rets = [
                    returns_history['rets'][s][t]
                    for s in active
                    if s in raw and len(raw[s]) == len(returns_history['rets'][s]) and t < len(returns_history['rets'][s])
                ]
                if len(scores) < 3: continue
                try:
                    ic = scipy.stats.spearmanr(scores, rets)[0]
                except Exception:
                    ic = 0
                if ic is None or np.isnan(ic): ic = 0
                ics.append(ic)
            ic_dict[k] = np.mean(ics) if ics else 0
        ic_arr = np.array([max(0, ic_dict[k]) for k in self.ic_history.keys()])
        w_arr = np.array([0.01 if ic_dict[k] < 0.02 else ic_dict[k] for k in self.ic_history.keys()])
        if w_arr.sum() > 0:
            w_arr = w_arr / w_arr.sum()
        else:
            w_arr = np.ones_like(w_arr) / len(w_arr)
        w_arr = np.clip(w_arr, 0.01, 0.35)
        w_arr = w_arr / w_arr.sum()
        for idx, k in enumerate(self.ic_history.keys()):
            self.factor_weights[k] = float(w_arr[idx])
        algorithm.Debug(f"因子IC动态权重: {self.factor_weights}")

    def Update(self, algorithm, data):
        algorithm.Debug("[AlphaModel] Update called")
        if algorithm.IsWarmingUp:
            return []
        active = self.securities if hasattr(self, 'securities') else []
        if not active:
            algorithm.Debug("[AlphaModel] 没有活跃股票")
            return []
        algorithm.Debug(f"[DEBUG] 初始股票池数量: {len(active)}")
        
        # 放宽数据可用性检查
        data_available = active  # 只要在active列表就认为可用
        algorithm.Debug(f"[DEBUG] 数据可用股票数: {len(data_available)}")
        
        etfs = [s for s in getattr(self.algo, 'etf_symbols', []) if s in active]
        stocks = [s for s in active if s not in etfs]
        algorithm.Debug(f"[DEBUG] ETF数量: {len(etfs)}, 股票数量: {len(stocks)}")
        
        mom_raw = {}
        for s in active:
            if not (data.ContainsKey(s) and data[s] is not None):
                continue
            try:
                history = self.algo.History([s], self.mom.windows[-1]+1, Resolution.DAILY)
                if history is None or len(history) == 0 or ('close' not in history):
                    closes = None
                elif hasattr(history, 'loc') and s in history.index and 'close' in history.columns:
                    closes = history.loc[s]['close']
                else:
                    closes = history['close']
                if closes is None or len(closes) < 2:
                    mom_raw[s] = 0.0
                else:
                    mom_raw[s] = float(float(closes[-1]) / float(closes[0]) - 1)
            except Exception:
                mom_raw[s] = 0.0
        algorithm.Debug(f"[DEBUG] 动量分数非零股票数: {sum(1 for v in mom_raw.values() if v != 0)}")
        
        def safe_dict(d, keys):
            return {k: d.get(k, 0) for k in keys}
        all_syms = set(active)
        # 放宽数据检查，确保所有股票都有分数
        vol_raw = safe_dict({s: -self.vol.get_score(s) for s in active}, all_syms)
        sent_raw = safe_dict({s: self.sent.get_score(s) for s in active}, all_syms)
        val_raw = safe_dict({s: self.val.get_score(s) for s in active}, all_syms)
        rev_raw = safe_dict({s: self.get_reversal_score(s) for s in active}, all_syms)
        vcb_raw = safe_dict({s: self.get_volatility_contraction_breakout_score(s) for s in active}, all_syms)
        
        algorithm.Debug(f"[DEBUG] 各因子非零股票数: 动量={sum(1 for v in mom_raw.values() if v != 0)}, "
                       f"波动率={sum(1 for v in vol_raw.values() if v != 0)}, "
                       f"情感={sum(1 for v in sent_raw.values() if v != 0)}, "
                       f"估值={sum(1 for v in val_raw.values() if v != 0)}")
        
        algorithm.Debug(f"[DEBUG] 波动率分数均值: {np.mean(list(vol_raw.values())) if vol_raw else 0:.4f}")
        
        def smooth_scores(d):
            arr = np.array(list(d.values()))
            mean, std = np.mean(arr), np.std(arr)
            if std == 0:
                arr = arr
            else:
                arr = np.clip(arr, mean-2*std, mean+2*std)
            arr = 0.7*arr + 0.3*mean
            return {k: arr[i] for i, k in enumerate(d.keys())}
        
        qm = smooth_scores(quantile_normalize(mom_raw))
        qv = smooth_scores(quantile_normalize(vol_raw))
        qs = smooth_scores(quantile_normalize(sent_raw))
        ql = smooth_scores(quantile_normalize(val_raw))
        qr = smooth_scores(quantile_normalize(rev_raw))
        qvcb = smooth_scores(quantile_normalize(vcb_raw))
        
        algorithm.Debug(f"[DEBUG] 归一化后动量分数均值: {np.mean(list(qm.values())) if qm else 0:.4f}")
        
        month = self.algo.Time.month if hasattr(self.algo, 'Time') else None
        if self.last_weight_update_month != month:
            returns_history: Dict[str, Dict[str, List[float]]] = {'rets': {}}
            factor_scores: Dict[str, Dict[str, List[float]]] = {k: {} for k in self.ic_history.keys()}
            for s in active:
                try:
                    history = self.algo.History([s], 31, Resolution.DAILY)
                    if history is None or len(history) == 0 or ('close' not in history):
                        closes = None
                    elif hasattr(history, 'loc') and s in history.index and 'close' in history.columns:
                        closes = history.loc[s]['close']
                    else:
                        closes = history['close']
                    if closes is None or len(closes) < 2: continue
                    rets = [float(closes[i+1])/float(closes[i])-1 for i in range(len(closes)-1)]
                    returns_history['rets'][s] = rets
                except Exception:
                    continue
                for k, qdict in zip(['mom','vol','sent','val','rev','vcb'], [qm,qv,qs,ql,qr,qvcb]):
                    if s in qdict:
                        if k not in factor_scores:
                            factor_scores[k] = {}
                        if s not in factor_scores[k]:
                            factor_scores[k][s] = [float(qdict[s])]*30  # type: ignore[assignment,index]
            returns_history['factor_scores'] = factor_scores
            self._update_factor_weights(algorithm, active, returns_history)
            self.last_weight_update_month = month
        
        w_mom = self.factor_weights['mom']
        w_vol = self.factor_weights['vol']
        w_sent = self.factor_weights['sent']
        w_val = self.factor_weights['val']
        w_rev = self.factor_weights['rev']
        w_vcb = self.factor_weights['vcb']
        
        valid_syms_set = set(qm) & set(qv) & set(qs) & set(ql) & set(qr) & set(qvcb)
        algorithm.Debug(f"[DEBUG] 归一化后有效股票数: {len(valid_syms_set)}")
        
        valid_syms = [s for s in valid_syms_set if self.trend.is_up(s)]
        algorithm.Debug(f"[DEBUG] 趋势过滤后股票数: {len(valid_syms)}")
        
        # 新增：如果趋势过滤后股票太少，强制选择前N只
        min_trend_stocks = 15  # 趋势过滤后至少保留15只股票
        if len(valid_syms) < min_trend_stocks:
            algorithm.Debug(f"[WARNING] 趋势过滤后股票数量过少({len(valid_syms)})，强制选择前{min_trend_stocks}只")
            # 重新计算所有股票的分数，不进行趋势过滤
            all_valid_set = set(qm) & set(qv) & set(qs) & set(ql) & set(qr) & set(qvcb)
            all_scores = {s: (
                w_mom * qm[s] +
                w_vol * qv[s] +
                w_sent * qs[s] +
                w_val * ql[s] +
                w_rev * qr[s] +
                w_vcb * qvcb[s]
            ) for s in all_valid_set}
            
            # 按分数排序，选择前N只
            sorted_stocks = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            valid_syms = [s for s, _ in sorted_stocks[:min_trend_stocks]]
            algorithm.Debug(f"[DEBUG] 强制选择后股票数: {len(valid_syms)}")
        
        ban_list = getattr(self.algo, 'ban_list', {})
        today = self.algo.Time.date() if hasattr(self.algo, 'Time') else None
        valid_syms = [s for s in valid_syms if s not in ban_list or (today and today > ban_list[s])]
        algorithm.Debug(f"[DEBUG] ban_list过滤后股票数: {len(valid_syms)}")
        
        scores = {s: (
            w_mom * qm[s] +
            w_vol * qv[s] +
            w_sent * qs[s] +
            w_val * ql[s] +
            w_rev * qr[s] +
            w_vcb * qvcb[s]
        ) for s in valid_syms}
        
        algorithm.Debug(f"[DEBUG] 综合打分后股票数: {len(scores)}，分数均值: {np.mean(list(scores.values())) if scores else 0:.4f}")
        
        # 彻底修复：如果股票数量太少，使用备用策略
        if len(scores) < 5:
            algorithm.Debug(f"[WARNING] 选股数量过少({len(scores)})，使用备用策略")
            
            # 备用策略1：使用所有有数据的股票
            if len(active) > 0:
                algorithm.Debug(f"[DEBUG] 备用策略1：使用所有活跃股票({len(active)})")
                scores = {s: 1.0 for s in active}  # 给所有股票相同权重
            else:
                # 备用策略2：使用默认股票池
                algorithm.Debug(f"[DEBUG] 备用策略2：使用默认股票池")
                default_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
                scores = {s: 1.0 for s in default_stocks if s in active}
            
            algorithm.Debug(f"[DEBUG] 备用策略后股票数: {len(scores)}")
        
        insights = []
        etfs = [s for s in getattr(self.algo, 'etf_symbols', []) if s in scores]
        stocks = [s for s in scores if s not in etfs]
        
        # 修复ETF权重分配
        top_etfs = [s for s in sorted(etfs, key=lambda s: scores.get(s, 0), reverse=True)][:3]  # 增加到3只ETF
        if len(top_etfs) > 0:
            etf_scores = {s: scores[s] for s in top_etfs}
            total_etf_score = sum(abs(v) for v in etf_scores.values())
            
            if total_etf_score > 0:
                # ETF权重标准化到0.15-0.25范围
                min_etf_weight = 0.15
                max_etf_weight = 0.25
                for s in top_etfs:
                    normalized_score = abs(etf_scores[s]) / total_etf_score
                    weight = min_etf_weight + (max_etf_weight - min_etf_weight) * normalized_score
                    insights.append(Insight.Price(s, timedelta(days=30), InsightDirection.UP, weight=weight))  # type: ignore
            else:
                # 如果ETF分数都为0，使用平均权重
                avg_etf_weight = 0.2  # 固定ETF权重
                for s in top_etfs:
                    insights.append(Insight.Price(s, timedelta(days=30), InsightDirection.UP, weight=avg_etf_weight))  # type: ignore
        
        algorithm.Debug(f"[DEBUG] ETF信号数: {len(top_etfs)}")
        if len(top_etfs) > 0:
            algorithm.Debug(f"[DEBUG] ETF选择: {[str(s) for s in top_etfs]}")
        
        industry_map = {}
        market_cap_map = {}
        for s in stocks:
            sec = self.algo.Securities[s]
            industry = 'Unknown'
            if hasattr(sec, 'Fundamentals') and sec.Fundamentals is not None:
                try:
                    industry = getattr(sec.Fundamentals.CompanyReference, 'IndustryTemplateCode', None)  # type: ignore[assignment]
                    if industry is None:
                        industry = 'Unknown'
                    else:
                        industry = str(industry) if industry is not None else 'Unknown'  # 确保是字符串类型
                except Exception:
                    industry = 'Unknown'
            industry_map[s] = industry
            try:
                if hasattr(sec, 'Fundamentals') and sec.Fundamentals is not None:
                    market_cap = getattr(sec.Fundamentals.ValuationRatios, 'MarketCap', None)
                    market_cap_map[s] = market_cap if market_cap and market_cap > 0 else 1e9
                else:
                    market_cap_map[s] = 1e9
            except Exception:
                market_cap_map[s] = 1e9
        
        industry_selected = {}
        industry_stocks = []
        industry_count = 0
        for s in sorted(stocks, key=lambda s: scores.get(s, 0), reverse=True):
            industry = industry_map[s]
            if industry_selected.get(industry, 0) < 2:  # 每行业最多2只
                industry_stocks.append(s)
                industry_selected[industry] = industry_selected.get(industry, 0) + 1
                industry_count += 1
            if len(industry_stocks) >= 20 or industry_count >= 5:
                break
        
        algorithm.Debug(f"[DEBUG] 行业分散后股票数: {len(industry_stocks)}")
        
        if len(industry_stocks) >= 6:
            large_cap = [s for s in industry_stocks if market_cap_map[s] > 1e10]
            mid_cap = [s for s in industry_stocks if 1e9 <= market_cap_map[s] <= 1e10]
            small_cap = [s for s in industry_stocks if market_cap_map[s] < 1e9]
            balanced_stocks = []
            if large_cap:
                balanced_stocks.append(max(large_cap, key=lambda s: scores.get(s, 0)))
            if mid_cap:
                balanced_stocks.append(max(mid_cap, key=lambda s: scores.get(s, 0)))
            if small_cap:
                balanced_stocks.append(max(small_cap, key=lambda s: scores.get(s, 0)))
            remaining = [s for s in industry_stocks if s not in balanced_stocks]
            remaining = sorted(remaining, key=lambda s: scores.get(s, 0), reverse=True)
            balanced_stocks.extend(remaining[:10 - len(balanced_stocks)])
            industry_stocks = balanced_stocks[:10]
        
        algorithm.Debug(f"[DEBUG] 市值平衡后股票数: {len(industry_stocks)}")
        
        correlation_threshold = 0.85
        returns_dict = {}
        for s in industry_stocks:
            try:
                history = self.algo.History([s], 61, Resolution.DAILY)
                if history is None or len(history) == 0 or ('close' not in history):
                    closes = None
                elif hasattr(history, 'loc') and s in history.index and 'close' in history.columns:
                    closes = history.loc[s]['close']
                else:
                    closes = history['close']
                if closes is None or len(closes) < 2: continue
                returns_dict[s] = pd.Series(np.diff(np.log(closes)))
            except Exception:
                continue
        
        final_stocks = []
        if len(returns_dict) >= 2:
            returns_df = pd.DataFrame(returns_dict)
            corr_matrix = returns_df.corr().fillna(0)
            for s in industry_stocks:
                if s not in returns_dict:
                    continue
                if not final_stocks:
                    final_stocks.append(s)
                else:
                    max_corr = 0
                    for fs in final_stocks:
                        if fs in corr_matrix.columns and s in corr_matrix.columns:
                            max_corr = max(max_corr, abs(corr_matrix[s][fs]))
                    if max_corr < correlation_threshold:
                        final_stocks.append(s)
                if len(final_stocks) >= 10:
                    break
            if len(final_stocks) < 10:
                supplement = [s for s in industry_stocks if s not in final_stocks]
                supplement = sorted(supplement, key=lambda s: scores.get(s, 0), reverse=True)
                final_stocks.extend(supplement[:10-len(final_stocks)])
        else:
            final_stocks = industry_stocks[:10]
        
        algorithm.Debug(f"[DEBUG] 相关性过滤后股票数: {len(final_stocks)}")
        
        min_stock_num = 5  # 强制最少5只股票
        available_cash = getattr(self.algo.Portfolio.CashBook, 'TotalValueInAccountCurrency', 10000)
        price_map = {s: self.algo.Securities[s].Price if s in self.algo.Securities else 0 for s in scores}
        if len(final_stocks) < min_stock_num:
            supplement = [s for s in industry_stocks if s not in final_stocks]
            # 优先补价格低于均分资金的股票
            supplement = [s for s in supplement if price_map[s] > 0 and price_map[s] <= available_cash / min_stock_num]
            supplement = sorted(supplement, key=lambda s: scores.get(s, 0), reverse=True)
            final_stocks.extend(supplement[:min_stock_num-len(final_stocks)])
        # 如果还不够，直接从所有分数最高且买得起的股票补齐
        if len(final_stocks) < min_stock_num:
            all_candidates = sorted(scores.keys(), key=lambda s: scores[s], reverse=True)
            for s in all_candidates:
                if s not in final_stocks and price_map[s] > 0 and price_map[s] <= available_cash / min_stock_num:
                    final_stocks.append(s)
                if len(final_stocks) >= min_stock_num:
                    break
        algorithm.Debug(f"[DEBUG] 最终选股数量: {len(final_stocks)}，可用现金: {available_cash:.2f}，选股: {[str(s) for s in final_stocks]}")
        
        # 修复权重分配问题
        if len(final_stocks) > 0:
            # 重新计算权重，确保权重合理
            final_scores = {s: scores[s] for s in final_stocks}
            
            # 标准化权重到合理范围
            total_score = sum(abs(v) for v in final_scores.values())
            if total_score > 0:
                # 将权重标准化到0.1-0.3范围
                min_weight = 0.1
                max_weight = 0.3
                normalized_weights = {}
                
                for s, score in final_scores.items():
                    # 将分数标准化到0-1范围
                    normalized_score = abs(score) / total_score
                    # 映射到目标权重范围
                    weight = min_weight + (max_weight - min_weight) * normalized_score
                    normalized_weights[s] = weight
                
                # 确保权重总和不超过1.0
                total_weight = sum(normalized_weights.values())
                if total_weight > 1.0:
                    scale_factor = 1.0 / total_weight
                    normalized_weights = {s: w * scale_factor for s, w in normalized_weights.items()}
                
                algorithm.Debug(f"[DEBUG] 权重分配: {[(str(s), f'{w:.3f}') for s, w in list(normalized_weights.items())[:5]]}")
                
                for s in final_stocks:
                    insights.append(Insight.Price(s, timedelta(days=30), InsightDirection.UP, weight=normalized_weights[s]))  # type: ignore
            else:
                # 如果所有分数都为0，使用平均权重
                avg_weight = 1.0 / len(final_stocks)
                for s in final_stocks:
                    insights.append(Insight.Price(s, timedelta(days=30), InsightDirection.UP, weight=avg_weight))  # type: ignore
        
        algorithm.Debug(f"[DEBUG] 最终信号数量: {len(insights)}")
        return insights

class SignalWeightedPCM(PortfolioConstructionModel):
    def __init__(self, safe_asset, spy, min_weight_threshold=0.05, max_safe_asset_pct=0.5, confirm_days=15, available_weight=0.98, min_order_value=100):
        super().__init__()
        self.safe_asset = safe_asset
        self.spy = spy
        self.min_weight_threshold = min_weight_threshold
        self.max_safe_asset_pct = max_safe_asset_pct
        self.confirm_days = int(confirm_days)
        self.available_weight = available_weight
        self.min_order_value = min_order_value
    def CreateTargets(self, algorithm, insights):
        algorithm.Debug(f"[PortfolioConstruction] CreateTargets called, insights count: {len(insights)}")
        if algorithm.IsWarmingUp:
            return []
        
        # 保证max_safe_asset_pct与主参数同步
        if hasattr(algorithm, 'params') and 'pcm_max_safe_asset_pct' in algorithm.params:
            self.max_safe_asset_pct = algorithm.params['pcm_max_safe_asset_pct']
        
        etfs = [s for s in getattr(algorithm, 'etf_symbols', [])]
        etf_insights = [i for i in insights if i.Symbol in etfs]
        stock_insights = [i for i in insights if i.Symbol not in etfs]
        
        algorithm.Debug(f"[PortfolioConstruction] ETF信号: {len(etf_insights)}, 股票信号: {len(stock_insights)}")
        
        # 新增：10%资金限制规则
        max_single_asset_pct = 0.10  # 每个投资标的不得超过总资产的10%
        algorithm.Debug(f"[PortfolioConstruction] 单标的最大资金比例: {max_single_asset_pct:.1%}")
        
        available_cash = algorithm.Portfolio.CashBook.TotalValueInAccountCurrency
        total_portfolio_value = algorithm.Portfolio.TotalPortfolioValue
        
        # 彻底修复：保证金和资金计算问题
        # 关键问题：保证金计算错误导致订单金额过大
        
        # 1. 获取真实的可用资金
        available_cash = algorithm.Portfolio.CashBook.TotalValueInAccountCurrency
        total_portfolio_value = algorithm.Portfolio.TotalPortfolioValue
        
        # 2. 修复保证金计算 - 这是关键问题所在
        # 不要使用MarginRemaining或BuyingPower，直接使用可用现金
        free_margin = available_cash
        
        # 3. 彻底修复资金计算逻辑
        # 如果可用现金为负数，尝试恢复资金状态
        if available_cash <= 0:
            algorithm.Debug(f"[PortfolioConstruction] 严重警告：可用现金为负数({available_cash:.2f})，尝试恢复资金状态")
            
            # 尝试使用总资产作为可用资金
            if total_portfolio_value > 0:
                available_cash = total_portfolio_value * 0.1  # 使用总资产的10%作为可用资金
                free_margin = available_cash
                algorithm.Debug(f"[PortfolioConstruction] 使用总资产恢复资金: {available_cash:.2f}")
            else:
                # 如果总资产也为0，使用固定金额
                available_cash = 1000  # 使用固定金额
                free_margin = available_cash
                algorithm.Debug(f"[PortfolioConstruction] 使用固定金额恢复资金: {available_cash:.2f}")
        
        # 4. 使用保守但合理的资金分配
        # 只使用可用现金的50%，确保有足够的安全边际
        safe_available_cash = available_cash * 0.5
        safe_free_margin = free_margin * 0.5
        
        algorithm.Debug(f"[PortfolioConstruction] 原始可用现金: {available_cash:.2f}, 安全可用现金: {safe_available_cash:.2f}")
        algorithm.Debug(f"[PortfolioConstruction] 原始可用保证金: {free_margin:.2f}, 安全可用保证金: {safe_free_margin:.2f}")
        
        algorithm.Debug(f"[PortfolioConstruction] 可用现金: {available_cash:.2f}, 总资产: {total_portfolio_value:.2f}, 可用保证金: {free_margin:.2f}")
        
        # 股票最多投资10只，ETF最多3只
        max_stocks_to_invest = min(10, len(stock_insights))
        max_etfs_to_invest = min(3, len(etf_insights))
        
        # ====== 关键修复：强制均匀分配资金给前N只股票，且不设金额门槛 ======
        sorted_stock_insights = sorted(stock_insights, key=lambda x: x.Weight if hasattr(x, 'Weight') else 0, reverse=True)
        candidates = sorted_stock_insights[:max_stocks_to_invest]
        n = len(candidates)
        targets = []
        used_cash = 0.0
        min_order_value = 1  # 彻底消除金额门槛
        
        # 设定资金利用率
        fund_utilization = 0.8
        available_cash = algorithm.Portfolio.CashBook.TotalValueInAccountCurrency * fund_utilization
        if n > 0:
            alloc_per_stock = available_cash / n
            for i in candidates:
                price = algorithm.Securities[i.Symbol].Price
                if price <= 0:
                    continue
                qty = int(alloc_per_stock / price)
                if qty < 1:
                    continue
                targets.append(PortfolioTarget(i.Symbol, qty))
                available_cash -= qty * price
                if available_cash < min(price, 1):
                    break
        
        # 彻底修复：ETF处理逻辑
        if max_etfs_to_invest > 0 and len(etf_insights) > 0:
            etf_cash = algorithm.Portfolio.CashBook.TotalValueInAccountCurrency * 0.15
            alloc_per_etf = etf_cash / max_etfs_to_invest
            for i in etf_insights[:max_etfs_to_invest]:
                price = algorithm.Securities[i.Symbol].Price
                if price <= 0:
                    continue
                qty = int(alloc_per_etf / price)
                if qty < 1:
                    continue
                targets.append(PortfolioTarget(i.Symbol, qty))
                etf_cash -= qty * price
                if etf_cash < min(price, 1):
                    break
        
        # SHV安全资产
        shv_cash = algorithm.Portfolio.CashBook.TotalValueInAccountCurrency - (algorithm.Portfolio.CashBook.TotalValueInAccountCurrency * fund_utilization) - (algorithm.Portfolio.CashBook.TotalValueInAccountCurrency * 0.15)
        price = algorithm.Securities[self.safe_asset].Price
        if price > 0 and shv_cash > price:
            qty = int(shv_cash / price)
            if qty >= 1:
                targets.append(PortfolioTarget(self.safe_asset, qty))
        
        # 最终安全检查：确保总分配金额不超过安全资金
        total_allocated = used_cash
        if total_allocated > safe_available_cash * 0.95:  # 如果使用了超过95%的安全可用现金
            algorithm.Debug(f"[PortfolioConstruction] 警告：总分配金额({total_allocated:.2f})接近安全可用现金({safe_available_cash:.2f})")
        
        algorithm.Debug(f"[PortfolioConstruction] 最终目标数量: {len(targets)}, 总使用资金: {used_cash:.2f}, 资金利用率: {used_cash/total_portfolio_value:.1%}")
        # 彻底修复：添加投资目标数量检查和强制投资
        algorithm.Debug(f"[PortfolioConstruction] 最终目标数量: {len(targets)}, 总使用资金: {used_cash:.2f}, 资金利用率: {used_cash/total_portfolio_value*100:.1f}%")
        
        # 如果投资目标太少，强制添加更多目标
        if len(targets) < 3:
            algorithm.Debug(f"[WARNING] 投资目标数量过少({len(targets)})，强制添加更多目标")
            
            # 强制添加SHV作为安全资产
            if self.safe_asset not in [t.symbol for t in targets]:  # 修复：Symbol -> symbol
                shv_qty = int(min_order_value / algorithm.Securities[self.safe_asset].Price) if algorithm.Securities[self.safe_asset].Price > 0 else 1
                if shv_qty > 0:
                    targets.append(PortfolioTarget(self.safe_asset, shv_qty))
                    algorithm.Debug(f"[PortfolioConstruction] 强制添加SHV目标: 股数={shv_qty}")
            
            # 强制添加SPY作为指数基金
            spy_symbol = getattr(self, 'spy', None)
            if spy_symbol is None:
                spy_symbol = algorithm.AddEquity("SPY", Resolution.DAILY).Symbol
            if spy_symbol not in [t.symbol for t in targets]:
                spy_price = algorithm.Securities[spy_symbol].Price if spy_symbol in algorithm.Securities else 0
                spy_qty = int(min_order_value / spy_price) if spy_price > 0 else 1
                if spy_qty > 0:
                    targets.append(PortfolioTarget(spy_symbol, spy_qty))
                    algorithm.Debug(f"[PortfolioConstruction] 强制添加SPY目标: 股数={spy_qty}")
        
        algorithm.Debug(f"[PortfolioConstruction] 最终目标数量: {len(targets)}")
        algorithm.Debug(f"[PortfolioConstruction] 剩余原始现金: {available_cash:.2f}, 剩余原始保证金: {free_margin:.2f}")
        algorithm.Debug(f"[PortfolioConstruction] 剩余安全现金: {safe_available_cash:.2f}, 剩余安全保证金: {safe_free_margin:.2f}")
        
        return targets

class ATRStopLossRiskModel(RiskManagementModel):
    def __init__(self, algorithm,
                 atr_period=14,
                 multiplier=2.0,  # 降低默认倍数
                 max_drawdown=0.05,  # 降低默认回撤阈值
                 trailing_stop_pct=0.015):  # 降低默认止盈阈值
        self.algorithm = algorithm
        self.atr_period = int(atr_period)
        self.multiplier = float(multiplier)
        self.max_drawdown = float(max_drawdown)
        self.trailing_stop_pct = float(trailing_stop_pct)
        self.entry_prices = {}
        self.peaks = {}
        self.atrs = {}  # 保存每个symbol的ATR指标
        
        # 新增：组合层面风控
        self.portfolio_peak = None
        # 彻底修复：更严格的止损参数
        self.portfolio_stop_loss = 0.05  # 组合层面5%止损（从8%降低）
        self.single_stock_max_loss = 0.02  # 单只股票最大2%损失（从3%降低）
        self.emergency_stop_loss = 0.10  # 紧急止损10%
        self.max_historical_drawdown = 0.15  # 历史最大回撤15%
        
    def ManageRisk(self, algorithm, targets):
        # 更新组合峰值
        current_portfolio_value = algorithm.Portfolio.TotalPortfolioValue
        if self.portfolio_peak is None or current_portfolio_value > self.portfolio_peak:
            self.portfolio_peak = current_portfolio_value
        
        # 彻底修复：组合层面止损检查
        if self.portfolio_peak is not None:
            portfolio_drawdown = (self.portfolio_peak - current_portfolio_value) / self.portfolio_peak
            
            # 紧急止损检查（10%）
            if portfolio_drawdown > self.emergency_stop_loss:
                algorithm.Debug(f"[RiskManagement] 紧急止损触发！组合回撤({portfolio_drawdown:.2%})超过{self.emergency_stop_loss:.1%}，立即清空所有持仓")
                # 立即清空所有持仓
                risk_targets = []
                for sym in algorithm.Portfolio.Keys:
                    if algorithm.Portfolio[sym].Quantity != 0:
                        risk_targets.append(PortfolioTarget(sym, 0))
                        algorithm.Debug(f"[RiskManagement] 紧急止损：清空{sym}")
                return risk_targets
            
            # 常规止损检查（5%）
            elif portfolio_drawdown > self.portfolio_stop_loss:
                algorithm.Debug(f"[RiskManagement] 组合止损触发！回撤({portfolio_drawdown:.2%})超过{self.portfolio_stop_loss:.1%}，清空所有持仓")
                # 清空所有持仓
                risk_targets = []
                for sym in algorithm.Portfolio.Keys:
                    if algorithm.Portfolio[sym].Quantity != 0:
                        risk_targets.append(PortfolioTarget(sym, 0))
                        algorithm.Debug(f"[RiskManagement] 组合止损：清空{sym}")
                return risk_targets
            
            # 历史最大回撤检查（15%）
            elif portfolio_drawdown > self.max_historical_drawdown:
                algorithm.Debug(f"[RiskManagement] 历史最大回撤触发！回撤({portfolio_drawdown:.2%})超过{self.max_historical_drawdown:.1%}，清空所有持仓")
                # 清空所有持仓
                risk_targets = []
                for sym in algorithm.Portfolio.Keys:
                    if algorithm.Portfolio[sym].Quantity != 0:
                        risk_targets.append(PortfolioTarget(sym, 0))
                        algorithm.Debug(f"[RiskManagement] 历史回撤止损：清空{sym}")
                return risk_targets
        
        # ATR止损+浮盈回撤止盈+最大回撤风控
        risk_targets = []
        for tgt in targets:
            sym = tgt.symbol  # 修复：Symbol -> symbol
            sec = algorithm.Securities[sym]
            if not sec.Invested or not sec.HasData or sec.Price <= 0:
                continue
                
            # 1. ATR止损
            if sym not in self.atrs:
                self.atrs[sym] = algorithm.ATR(sym, self.atr_period, MovingAverageType.WILDERS, Resolution.DAILY)
            atr = self.atrs[sym]
            if not atr.IsReady:
                continue
                
            if sym not in self.entry_prices:
                avg_price = sec.Holdings.AveragePrice if hasattr(sec, 'Holdings') and hasattr(sec.Holdings, 'AveragePrice') else sec.Price
                self.entry_prices[sym] = avg_price
                self.peaks[sym] = avg_price
                
            price = sec.Price
            entry = self.entry_prices[sym]
            peak_candidates = [x for x in [self.peaks.get(sym, entry), price] if x is not None]
            peak = max(peak_candidates) if peak_candidates else price
            self.peaks[sym] = peak
            
            # 计算各种止损价格
            stop_loss = entry - self.multiplier * atr.Current.Value
            trailing_stop = peak * (1 - self.trailing_stop_pct)
            max_drawdown_price = entry * (1 - self.max_drawdown)
            
            # 彻底修复：单只股票风险控制
            current_loss_pct = (entry - price) / entry if entry > 0 else 0
            
            # 单只股票紧急止损（2%）
            if current_loss_pct > self.single_stock_max_loss:
                algorithm.Debug(f"[RiskManagement] {sym} 单只股票紧急止损触发！损失({current_loss_pct:.2%})超过{self.single_stock_max_loss:.1%}")
                risk_targets.append(PortfolioTarget(sym, 0))
                self.entry_prices.pop(sym, None)
                self.peaks.pop(sym, None)
                self.atrs.pop(sym, None)
                continue
            
            # 单只股票快速止损（1%）
            elif current_loss_pct > 0.01:  # 1%快速止损
                algorithm.Debug(f"[RiskManagement] {sym} 单只股票快速止损触发！损失({current_loss_pct:.2%})超过1%")
                risk_targets.append(PortfolioTarget(sym, 0))
                self.entry_prices.pop(sym, None)
                self.peaks.pop(sym, None)
                self.atrs.pop(sym, None)
                continue
            
            # 彻底修复：ATR止损检查
            if price < stop_loss:
                algorithm.Debug(f"[RiskManagement] {sym} ATR止损触发！价格={price:.2f}, 止损价={stop_loss:.2f}, ATR={atr.Current.Value:.4f}")
                risk_targets.append(PortfolioTarget(sym, 0))
                self.entry_prices.pop(sym, None)
                self.peaks.pop(sym, None)
                self.atrs.pop(sym, None)
                continue
                
            # 新增：浮盈回撤止盈
            if price < trailing_stop and price > entry:  # 只有在有盈利的情况下才检查止盈
                algorithm.Debug(f"[RiskManagement] {sym} 浮盈回撤止盈触发: 价格={price:.2f}, 止盈价={trailing_stop:.2f}")
                risk_targets.append(PortfolioTarget(sym, 0))
                self.entry_prices.pop(sym, None)
                self.peaks.pop(sym, None)
                self.atrs.pop(sym, None)
                continue
                
            # 新增：最大回撤止损
            if price < max_drawdown_price:
                algorithm.Debug(f"[RiskManagement] {sym} 最大回撤止损触发: 价格={price:.2f}, 回撤价={max_drawdown_price:.2f}")
                risk_targets.append(PortfolioTarget(sym, 0))
                self.entry_prices.pop(sym, None)
                self.peaks.pop(sym, None)
                self.atrs.pop(sym, None)
                continue
                
        return risk_targets

class SlippageControlledExecution(ExecutionModel):
    def __init__(self, slippage_rate=0.001):
        super().__init__()
        self.slippage_rate = float(slippage_rate)
    def Execute(self, algorithm, targets):
        for tgt in targets:
            sym = tgt.symbol  # 修复：Symbol -> symbol
            w = tgt.Quantity
            if w == 0: continue
            sec = algorithm.Securities[sym]
            if not sec.HasData or sec.Price <= 0: continue
            market_cap = 1e9
            if hasattr(sec, 'Fundamentals') and sec.Fundamentals is not None:
                try:
                    market_cap = getattr(sec.Fundamentals.ValuationRatios, 'MarketCap', 1e9)
                except Exception:
                    market_cap = 1e9
            if market_cap > 1e10:
                slip = 0.0005
            elif market_cap > 2e9:
                slip = 0.001
            else:
                slip = 0.002
            qty = algorithm.CalculateOrderQuantity(sym, w)
            if abs(qty) < 1: continue
            direction = 1 if qty > 0 else -1
            limit_price = sec.Price * (1 + direction * slip)
            algorithm.LimitOrder(sym, qty, limit_price, tag="slippage_limit") 