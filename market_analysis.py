import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MarketEnvironmentAnalyzer:
    """市场环境分析器"""
    
    def __init__(self):
        # 市场环境定义
        self.market_periods = {
            'bull_market': [
                ('2020-04-01', '2021-12-31'),  # 疫情后复苏
                ('2016-02-01', '2018-09-30'),  # 特朗普牛市
            ],
            'bear_market': [
                ('2020-02-01', '2020-03-31'),  # 疫情冲击
                ('2018-10-01', '2018-12-31'),  # 年末调整
                ('2022-01-01', '2022-10-31'),  # 通胀加息
            ],
            'sideways_market': [
                ('2019-01-01', '2019-12-31'),  # 震荡整理
                ('2021-01-01', '2021-03-31'),  # 年初调整
            ]
        }
        
        # 回测参数
        self.base_params = {
            'start_date': '2020-01-01',
            'end_date': '2025-12-01',
            'initial_cash': 100000,
            'final_universe_size': 30,
            'max_drawdown_threshold': 0.08,
            'pcm_min_weight_threshold': 0.05,
            'atr_multiplier': 2.5,
            'trend_fast_period': 50,
            'trend_slow_period': 200
        }
    
    def run_market_environment_tests(self):
        """运行不同市场环境下的回测"""
        results = {}
        
        for env_name, periods in self.market_periods.items():
            print(f"\n=== 测试 {env_name} 市场环境 ===")
            env_results = []
            
            for start_date, end_date in periods:
                print(f"测试期间: {start_date} ~ {end_date}")
                
                # 这里应该调用实际的回测引擎
                # 暂时用模拟数据
                result = self.simulate_backtest(start_date, end_date, env_name)
                env_results.append(result)
            
            results[env_name] = env_results
        
        return results
    
    def simulate_backtest(self, start_date, end_date, env_name):
        """模拟回测结果（实际使用时替换为真实回测调用）"""
        # 根据市场环境调整预期收益
        env_multipliers = {
            'bull_market': 1.2,
            'bear_market': 0.8,
            'sideways_market': 1.0
        }
        
        # 模拟回测结果
        days = (datetime.datetime.strptime(end_date, '%Y-%m-%d') - 
                datetime.datetime.strptime(start_date, '%Y-%m-%d')).days
        
        base_return = 0.15  # 年化15%
        env_return = base_return * env_multipliers[env_name] * (days / 365)
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'total_return': env_return,
            'sharpe_ratio': 1.2 if env_name == 'bull_market' else 0.8,
            'max_drawdown': 0.05 if env_name == 'bull_market' else 0.12,
            'volatility': 0.12 if env_name == 'bull_market' else 0.18,
            'win_rate': 0.65 if env_name == 'bull_market' else 0.45
        }
    
    def analyze_results(self, results):
        """分析不同市场环境下的表现"""
        analysis = {}
        
        for env_name, env_results in results.items():
            if not env_results:
                continue
                
            # 计算平均表现
            avg_return = np.mean([r['total_return'] for r in env_results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in env_results])
            avg_drawdown = np.mean([r['max_drawdown'] for r in env_results])
            avg_vol = np.mean([r['volatility'] for r in env_results])
            avg_win_rate = np.mean([r['win_rate'] for r in env_results])
            
            analysis[env_name] = {
                'avg_return': avg_return,
                'avg_sharpe': avg_sharpe,
                'avg_drawdown': avg_drawdown,
                'avg_volatility': avg_vol,
                'avg_win_rate': avg_win_rate,
                'consistency': np.std([r['total_return'] for r in env_results])  # 一致性
            }
        
        return analysis
    
    def generate_report(self, analysis):
        """生成市场环境分析报告"""
        print("\n" + "="*60)
        print("市场环境表现分析报告")
        print("="*60)
        
        # 创建对比表格
        df = pd.DataFrame(analysis).T
        df = df.round(4)
        
        print("\n各市场环境平均表现:")
        print(df)
        
        # 鲁棒性评分
        robustness_score = self.calculate_robustness_score(analysis)
        print(f"\n策略鲁棒性评分: {robustness_score:.2f}/10")
        
        # 建议
        self.generate_recommendations(analysis)
        
        return df, robustness_score
    
    def calculate_robustness_score(self, analysis):
        """计算策略鲁棒性评分"""
        if not analysis:
            return 0
        
        scores = []
        
        for env_name, metrics in analysis.items():
            env_score = 0
            
            # 收益率评分 (0-3分)
            if metrics['avg_return'] > 0.1:
                env_score += 3
            elif metrics['avg_return'] > 0.05:
                env_score += 2
            elif metrics['avg_return'] > 0:
                env_score += 1
            
            # 夏普比率评分 (0-3分)
            if metrics['avg_sharpe'] > 1.0:
                env_score += 3
            elif metrics['avg_sharpe'] > 0.5:
                env_score += 2
            elif metrics['avg_sharpe'] > 0:
                env_score += 1
            
            # 回撤控制评分 (0-2分)
            if metrics['avg_drawdown'] < 0.05:
                env_score += 2
            elif metrics['avg_drawdown'] < 0.10:
                env_score += 1
            
            # 一致性评分 (0-2分)
            if metrics['consistency'] < 0.05:
                env_score += 2
            elif metrics['consistency'] < 0.10:
                env_score += 1
            
            scores.append(env_score)
        
        # 平均分并归一化到10分制
        avg_score = np.mean(scores) if scores else 0
        return min(10, avg_score)
    
    def generate_recommendations(self, analysis):
        """生成优化建议"""
        print("\n优化建议:")
        
        if 'bear_market' in analysis:
            bear_metrics = analysis['bear_market']
            if bear_metrics['avg_return'] < 0:
                print("- 熊市表现不佳，建议增强防御性资产配置")
            if bear_metrics['avg_drawdown'] > 0.10:
                print("- 熊市回撤过大，建议收紧止损参数")
        
        if 'sideways_market' in analysis:
            side_metrics = analysis['sideways_market']
            if side_metrics['avg_return'] < 0.05:
                print("- 震荡市表现一般，建议优化选股因子")
        
        # 一致性建议
        consistency_issues = []
        for env_name, metrics in analysis.items():
            if metrics['consistency'] > 0.10:
                consistency_issues.append(env_name)
        
        if consistency_issues:
            print(f"- {', '.join(consistency_issues)}市场表现不稳定，建议参数优化")

def main():
    """主函数"""
    analyzer = MarketEnvironmentAnalyzer()
    
    # 运行市场环境测试
    results = analyzer.run_market_environment_tests()
    
    # 分析结果
    analysis = analyzer.analyze_results(results)
    
    # 生成报告
    df, robustness_score = analyzer.generate_report(analysis)
    
    print(f"\n分析完成！策略鲁棒性评分: {robustness_score:.2f}/10")

if __name__ == "__main__":
    main() 