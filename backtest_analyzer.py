import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import glob

class BacktestAnalyzer:
    """回测结果分析器"""
    
    def __init__(self, results_dir="backtest_results"):
        self.results_dir = results_dir
        self.results = {}
        self.comparison_data = []
        
        # 确保结果目录存在
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
    
    def save_backtest_result(self, result_id, backtest_data):
        """保存单次回测结果"""
        result_file = os.path.join(self.results_dir, f"{result_id}.json")
        
        # 提取关键指标
        result_summary = {
            'result_id': result_id,
            'timestamp': datetime.now().isoformat(),
            'total_return': backtest_data.get('Total Return', 0),
            'sharpe_ratio': backtest_data.get('Sharpe Ratio', 0),
            'max_drawdown': backtest_data.get('Maximum Drawdown', 0),
            'volatility': backtest_data.get('Volatility', 0),
            'win_rate': backtest_data.get('Win Rate', 0),
            'profit_factor': backtest_data.get('Profit Factor', 0),
            'calmar_ratio': backtest_data.get('Calmar Ratio', 0),
            'start_date': backtest_data.get('Start Date', ''),
            'end_date': backtest_data.get('End Date', ''),
            'parameters': backtest_data.get('Parameters', {}),
            'equity_curve': backtest_data.get('Equity Curve', []),
            'trades': backtest_data.get('Trades', [])
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_summary, f, indent=2, ensure_ascii=False)
        
        print(f"回测结果已保存: {result_file}")
        return result_summary
    
    def load_all_results(self):
        """加载所有回测结果"""
        result_files = glob.glob(os.path.join(self.results_dir, "*.json"))
        
        for file_path in result_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    self.results[result['result_id']] = result
                    self.comparison_data.append(result)
            except Exception as e:
                print(f"加载文件 {file_path} 失败: {e}")
        
        print(f"已加载 {len(self.results)} 个回测结果")
        return self.results
    
    def create_comparison_table(self):
        """创建对比表格"""
        if not self.comparison_data:
            print("没有可对比的数据")
            return None
        
        # 提取关键指标
        df_data = []
        for result in self.comparison_data:
            row = {
                '回测ID': result['result_id'],
                '开始日期': result['start_date'],
                '结束日期': result['end_date'],
                '总收益率': f"{result['total_return']:.2%}",
                '夏普比率': f"{result['sharpe_ratio']:.2f}",
                '最大回撤': f"{result['max_drawdown']:.2%}",
                '波动率': f"{result['volatility']:.2%}",
                '胜率': f"{result['win_rate']:.2%}",
                '盈亏比': f"{result['profit_factor']:.2f}",
                '卡玛比率': f"{result['calmar_ratio']:.2f}"
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        return df
    
    def generate_performance_report(self):
        """生成性能报告"""
        if not self.comparison_data:
            print("没有数据可分析")
            return
        
        print("\n" + "="*80)
        print("回测结果对比分析报告")
        print("="*80)
        
        # 创建对比表格
        df = self.create_comparison_table()
        if df is not None:
            print("\n回测结果对比表:")
            print(df.to_string(index=False))
        
        # 统计分析
        self.analyze_statistics()
        
        # 生成可视化
        self.create_visualizations()
        
        return df
    
    def analyze_statistics(self):
        """统计分析"""
        if not self.comparison_data:
            return
        
        # 提取数值指标
        returns = [r['total_return'] for r in self.comparison_data]
        sharpes = [r['sharpe_ratio'] for r in self.comparison_data]
        drawdowns = [r['max_drawdown'] for r in self.comparison_data]
        
        print(f"\n统计分析:")
        print(f"平均收益率: {np.mean(returns):.2%} ± {np.std(returns):.2%}")
        print(f"平均夏普比率: {np.mean(sharpes):.2f} ± {np.std(sharpes):.2f}")
        print(f"平均最大回撤: {np.mean(drawdowns):.2%} ± {np.std(drawdowns):.2%}")
        print(f"最佳收益率: {max(returns):.2%}")
        print(f"最差收益率: {min(returns):.2%}")
        print(f"收益率标准差: {np.std(returns):.2%}")
        
        # 稳定性分析
        positive_returns = sum(1 for r in returns if r > 0)
        print(f"正收益次数: {positive_returns}/{len(returns)} ({positive_returns/len(returns):.1%})")
    
    def create_visualizations(self):
        """创建可视化图表"""
        if not self.comparison_data:
            return
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('回测结果分析', fontsize=16)
        
        # 1. 收益率分布
        returns = [r['total_return'] for r in self.comparison_data]
        axes[0, 0].hist(returns, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(np.mean(returns), color='red', linestyle='--', label=f'平均值: {np.mean(returns):.2%}')
        axes[0, 0].set_title('收益率分布')
        axes[0, 0].set_xlabel('收益率')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].legend()
        
        # 2. 夏普比率 vs 最大回撤
        sharpes = [r['sharpe_ratio'] for r in self.comparison_data]
        drawdowns = [r['max_drawdown'] for r in self.comparison_data]
        axes[0, 1].scatter(drawdowns, sharpes, alpha=0.6, s=50)
        axes[0, 1].set_xlabel('最大回撤')
        axes[0, 1].set_ylabel('夏普比率')
        axes[0, 1].set_title('风险收益散点图')
        
        # 3. 收益率时间序列
        dates = [r['start_date'] for r in self.comparison_data]
        axes[1, 0].plot(range(len(returns)), returns, marker='o', linewidth=2, markersize=6)
        axes[1, 0].set_title('收益率时间序列')
        axes[1, 0].set_xlabel('回测序号')
        axes[1, 0].set_ylabel('收益率')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 指标对比箱线图
        data_for_box = [returns, sharpes, drawdowns]
        labels = ['收益率', '夏普比率', '最大回撤']
        axes[1, 1].boxplot(data_for_box, labels=labels)
        axes[1, 1].set_title('关键指标分布')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'backtest_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n可视化图表已保存: {os.path.join(self.results_dir, 'backtest_analysis.png')}")
    
    def find_best_performing(self, metric='total_return'):
        """找出表现最好的回测"""
        if not self.comparison_data:
            return None
        
        best_result = max(self.comparison_data, key=lambda x: x[metric])
        print(f"\n最佳表现回测 (基于{metric}):")
        print(f"回测ID: {best_result['result_id']}")
        print(f"总收益率: {best_result['total_return']:.2%}")
        print(f"夏普比率: {best_result['sharpe_ratio']:.2f}")
        print(f"最大回撤: {best_result['max_drawdown']:.2%}")
        print(f"参数设置: {best_result['parameters']}")
        
        return best_result
    
    def export_to_excel(self, filename="backtest_comparison.xlsx"):
        """导出结果到Excel"""
        if not self.comparison_data:
            print("没有数据可导出")
            return
        
        # 创建详细数据表
        detailed_data = []
        for result in self.comparison_data:
            row = {
                '回测ID': result['result_id'],
                '开始日期': result['start_date'],
                '结束日期': result['end_date'],
                '总收益率': result['total_return'],
                '夏普比率': result['sharpe_ratio'],
                '最大回撤': result['max_drawdown'],
                '波动率': result['volatility'],
                '胜率': result['win_rate'],
                '盈亏比': result['profit_factor'],
                '卡玛比率': result['calmar_ratio'],
                '时间戳': result['timestamp']
            }
            # 添加参数
            for key, value in result['parameters'].items():
                row[f'参数_{key}'] = value
            
            detailed_data.append(row)
        
        df = pd.DataFrame(detailed_data)
        
        # 保存到Excel
        excel_path = os.path.join(self.results_dir, filename)
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='详细对比', index=False)
            
            # 创建汇总表
            summary_data = {
                '指标': ['平均收益率', '平均夏普比率', '平均最大回撤', '平均波动率', '平均胜率'],
                '数值': [
                    f"{np.mean([r['total_return'] for r in self.comparison_data]):.2%}",
                    f"{np.mean([r['sharpe_ratio'] for r in self.comparison_data]):.2f}",
                    f"{np.mean([r['max_drawdown'] for r in self.comparison_data]):.2%}",
                    f"{np.mean([r['volatility'] for r in self.comparison_data]):.2%}",
                    f"{np.mean([r['win_rate'] for r in self.comparison_data]):.2%}"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='汇总统计', index=False)
        
        print(f"结果已导出到: {excel_path}")
        return excel_path

def main():
    """主函数 - 演示使用方法"""
    analyzer = BacktestAnalyzer()
    
    # 加载现有结果
    analyzer.load_all_results()
    
    # 生成分析报告
    analyzer.generate_performance_report()
    
    # 找出最佳表现
    analyzer.find_best_performing()
    
    # 导出到Excel
    analyzer.export_to_excel()

if __name__ == "__main__":
    main() 