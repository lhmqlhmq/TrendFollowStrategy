import os
import datetime
import subprocess
import random
import copy

# 回测窗口参数
START_DATE = datetime.date(2020, 1, 1)
END_DATE = datetime.date(2025, 12, 1)
WINDOW_DAYS = 365 * 2   # 每个窗口2年
STEP_DAYS = 180         # 每次前移6个月

# 交叉验证/蒙特卡洛参数
N_MONTE_CARLO = 5  # 每个窗口扰动5组参数

# 参数扰动范围（可根据实际情况调整）
PARAM_RANGES = {
    'final_universe_size': [20, 30, 40],
    'max_drawdown_threshold': [0.06, 0.08, 0.10],
    'atr_multiplier': [2.0, 2.5, 3.0],
    'trend_fast_period': [30, 50, 70],
    'trend_slow_period': [150, 200, 250],
    'rebalance_interval_days': [30],  # 可选扰动
    'pcm_max_safe_asset_pct': [0.3, 0.5, 0.7],  # SHV最大投资比例
    'spy_weight': [0.3, 0.5, 0.7]               # 股票指数最大投资比例
}

def run_batch_backtests():
    """运行批量回测"""
    # 生成所有窗口
    windows = []
    cur_start = START_DATE
    while True:
        cur_end = cur_start + datetime.timedelta(days=WINDOW_DAYS)
        if cur_end > END_DATE:
            break
        windows.append((cur_start, cur_end))
        cur_start = cur_start + datetime.timedelta(days=STEP_DAYS)

    # 运行每个窗口的多组参数回测
    total_tests = len(windows) * N_MONTE_CARLO
    completed_tests = 0
    
    for idx, (s, e) in enumerate(windows):
        for mc in range(N_MONTE_CARLO):
            # 随机采样参数
            params = {k: random.choice(v) for k, v in PARAM_RANGES.items()}
            completed_tests += 1
            
            print(f"\n=== 进度: {completed_tests}/{total_tests} | 窗口 {idx+1} 组 {mc+1}: {s} ~ {e} | params: {params} ===")
            cmd = [
                "python", "main.py",
                f"--start_date={s}",
                f"--end_date={e}",
                f"--initial_cash=100000",
                f"--final_universe_size={params['final_universe_size']}",
                f"--max_drawdown_threshold={params['max_drawdown_threshold']}",
                f"--atr_multiplier={params['atr_multiplier']}",
                f"--trend_fast_period={params['trend_fast_period']}",
                f"--trend_slow_period={params['trend_slow_period']}",
                f"--rebalance_interval_days={params['rebalance_interval_days']}"
            ]
            try:
                subprocess.run(cmd, check=True)
                print(f"Ok 窗口 {s}~{e} 组 {mc+1} 回测完成")
            except Exception as ex:
                print(f"X 窗口 {s}~{e} 组 {mc+1} 回测失败: {ex}")

def analyze_results():
    """分析回测结果"""
    print("\n" + "="*60)
    print("开始分析回测结果...")
    print("="*60)
    
    try:
        from backtest_analyzer import BacktestAnalyzer
        
        # 创建分析器
        analyzer = BacktestAnalyzer()
        
        # 加载所有结果
        analyzer.load_all_results()
        
        # 生成分析报告
        analyzer.generate_performance_report()
        
        # 找出最佳表现
        analyzer.find_best_performing()
        
        # 导出到Excel
        analyzer.export_to_excel()
        
        print("\n 回测结果分析完成！")
        
    except Exception as e:
        print(f"X 分析结果失败: {e}")

def main():
    """主函数"""
    print(" 开始批量回测...")

    window_list = []
    for i in range((END_DATE - START_DATE).days // STEP_DAYS + 1):
        s = START_DATE + datetime.timedelta(days=i*STEP_DAYS)
        e = START_DATE + datetime.timedelta(days=i*STEP_DAYS + WINDOW_DAYS)
        if e <= END_DATE:
            window_list.append((s, e))
    print(f"总窗口数: {len(window_list)}")
    print(f"每窗口测试组数: {N_MONTE_CARLO}")
    
    # 运行批量回测
    run_batch_backtests()
    
    # 分析结果
    analyze_results()

if __name__ == "__main__":
    main() 