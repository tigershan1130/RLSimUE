import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from collections import Counter
import argparse
import os
import glob

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def parse_training_log(file_path):
    """Parse the training log file and extract episode data"""
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Log file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split into episodes (each episode summary starts with "Episode \n")
    # Using lookahead to keep the delimiter pattern
    episodes = re.split(r'Episode \n', content)
    
    # Filter out empty episodes and the initial header
    episode_data = []
    
    for i, episode in enumerate(episodes):
        if not episode.strip() or 'Training Log Started' in episode:
            continue
        
        data = {'episode': len(episode_data) + 1}
        
        # Extract result
        result_match = re.search(r'Result: (\w+)', episode)
        if result_match:
            data['result'] = result_match.group(1).upper()
        else:
            continue  # Skip if no result found (incomplete episode)
        
        # Extract total steps
        steps_match = re.search(r'Total Steps: (\d+)', episode)
        if steps_match:
            data['total_steps'] = int(steps_match.group(1))
        else:
            data['total_steps'] = 0
        
        # Extract episode reward
        reward_match = re.search(r'Episode Reward: ([-\d.]+)', episode)
        if reward_match:
            data['episode_reward'] = float(reward_match.group(1))
        else:
            data['episode_reward'] = 0.0
        
        # Extract duration
        duration_match = re.search(r'Episode Duration: ([-\d.]+)s', episode)
        if duration_match:
            data['duration'] = float(duration_match.group(1))
        else:
            data['duration'] = 0.0
        
        # Extract average step time
        step_time_match = re.search(r'Average Step Time: ([-\d.]+)s', episode)
        if step_time_match:
            data['avg_step_time'] = float(step_time_match.group(1))
        else:
            data['avg_step_time'] = 0.0
        
        # Extract steps per second
        sps_match = re.search(r'Steps per Second: ([-\d.]+)', episode)
        if sps_match:
            data['steps_per_sec'] = float(sps_match.group(1))
        else:
            data['steps_per_sec'] = 0.0
        
        # Extract mean reward (last N) - can be "last 1" or "last 10"
        mean_reward_match = re.search(r'Mean Reward \(last \d+\): ([-\d.]+)', episode)
        if mean_reward_match:
            data['mean_reward_10'] = float(mean_reward_match.group(1))
        else:
            data['mean_reward_10'] = None
        
        # Extract reward breakdown from the episode (search for latest reward breakdown)
        # Look for reward breakdown in the episode text
        boundary_match = re.search(r'boundary: ([-\d.]+)', episode)
        collision_match = re.search(r'collision: ([-\d.]+)', episode)
        target_match = re.search(r'target_reached: ([-\d.]+)', episode)
        time_bonus_match = re.search(r'time_bonus: ([-\d.]+)', episode)
        
        data['boundary_penalty'] = float(boundary_match.group(1)) if boundary_match else 0.0
        data['collision_penalty'] = float(collision_match.group(1)) if collision_match else 0.0
        data['target_reward'] = float(target_match.group(1)) if target_match else 0.0
        data['time_bonus'] = float(time_bonus_match.group(1)) if time_bonus_match else 0.0
        
        episode_data.append(data)
    
    if len(episode_data) == 0:
        raise ValueError(f"No episode data found in log file: {file_path}")
    
    return pd.DataFrame(episode_data)

def generate_summary_analysis(df):
    """Generate comprehensive summary analysis"""
    
    print("=" * 60)
    print("TRAINING LOG ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Basic statistics
    print(f"\n BASIC STATISTICS:")
    print(f"   Total Episodes: {len(df)}")
    print(f"   Total Training Steps: {df['total_steps'].sum():,}")
    print(f"   Total Training Time: {df['duration'].sum()/3600:.2f} hours")
    
    # Result distribution
    result_counts = df['result'].value_counts()
    print(f"\n EPISODE RESULTS:")
    for result, count in result_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {result}: {count} episodes ({percentage:.1f}%)")
    
    # Reward statistics
    print(f"\n REWARD ANALYSIS:")
    print(f"   Average Episode Reward: {df['episode_reward'].mean():.2f}")
    print(f"   Best Episode Reward: {df['episode_reward'].max():.2f}")
    print(f"   Worst Episode Reward: {df['episode_reward'].min():.2f}")
    print(f"   Success Rate: {(result_counts.get('TARGET_REACHED', 0) / len(df)) * 100:.1f}%")
    
    # Performance metrics
    print(f"\n PERFORMANCE METRICS:")
    print(f"   Average Steps per Episode: {df['total_steps'].mean():.1f}")
    print(f"   Average Episode Duration: {df['duration'].mean():.2f}s")
    print(f"   Average Steps per Second: {df['steps_per_sec'].mean():.2f}")
    
    # Penalty analysis
    print(f"\n PENALTY ANALYSIS:")
    total_boundary = df['boundary_penalty'].sum()
    total_collision = df['collision_penalty'].sum()
    print(f"   Total Boundary Penalties: {total_boundary:.0f}")
    print(f"   Total Collision Penalties: {total_collision:.0f}")
    
    # Learning progress
    if len(df) >= 20:
        first_10 = df.head(10)['episode_reward'].mean()
        last_10 = df.tail(10)['episode_reward'].mean()
        improvement = ((last_10 - first_10) / abs(first_10)) * 100 if first_10 != 0 else 0
        print(f"\n LEARNING PROGRESS:")
        print(f"   First 10 episodes avg reward: {first_10:.2f}")
        print(f"   Last 10 episodes avg reward: {last_10:.2f}")
        print(f"   Improvement: {improvement:+.1f}%")

def create_visualizations(df, output_prefix='training'):
    """Create comprehensive visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training Log Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Episode Rewards Over Time
    axes[0, 0].plot(df['episode'], df['episode_reward'], alpha=0.7, linewidth=1)
    axes[0, 0].set_title('Episode Rewards Over Time')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add moving average
    window = min(20, len(df) // 10)
    if window > 0:
        df['reward_ma'] = df['episode_reward'].rolling(window=window).mean()
        axes[0, 0].plot(df['episode'], df['reward_ma'], color='red', linewidth=2, 
                       label=f'{window}-episode MA')
        axes[0, 0].legend()
    
    # 2. Result Distribution
    result_counts = df['result'].value_counts()
    axes[0, 1].pie(result_counts.values, labels=result_counts.index, autopct='%1.1f%%')
    axes[0, 1].set_title('Episode Results Distribution')
    
    # 3. Steps per Episode
    axes[0, 2].scatter(df['episode'], df['total_steps'], alpha=0.6, s=20)
    axes[0, 2].set_title('Steps per Episode')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Steps')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Reward Distribution Histogram
    axes[1, 0].hist(df['episode_reward'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Reward Distribution')
    axes[1, 0].set_xlabel('Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(df['episode_reward'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {df["episode_reward"].mean():.2f}')
    axes[1, 0].legend()
    
    # 5. Performance Metrics Over Time
    axes[1, 1].plot(df['episode'], df['steps_per_sec'], label='Steps/sec', alpha=0.7)
    axes[1, 1].plot(df['episode'], df['avg_step_time'], label='Avg Step Time (s)', alpha=0.7)
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Metric Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Cumulative Rewards
    df['cumulative_reward'] = df['episode_reward'].cumsum()
    axes[1, 2].plot(df['episode'], df['cumulative_reward'])
    axes[1, 2].set_title('Cumulative Reward')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Cumulative Reward')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional detailed plots
    create_detailed_plots(df, output_prefix)

def create_detailed_plots(df, output_prefix='training'):
    """Create additional detailed visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Reward by Result Type
    result_rewards = df.groupby('result')['episode_reward'].mean().sort_values()
    axes[0, 0].bar(result_rewards.index, result_rewards.values)
    axes[0, 0].set_title('Average Reward by Result Type')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Episode Duration Distribution
    axes[0, 1].hist(df['duration'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Episode Duration Distribution')
    axes[0, 1].set_xlabel('Duration (seconds)')
    axes[0, 1].set_ylabel('Frequency')
    
    # 3. Learning Progress (grouped by blocks)
    block_size = max(1, len(df) // 10)
    df['block'] = (df['episode'] - 1) // block_size
    block_rewards = df.groupby('block')['episode_reward'].mean()
    
    axes[1, 0].plot(block_rewards.index * block_size, block_rewards.values, marker='o')
    axes[1, 0].set_title('Learning Progress (Grouped by Blocks)')
    axes[1, 0].set_xlabel('Episode Block')
    axes[1, 0].set_ylabel('Average Reward')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Success Rate Over Time (rolling)
    df['is_success'] = (df['result'] == 'TARGET_REACHED').astype(int)
    window = min(50, len(df) // 5)
    if window > 0:
        df['success_rate'] = df['is_success'].rolling(window=window).mean() * 100
        axes[1, 1].plot(df['episode'], df['success_rate'])
        axes[1, 1].set_title(f'Success Rate ({window}-episode Moving Average)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Success Rate (%)')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def export_analysis_report(df, output_file='training_analysis_report.txt'):
    """Export a comprehensive analysis report"""
    
    with open(output_file, 'w') as f:
        f.write("TRAINING LOG ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Episodes Analyzed: {len(df)}\n")
        f.write(f"Total Training Steps: {df['total_steps'].sum():,}\n")
        f.write(f"Total Training Duration: {df['duration'].sum()/3600:.2f} hours\n\n")
        
        f.write("EPISODE RESULTS SUMMARY:\n")
        f.write("-" * 30 + "\n")
        result_summary = df['result'].value_counts()
        for result, count in result_summary.items():
            percentage = (count / len(df)) * 100
            f.write(f"{result}: {count} ({percentage:.1f}%)\n")
        
        f.write(f"\nSuccess Rate: {(result_summary.get('TARGET_REACHED', 0) / len(df)) * 100:.1f}%\n\n")
        
        f.write("REWARD STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Average Reward: {df['episode_reward'].mean():.2f}\n")
        f.write(f"Median Reward: {df['episode_reward'].median():.2f}\n")
        f.write(f"Standard Deviation: {df['episode_reward'].std():.2f}\n")
        f.write(f"Minimum Reward: {df['episode_reward'].min():.2f}\n")
        f.write(f"Maximum Reward: {df['episode_reward'].max():.2f}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Average Steps per Episode: {df['total_steps'].mean():.1f}\n")
        f.write(f"Average Episode Duration: {df['duration'].mean():.2f}s\n")
        f.write(f"Average Steps per Second: {df['steps_per_sec'].mean():.2f}\n")

def find_available_logs(log_dir='eval_logs'):
    """Find all available training log files"""
    log_dir_path = os.path.join(os.path.dirname(__file__), log_dir)
    if not os.path.exists(log_dir_path):
        return []
    
    log_files = glob.glob(os.path.join(log_dir_path, 'training_log_*.txt'))
    return sorted(log_files, reverse=True)  # Most recent first

def list_available_logs(log_dir='eval_logs'):
    """List all available training log files"""
    log_files = find_available_logs(log_dir)
    
    if not log_files:
        print(f"No training log files found in {log_dir}/")
        return
    
    print(f"\nAvailable training log files in {log_dir}/:\n")
    for i, log_file in enumerate(log_files, 1):
        filename = os.path.basename(log_file)
        # Extract timestamp from filename
        timestamp_match = re.search(r'training_log_(\d{8}_\d{6})\.txt', filename)
        if timestamp_match:
            timestamp_str = timestamp_match.group(1)
            try:
                dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                print(f"  {i}. {filename}")
                print(f"     Created: {formatted_time}")
            except:
                print(f"  {i}. {filename}")
        else:
            print(f"  {i}. {filename}")
        print()
    
    return log_files

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze training log files and generate visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a specific log file
  python analyze_results.py eval_logs/training_log_20251104_130453.txt
  
  # Analyze latest log file (auto-detect)
  python analyze_results.py --latest
  
  # List all available log files
  python analyze_results.py --list
  
  # Analyze with custom output prefix
  python analyze_results.py eval_logs/training_log_20251104_130453.txt --output-prefix my_analysis
        """
    )
    
    parser.add_argument(
        'log_file',
        nargs='?',
        default=None,
        help='Path to training log file to analyze (e.g., "eval_logs/training_log_20251104_130453.txt")'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available training log files and exit'
    )
    
    parser.add_argument(
        '--latest',
        action='store_true',
        help='Analyze the most recent training log file'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots (only print summary and export report)'
    )
    
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='training',
        help='Prefix for output files (default: "training")'
    )
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        list_available_logs()
        exit(0)
    
    # Determine which log file to use
    log_file_path = None
    
    if args.latest:
        log_files = find_available_logs()
        if log_files:
            log_file_path = log_files[0]
            print(f"Using latest log file: {os.path.basename(log_file_path)}\n")
        else:
            print("ERROR: No training log files found.")
            exit(1)
    elif args.log_file:
        log_file_path = args.log_file
        # If relative path, try to find in eval_logs directory
        if not os.path.exists(log_file_path):
            eval_logs_path = os.path.join(os.path.dirname(__file__), 'eval_logs', log_file_path)
            if os.path.exists(eval_logs_path):
                log_file_path = eval_logs_path
            else:
                # Try with just the filename
                eval_logs_path = os.path.join(os.path.dirname(__file__), 'eval_logs', os.path.basename(log_file_path))
                if os.path.exists(eval_logs_path):
                    log_file_path = eval_logs_path
    else:
        # No file specified, try to use latest
        log_files = find_available_logs()
        if log_files:
            log_file_path = log_files[0]
            print(f"No log file specified. Using latest: {os.path.basename(log_file_path)}\n")
        else:
            print("ERROR: No log file specified and no training logs found.")
            print("\nAvailable options:")
            print("  - Specify a log file: python analyze_results.py eval_logs/training_log_XXX.txt")
            print("  - Use latest: python analyze_results.py --latest")
            print("  - List available: python analyze_results.py --list")
            exit(1)
    
    # Validate log file exists
    if not os.path.exists(log_file_path):
        print(f"ERROR: Log file not found: {log_file_path}")
        print("\nAvailable log files:")
        list_available_logs()
        exit(1)
    
    try:
        # Parse the training log
        print(f"Parsing training log: {os.path.basename(log_file_path)}")
        df = parse_training_log(log_file_path)
        print(f"Found {len(df)} episodes\n")
        
        # Generate summary analysis
        print("Generating analysis...")
        generate_summary_analysis(df)
        
        # Create visualizations
        if not args.no_plots:
            print("\nCreating visualizations...")
            create_visualizations(df, output_prefix=args.output_prefix)
            print(f"   - {args.output_prefix}_analysis_dashboard.png")
            print(f"   - {args.output_prefix}_detailed_analysis.png")
        else:
            print("\nSkipping plots (--no-plots specified)")
        
        # Export report
        report_file = f'{args.output_prefix}_analysis_report.txt'
        print(f"\nExporting report...")
        export_analysis_report(df, output_file=report_file)
        print(f"   - {report_file}")
        
        print("\nAnalysis complete!")
        
        # Display final dataframe info
        print(f"\nData Overview:")
        print(f"   Episodes: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Date range: Episodes 1 to {len(df)}")
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        list_available_logs()
        exit(1)
    except ValueError as e:
        print(f"ERROR: {e}")
        exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)