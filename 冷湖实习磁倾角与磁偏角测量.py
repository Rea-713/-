import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
import sys

def read_magnetic_data(file_path):
    try:
        # 检查文件扩展名以确定格式
        if file_path.endswith('.xls') or file_path.endswith('.xlsx'):
            # 读取Excel文件，直接获取第一个工作表
            df = pd.read_excel(file_path)
            # 重命名列使其一致
            df.columns = ['Time_s', 'Bx_uT', 'By_uT', 'Bz_uT', 'Absolute_field_uT']
        else:
            # 尝试读取文本文件
            column_names = ["Time_s", "Bx_uT", "By_uT", "Bz_uT", "Absolute_field_uT"]
            # 跳过前两行（文件名行和<Raw Data>行）
            df = pd.read_csv(file_path, skiprows=2, header=None, sep=r'\s+', 
                             names=column_names, engine='python')
        return df
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None

def calculate_magnetic_parameters(df):
    
    # 计算磁倾角（Dip Angle）和磁偏角（Declination）
    # Bx = north component (手机指向真北)
    # By = east component
    # Bz = vertical component (向下为正)
    # 确保数值列是浮点型
    for col in ['Time_s', 'Bx_uT', 'By_uT', 'Bz_uT', 'Absolute_field_uT']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 计算水平磁场强度
    df['H_uT'] = np.sqrt(df['Bx_uT']**2 + df['By_uT']**2)
    
    # 计算磁倾角（度）- 地磁向量与水平面的夹角
    df['Dip_Angle_deg'] = np.degrees(np.arctan2(df['Bz_uT'], df['H_uT']))
    
    # 计算磁偏角（度）- 地磁北与地理北之间的水平夹角
    # 由于手机指向地理北，磁偏角计算为 arctan(By/Bx)
    df['Declination_deg'] = np.degrees(np.arctan2(df['By_uT'], df['Bx_uT']))
    
    # 应用Savitzky-Golay滤波器平滑数据
    n = len(df)
    if n > 10:  # 确保有足够的数据点进行平滑处理
        # 动态确定窗口大小
        window_length = min(21, n // 4 if n // 4 % 2 == 1 else n // 4 + 1)
        window_length = max(5, window_length)  # 最小窗口大小为5
        
        # 使用Savitzky-Golay滤波器平滑数据
        if window_length >= 5:
            df['Dip_Angle_smooth'] = savgol_filter(df['Dip_Angle_deg'], 
                                                  window_length=window_length, 
                                                  polyorder=2)
            df['Declination_smooth'] = savgol_filter(df['Declination_deg'], 
                                                    window_length=window_length, 
                                                    polyorder=2)
            df['Bz_smooth'] = savgol_filter(df['Bz_uT'], 
                                           window_length=window_length, 
                                           polyorder=2)
        else:
            # 窗口大小不足时不应用平滑
            df['Dip_Angle_smooth'] = df['Dip_Angle_deg']
            df['Declination_smooth'] = df['Declination_deg']
            df['Bz_smooth'] = df['Bz_uT']
    else:
        # 数据点不足时使用原始值
        df['Dip_Angle_smooth'] = df['Dip_Angle_deg']
        df['Declination_smooth'] = df['Declination_deg']
        df['Bz_smooth'] = df['Bz_uT']
    
    return df

def plot_magnetic_analysis(df, location_name):
    # 绘制磁参数随时间变化的图表
    # 创建画布
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # 1. 磁倾角（Dip Angle）
    axs[0].plot(df['Time_s'], df['Dip_Angle_deg'], 'c-', alpha=0.4, label='Raw')
    if 'Dip_Angle_smooth' in df:
        axs[0].plot(df['Time_s'], df['Dip_Angle_smooth'], 'b-', label='Smoothed', linewidth=2)
    axs[0].set_title(f'Magnetic Dip Angle over Time - {location_name}', fontsize=14)
    axs[0].set_ylabel('Dip Angle (°)', fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].legend(fontsize=10)
    
    # 2. 磁偏角（Declination）
    axs[1].plot(df['Time_s'], df['Declination_deg'], 'm-', alpha=0.4, label='Raw')
    if 'Declination_smooth' in df:
        axs[1].plot(df['Time_s'], df['Declination_smooth'], 'r-', label='Smoothed', linewidth=2)
    axs[1].set_title(f'Magnetic Declination over Time - {location_name}', fontsize=14)
    axs[1].set_ylabel('Declination (°)', fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].legend(fontsize=10)
    
    # 3. 磁场分量
    axs[2].plot(df['Time_s'], df['Bx_uT'], 'b-', label='Bx (North)')
    axs[2].plot(df['Time_s'], df['By_uT'], 'g-', label='By (East)')
    if 'Bz_smooth' in df:
        axs[2].plot(df['Time_s'], df['Bz_smooth'], 'r-', label='Bz (Down)')
    else:
        axs[2].plot(df['Time_s'], df['Bz_uT'], 'r-', label='Bz (Down)')
    axs[2].plot(df['Time_s'], df['Absolute_field_uT'], 'k-', linewidth=2, label='Total Field')
    axs[2].set_title('Magnetic Field Components', fontsize=14)
    axs[2].set_xlabel('Time (s)', fontsize=12)
    axs[2].set_ylabel('Magnetic Field (µT)', fontsize=12)
    axs[2].grid(True, linestyle='--', alpha=0.7)
    axs[2].legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    # 替换文件名中的空格为下划线
    clean_name = location_name.replace(' ', '_')
    plt.savefig(f'Magnetic_Analysis_{clean_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存处理后的数据
    output_file = f'Magnetic_Results_{clean_name}.csv'
    df.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to: {output_file}")

# 主程序
if __name__ == "__main__":
    file_path = "/Users/rea713/Downloads/phyphox 2025-06-25 19-40-06.xls"  # Excel文件路径
    location_name = "Location 1"  #
    
    print(f"Reading data from: {file_path}")
    df = read_magnetic_data(file_path)
    
    if df is None:
        print("Failed to read data file. Exiting.")
        sys.exit(1)
    
    # 检查数据完整性
    if df.empty:
        print("Error: Data file is empty.")
        sys.exit(1)
    
    print(f"Successfully read {len(df)} data points")
    
    # 显示前几行数据以验证读取
    print("\nFirst few rows of data:")
    print(df.head())
    
    # 计算磁参数
    print("\nCalculating magnetic parameters...")
    df = calculate_magnetic_parameters(df)
    
    # 结果统计
    print("\nMagnetic Parameter Statistics:")
    print(f"  Average Dip Angle: {df['Dip_Angle_deg'].mean():.2f}°")
    print(f"  Min Dip Angle: {df['Dip_Angle_deg'].min():.2f}°, Max Dip Angle: {df['Dip_Angle_deg'].max():.2f}°")
    print(f"  Average Declination: {df['Declination_deg'].mean():.2f}°")
    print(f"  Min Declination: {df['Declination_deg'].min():.2f}°, Max Declination: {df['Declination_deg'].max():.2f}°")
    print(f"  Vertical Component (Bz) Range: {df['Bz_uT'].min():.2f} to {df['Bz_uT'].max():.2f} µT")
    print(f"  Total Field Range: {df['Absolute_field_uT'].min():.2f} to {df['Absolute_field_uT'].max():.2f} µT")
    
    # 绘图
    plot_magnetic_analysis(df, location_name)
    print("Analysis complete!")