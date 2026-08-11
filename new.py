import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import find_peaks
import os

fileName = "Sample_Data.csv"
outputFolder = "output"

timestampInterval = 180

if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)

print("-" * 70)
print("LOADING DATA")
print("-" * 70)

df = pd.read_csv(fileName)
df = df.rename(columns={"Values": "Voltage"})

df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
df["Voltage"] = pd.to_numeric(df["Voltage"], errors="coerce")

df = df.dropna(subset=["Timestamp", "Voltage"])
df = df.sort_values("Timestamp").reset_index(drop=True)

print(f"Number of observations : {len(df)}")
print(f"Start timestamp        : {df['Timestamp'].min()}")
print(f"End timestamp          : {df['Timestamp'].max()}")
print(f"Minimum voltage        : {df['Voltage'].min()}")
print(f"Maximum voltage        : {df['Voltage'].max()}")
print()
print("First 5 rows:")
print(df.head())
print()


# MOVING AVERAGES (1000 and 5000 values)

df["MA_1000"] = df["Voltage"].rolling(window=1000).mean()
df["MA_5000"] = df["Voltage"].rolling(window=5000).mean()

# TIMESTAMP AXIS FORMATTING

def format_timestamp_axis(ax):
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=timestampInterval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y %H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")


# FIGURE 1: VOLTAGE + MOVING AVERAGES

print("-" * 70)
print("CREATING FIGURE 1 STYLE CHART")
print("-" * 70)

plt.figure(figsize=(20, 9))
plt.plot(df["Timestamp"], df["Voltage"], label="Original Value", linewidth=0.8)
plt.plot(df["Timestamp"], df["MA_1000"], label="1000 Value Moving Average", linewidth=1.5)
plt.plot(df["Timestamp"], df["MA_5000"], label="5000 Value Moving Average", linewidth=2)

plt.xlabel("Timestamp", fontsize=12)
plt.ylabel("Voltage", fontsize=12)
plt.title("Voltage with 1000 and 5000 Value Moving Averages", fontsize=15)

ax = plt.gca()
format_timestamp_axis(ax)

plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(outputFolder, "figure1_moving_averages.png"), dpi=300, bbox_inches="tight")
plt.show()


# 5-DAY MOVING AVERAGE

print("-" * 70)
print("CREATING 5-DAY MOVING AVERAGE")
print("-" * 70)

df_time = df.set_index("Timestamp").copy()
df_time["MA_5Day"] = df_time["Voltage"].rolling("5D").mean()

plt.figure(figsize=(20, 9))
plt.plot(df_time.index, df_time["Voltage"], label="Voltage", linewidth=0.8)
plt.plot(df_time.index, df_time["MA_5Day"], label="5-Day Moving Average", linewidth=2)

plt.xlabel("Timestamp", fontsize=12)
plt.ylabel("Voltage", fontsize=12)
plt.title("Voltage with 5-Day Moving Average", fontsize=15)

ax = plt.gca()
format_timestamp_axis(ax)

plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(outputFolder, "voltage_5day_moving_average.png"), dpi=300, bbox_inches="tight")
plt.show()


# LOCAL PEAKS AND LOWS

print("-" * 70)
print("FINDING LOCAL PEAKS")
print("-" * 70)

peaks, peak_properties = find_peaks(df["Voltage"].values, prominence=2)
peak_table = df.iloc[peaks][["Timestamp", "Voltage"]].copy().reset_index(drop=True)

print(f"Number of local peaks found: {len(peak_table)}")
print()
print(peak_table.to_string(index=False))

peak_table.to_csv(os.path.join(outputFolder, "local_peaks.csv"), index=False)

print()
print("-" * 70)
print("FINDING LOCAL LOWS")
print("-" * 70)

lows, low_properties = find_peaks(-df["Voltage"].values, prominence=2)
low_table = df.iloc[lows][["Timestamp", "Voltage"]].copy().reset_index(drop=True)

print(f"Number of local lows found: {len(low_table)}")
print()
print(low_table.to_string(index=False))

low_table.to_csv(os.path.join(outputFolder, "local_lows.csv"), index=False)


# VOLTAGE BELOW 20

print()
print("-" * 70)
print("VOLTAGE BELOW 20")
print("-" * 70)

below_20 = df[df["Voltage"] < 20][["Timestamp", "Voltage"]].copy()

if below_20.empty:
    print("No instances found where Voltage < 20.")
else:
    print(below_20.to_string(index=False))

below_20.to_csv(os.path.join(outputFolder, "voltage_below_20.csv"), index=False)


# PLOT LOCAL PEAKS AND LOWS

print()
print("-" * 70)
print("CREATING PEAKS AND LOWS CHART")
print("-" * 70)

plt.figure(figsize=(20, 9))
plt.plot(df["Timestamp"], df["Voltage"], label="Voltage", linewidth=0.8)
plt.scatter(peak_table["Timestamp"], peak_table["Voltage"], label="Local Peaks", marker="^", s=50)
plt.scatter(low_table["Timestamp"], low_table["Voltage"], label="Local Lows", marker="v", s=50)
plt.axhline(y=20, linestyle="--", label="20 Voltage Threshold")

plt.xlabel("Timestamp", fontsize=12)
plt.ylabel("Voltage", fontsize=12)
plt.title("Voltage with Local Peaks and Lows", fontsize=15)

ax = plt.gca()
format_timestamp_axis(ax)

plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(outputFolder, "peaks_and_lows.png"), dpi=300, bbox_inches="tight")
plt.show()


# DOWNWARD SLOPE ACCELERATION

print()
print("-" * 70)
print("BONUS: DOWNWARD SLOPE ACCELERATION")
print("-" * 70)

bonus_df = df[["Timestamp", "Voltage"]].copy()

bonus_df["dt"] = bonus_df["Timestamp"].diff().dt.total_seconds()

bonus_df.loc[bonus_df["dt"] == 0, "dt"] = pd.NA

bonus_df["dV"] = bonus_df["Voltage"].diff()
bonus_df["Slope"] = bonus_df["dV"] / bonus_df["dt"]

bonus_df["Slope_Change"] = bonus_df["Slope"].diff()
bonus_df["Acceleration"] = bonus_df["Slope_Change"] / bonus_df["dt"]

bonus_df = bonus_df.replace([float("inf"), float("-inf")], pd.NA)

cycle_peaks, _ = find_peaks(df["Voltage"].values, prominence=10, distance=500)
cycle_lows, _ = find_peaks(-df["Voltage"].values, prominence=10, distance=500)

print(f"Significant peaks used for cycles: {len(cycle_peaks)}")
print(f"Significant lows used for cycles: {len(cycle_lows)}")

results = []
cycle_number = 0

for peak_index in cycle_peaks:
    next_lows = cycle_lows[cycle_lows > peak_index]
    if len(next_lows) == 0:
        continue

    low_index = next_lows[0]
    cycle = bonus_df.iloc[peak_index:low_index + 1].copy()

    accelerating = cycle[(cycle["Slope"] < 0) & (cycle["Acceleration"] < 0)].copy()
    cycle_number += 1

    for _, row in accelerating.iterrows():
        results.append({
            "Cycle": cycle_number,
            "Peak Timestamp": df.iloc[peak_index]["Timestamp"],
            "Low Timestamp": df.iloc[low_index]["Timestamp"],
            "Acceleration Timestamp": row["Timestamp"],
            "Voltage": row["Voltage"],
            "Slope": row["Slope"],
            "Acceleration": row["Acceleration"],
        })

acceleration_table = pd.DataFrame(results)

if acceleration_table.empty:
    print()
    print("No downward acceleration instances found.")
else:
    print()
    print(acceleration_table[["Cycle", "Peak Timestamp", "Low Timestamp", "Acceleration Timestamp"]].to_string(index=False))
    acceleration_table.to_csv(os.path.join(outputFolder, "downward_acceleration_by_cycle.csv"), index=False)


# FINAL

print()
print("-" * 70)
print("ANALYSIS COMPLETE")
print("-" * 70)

print(f"Total observations: {len(df)}")
print(f"Local peaks found: {len(peak_table)}")
print(f"Local lows found: {len(low_table)}")
print(f"Voltage < 20 instances: {len(below_20)}")

if acceleration_table.empty:
    print("Downward acceleration instances: 0")
else:
    print(f"Downward acceleration instances: {len(acceleration_table)}")

print()
print("Output files:")
print("1. figure1_moving_averages.png")
print("2. voltage_5day_moving_average.png")
print("3. peaks_and_lows.png")
print("4. local_peaks.csv")
print("5. local_lows.csv")
print("6. voltage_below_20.csv")
print("7. downward_acceleration_by_cycle.csv")
print()
print("Execution Complete")