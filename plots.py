## Plots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
from scipy import stats
from matplotlib.lines import Line2D

# DATA
def read_files_output(folder_path):
  files = glob.glob(os.path.join(folder_path, "output_model*"))
  data_frame = pd.DataFrame()
  for file in files:
      data = pd.read_csv(file, sep=";")
      data_frame = pd.concat([data_frame, data], ignore_index=True)
      
  return data_frame    

def read_files_agents(folder_path):
  files = glob.glob(os.path.join(folder_path, "output_agents*"))
  data_frame = pd.DataFrame()
  for file in files:
      data = pd.read_csv(file, sep=";")
      data_frame = pd.concat([data_frame, data], ignore_index=True)
      
  return data_frame


def plot_policy_comparison(
    baseline_data,
    prudential_data,
    variable,
    xlabel,
    filename,
    bins=20,
    figsize=(12, 8),
    baseline_color="skyblue",
    prudential_color="red"
):
    """
    Plot histogram comparison with median, std, and 95% interval.
    """

    # -----------------------------
    # Extract data
    # -----------------------------
    b_data = baseline_data[variable]
    p_data = prudential_data[variable]

    # -----------------------------
    # Statistics
    # -----------------------------
    b_mean = b_data.mean()
    b_std = b_data.std()
    b_low, b_high = np.percentile(b_data, [2.5, 97.5])
    b_median = b_data.median()

    p_mean = p_data.mean()
    p_std = p_data.std()
    p_low, p_high = np.percentile(p_data, [2.5, 97.5])
    p_median = p_data.median()

    # -----------------------------
    # Plot
    # -----------------------------
    plt.clf()
    plt.figure(figsize=figsize)

    # Histograms
    plt.hist(
        b_data,
        bins=bins,
        color=baseline_color,
        alpha=0.6,
        label="No Prudential Policy"
    )

    plt.hist(
        p_data,
        bins=bins,
        color=prudential_color,
        alpha=0.6,
        label="Prudential Policy"
    )

    # Medians
    plt.axvline(b_median, color=baseline_color, lw=1.5, alpha=0.6)
    plt.axvline(p_median, color=prudential_color, lw=1.5, alpha=0.6)

    # ±1 Std. Dev.
    plt.axvline(b_mean - b_std, color=baseline_color, linestyle="--", alpha=0.7)
    plt.axvline(b_mean + b_std, color=baseline_color, linestyle="--", alpha=0.7)

    plt.axvline(p_mean - p_std, color=prudential_color, linestyle="--", alpha=0.7)
    plt.axvline(p_mean + p_std, color=prudential_color, linestyle="--", alpha=0.7)

    # 95% Interval
    plt.axvline(b_low, color=baseline_color, linestyle=":", alpha=0.9)
    plt.axvline(b_high, color=baseline_color, linestyle=":", alpha=0.9)

    plt.axvline(p_low, color=prudential_color, linestyle=":", alpha=0.9)
    plt.axvline(p_high, color=prudential_color, linestyle=":", alpha=0.9)

    # Labels
    plt.xlabel(xlabel)
    plt.ylabel("")

    # -----------------------------
    # Legends
    # -----------------------------

    # Policy legend
    dist_legend = plt.legend(loc="upper left")

    # Line-style legend
    style_legend = [
        Line2D([0], [0], color="black", lw=1.5, label="Median"),
        Line2D([0], [0], color="black", lw=1.2, linestyle="--", label="±1 Std. Dev."),
        Line2D([0], [0], color="black", lw=1.2, linestyle=":", label="95% Interval")
    ]

    plt.legend(handles=style_legend, loc="upper right")

    # Keep both legends
    plt.gca().add_artist(dist_legend)

    # -----------------------------
    # Save & show
    # -----------------------------
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16
})


# Baseline ---------------------------------------------------------------------
baseline_folder = "./baseline"
baseline_data = read_files_output(baseline_folder)
    
# Prudential -------------------------------------------------------------------
prudential_folder = "./prudential_model"
prudential_data = read_files_output(prudential_folder)

# Convergence ------------------------------------------------------------------
convergence_folder = "./convergence_test"
files = glob.glob(os.path.join(convergence_folder, "output_model*"))
convergence_data = pd.DataFrame()

for i, file in enumerate(files):
    data = pd.read_csv(file, sep=";")[["Convergence_Percentage"]]
    # Rename column to keep track of simulations
    data.columns = [f"sim_{i+1}"]
    convergence_data = pd.concat([convergence_data, data], axis=1)
    
# Plots ------------------------------------------------------------------------
# Convergence over time (all simulations + average)
convergence_data["average_convergence"] = convergence_data.mean(axis=1)

plt.clf()
plt.figure(figsize=(12, 8))

# Plot each simulation
for col in convergence_data.columns:
    if col != "average_convergence":
        plt.plot(
            convergence_data.index,
            convergence_data[col],
            color="skyblue",
            alpha=0.4,
            linewidth=1
        )

# Plot average convergence
plt.plot(
    convergence_data.index,
    convergence_data["average_convergence"],
    color="red",
    linewidth=3,
    label="Average Convergence"
)

plt.xlabel("Simulation Step")
plt.xticks([0, 5000, 10000, 15000, 20000])
plt.ylabel("Percentage of agents with converged strategies")
plt.legend()
plt.tight_layout()

plt.savefig("figures/convergence_over_time.png", dpi=300, bbox_inches="tight")
plt.show()

# First, I'll just group the values for all the 4 Monte Carlo Simulations
# Plots ------------------------------------------------------------------------
# 1) No Prudential x Prudential Policy
# Real Sector interest rate
plot_policy_comparison(
baseline_data,
prudential_data,
variable="Real_Sector_Interest_Rate",
xlabel="Interest rate",
filename="figures/real_sector_interest_rate.png"
)

# Average Risk
plot_policy_comparison(
baseline_data,
prudential_data,
variable="average_risk",
xlabel="Average risk",
filename="figures/average_risk.png"
)

# Real Sector Loans
plot_policy_comparison(
baseline_data,
prudential_data,
variable="Real_Sector_Loans",
xlabel="Real sector loans",
filename="figures/real_sector_loans.png"
)

# High Risk loan ratio
plot_policy_comparison(
baseline_data,
prudential_data,
variable="High risk loan Ratio",
xlabel="High risk loans to total loans ratio",
filename="figures/high_risk_loan_ratio.png"
)

# HHI
plot_policy_comparison(
baseline_data,
prudential_data,
variable="HHI",
xlabel="HHI",
filename="figures/hhi.png"
)

# CR5
plot_policy_comparison(
baseline_data,
prudential_data,
variable="CR5",
xlabel="CR5",
filename="figures/cr5.png"
)

# Capital
plot_policy_comparison(
baseline_data,
prudential_data,
variable="Capital",
xlabel="Capital",
filename="figures/capital.png"
)

# Insolvencies
plot_policy_comparison(
baseline_data,
prudential_data,
variable="Insolvencies",
xlabel="Insolvencies ratio",
filename="figures/insolvencies.png"
)

# 1.2) Table with the distributional values
# Variables to include
variables = [
    'Real_Sector_Interest_Rate',
    'average_risk',
    'Real_Sector_Loans',
    'High risk loan Ratio',
    'HHI',
    'CR5',
    'Capital',
    'Insolvencies'
]

# Variable display names for the table
variable_names = {
    'Real_Sector_Interest_Rate': 'Real Sector Interest Rate',
    'average_risk': 'Average Risk',
    'Real_Sector_Loans': 'Real Sector Loans',
    'High risk loan Ratio': 'High Risk Loan Ratio',
    'HHI': 'HHI',
    'CR5': 'CR5',
    'Capital': 'Capital',
    'Insolvencies': 'Insolvencies'
}

def calculate_stats(data, variable):
    """
    Calculate median, mean, std, and 95% distributional interval
    (2.5th–97.5th percentiles)
    """
    values = data[variable].dropna()

    median = np.median(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)

    # 95% distributional interval
    p_lower, p_upper = np.percentile(values, [2.5, 97.5])

    return {
        'median': median,
        'mean': mean,
        'std': std,
        'p_lower': p_lower,
        'p_upper': p_upper
    }

# Initialize LaTeX table
latex_table = r"""\begin{table}[htbp]
\centering
\caption{Distributional Statistics: Prudential vs No Prudential Policy}
\label{tab:dist_stats}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccc}
\hline\hline
 & \multicolumn{4}{c}{\textbf{No Prudential Policy}} & \multicolumn{4}{c}{\textbf{Prudential Policy}} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-9}
\textbf{Variable} & \textbf{Median} & \textbf{Mean} & \textbf{Std. Dev.} & \textbf{95\% Interval} &
\textbf{Median} & \textbf{Mean} & \textbf{Std. Dev.} & \textbf{95\% Interval} \\
\hline
"""

# Populate table rows
for var in variables:
    baseline_stats = calculate_stats(baseline_data, var)
    prudential_stats = calculate_stats(prudential_data, var)

    latex_table += (
        f"{variable_names[var]} & "
        f"{baseline_stats['median']:.4f} & "
        f"{baseline_stats['mean']:.4f} & "
        f"{baseline_stats['std']:.4f} & "
        f"[{baseline_stats['p_lower']:.4f}, {baseline_stats['p_upper']:.4f}] & "
        f"{prudential_stats['median']:.4f} & "
        f"{prudential_stats['mean']:.4f} & "
        f"{prudential_stats['std']:.4f} & "
        f"[{prudential_stats['p_lower']:.4f}, {prudential_stats['p_upper']:.4f}] \\\\\n"
    )

# Close table
latex_table += r"""\hline\hline
\end{tabular}%
}
\end{table}
"""

# Print or write to file
print(latex_table)

# Print the LaTeX table
print(latex_table)

# LogNormal Firm Size Distribution
# Baseline ---------------------------------------------------------------------
baseline_folder = "./lognormal_distribution_baseline"
baseline_data = read_files_output(baseline_folder)
    
# Prudential -------------------------------------------------------------------
prudential_folder = "./lognormal_distribution_prudential"
prudential_data = read_files_output(prudential_folder)

# Real Sector interest rate
plot_policy_comparison(
baseline_data,
prudential_data,
variable="Real_Sector_Interest_Rate",
xlabel="Interest rate",
filename="figures/real_sector_interest_rate_lognorm.png"
)

# Average Risk
plot_policy_comparison(
baseline_data,
prudential_data,
variable="average_risk",
xlabel="Average risk",
filename="figures/average_risk_lognorm.png"
)

# Real Sector Loans
plot_policy_comparison(
baseline_data,
prudential_data,
variable="Real_Sector_Loans",
xlabel="Real sector loans",
filename="figures/real_sector_loans_lognorm.png"
)

# High Risk loan ratio
plot_policy_comparison(
baseline_data,
prudential_data,
variable="High risk loan Ratio",
xlabel="High risk loans to total loans ratio",
filename="figures/high_risk_loan_ratio_lognorm.png"
)

# HHI
plot_policy_comparison(
baseline_data,
prudential_data,
variable="HHI",
xlabel="HHI",
filename="figures/hhi_lognorm.png"
)

# CR5
plot_policy_comparison(
baseline_data,
prudential_data,
variable="CR5",
xlabel="CR5",
filename="figures/cr5_lognorm.png"
)

# Capital
plot_policy_comparison(
baseline_data,
prudential_data,
variable="Capital",
xlabel="Capital",
filename="figures/capital_lognorm.png"
)

# Insolvencies
plot_policy_comparison(
baseline_data,
prudential_data,
variable="Insolvencies",
xlabel="Insolvencies ratio",
filename="figures/insolvencies_lognorm.png"
)

# 2) Agents Data
# Baseline ---------------------------------------------------------------------
baseline_folder = "./baseline"
baseline_data_agents = read_files_agents(baseline_folder)

baseline_data_agents = baseline_data_agents[baseline_data_agents['Firm Size'] > 0]
                                                 
baseline_data_agents['weighted_risk'] = np.where(baseline_data_agents['Predicted Risk Type'] == "LowRisk",
                                                 baseline_data_agents['Loan Total'] * 2,
                                                 baseline_data_agents['Loan Total'] * 5)
                                                 
plt.clf()
plt.figure(figsize=(10, 6))
plt.scatter(baseline_data_agents['Firm Size'], baseline_data_agents['Loan Total'], alpha=0.5, edgecolor='k')
plt.title('Firm Size vs. Loan Total', fontsize=14)
plt.xlabel('Firm Size', fontsize=12)
plt.ylabel('Loan Total', fontsize=12)
plt.grid(True)
plt.show()

# Firm size distribution
plt.clf()
plt.figure(figsize=(10, 6))
plt.hist(baseline_data_agents['Firm Size'].unique(), bins=20, color='skyblue', edgecolor='black', alpha=0.7, label='Firm Size')
plt.title('Distribution of Firms Size (Pareto)')
plt.xlabel('Firm Size')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.show()

# Prudential -------------------------------------------------------------------
prudential_folder = "./prudential_model"
prudential_data_agents = read_files_agents(prudential_folder)

prudential_data_agents = prudential_data_agents[prudential_data_agents['Firm Size'] > 0]

prudential_data_agents['weighted_risk'] = np.where(prudential_data_agents['Predicted Risk Type'] == "LowRisk",
                                                 prudential_data_agents['Loan Total'] * 2,
                                                 prudential_data_agents['Loan Total'] * 5)

plt.clf()
plt.figure(figsize=(10, 6))
plt.scatter(prudential_data_agents['Firm Size'], prudential_data_agents['Loan Total'], alpha=0.5, edgecolor='k')
plt.title('Firm Size vs. Loan Total', fontsize=14)
plt.xlabel('Firm Size', fontsize=12)-
plt.ylabel('Loan Total', fontsize=12)
plt.grid(True)
plt.show()

# Firm size distribution
plt.clf()
plt.figure(figsize=(10, 6))
_ = plt.hist(prudential_data_agents['Firm Size'], bins=20, color='skyblue', edgecolor='black', alpha=0.7, label='Firm Size')
plt.title('Distribution of Firms Size (Pareto)')
plt.xlabel('Firm Size')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.show()

# Risk Distribution x Firm Size -------------------------------------------------
plt.clf()
plt.figure(figsize=(10, 6))
plt.scatter(baseline_data_agents['Firm Size'], baseline_data_agents['weighted_risk'], alpha=0.5, edgecolor='blue')
plt.scatter(prudential_data_agents['Firm Size'], prudential_data_agents['weighted_risk'], alpha=0.5, edgecolor='red')
plt.title('Firm Size vs. Weighted Risk', fontsize=14)
plt.xlabel('Firm Size', fontsize=12)
plt.ylabel('Weighted Risk', fontsize=12)
plt.grid(True)
plt.show()


#
import seaborn as sns
import matplotlib.patches as  mpatches

density_baseline = baseline_data_agents[(baseline_data_agents['Loan Total'] > 0) & (baseline_data_agents['Firm Size'] < 8)]
density_prudential = prudential_data_agents[(prudential_data_agents['Loan Total'] > 0) & (prudential_data_agents['Firm Size'] < 8)]

# Baseline
plt.clf()
plt.figure(figsize=(10, 6))
sns.kdeplot(x=density_baseline['Firm Size'], y=density_baseline['Loan Total'], cmap='Blues', fill=True)
plt.title('Density Plot: Firm Size vs. Loan Total', fontsize=14)
plt.xlabel('Firm Size', fontsize=12)
plt.ylabel('Loan Total', fontsize=12)
plt.show()

# Prudential
plt.clf()
plt.figure(figsize=(10, 6))
sns.kdeplot(x=density_prudential['Firm Size'], y=density_prudential['Loan Total'], cmap='Blues', fill=True)
plt.title('Density Plot: Firm Size vs. Loan Total', fontsize=14)
plt.xlabel('Firm Size', fontsize=12)
plt.ylabel('Loan Total', fontsize=12)
plt.show()

plt.clf()
fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

# Plot for baseline
sns.kdeplot(
    x=density_baseline['Firm Size'], 
    y=density_baseline['Loan Total'], 
    cmap='Blues', 
    fill=True, 
    ax=axes[0]
)
axes[0].set_title('Baseline: Firm Size vs. Loan Total', fontsize=8)
axes[0].set_xlabel('')  # Remove x-label
axes[0].set_ylabel('')  # Remove y-label

# Plot for prudential
sns.kdeplot(
    x=density_prudential['Firm Size'], 
    y=density_prudential['Loan Total'], 
    cmap='Blues', 
    fill=True, 
    ax=axes[1]
)
axes[1].set_title('Prudential: Firm Size vs. Loan Total', fontsize=8)
axes[1].set_xlabel('')  # Remove x-label
axes[1].set_ylabel('')  # Remove y-label
axes[1].set_yticks([])

# Set common labels for the entire figure
fig.text(0.5, 0.04, 'Firm Size', ha='center', fontsize=12)  # x-label
fig.text(0.02, 0.5, 'Loan Total', va='center', rotation='vertical', fontsize=12)  # y-label

# Adjust layout and display
plt.tight_layout(rect=[0.05, 0.05, 1, 1])  # Adjust layout to fit axis labels
plt.show()


##
baseline_folder = "./baseline"
baseline_data_agents = read_files_agents(baseline_folder)

baseline_data_agents = baseline_data_agents[baseline_data_agents['Firm Size'] > 0]
baseline_data_agents['Quartil'] = pd.qcut(x = baseline_data_agents['Firm Size'], q = 4, labels = False, duplicates = "drop")

prudential_folder = "./prudential_model"
prudential_data_agents = read_files_agents(prudential_folder)

prudential_data_agents = prudential_data_agents[prudential_data_agents['Firm Size'] > 0]
prudential_data_agents['Quartil'] = pd.qcut(x = prudential_data_agents['Firm Size'], q = 4, labels = False, duplicates = "drop")

for i in range(4):
    baseline_boxplot = baseline_data_agents[(baseline_data_agents['Loan Total'] > 0) & (baseline_data_agents['Quartil'] == i)]['Loan Total']
    prudential_boxplot = prudential_data_agents[(prudential_data_agents['Loan Total'] > 0) & (prudential_data_agents['Quartil'] == i)]['Loan Total']
    print(np.median(baseline_boxplot), np.median(prudential_boxplot))
    print(np.var(baseline_boxplot), np.var(prudential_boxplot))
    boxplot_data = [baseline_boxplot,prudential_boxplot]
    plt.clf()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_ylabel('Loan Size')

    bplot = ax.boxplot(boxplot_data,
                      labels=['No Prudential Policy', 'Prudential Policy'],
                      widths = 0.4)

    plt.title(f'Box plot for quartil {i+1}', fontsize=14)               
    plt.show()
    plt.savefig(f'figures/box_plot_quartile{i+1}.png')

# Banks CAR
baseline_data_banks = baseline_data_agents[baseline_data_agents['is_bank'] == True]
baseline_data_banks = baseline_data_banks[baseline_data_banks['Loan Value'] > 0.01]

prudential_data_banks = prudential_data_agents[prudential_data_agents['is_bank'] == True]
prudential_data_banks = prudential_data_banks[prudential_data_banks['Loan Value'] > 0.01]


plt.clf()
plt.figure(figsize=(12, 8))
plt.hist(baseline_data_banks['CAR'], bins=100, color='skyblue', label='No Prudential Policy')
plt.hist(prudential_data_banks['CAR'], bins=100, color='red', alpha = 0.7,  label='Prudential Policy')
plt.axvline(x=0.08, color='black', linestyle='--', linewidth=1.5, label='Regulatory Threshold')
plt.ylabel('')
plt.xlabel('CAR')
plt.legend()
plt.tight_layout()
plt.savefig('figures/car.png', dpi = 300, bbox_inches="tight")
plt.show()


## Policy trade-off
# =============================================================================
# POLICY TRADE-OFF TABLE: % CHANGE (PRUDENTIAL VS BASELINE)
# =============================================================================
# -----------------------------
# Helper functions
# -----------------------------
def pct_change(policy, baseline):
    """Percentage change: (policy - baseline) / baseline * 100"""
    return 100 * (policy - baseline) / baseline


def credit_share_bottom_quartile(data_agents):
    """
    Share of total credit going to bottom firm-size quartile
    """
    df = data_agents[data_agents["Loan Total"] > 0].copy()
    df["quartile"] = pd.qcut(df["Firm Size"], q=4, labels=False, duplicates="drop")

    total_credit = df["Loan Total"].sum()
    bottom_credit = df[df["quartile"] == 0]["Loan Total"].sum()

    return bottom_credit / total_credit


def average_car_banks(data_agents):
    """
    Average CAR for banks with positive loan value
    """
    banks = data_agents[
        (data_agents["is_bank"] == True) &
        (data_agents["Loan Value"] > 0.01)
    ]
    return banks["CAR"].mean()


# -----------------------------
# Compute baseline statistics
# -----------------------------
baseline_total_credit = baseline_data["Real_Sector_Loans"].mean()
prudential_total_credit = prudential_data["Real_Sector_Loans"].mean()

baseline_credit_share_bottom = credit_share_bottom_quartile(baseline_data_agents)
prudential_credit_share_bottom = credit_share_bottom_quartile(prudential_data_agents)

baseline_hhi = baseline_data["HHI"].mean()
prudential_hhi = prudential_data["HHI"].mean()

baseline_cr5 = baseline_data["CR5"].mean()
prudential_cr5 = prudential_data["CR5"].mean()

baseline_insolvencies = baseline_data["Insolvencies"].mean()
prudential_insolvencies = prudential_data["Insolvencies"].mean()

baseline_car = average_car_banks(baseline_data_agents)
prudential_car = average_car_banks(prudential_data_agents)


# -----------------------------
# Assemble trade-off table
# -----------------------------
policy_tradeoff = pd.DataFrame({
    "Metric": [
        "Total Credit",
        "Credit Share (Bottom Firm-Size Quartile)",
        "HHI",
        "CR5",
        "Insolvency Rate",
        "Average CAR (Banks)"
    ],
    "Baseline": [
        baseline_total_credit,
        baseline_credit_share_bottom,
        baseline_hhi,
        baseline_cr5,
        baseline_insolvencies,
        baseline_car
    ],
    "Prudential": [
        prudential_total_credit,
        prudential_credit_share_bottom,
        prudential_hhi,
        prudential_cr5,
        prudential_insolvencies,
        prudential_car
    ]
})

policy_tradeoff["% Change (Policy vs Baseline)"] = pct_change(
    policy_tradeoff["Prudential"],
    policy_tradeoff["Baseline"]
)

# Optional: nicer formatting
policy_tradeoff["% Change (Policy vs Baseline)"] = (
    policy_tradeoff["% Change (Policy vs Baseline)"].round(2)
)

print(policy_tradeoff)


# -----------------------------
# (Optional) LaTeX export
# -----------------------------
latex_tradeoff = r"""\begin{table}[htbp]
\centering
\caption{Policy Trade-Offs: Percentage Change Relative to Baseline}
\label{tab:policy_tradeoffs}
\resizebox{0.85\textwidth}{!}{%
\begin{tabular}{lc}
\hline\hline
\textbf{Metric} & \textbf{Prudential vs Baseline (\%)} \\
\hline
"""

for _, row in policy_tradeoff.iterrows():
    latex_tradeoff += (
        f"{row['Metric']} & {row['% Change (Policy vs Baseline)']:.2f} \\\\\n"
    )

latex_tradeoff += r"""\hline\hline
\end{tabular}%
}
\end{table}
"""

print(latex_tradeoff)
