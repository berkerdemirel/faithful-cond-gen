"""Generate HTML report from faithfulness evaluation results.

This script creates an interactive HTML page with:
- Summary statistics (AUROC, FPR@95, correlations)
- Embedded visualization plots
- Top-K best/worst conditions tables
- Downloadable data links

Usage:
    python scripts/generate_report.py \
        <score_payload.pt> \
        <output_report.html>

Example:
    python scripts/generate_report.py \
        outputs/scores/celeba_dinov2_scores.pt \
        outputs/reports/celeba_report.html
"""

import argparse
import base64
import logging
import os
import sys
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import torch
from jinja2 import Template

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Faithfulness Evaluation Report</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }

        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }

        h2 {
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-bottom: 8px;
            border-bottom: 2px solid #e0e0e0;
        }

        h3 {
            color: #7f8c8d;
            margin-top: 25px;
            margin-bottom: 15px;
        }

        .summary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .summary h2 {
            color: white;
            border: none;
            margin: 0 0 15px 0;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }

        .metric {
            background: rgba(255,255,255,0.15);
            padding: 15px;
            border-radius: 6px;
            backdrop-filter: blur(10px);
        }

        .metric-label {
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 5px;
        }

        .metric-value {
            font-size: 1.8em;
            font-weight: bold;
        }

        .plot-container {
            margin: 25px 0;
            text-align: center;
        }

        .plot-container img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.95em;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            border-radius: 4px;
            overflow: hidden;
        }

        th {
            background-color: #3498db;
            color: white;
            text-align: left;
            padding: 12px;
            font-weight: 600;
        }

        td {
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }

        tr:nth-child(even) {
            background-color: #f8f9fa;
        }

        tr:hover {
            background-color: #e3f2fd;
        }

        .good-score {
            color: #27ae60;
            font-weight: bold;
        }

        .bad-score {
            color: #e74c3c;
            font-weight: bold;
        }

        .info-box {
            background: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }

        .warning-box {
            background: #fff3cd;
            border-left: 4px solid #f39c12;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }

        .download-links {
            background: #f8f9fa;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
            border: 1px solid #dee2e6;
        }

        .download-links a {
            color: #3498db;
            text-decoration: none;
            margin-right: 20px;
            font-weight: 500;
        }

        .download-links a:hover {
            text-decoration: underline;
        }

        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Faithfulness Evaluation Report</h1>

        <div class="info-box">
            <strong>Dataset:</strong> {{ dataset_name }}<br>
            <strong>Encoder:</strong> {{ encoder_name }}<br>
            <strong>Generated:</strong> {{ timestamp }}
        </div>

        <div class="summary">
            <h2>📊 Global Metrics</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-label">AUROC</div>
                    <div class="metric-value">{{ "%.4f"|format(auroc) }}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">FPR@95% TPR</div>
                    <div class="metric-value">{{ "%.4f"|format(fpr95) }}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total Conditions</div>
                    <div class="metric-value">{{ n_conditions }}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Mean Score</div>
                    <div class="metric-value">{{ "%.3f"|format(mean_score) }}</div>
                </div>
            </div>
        </div>

        {% if viz_plots %}
        <h2>📈 Visualizations</h2>

        {% for plot_title, plot_data in viz_plots.items() %}
        <div class="plot-container">
            <h3>{{ plot_title }}</h3>
            <img src="data:image/png;base64,{{ plot_data }}" alt="{{ plot_title }}">
        </div>
        {% endfor %}
        {% endif %}

        <h2>🏆 Best 10 Conditions (Lowest Scores)</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Condition</th>
                    <th>Score</th>
                    <th>KID Δ</th>
                    <th>Real Pool</th>
                    <th>Difficulty</th>
                </tr>
            </thead>
            <tbody>
                {% for row in best_conditions %}
                <tr>
                    <td>{{ loop.index }}</td>
                    <td style="max-width: 300px; overflow: hidden; text-overflow: ellipsis;">
                        {{ row.condition_hash[:60] }}...
                    </td>
                    <td class="good-score">{{ "%.4f"|format(row.mean_score) }}</td>
                    <td>{{ "%.4f"|format(row.kid_delta_mean) if row.kid_delta_mean == row.kid_delta_mean else 'N/A' }}</td>
                    <td>{{ row.n_real_pool }}</td>
                    <td>{{ "%.2f"|format(row.difficulty) if row.difficulty == row.difficulty else 'N/A' }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <h2>⚠️ Worst 10 Conditions (Highest Scores)</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Condition</th>
                    <th>Score</th>
                    <th>KID Δ</th>
                    <th>Real Pool</th>
                    <th>Difficulty</th>
                </tr>
            </thead>
            <tbody>
                {% for row in worst_conditions %}
                <tr>
                    <td>{{ loop.index }}</td>
                    <td style="max-width: 300px; overflow: hidden; text-overflow: ellipsis;">
                        {{ row.condition_hash[:60] }}...
                    </td>
                    <td class="bad-score">{{ "%.4f"|format(row.mean_score) }}</td>
                    <td>{{ "%.4f"|format(row.kid_delta_mean) if row.kid_delta_mean == row.kid_delta_mean else 'N/A' }}</td>
                    <td>{{ row.n_real_pool }}</td>
                    <td>{{ "%.2f"|format(row.difficulty) if row.difficulty == row.difficulty else 'N/A' }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        {% if key_correlations %}
        <h2>🔗 Key Correlations</h2>
        <div class="info-box">
            {% for corr_name, corr_value in key_correlations.items() %}
            <strong>{{ corr_name }}:</strong> {{ "%.3f"|format(corr_value) }}<br>
            {% endfor %}
        </div>
        {% endif %}

        <div class="download-links">
            <h3>📥 Download Data</h3>
            <a href="{{ csv_filename }}" download>Download Full CSV</a>
            <a href="{{ corr_filename }}" download>Download Correlation Matrix</a>
        </div>

        <div class="footer">
            Generated by faithful-cond-gen evaluation pipeline
        </div>
    </div>
</body>
</html>
"""


def encode_plot_base64(plot_path: str) -> str:
    """Encode a plot image as base64 string for HTML embedding."""
    if not os.path.exists(plot_path):
        return ""

    with open(plot_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_html_report(score_payload_path: str, output_html: str):
    """Generate comprehensive HTML report from scoring results.

    Args:
        score_payload_path: Path to .pt file from run_scoring.py
        output_html: Path to save HTML report
    """
    log.info(f"Loading scoring results from {score_payload_path}...")
    data = torch.load(score_payload_path, map_location="cpu", weights_only=False)

    df = data["df"]
    global_metrics = data["global_metrics"]
    corr_df = data.get("spearman_corr", pd.DataFrame())

    log.info(f"Loaded {len(df)} conditions")

    # Extract metadata
    dataset_name = Path(score_payload_path).parent.parent.name
    encoder_name = "Unknown"
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Compute summary stats
    mean_score = df["mean_score"].mean() if "mean_score" in df.columns else 0.0
    n_conditions = len(df)

    # Get best and worst conditions
    best_conditions = df.nsmallest(10, "mean_score").to_dict("records")
    worst_conditions = df.nlargest(10, "mean_score").to_dict("records")

    # Extract key correlations
    key_correlations = {}
    if not corr_df.empty:
        if "mean_score" in corr_df.index:
            for col in ["kid_delta_mean", "kid_rel", "relative_fid", "n_real_pool"]:
                if col in corr_df.columns:
                    key_correlations[f"Score vs {col}"] = corr_df.loc["mean_score", col]

    # Load visualization plots if available
    viz_plots = {}
    viz_dir = os.path.join(os.path.dirname(score_payload_path), "visualizations")

    if os.path.exists(viz_dir):
        log.info(f"Loading visualizations from {viz_dir}...")
        plot_files = {
            "Score vs KID": "score_vs_kid.png",
            "Correlation Heatmap": "correlations.png",
            "Score vs Pool Size": "score_vs_pool_size.png",
            "Difficulty Distribution": "difficulty_distribution.png",
        }

        for title, filename in plot_files.items():
            plot_path = os.path.join(viz_dir, filename)
            if os.path.exists(plot_path):
                viz_plots[title] = encode_plot_base64(plot_path)

    # Prepare download filenames (relative paths)
    csv_filename = os.path.basename(score_payload_path).replace(".pt", "_analysis.csv")
    corr_filename = os.path.basename(score_payload_path).replace(".pt", "_corr_spearman.csv")

    # Render template
    template = Template(HTML_TEMPLATE)
    html = template.render(
        dataset_name=dataset_name,
        encoder_name=encoder_name,
        timestamp=timestamp,
        auroc=global_metrics.get("auroc", 0.0),
        fpr95=global_metrics.get("fpr95", 0.0),
        n_conditions=n_conditions,
        mean_score=mean_score,
        best_conditions=best_conditions,
        worst_conditions=worst_conditions,
        key_correlations=key_correlations,
        viz_plots=viz_plots,
        csv_filename=csv_filename,
        corr_filename=corr_filename,
    )

    # Save HTML
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    with open(output_html, "w") as f:
        f.write(html)

    log.info(f"✅ Report saved to {output_html}")


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report from scoring results")
    parser.add_argument("score_payload", type=str, help="Path to scoring results (.pt file)")
    parser.add_argument("output_html", type=str, help="Path to output HTML report")

    args = parser.parse_args()

    if not os.path.exists(args.score_payload):
        log.error(f"Score payload not found: {args.score_payload}")
        sys.exit(1)

    generate_html_report(args.score_payload, args.output_html)


if __name__ == "__main__":
    main()
