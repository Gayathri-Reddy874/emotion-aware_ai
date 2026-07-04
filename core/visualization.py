"""
Chart-building helpers, kept separate from app.py so the plotting logic
(and its Plotly dependency) can change without touching UI flow control.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from core.utils import get_emotion_style


def build_emotion_bar_chart(predictions: list[dict[str, Any]]) -> go.Figure:
    """
    Build a horizontal bar chart of emotion confidence scores, colored
    per-emotion, sorted with the strongest emotion on top.
    """
    ordered = sorted(predictions, key=lambda p: p["score"])  # ascending for horizontal bar top-down
    labels = [p["label"].title() for p in ordered]
    scores = [round(p["score"] * 100, 2) for p in ordered]
    colors = [get_emotion_style(p["label"])["color"] for p in ordered]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{s}%" for s in scores],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis_title="Confidence (%)",
        yaxis_title=None,
        xaxis_range=[0, 100],
        height=90 + 40 * len(labels),
        margin=dict(l=10, r=30, t=20, b=30),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
