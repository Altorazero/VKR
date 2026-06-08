from __future__ import annotations

from typing import Any

import plotly.graph_objects as go


def tempo_figure(times: list[float], tempo: list[float], acceleration: list[float], speech_segments: list[list[float]], pauses: list[list[float]], labels: list[str] = None, y_label: str = "Tempo") -> dict[str, Any]:
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=times, 
        y=tempo, 
        name="Tempo", 
        mode="lines+markers+text" if labels else "lines",
        text=labels,
        textposition="top center",
        hoverinfo="x+y+text",
        textfont=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=times, 
        y=acceleration, 
        name="Acceleration", 
        mode="lines", 
        line=dict(dash='dash'),
        opacity=0.5
    ))

    for start, end in speech_segments:
        fig.add_vrect(x0=start, x1=end, fillcolor="green", opacity=0.08, line_width=0)
    for start, end in pauses:
        fig.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.08, line_width=0)

    fig.update_layout(
        title="Tempo Analysis", 
        xaxis_title="Time (s)", 
        yaxis_title=y_label,
        height=500, # Увеличиваем высоту для читаемости
        margin=dict(t=50, b=50, l=50, r=50)
    )
    return fig.to_plotly_json()


def spectrum_figure(freqs: list[float], magnitude: list[float]) -> dict[str, Any]:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=magnitude, mode="lines", name="FFT"))
    fig.update_layout(title="Tempo FFT Spectrum", xaxis_title="Frequency", yaxis_title="Magnitude")
    return fig.to_plotly_json()


def wavelet_figure(scales: list[float], power: list[list[float]], time: list[float]) -> dict[str, Any]:
    fig = go.Figure(
        data=go.Heatmap(
            z=power,
            x=time,
            y=scales,
            colorscale="Viridis",
            colorbar={"title": "Intensity"},
        )
    )
    fig.update_layout(title="Tempo Time-Frequency (Wavelet)", xaxis_title="Time (s)", yaxis_title="Scale")
    return fig.to_plotly_json()


def text_heatmap(words: list[str], values: list[float]) -> dict[str, Any]:
    fig = go.Figure(data=go.Heatmap(z=[values], x=words, y=["Tempo"], colorscale="RdYlGn"))
    fig.update_layout(title="Text Tempo Heatmap")
    return fig.to_plotly_json()


def compare_figure(speaker_series: list[dict[str, Any]]) -> dict[str, Any]:
    fig = go.Figure()
    for series in speaker_series:
        fig.add_trace(
            go.Scatter(
                x=series["time"],
                y=series["tempo"],
                mode="lines",
                name=series["label"],
            )
        )
    fig.update_layout(title="Tempo Comparison", xaxis_title="Time (s)", yaxis_title="Tempo")
    return fig.to_plotly_json()
