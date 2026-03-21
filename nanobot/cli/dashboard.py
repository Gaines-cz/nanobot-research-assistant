"""Gradio Dashboard for real-time monitoring."""

import gradio as gr

from nanobot.agent.monitor import MonitorCollector

# CSS 样式 - 简洁专业风格
DASHBOARD_CSS = """
.dashboard-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}
.dashboard-header h1 {
    color: white !important;
    margin: 0 !important;
    font-size: 24px !important;
}
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 14px;
    background: rgba(255,255,255,0.2);
}
.stat-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border: 1px solid #e8e8e8;
    text-align: center;
}
.stat-card:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    transform: translateY(-2px);
    transition: all 0.2s ease;
}
.stat-value {
    font-size: 32px;
    font-weight: 600;
    color: #1a1a2e;
    margin: 8px 0;
}
.stat-label {
    font-size: 13px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.token-bar {
    background: #f3f4f6;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 16px 0;
}
.event-item {
    padding: 10px 16px;
    border-radius: 8px;
    margin: 6px 0;
    background: white;
    border-left: 4px solid #10b981;
    font-family: 'SF Mono', monospace;
    font-size: 13px;
}
.event-item.error {
    border-left-color: #ef4444;
}
.event-time {
    color: #9ca3af;
    font-size: 11px;
    margin-right: 8px;
}
"""


def create_dashboard():
    """创建 Dashboard - 简洁专业风格"""

    with gr.Blocks(title="nanobot Monitor") as demo:
        # 顶部 Header
        with gr.Row():
            with gr.Column(scale=4):
                gr.HTML("""
                <div class="dashboard-header">
                    <h1>🐈 nanobot Monitor</h1>
                </div>
                """)
            with gr.Column(scale=1):
                status_display = gr.HTML("<span class='status-badge'>● Loading...</span>")

        # 统计卡片行
        with gr.Row():
            total_card = gr.HTML("""
            <div class="stat-card">
                <div class="stat-label">Total Events</div>
                <div class="stat-value" id="total-value">-</div>
            </div>
            """)
            llm_card = gr.HTML("""
            <div class="stat-card">
                <div class="stat-label">LLM Calls</div>
                <div class="stat-value" id="llm-value">-</div>
            </div>
            """)
            tool_card = gr.HTML("""
            <div class="stat-card">
                <div class="stat-label">Tool Calls</div>
                <div class="stat-value" id="tool-value">-</div>
            </div>
            """)
            latency_card = gr.HTML("""
            <div class="stat-card">
                <div class="stat-label">Avg Latency</div>
                <div class="stat-value" id="latency-value">-</div>
            </div>
            """)

        # Token 统计条
        token_display = gr.HTML("""
        <div class="token-bar">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 13px; color: #6b7280; font-weight: 500;">Token Usage</span>
                <span style="font-size: 18px; font-weight: 600; color: #1a1a2e;" id="token-total">0</span>
            </div>
        </div>
        """)

        # 最近事件
        gr.HTML("<div style='font-size: 14px; font-weight: 600; color: #374151; margin: 16px 0 8px 0;'>Recent Events</div>")
        events_container = gr.HTML("<div id='events-list'>No events yet</div>")

        def update_display():
            """更新所有显示组件"""
            import time

            mc = MonitorCollector.instance()
            stats = mc.get_stats()
            events = mc.get_events(limit=10)

            total = stats.get("total_events", 0)
            llm = stats.get("llm_calls", 0)
            tool = stats.get("tool_calls", 0)
            latency = round(stats.get("avg_llm_latency_ms", 0), 1)
            tokens = stats.get("total_tokens", 0)

            # 状态
            if not events:
                status_html = "<span class='status-badge'>● Waiting for events...</span>"
            else:
                age = time.time() - events[-1].timestamp
                if age < 10:
                    status_html = "<span class='status-badge' style='background: #d1fae5; color: #065f46;'>🟢 Running</span>"
                elif age < 60:
                    status_html = "<span class='status-badge' style='background: #fef3c7; color: #92400e;'>🟡 Idle</span>"
                else:
                    status_html = "<span class='status-badge' style='background: #fee2e2; color: #991b1b;'>🔴 Inactive</span>"

            # Token 格式化
            token_str = f"{tokens:,}" if tokens else "0"

            # 事件列表
            events_html = ""
            for e in reversed(events):
                icon = "✓" if e.success else "✗"
                cls = "" if e.success else "error"
                duration = f"{e.duration_ms:.0f}ms" if e.duration_ms else "-"
                tool_name = e.metadata.get('tool_name', e.metadata.get('model', '-'))
                age = int(time.time() - e.timestamp)
                age_str = f"{age}s ago" if age < 60 else f"{age//60}m ago"
                events_html += f"""
                <div class="event-item {cls}">
                    <span class="event-time">{age_str}</span>
                    <span style="color: #10b981;">{icon}</span>
                    <span style="color: #6366f1; font-weight: 500;">{e.event_type.value}</span>
                    <span style="color: #9ca3af;">|</span>
                    <span style="color: #f59e0b;">{duration}</span>
                    <span style="color: #9ca3af;">|</span>
                    <span style="color: #374151;">{tool_name}</span>
                </div>
                """

            no_events = '<div style="color: #9ca3af; padding: 20px; text-align: center;">No events yet</div>'

            return (
                status_html,
                f"<div class='stat-card'><div class='stat-label'>Total Events</div><div class='stat-value'>{total:,}</div></div>",
                f"<div class='stat-card'><div class='stat-label'>LLM Calls</div><div class='stat-value'>{llm:,}</div></div>",
                f"<div class='stat-card'><div class='stat-label'>Tool Calls</div><div class='stat-value'>{tool:,}</div></div>",
                f"<div class='stat-card'><div class='stat-label'>Avg Latency</div><div class='stat-value'>{latency}ms</div></div>",
                f"<div class='token-bar'><div style='display: flex; justify-content: space-between; align-items: center;'><span style='font-size: 13px; color: #6b7280; font-weight: 500;'>Token Usage</span><span style='font-size: 18px; font-weight: 600; color: #1a1a2e;'>{token_str}</span></div></div>",
                f"<div id='events-list'>{events_html or no_events}</div>",
            )

        # 定时刷新
        timer = gr.Timer(value=2, active=True)
        timer.tick(
            fn=update_display,
            outputs=[status_display, total_card, llm_card, tool_card, latency_card, token_display, events_container],
        )

    return demo


def launch_dashboard(server_name: str = "0.0.0.0", port: int = 7860):
    """启动 Dashboard"""
    demo = create_dashboard()
    demo.launch(
        server_name=server_name,
        server_port=port,
        prevent_thread_lock=True,
        css=DASHBOARD_CSS,
    )
