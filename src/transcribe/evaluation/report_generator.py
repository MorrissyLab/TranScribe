"""
report_generator.py
====================
Generates a single-file interactive HTML evaluation report from TranScribe
eval output directories. Uses base.html + report.css templates and injects
computed data via simple placeholder replacement.
"""
import json
from pathlib import Path
from typing import Optional, List
from transcribe.config import logger


# ---------------------------------------------------------------------------
# Helpers – badges / confidence
# ---------------------------------------------------------------------------

def _acc_cls(v: float) -> str:
    return "badge-high" if v >= 0.8 else ("badge-med" if v >= 0.5 else "badge-low")


def _conf_badge(conf) -> str:
    if isinstance(conf, str):
        c = conf.lower()
        if c == "high":   return '<span class="badge badge-high" style="font-size:.75rem;padding:2px 9px">High</span>'
        if c == "medium": return '<span class="badge badge-med"  style="font-size:.75rem;padding:2px 9px">Medium</span>'
        return '<span class="badge badge-low" style="font-size:.75rem;padding:2px 9px">Low</span>'
    if isinstance(conf, (int, float)):
        c = float(conf)
        cl = _acc_cls(c)
        return f'<span class="badge {cl}" style="font-size:.75rem;padding:2px 9px">{c*100:.0f}%</span>'
    return '<span class="badge badge-low" style="font-size:.75rem;padding:2px 9px">Unknown</span>'


def _conf_bar_width(conf) -> int:
    if isinstance(conf, str):
        return {"high": 100, "medium": 60}.get(conf.lower(), 30)
    if isinstance(conf, (int, float)):
        return int(min(100, max(0, float(conf) * 100)))
    return 0


# ---------------------------------------------------------------------------
# Load one dataset from a subdirectory
# ---------------------------------------------------------------------------

def _load_dataset(item: Path) -> Optional[dict]:
    report_path = item / "eval_report.json"
    if not report_path.exists():
        return None

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    traces: dict = {}
    trace_path = item / "eval_communication_trace.json"
    if trace_path.exists():
        try:
            with open(trace_path, "r", encoding="utf-8") as f:
                traces = json.load(f)
        except Exception:
            pass

    umap_path    = item / "umap_predicted.png"
    spatial_path = item / "spatial_predicted.png"
    metrics      = data.get("metrics", {})

    return {
        "run_id":          item.name.replace(" ", "_").lower(),
        "name":            data.get("dataset_name", item.name),
        "accuracy":        metrics.get("accuracy", 0.0),
        "eval_accuracy":   metrics.get("evaluator_accuracy", metrics.get("accuracy", 0.0)),
        "mapping":         data.get("cluster_mapping", {}),
        "degs":            data.get("cluster_degs", {}),
        "raw":             data.get("raw_results", {}),
        "traces":          traces,
        "evaluator_results": data.get("evaluator_results", {}),
        "cluster_colors":  data.get("cluster_colors", {}),
        "umap_path":       f"{item.name}/umap_predicted.png" if umap_path.exists() else None,
        "spatial_path":    f"{item.name}/spatial_predicted.png" if spatial_path.exists() else None,
        "metadata":        data.get("metadata", {}),
    }


# ---------------------------------------------------------------------------
# HTML fragment builders
# ---------------------------------------------------------------------------

def _sidebar_link(ds: dict) -> str:
    model = ds["metadata"].get("model_name", "Unknown")
    return (
        f'<div class="tab-link" id="link_{ds["run_id"]}" '
        f'onclick="showTab(\'tab_exp_{ds["run_id"]}\', this)" '
        f'style="line-height:1.3">'
        f'{ds["name"]}<br>'
        f'<span style="font-size:0.7rem;color:#6e7681;font-weight:400">{model}</span>'
        f'</div>\n'
    )


def _summary_tab(datasets: List[dict]) -> str:
    has_eval   = any(d["metadata"].get("is_eval", True) for d in datasets)
    all_infer  = not has_eval
    total_dur  = sum(d["metadata"].get("duration_seconds", 0) for d in datasets)
    avg_dur    = total_dur / len(datasets) if datasets else 0

    rows = ""
    for ds in datasets:
        is_eval  = ds["metadata"].get("is_eval", True)
        type_str = "Evaluation" if is_eval else "Inference"
        model    = ds["metadata"].get("model_name", "Unknown")
        run_id   = ds["run_id"]
        rows += (
            f'<tr class="clickable-row" onclick="document.getElementById(\'link_{run_id}\').click()">'
            f'<td>{ds["name"]}</td>'
            f'<td><code style="color:var(--blue)">{model}</code></td>'
            f'<td>{len(ds["mapping"])}</td>'
            f'<td>{type_str}</td></tr>\n'
        )

    # Accuracy badges – only show when there's at least one eval dataset
    badges_html = ""
    if not all_infer:
        badges_html = """
            <div style="display:flex;gap:30px;justify-content:center;margin-top:16px;flex-wrap:wrap">
                <div style="text-align:center">
                    <div style="font-size:.85rem;color:#8b949e;margin-bottom:6px">Avg Naive Accuracy</div>
                    <div id="badge_acc" class="badge badge-high" style="font-size:1.7rem;padding:8px 24px">—</div>
                </div>
                <div style="text-align:center">
                    <div style="font-size:.85rem;color:var(--purple);margin-bottom:6px">Avg Biological Accuracy</div>
                    <div id="badge_bio" class="badge badge-high" style="font-size:1.7rem;padding:8px 24px;border:2px solid var(--purple);background:rgba(188,140,255,.08)">—</div>
                </div>
            </div>"""

    barplot_html = ""
    if has_eval:
        barplot_html = """
        <div style="margin-top:32px;background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px">
            <div style="display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--border);padding-bottom:12px;margin-bottom:16px;flex-wrap:wrap;gap:10px">
                <h3 style="margin:0">Biological Accuracy — <span id="barplot_lbl" style="color:var(--blue)"></span></h3>
                <div style="display:flex;gap:7px;align-items:center">
                    <span style="font-size:0.78rem;color:#8b949e">View by:</span>
                    <button class="btn-toggle filter-btn" onclick="setPlotMode('model',this)">Model</button>
                    <button class="btn-toggle filter-btn" onclick="setPlotMode('dataset',this)">Dataset</button>
                    <button class="btn-toggle active filter-btn" onclick="setPlotMode('all',this)">All</button>
                </div>
            </div>
            <div id="sec_filters" style="display:flex;gap:8px;justify-content:center;margin-bottom:14px;flex-wrap:wrap"></div>
            <div id="bar_area" class="bar-area"></div>
            <div id="bar_legend" style="display:none;justify-content:center;gap:10px;padding:10px;flex-wrap:wrap"></div>
        </div>"""

    heading = "Overall Inference Summary" if all_infer else "Overall Evaluation Summary"
    run_label = "Runs" if all_infer else "Experiments"

    perf = f"""
        <div style="margin-top:26px;background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px">
            <h3 style="border-bottom:1px solid var(--border);padding-bottom:11px;margin-bottom:16px">Run Performance</h3>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px">
                <div class="stat-tile"><div class="stat-label">Avg Duration</div>
                    <div class="stat-value" style="color:var(--blue)">{avg_dur:.1f}s</div></div>
                <div class="stat-tile"><div class="stat-label">Total Run Time</div>
                    <div class="stat-value" style="color:var(--purple)">{total_dur:.1f}s</div></div>
                <div class="stat-tile"><div class="stat-label">{run_label}</div>
                    <div class="stat-value">{len(datasets)}</div></div>
            </div>
        </div>"""

    return f"""
    <div id="tab_summary" class="tab-pane">
        <div style="text-align:center;margin-bottom:28px">
            <h1>{heading}</h1>
            {badges_html}
        </div>

        {barplot_html}

        <table class="data-table" style="margin-top:28px">
            <thead><tr><th>Dataset</th><th>Model</th><th>Clusters</th><th>Type</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>

        {perf}
    </div>
    """


def _experiment_tab(ds: dict, all_traces: dict, all_eval: dict, all_ann: dict) -> str:
    run_id   = ds["run_id"]
    is_eval  = ds["metadata"].get("is_eval", True)
    meta     = ds["metadata"]
    acc      = ds["accuracy"]
    eval_acc = ds["eval_accuracy"]
    dur      = meta.get("duration_seconds", 0)
    st_raw   = meta.get("start_time", "")
    start_t  = st_raw.split("T")[1].split(".")[0] if "T" in st_raw else "N/A"
    num_tries = meta.get("num_tries", 1)
    model    = meta.get("model_name", "N/A")

    acc_html = ""
    if is_eval:
        ac  = _acc_cls(acc)
        eac = _acc_cls(eval_acc)
        acc_html = (
            f'<span class="badge {ac}" style="padding:4px 13px;font-size:.83rem">Naive: {acc*100:.1f}%</span>'
            f'&nbsp;<span class="badge {eac}" style="padding:4px 13px;font-size:.83rem;border:1px solid var(--purple);background:rgba(188,140,255,.08)">Bio: {eval_acc*100:.1f}%</span>'
        )

    tag_parts = [p for p in [
        meta.get("organism"), meta.get("tissue"), meta.get("disease"),
        "[TOY]" if meta.get("is_toy") else None,
        "[INFERENCE]" if not is_eval else None,
        "[SPATIAL]" if meta.get("modality") == "spatial" else None,
    ] if p]
    tag_line = " · ".join(tag_parts)

    # Plot
    plot_html = ""
    is_factorized = meta.get("modality") == "factorized"
    is_spatial = meta.get("modality") == "spatial" or (is_factorized and meta.get("factorized_type") == "spatial")
    
    # Global plot if not factorized
    if not is_factorized:
        plot_path = ds.get("spatial_path") if is_spatial else ds.get("umap_path")
        plot_label = "Spatial Plot" if is_spatial else "Annotated UMAP"
        
        if plot_path:
            plot_html = f'<div class="umap-wrap"><h3>{plot_label}</h3><img src="{plot_path}" alt="{plot_label} {ds["name"]}"></div>'

    # Cluster rows + cards
    tbl_rows  = ""
    card_html = ""

    for cid, m in ds["mapping"].items():
        true_lbl = m.get("true", "Unknown")
        pred_lbl = m.get("pred", "Error")
        ev_res   = ds["evaluator_results"].get(cid, {})
        is_match = ev_res.get("is_match", False)
        c_color  = ds["cluster_colors"].get(cid, "#ffffff")
        dot      = f'<span class="color-dot" style="background:{c_color}"></span>'

        raw_ann  = ds["raw"].get(cid, {})
        if isinstance(raw_ann, str):
            confidence, reasoning = "low", raw_ann
        elif isinstance(raw_ann, dict):
            confidence = raw_ann.get("confidence", "low")
            reasoning  = raw_ann.get("reasoning_chain", "N/A")
        else:
            confidence, reasoning = "low", "N/A"

        key = f"{run_id}_{cid}"
        all_traces[key] = ds["traces"].get(cid, [])
        all_eval[key]   = ev_res
        all_ann[key]    = reasoning

        conf_badge = _conf_badge(confidence)
        bar_w      = _conf_bar_width(confidence)
        genes5     = ds["degs"].get(cid, [])[:5]
        genes_html = "".join(f'<span class="gene-chip">{g}</span>' for g in genes5)
        pred_color = "white" if (is_match or not is_eval) else "#f85149"

        status_td = f'<td style="text-align:center;font-size:1.1rem">{"✅" if is_match else "❌"}</td>' if is_eval else ""
        truth_td  = f"<td>{true_lbl}</td>" if is_eval else ""
        ann_btn   = f'<button class="btn" onclick="openAnnotation(\'{key}\')">Ann. Reason</button>'
        trace_btn = f'<button class="btn" onclick="openTrace(\'{key}\',\'Cluster {cid} Trace\')">Trace</button>'
        bio_btn   = f'<button class="btn btn-purple" onclick="openBioMatch(\'{key}\')">Bio-Match</button>' if is_eval else ""

        tbl_rows += f"""
            <tr>
                {status_td}
                <td><code>{cid}</code></td>
                {truth_td}
                <td style="color:{pred_color}">{dot}{pred_lbl}</td>
                <td>{conf_badge}</td>
                <td>{genes_html}</td>
                <td style="display:flex;flex-direction:column;gap:5px">{ann_btn}{trace_btn}{bio_btn}</td>
            </tr>"""

        # Card
        genes_all = "".join(f'<span class="gene-chip">{g}</span>' for g in ds["degs"].get(cid, []))
        match_cls  = "tag-match" if is_match else "tag-miss"
        pred_tcls  = f"tag {match_cls}" if is_eval else "tag tag-plain"
        emo        = ("✅ " if is_match else "❌ ") if is_eval else ""
        truth_tag  = f'<div class="tag tag-true"><span>True:</span><span>{true_lbl}</span></div>' if is_eval else ""

        factor_img_html = ""
        if is_factorized:
            f_img = ds.get("run_id","") + f"/plots/factor_{cid}_usage.png"
            factor_img_html = f'<div style="text-align:center;margin-bottom:10px;"><img src="{f_img}" alt="Factor {cid} Usage Plot" style="max-width:100%;border-radius:6px;border:1px solid var(--border);"></div>'

        card_html += f"""
            <div class="cluster-card">
                {factor_img_html}
                <div style="display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--border);padding-bottom:9px">
                    <span style="font-size:1.05rem;font-weight:800;color:#fff">{emo}Cluster {cid}</span>
                    <span>{conf_badge}</span>
                </div>
                <div class="tag-row">
                    {truth_tag}
                    <div class="{pred_tcls}"><span>Pred:</span><span>{dot}{pred_lbl}</span></div>
                </div>
                <div class="conf-bar-wrap"><div class="conf-bar" style="width:{bar_w}%"></div></div>
                <div class="reasoning"><strong>Reason:</strong><br>{reasoning}</div>
                <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:auto">
                    <button class="btn" onclick="toggleElement('genes_{key}')">DEGs</button>
                    {trace_btn}{bio_btn}
                </div>
                <div id="genes_{key}" class="genes-list hidden">{genes_all}</div>
            </div>"""

    status_th = "<th>✓</th>" if is_eval else ""
    truth_th  = "<th>Ground Truth</th>" if is_eval else ""
    thead     = f"<tr>{status_th}<th>C</th>{truth_th}<th>Predicted</th><th>Conf.</th><th>Top DEGs</th><th>Actions</th></tr>"

    return f"""
    <div id="tab_exp_{run_id}" class="tab-pane">
        <div style="text-align:center;margin-bottom:22px">
            <h1>{ds['name']} <span style="font-size:1.05rem;color:#8b949e">({model})</span></h1>
            <div style="font-size:.76rem;color:#6e7681;margin:6px 0">{tag_line}</div>
            <div style="font-size:.8rem;margin-bottom:10px">Path: <code style="color:#8b949e">{meta.get('data_path','N/A')}</code></div>
        </div>

        <div class="stats-row">
            <div class="stat-tile"><div class="stat-label">Duration</div><div class="stat-value">{dur:.1f}s</div></div>
            <div class="stat-tile"><div class="stat-label">Start</div><div class="stat-value">{start_t}</div></div>
            <div class="stat-tile"><div class="stat-label">Tries</div><div class="stat-value">{num_tries}</div></div>
            <div class="stat-tile" style="display:flex;flex-direction:column;gap:6px;align-items:center">{acc_html}</div>
        </div>

        <div class="experiment-grid">
            <div class="experiment-left">
                {plot_html}
            </div>
            <div class="experiment-right">
                <div class="view-toggle-row">
                    <button class="btn-toggle active vtbtn_{run_id}" onclick="switchView('table','{run_id}')">Table View</button>
                    <button class="btn-toggle vtbtn_{run_id}" onclick="switchView('cards','{run_id}')">Card View</button>
                </div>

                <div id="tbl_{run_id}" style="background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px">
                    <h3 style="border-bottom:1px solid var(--border);padding-bottom:10px;margin-bottom:14px">Cluster Details</h3>
                    <table class="data-table">
                        <thead>{thead}</thead>
                        <tbody>{tbl_rows}</tbody>
                    </table>
                </div>

                <div id="cards_{run_id}" class="cluster-grid" style="display:none">{card_html}</div>
            </div>
        </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_html_report(eval_dir: str):
    """
    Scans eval_dir for subdirectories containing eval_report.json and
    generates eval_dir/index.html using the HTML + CSS templates.
    """
    base_dir = Path(eval_dir)
    if not base_dir.exists() or not base_dir.is_dir():
        logger.error(f"Cannot generate report: {eval_dir} is not a valid directory.")
        return

    tpl_dir = Path(__file__).parent / "templates"
    base_html = tpl_dir / "base.html"
    css_file  = tpl_dir / "report.css"

    if not base_html.exists():
        logger.error(f"Template not found: {base_html}"); return

    template = base_html.read_text(encoding="utf-8")
    css_content = css_file.read_text(encoding="utf-8") if css_file.exists() else ""

    # Embed CSS inline so the report is fully self-contained
    css_tag = f"<style>\n{css_content}\n</style>" if css_content else ""

    # Collect datasets
    datasets: List[dict] = []
    for item in sorted(base_dir.iterdir()):
        if item.is_dir():
            ds = _load_dataset(item)
            if ds:
                datasets.append(ds)

    if not datasets:
        logger.warning("No eval_report.json files found – report not generated.")
        return

    all_traces: dict = {}
    all_eval:   dict = {}
    all_ann:    dict = {}

    sidebar_links  = "".join(_sidebar_link(ds) for ds in datasets)
    summary_pane   = _summary_tab(datasets)
    experiment_tabs = "".join(_experiment_tab(ds, all_traces, all_eval, all_ann) for ds in datasets)
    tab_contents   = summary_pane + experiment_tabs

    js_experiments = json.dumps([
        {
            "run_id":        ds["run_id"],
            "name":          ds["name"],
            "model":         ds["metadata"].get("model_name", "Unknown"),
            "accuracy":      ds["accuracy"],
            "eval_accuracy": ds["eval_accuracy"],
            "is_eval":       ds["metadata"].get("is_eval", True),
        }
        for ds in datasets
    ], ensure_ascii=True)

    # Escape </script> in JSON blobs to prevent premature script tag closing
    def _safe_json(obj: dict) -> str:
        return json.dumps(obj, ensure_ascii=True).replace("</", "<\\/")

    html = (
        template
        .replace("<!--CSS_LINK-->",       css_tag)
        .replace("<!--SIDEBAR_LINKS-->",  sidebar_links)
        .replace("<!--TAB_CONTENTS-->",   tab_contents)
        .replace("<!--JS_EXPERIMENTS-->", js_experiments)
        .replace("<!--JS_TRACES-->",      _safe_json(all_traces))
        .replace("<!--JS_EVAL_DATA-->",   _safe_json(all_eval))
        .replace("<!--JS_ANN_DATA-->",    _safe_json(all_ann))
    )

    out_path = base_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info(f"HTML report written to {out_path}")
    print(f"[TranScribe] Report generated: {out_path}")
