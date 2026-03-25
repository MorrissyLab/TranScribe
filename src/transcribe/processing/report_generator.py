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
from transcribe.tools.exporter import export_summary_to_csv, export_experiment_degs_to_csv, export_batch_degs_to_excel


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
        "run_id":          item.name.replace(" ", "_").replace(".", "_").replace("-", "_").lower(),
        "name":            data.get("dataset_name", item.name),
        "llmaj_accuracy":  metrics.get("llmaj_accuracy", metrics.get("inference_accuracy", metrics.get("accuracy", 0.0))),
        "mapping":         data.get("cluster_mapping", {}),
        "degs":            data.get("cluster_degs", {}),
        "raw":             data.get("raw_results", {}),
        "traces":          traces,
        "inference_results": data.get("inference_results", {}),
        "cluster_colors":  data.get("cluster_colors", {}),
        "umap_path":       f"{item.name}/umap_predicted.png" if umap_path.exists() else None,
        "spatial_path":    f"{item.name}/spatial_predicted.png" if spatial_path.exists() else None,
        "metadata":        data.get("metadata", {}),
        "hierarchical_summary": data.get("hierarchical_summary", {}),
        "pathway_activity": {cid: r.get("pathway_activity") for cid, r in data.get("raw_results", {}).items() if isinstance(r, dict)},
        "dir_name":        item.name
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
    has_eval   = any(d["metadata"].get("is_eval", False) for d in datasets)
    all_infer  = not has_eval
    total_dur  = sum(d["metadata"].get("duration_seconds", 0) for d in datasets)
    avg_dur    = total_dur / len(datasets) if datasets else 0
    
    # Calculate global average accuracy
    eval_ds = [d for d in datasets if d["metadata"].get("is_eval", False)]
    avg_acc = sum(d["llmaj_accuracy"] for d in eval_ds) / len(eval_ds) if eval_ds else 0.0
    acc_color = "badge-high" if avg_acc >= 0.8 else ("badge-med" if avg_acc >= 0.5 else "badge-low")

    rows = ""
    for ds in datasets:
        is_eval  = ds["metadata"].get("is_eval", False)
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
        badges_html = f"""
            <div style="display:flex;gap:40px;justify-content:center;margin-top:20px;flex-wrap:wrap">
                <div style="text-align:center">
                    <div style="font-size:0.8rem;color:#8b949e;margin-bottom:6px;text-transform:uppercase;letter-spacing:1px">Avg Biological Accuracy </div>
                    <div id="badge_bio" class="badge {acc_color}" style="font-size:2.2rem;padding:12px 32px; border-radius:12px">{avg_acc*100:.1f}%</div>
                </div>
                <div style="text-align:center">
                    <div style="font-size:0.8rem;color:#8b949e;margin-bottom:6px;text-transform:uppercase;letter-spacing:1px">Total Experiments</div>
                    <div class="badge" style="background:rgba(255,255,255,0.1);font-size:2.2rem;padding:12px 32px; border-radius:12px">{len(datasets)}</div>
                </div>
            </div>"""

    barplot_html = ""
    if has_eval:
        barplot_html = f"""
        <div style="margin-top:40px;background:var(--card);border:1px solid var(--border);border-radius:16px;padding:24px;box-shadow:0 8px 32px rgba(0,0,0,0.3)">
            <div style="display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--border);padding-bottom:14px;margin-bottom:20px;flex-wrap:wrap;gap:15px">
                <h3 style="margin:0; font-size:1.4rem">Biological Performance — <span id="barplot_lbl" style="color:var(--blue)"></span></h3>
                <div style="display:flex;gap:10px;align-items:center">
                    <span style="font-size:0.8rem;color:#8b949e">Group By:</span>
                    <button class="btn-toggle filter-btn" onclick="setPlotMode('model',this)">Model</button>
                    <button class="btn-toggle filter-btn" onclick="setPlotMode('dataset',this)">Dataset</button>
                    <button class="btn-toggle active filter-btn" onclick="setPlotMode('all',this)">Overview</button>
                </div>
            </div>
            <div id="sec_filters" style="display:flex;gap:10px;justify-content:center;margin-bottom:20px;flex-wrap:wrap"></div>
            <div id="bar_area" class="bar-area" style="height:250px"></div>
            <div id="bar_legend" style="display:none;justify-content:center;gap:15px;padding:15px;flex-wrap:wrap"></div>
        </div>"""

    heading = "Multi-Agent Evaluation Dashboard" if not all_infer else "Multi-Agent Annotation Dashboard"
    perf = f"""
        <div style="margin-top:32px;background:var(--card);border:1px solid var(--border);border-radius:16px;padding:24px; display:grid; grid-template-columns: 1fr 2fr; gap:24px">
            <div>
                <h3 style="border-bottom:1px solid var(--border);padding-bottom:12px;margin-bottom:16px; font-size:1.1rem">Global Stats</h3>
                <div style="display:grid;grid-template-columns:1fr;gap:12px">
                    <div class="stat-tile"><div class="stat-label">Total Execution Time</div>
                        <div class="stat-value" style="color:var(--purple)">{total_dur:.2f}s</div></div>
                    <div class="stat-tile"><div class="stat-label">Avg Duration / Exp</div>
                        <div class="stat-value" style="color:var(--blue)">{avg_dur:.2f}s</div></div>
                </div>
            </div>
            <div>
                <h3 style="border-bottom:1px solid var(--border);padding-bottom:12px;margin-bottom:16px; font-size:1.1rem">Experience Inventory</h3>
                <table class="data-table">
                    <thead><tr><th>Dataset Name</th><th>Language Model</th><th>Clusters</th><th>Workflow</th></tr></thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
        </div>"""

    return f"""
    <div id="tab_summary" class="tab-pane">
        <div style="text-align:center;margin-bottom:34px">
            <div style="display:flex; justify-content:center; align-items:center; gap:25px; margin-bottom:15px">
                <h1 style="margin:0; font-size:2.5rem; letter-spacing:-1px">{heading}</h1>
                <a href="summary_results.csv" class="btn btn-purple" style="text-decoration:none; padding:10px 20px; font-size:1rem; display:inline-flex; align-items:center; gap:8px">
                    <span>📊</span> Export Full Results
                </a>
            </div>
            {badges_html}
        </div>

        {barplot_html}
        {perf}
    </div>
    """


def _experiment_tab(ds: dict, all_traces: dict, all_eval: dict, all_ann: dict, all_pathway: dict, all_zeta: dict) -> str:
    run_id   = ds["run_id"]
    dir_name = ds.get("dir_name", run_id)
    is_eval  = ds["metadata"].get("is_eval", False)
    meta     = ds["metadata"]
    llmaj_acc = ds["llmaj_accuracy"]
    dur      = meta.get("duration_seconds", 0)
    st_raw   = meta.get("start_time", "")
    start_t  = st_raw.split("T")[1].split(".")[0] if "T" in st_raw else "N/A"
    num_tries = meta.get("num_tries", 1)
    model    = meta.get("model_name", "N/A")

    acc_html = ""
    if is_eval:
        eac = _acc_cls(llmaj_acc)
        acc_html = f'<span class="badge {eac}" style="padding:4px 13px;font-size:.83rem">LLMaJ: {llmaj_acc*100:.1f}%</span>'

    tag_parts = [p for p in [
        meta.get("organism"), meta.get("tissue"), meta.get("disease"),
        "[TOY]" if meta.get("is_toy") else None,
        "[INFERENCE]" if not is_eval else None,
        "[SPATIAL]" if meta.get("modality") == "spatial" else None,
    ] if p]
    tag_line = " · ".join(tag_parts)

    # Singleton Warning
    singleton_warning = ""
    singletons = meta.get("singleton_clusters", [])
    if singletons:
        singletons_str = ", ".join([f"<code>{s}</code>" for s in singletons])
        singleton_warning = f"""
        <div style="background:rgba(255,165,0,0.1); border:1px solid rgba(255,165,0,0.3); border-radius:8px; padding:12px 16px; margin-bottom:20px; color:#ffca28; display:flex; align-items:center; gap:12px">
            <span style="font-size:1.5rem">⚠️</span>
            <div style="font-size:0.85rem; line-height:1.4">
                <strong>Singleton Cluster Warning:</strong> Cluster(s) {singletons_str} have fewer than 2 cells and were 
                excluded from differential expression (DEG) computation. They are still included in the final annotation results.
            </div>
        </div>"""

    # Plot
    plot_html = ""
    is_factorized = meta.get("modality") == "factorized"
    
    # Check for spatial files explicitly
    has_spatial_file = ds.get("spatial_path") is not None
    has_umap_file = ds.get("umap_path") is not None
    
    # Determine if we should treat this as a spatial display
    # (either modality says so, or we ONLY have a spatial file)
    is_spatial = meta.get("modality") == "spatial" or (is_factorized and meta.get("factorized_type") == "spatial")
    if not is_spatial and has_spatial_file and not has_umap_file:
        is_spatial = True

    # Global plot if not factorized
    if not is_factorized:
        # Prefer spatial if in spatial mode and file exists, otherwise fallback to umap if it exists
        if is_spatial and has_spatial_file:
            plot_path = ds.get("spatial_path")
            plot_label = "Spatial Plot"
        elif has_umap_file:
            plot_path = ds.get("umap_path")
            plot_label = "Annotated UMAP"
        else:
            plot_path = None
            plot_label = ""
            
        if plot_path:
            plot_html = f'<div class="umap-wrap"><h3>{plot_label}</h3><img src="{plot_path}" alt="{plot_label} {ds["name"]}"></div>'

    # Eta Hierarchical Summary
    eta_html = ""
    eta_summary = ds.get("hierarchical_summary")
    if eta_summary and eta_summary.get("groups"):
        groups_html = ""
        for g in eta_summary.get("groups", []):
            cl_list = g.get('member_clusters', [])
            cl_str = ", ".join([f"<code>{c}</code>" for c in cl_list])
            groups_html += f"<li><strong>{g.get('group_name')}</strong> (Parent: {g.get('parent_group')}): Clusters [{cl_str}] &mdash; <span style='color:#8b949e'>{g.get('description')}</span></li>"
        
        gamma_trace_key = f"{run_id}___GLOBAL_GAMMA__"
        all_traces[gamma_trace_key] = ds["traces"].get("__GLOBAL_GAMMA__", [])

        eta_html = f"""
        <div style="font-size:0.9rem;line-height:1.5;margin-bottom:12px;">{eta_summary.get("narrative_summary", "")}</div>
        <ul style="font-size:0.85rem;line-height:1.5;margin:0;padding-left:20px;">
            {groups_html}
        </ul>
        """

    # Cluster rows + cards
    tbl_rows  = ""
    card_html = ""

    for cid, m in ds["mapping"].items():
        true_lbl = m.get("true", "Unknown")
        pred_lbl = m.get("pred", "Error")
        ev_res   = ds["inference_results"].get(cid, {})
        is_match = ev_res.get("is_match", False)
        c_color  = ds["cluster_colors"].get(cid, "#ffffff")
        dot      = f'<span class="color-dot" style="background:{c_color}"></span>'

        raw_ann  = ds["raw"].get(cid, {})
        if isinstance(raw_ann, str):
            confidence, reasoning = "low", raw_ann
            zeta_data, pathway_data = None, None
        elif isinstance(raw_ann, dict):
            confidence = raw_ann.get("confidence", "low")
            reasoning  = raw_ann.get("reasoning_chain", "N/A")
            zeta_data  = raw_ann.get("confidence_assessment")
            pathway_data = raw_ann.get("pathway_activity")
        else:
            confidence, reasoning, zeta_data, pathway_data = "low", "N/A", None, None

        key = f"{run_id}_{cid}"
        all_traces[key] = ds["traces"].get(cid, [])
        all_eval[key]   = ev_res
        all_ann[key]    = reasoning
        all_pathway[key]= pathway_data
        all_zeta[key]   = zeta_data

        display_conf = confidence
        if zeta_data and "overlap_score" in zeta_data:
            display_conf = zeta_data["overlap_score"]
            
        conf_badge = _conf_badge(display_conf)
        bar_w      = _conf_bar_width(display_conf)
        
        # Wrap badge to be clickable for Zeta details
        conf_badge_clickable = f'<div style="cursor:pointer; display:inline-block; transition: opacity 0.2s" onclick="openZeta(\'{key}\')" onmouseover="this.style.opacity=0.8" onmouseout="this.style.opacity=1" title="Click to view Zeta reasoning">{conf_badge}</div>'
        genes5     = ds["degs"].get(cid, [])[:5]
        genes_html = "".join(f'<span class="gene-chip">{g}</span>' for g in genes5)
        pred_color = "white" if (is_match or not is_eval) else "#f85149"

        status_td = f'<td style="text-align:center;font-size:1.1rem">{"✅" if is_match else "❌"}</td>' if is_eval else ""
        truth_td  = f"<td>{true_lbl}</td>" if is_eval else ""
        ann_btn   = f'<button class="btn" onclick="openAnnotation(\'{key}\')">Ann. Reason</button>'
        trace_btn = f'<button class="btn" onclick="openTrace(\'{key}\')">Trace</button>'
        bio_btn   = f'<button class="btn btn-purple" onclick="openBioMatch(\'{key}\')">Bio-Match</button>' if is_eval else ""
        pathway_btn = f'<button class="btn btn-purple" style="background:rgba(210,153,34,0.15);color:#d29922;border:1px solid #d29922" onclick="openPathway(\'{key}\')">Pathway Activity</button>' if pathway_data else ""

        tbl_rows += f"""
            <tr>
                {status_td}
                <td><code>{cid}</code></td>
                {truth_td}
                <td style="color:{pred_color}">{dot}{pred_lbl}</td>
                <td>{conf_badge_clickable}</td>
                <td>{genes_html}</td>
                <td style="display:flex;flex-direction:column;gap:5px">{ann_btn}{pathway_btn}{trace_btn}{bio_btn}</td>
            </tr>"""

        # Card
        genes_all = "".join(f'<span class="gene-chip">{g}</span>' for g in ds["degs"].get(cid, []))
        match_cls  = "tag-match" if is_match else "tag-miss"
        pred_tcls  = f"tag {match_cls}" if is_eval else "tag tag-plain"
        emo        = ("✅ " if is_match else "❌ ") if is_eval else ""
        truth_tag  = f'<div class="tag tag-true"><span>True:</span><span>{true_lbl}</span></div>' if is_eval else ""

        factor_img_html = ""
        if is_factorized:
            f_img_rel = f"{dir_name}/plots/factor_{cid}_usage.png"
            # We don't have easy access to filesystem here to check existence for every cluster, 
            # but we can assume if it's factorized, we want to try showing them.
            # However, to be safe and avoid broken images if they really are missing:
            factor_img_html = f"""
            <div style="width:100%; aspect-ratio:1/1; overflow:hidden; border-radius:6px; border:1px solid var(--border); margin-bottom:12px; display:flex; align-items:center; justify-content:center; background:#1e1e1e;">
                <img src="{f_img_rel}" alt="Factor {cid} Usage Plot" 
                     style="width:100%; height:100%; object-fit:contain;" 
                     onerror="this.parentElement.style.display='none'">
            </div>"""

        zeta_html = ""
        if zeta_data:
             overlap = zeta_data.get('overlap_score', 0)
             narrative = zeta_data.get('agreement_narrative', '')
             z_col = "#3fb950" if overlap >= 0.7 else ("#d29922" if overlap >= 0.4 else "#f85149")
             zeta_html = f'<div style="margin-top:10px;padding:10px;border:1px dashed {z_col};border-radius:6px;font-size:0.8rem;color:#8b949e"><strong>Zeta Confidence Score:</strong> <span style="color:{z_col};font-weight:bold">{overlap:.2f}</span><br>{narrative}</div>'

        card_html += f"""
            <div class="cluster-card">
                {factor_img_html}
                <div style="display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--border);padding-bottom:9px">
                    <span style="font-size:1.05rem;font-weight:800;color:#fff">{emo}Cluster {cid}</span>
                    <span>{conf_badge_clickable}</span>
                </div>
                <div class="tag-row">
                    {truth_tag}
                    <div class="{pred_tcls}"><span>Pred:</span><span>{dot}{pred_lbl}</span></div>
                </div>
                <div class="conf-bar-wrap"><div class="conf-bar" style="width:{bar_w}%"></div></div>
                <div class="reasoning"><strong>Reason:</strong><br>{reasoning}{zeta_html}</div>
                <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:auto">
                    <button class="btn" onclick="toggleElement('genes_{key}')">DEGs</button>
                    {pathway_btn}{trace_btn}{bio_btn}
                </div>
                <div id="genes_{key}" class="genes-list hidden">{genes_all}</div>
            </div>"""

    status_th = "<th>✓</th>" if is_eval else ""
    truth_th  = "<th>Ground Truth</th>" if is_eval else ""
    thead     = f"<tr>{status_th}<th>C</th>{truth_th}<th>Predicted</th><th>Conf.</th><th>Top DEGs</th><th>Actions</th></tr>"

    return f"""
    <div id="tab_exp_{run_id}" class="tab-pane">
        <div style="text-align:center;margin-bottom:22px">
            <h1 style="font-size:2rem">{ds['name']} <span style="font-size:1.1rem;color:#8b949e;font-weight:400">| {model}</span></h1>
            <div style="font-size:.76rem;color:#6e7681;margin:6px 0">{tag_line}</div>
            <div style="font-size:.8rem;margin-bottom:10px">Path: <code style="color:#8b949e">{meta.get('data_path','N/A')}</code></div>
        </div>

        <div class="stats-row">
            <div class="stat-tile"><div class="stat-label">Duration</div><div class="stat-value">{dur:.1f}s</div></div>
            <div class="stat-tile"><div class="stat-label">Start</div><div class="stat-value">{start_t}</div></div>
            <div class="stat-tile"><div class="stat-label">Tries</div><div class="stat-value">{num_tries}</div></div>
            <div class="stat-tile" style="display:flex;flex-direction:column;gap:6px;align-items:center">{acc_html}</div>
        </div>

        {singleton_warning}

        <div class="experiment-grid">
            <!-- LEFT PANEL: TABLE & CARDS -->
            <div class="experiment-left">
                <div class="view-toggle-row">
                    <button class="btn-toggle active vtbtn_{run_id}" onclick="switchView('table','{run_id}')">Table View</button>
                    <button class="btn-toggle vtbtn_{run_id}" onclick="switchView('cards','{run_id}')">Card View</button>
                </div>

                <div id="tbl_{run_id}" style="background:var(--card); border:1px solid var(--border); border-radius:12px; padding:20px; box-shadow: 0 4px 15px rgba(0,0,0,0.2)">
                    <h3 style="margin-bottom:16px; font-size: 1.1rem; border-bottom: 1px solid var(--border); padding-bottom: 10px">Cluster Inventory</h3>
                    <table class="data-table">
                        <thead>{thead}</thead>
                        <tbody>{tbl_rows}</tbody>
                    </table>
                </div>

                <div id="cards_{run_id}" class="cluster-grid" style="display:none">
                    {card_html}
                </div>
            </div>

            <!-- RIGHT PANEL: PLOTS & ETA SUMMARY -->
            <div class="experiment-right">
                <div class="umap-wrap">
                    <h3 style="margin-bottom:12px; font-size: 1rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px">Visualization</h3>
                    {plot_html if plot_html else '<div style="padding:40px; background:rgba(255,255,255,0.02); border-radius:12px; border:1px dashed var(--border); text-align:center; color:#484f58">No Plots Available</div>'}
                </div>
                
                <div class="eta-summary-box">
                    <h3>Dataset Biological Insight | Agent Eta</h3>
                    {eta_html if eta_html else '<div style="color:#484f58; font-style:italic">No hierarchical summary available for this experiment.</div>'}
                    <div style="margin-top:15px; border-top:1px solid rgba(188,140,255,0.2); padding-top:12px">
                        <button class="btn btn-purple" onclick="openTrace('{run_id}___GLOBAL_GAMMA__','Batch Gamma (Final Decision) Trace')">View Batch Gamma Trace</button>
                    </div>
                </div>
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
                # Export individual CSV for this experiment
                csv_deg_path = item / "degs.csv"
                export_experiment_degs_to_csv(ds, csv_deg_path)

    if not datasets:
        logger.warning("No eval_report.json files found – report not generated.")
        return

    all_traces: dict = {}
    all_eval:   dict = {}
    all_ann:    dict = {}
    all_pathway: dict = {}
    all_zeta:    dict = {}

    sidebar_links  = "".join(_sidebar_link(ds) for ds in datasets)
    summary_pane   = _summary_tab(datasets)
    experiment_tabs = "".join(_experiment_tab(ds, all_traces, all_eval, all_ann, all_pathway, all_zeta) for ds in datasets)
    tab_contents   = summary_pane + experiment_tabs

    js_experiments = json.dumps([
        {
            "run_id":        ds["run_id"],
            "name":          ds["name"],
            "model":         ds["metadata"].get("model_name", "Unknown"),
            "llmaj_accuracy": ds["llmaj_accuracy"],
            "is_eval":        ds["metadata"].get("is_eval", False),
        }
        for ds in datasets
    ], ensure_ascii=True)

    # Escape </script> in JSON blobs to prevent premature script tag closing
    def _safe_json(obj: dict) -> str:
        return json.dumps(obj, ensure_ascii=True).replace("</", "<\\/")

    html = (
        template
        .replace("<!--CSS_LINK-->",       css_tag)
        .replace("__SIDEBAR_LINKS__",     sidebar_links)
        .replace("__SUMMARY_PANE__",       summary_pane)
        .replace("__EXPERIMENT_TABS__",    experiment_tabs)
        .replace("__JS_EXPERIMENTS__",    js_experiments)
        .replace("__JS_TRACES__",         _safe_json(all_traces))
        .replace("__JS_EVAL_DATA__",      _safe_json(all_eval))
        .replace("__JS_ANN_DATA__",       _safe_json(all_ann))
        .replace("__JS_PATHWAY_DATA__",   _safe_json(all_pathway))
        .replace("__JS_ZETA_DATA__",      _safe_json(all_zeta))
    )

    out_path = base_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")
    
    # Export CSV summary
    csv_path = base_dir / "summary_results.csv"
    export_summary_to_csv(datasets, csv_path)
    
    # Export Batch Excel
    export_batch_degs_to_excel(datasets, base_dir)

    logger.info(f"HTML report written to {out_path}")
    logger.info(f"[TranScribe] Report generated: {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate TranScribe HTML report.")
    parser.add_argument("eval_dir", type=str, help="Directory containing experiment subfolders.")
    args = parser.parse_args()
    generate_html_report(args.eval_dir)
