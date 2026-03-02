import json
import os
from pathlib import Path
from transcribe.config import logger

def generate_html_report(eval_dir: str):
    """
    Generates a vibrant, interactive HTML report from the evaluation outputs.
    Supports multiple datasets via tabs, and a modal for agent communication traces.
    """
    base_dir = Path(eval_dir)
    if not base_dir.exists() or not base_dir.is_dir():
        logger.error(f"Cannot generate HTML report: {eval_dir} is not a valid directory.")
        return

    datasets = []
    
    for item in base_dir.iterdir():
        if item.is_dir():
            report_json_path = item / "eval_report.json"
            if report_json_path.exists():
                with open(report_json_path, "r") as f:
                    data = json.load(f)
                
                dataset_name = data.get("dataset_name", item.name)
                metrics = data.get("metrics", {})
                acc = metrics.get("accuracy", 0.0)
                mapping = data.get("cluster_mapping", {})
                degs = data.get("cluster_degs", {})
                raw = data.get("raw_results", {})
                
                traces = {}
                trace_json_path = item / "eval_communication_trace.json"
                if trace_json_path.exists():
                    try:
                        with open(trace_json_path, "r") as f:
                            traces = json.load(f)
                    except Exception:
                        pass
                
                umap_path = item / "umap_predicted.png"
                rel_umap_path = f"{item.name}/umap_predicted.png" if umap_path.exists() else None
                
                datasets.append({
                    "id": item.name.replace(" ", "_").lower(),
                    "name": dataset_name,
                    "accuracy": acc,
                    "eval_accuracy": data.get("metrics", {}).get("evaluator_accuracy", acc),
                    "mapping": mapping,
                    "degs": degs,
                    "raw": raw,
                    "traces": traces,
                    "evaluator_results": data.get("evaluator_results", {}),
                    "cluster_colors": data.get("cluster_colors", {}),
                    "umap_path": rel_umap_path,
                    "metadata": data.get("metadata", {})
                })
                
    if not datasets:
        logger.warning(f"No evaluated datasets found in {eval_dir}.")
        return

    total_acc = sum(d["accuracy"] for d in datasets) / len(datasets)
    total_eval_acc = sum(d["eval_accuracy"] for d in datasets) / len(datasets)
    overall_acc_class = "acc-high" if total_acc >= 0.8 else ("acc-med" if total_acc >= 0.5 else "acc-low")
    overall_eval_class = "acc-high" if total_eval_acc >= 0.8 else ("acc-med" if total_eval_acc >= 0.5 else "acc-low")

    # Global Performance Metrics
    total_duration = sum(d["metadata"].get("duration_seconds", 0) for d in datasets)
    avg_duration = total_duration / len(datasets) if datasets else 0
    global_model = datasets[0]["metadata"].get("model_name", "N/A") if datasets else "N/A"

    # HTML Generation
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TranScribe Evaluation Report</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg-color: #0d1117;
                --text-color: #c9d1d9;
                --card-bg: rgba(22, 27, 34, 0.7);
                --card-border: rgba(255,255,255,0.1);
                --accent-blue: #58a6ff;
                --success-green: #2ea043;
                --error-red: #f85149;
                --hover-bg: rgba(255,255,255,0.05);
            }}
            * {{ box-sizing: border-box; }}
            body {{
                font-family: 'Inter', sans-serif;
                background-color: var(--bg-color);
                color: var(--text-color);
                margin: 0;
                display: flex;
                height: 100vh;
                overflow: hidden;
            }}
            h1, h2, h3 {{ color: #ffffff; margin-top: 0; }}
            
            /* Sidebar Navigation */
            .sidebar {{
                width: 250px;
                background: #161b22;
                border-right: 1px solid var(--card-border);
                padding: 20px;
                display: flex;
                flex-direction: column;
                overflow-y: auto;
            }}
            .sidebar-title {{
                font-size: 1.5rem;
                font-weight: 800;
                background: linear-gradient(90deg, var(--accent-blue), #bc8cff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 30px;
                text-align: center;
            }}
            .tab-link {{
                padding: 12px 15px;
                cursor: pointer;
                border-radius: 8px;
                margin-bottom: 5px;
                color: #8b949e;
                transition: all 0.2s;
                font-weight: 600;
            }}
            .tab-link:hover {{ background: var(--hover-bg); color: white; }}
            .tab-link.active {{ background: var(--accent-blue); color: white; }}
            
            /* Main Content */
            .main-content {{
                flex: 1;
                padding: 40px;
                overflow-y: auto;
            }}
            .tab-content {{ display: none; animation: fadeIn 0.4s ease-in-out; }}
            .tab-content.active {{ display: block; }}
            
            /* Summary Table */
            .summary-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
                background: var(--card-bg);
                border-radius: 12px;
                overflow: hidden;
                border: 1px solid var(--card-border);
            }}
            .summary-table th, .summary-table td {{
                padding: 15px;
                text-align: left;
                border-bottom: 1px solid var(--card-border);
            }}
            .summary-table th {{ color: white; background: rgba(255,255,255,0.05); }}
            .clickable-row {{ cursor: pointer; transition: background 0.2s; }}
            .clickable-row:hover {{ background: var(--hover-bg); }}
            
            .accuracy-badge {{
                display: inline-block;
                padding: 5px 15px;
                border-radius: 20px;
                color: white;
                font-weight: 600;
            }}
            .acc-high {{ background: var(--success-green); }}
            .acc-low {{ background: var(--error-red); }}
            .acc-med {{ background: #d29922; }}
            
            /* Dataset Content Elements */
            .dataset-header {{ margin-bottom: 40px; text-align: center; }}
            .umap-container {{ text-align: center; margin-bottom: 50px; }}
            .umap-container img {{ max-width: 800px; width: 100%; border-radius: 12px; border: 1px solid var(--card-border); }}
            
            .cluster-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
                gap: 20px;
            }}
            .cluster-card {{
                background: var(--card-bg);
                border: 1px solid var(--card-border);
                border-radius: 16px;
                padding: 25px;
                display: flex;
                flex-direction: column;
                gap: 15px;
            }}
            .tag-row {{ display: flex; flex-direction: column; gap: 8px; }}
            .tag {{ padding: 4px 10px; border-radius: 6px; font-size: 0.9rem; font-weight: 600; display: inline-flex; justify-content: space-between; }}
            .tag.true {{ background: rgba(255, 255, 255, 0.1); border: 1px solid rgba(255,255,255,0.2); }}
            .tag.pred.match {{ background: rgba(46, 160, 67, 0.2); border: 1px solid var(--success-green); color: #56d364; }}
            .tag.pred.mismatch {{ background: rgba(248, 81, 73, 0.2); border: 1px solid var(--error-red); color: #ff7b72; }}
            
            .confidence-bar-container {{ width: 100%; height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px; }}
            .confidence-bar {{ height: 100%; background: linear-gradient(90deg, var(--accent-blue), #bc8cff); border-radius: 3px; }}
            .reasoning {{ font-size: 0.85rem; color: #8b949e; background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px; max-height: 100px; overflow-y: auto; }}
            
            .action-btn {{
                background: rgba(88, 166, 255, 0.1);
                border: 1px solid var(--accent-blue);
                color: var(--accent-blue);
                padding: 8px 12px;
                border-radius: 6px;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.2s;
                flex: 1;
            }}
            .action-btn:hover {{ background: rgba(88, 166, 255, 0.2); }}
            
            .genes-list {{
                margin-top: 10px;
                font-size: 0.8rem;
                background: rgba(0,0,0,0.3);
                padding: 10px;
                border-radius: 8px;
                display: flex;
                flex-wrap: wrap;
                gap: 5px;
            }}
            .gene-chip {{ background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px; }}
            .hidden {{ display: none !important; }}
            
            .color-swatch {{
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
                vertical-align: middle;
            }}

            .view-toggle-container {{
                display: flex;
                justify-content: flex-end;
                margin-bottom: 15px;
                gap: 10px;
            }}
            .view-toggle-btn {{
                background: rgba(255,255,255,0.05);
                border: 1px solid var(--card-border);
                color: #8b949e;
                padding: 6px 15px;
                border-radius: 6px;
                cursor: pointer;
                transition: all 0.2s;
            }}
            .view-toggle-btn.active {{
                background: rgba(88, 166, 255, 0.2);
                border-color: var(--accent-blue);
                color: white;
            }}
            
            /* Modal / Panel for Communication Trace */
            .modal {{
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.8);
                backdrop-filter: blur(5px);
                justify-content: center;
                align-items: center;
            }}
            .modal.show {{ display: flex; }}
            .modal-content {{
                background-color: #161b22;
                border: 1px solid var(--card-border);
                width: 90%;
                max-width: 800px;
                height: 80%;
                border-radius: 12px;
                display: flex;
                flex-direction: column;
                box-shadow: 0 10px 30px rgba(0,0,0,0.5);
                animation: slideDown 0.3s ease-out;
            }}
            .modal-header {{
                padding: 20px;
                border-bottom: 1px solid var(--card-border);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .close-modal {{
                color: #8b949e;
                font-size: 24px;
                cursor: pointer;
                transition: color 0.2s;
            }}
            .close-modal:hover {{ color: white; }}
            
            .trace-container {{
                padding: 20px;
                overflow-y: auto;
                flex: 1;
                display: flex;
                flex-direction: column;
                gap: 15px;
            }}
            .trace-message {{
                padding: 15px;
                border-radius: 8px;
                font-size: 0.85rem;
                background: rgba(0,0,0,0.3);
            }}
            .trace-message.alpha {{ border-left: 4px solid var(--accent-blue); }}
            .trace-message.beta {{ border-left: 4px solid var(--success-green); }}
            .trace-message.gamma {{ border-left: 4px solid #bc8cff; }}
            .trace-agent-name {{
                font-weight: 800;
                margin-bottom: 10px;
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: white;
            }}
            .trace-message pre {{
                margin: 0;
                white-space: pre-wrap;
                word-wrap: break-word;
                font-family: 'Inter', sans-serif;
                color: #c9d1d9;
            }}

            @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
            @keyframes slideDown {{ from {{ opacity: 0; transform: translateY(-20px); }} to {{ opacity: 1; transform: translateY(0); }} }}
            ::-webkit-scrollbar {{ width: 8px; }}
            ::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.2); border-radius: 4px; }}
        </style>
    </head>
    <body>
        
        <!-- Sidebar -->
        <div class="sidebar">
            <div class="sidebar-title">TranScribe</div>
            <div class="tab-link active" onclick="openTab('summary_tab', this)">Overall Summary</div>
    """
    
    for ds in datasets:
        html_content += f'<div class="tab-link" onclick="openTab(\'ds_{ds["id"]}\', this)">{ds["name"]}</div>\n'
        
    html_content += f"""
        </div>
        
        <!-- Main Content -->
        <div class="main-content">
            <!-- Overall Summary Tab -->
            <div id="summary_tab" class="tab-content active">
                <div class="dataset-header">
                    <h1>Overall Evaluation Summary</h1>
                    <div style="font-size: 1.1rem; color: #8b949e; margin-bottom: 5px;">Model: <code style="color: var(--accent-blue);">{global_model}</code></div>
                    
                    <div style="display: flex; gap: 40px; justify-content: center; margin-top: 20px;">
                        <div style="text-align: center;">
                            <div style="font-size: 1rem; color: #8b949e; margin-bottom: 8px;">Avg. Naive Accuracy (Exact)</div>
                            <div class="accuracy-badge {overall_acc_class}" style="font-size: 2rem; padding: 10px 30px;">{(total_acc * 100):.1f}%</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 1rem; color: #bc8cff; margin-bottom: 8px;">Avg. Evaluator Accuracy (Biological)</div>
                            <div class="accuracy-badge {overall_eval_class}" style="font-size: 2rem; padding: 10px 30px; border: 2px solid #bc8cff; background: rgba(188, 140, 255, 0.1);">{(total_eval_acc * 100):.1f}%</div>
                        </div>
                    </div>
                </div>

                <!-- Run Performance Panel -->
                <div style="margin-top: 40px; background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 25px;">
                    <h3 style="margin-top: 0; border-bottom: 1px solid var(--card-border); padding-bottom: 15px; margin-bottom: 20px;">Run Performance</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
                        <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px; text-align: center;">
                            <div style="font-size: 0.8rem; color: #8b949e; text-transform: uppercase;">Average Duration</div>
                            <div style="font-size: 1.8rem; font-weight: 800; margin-top: 10px; color: var(--accent-blue);">{avg_duration:.1f}s</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px; text-align: center;">
                            <div style="font-size: 0.8rem; color: #8b949e; text-transform: uppercase;">Total Run Time</div>
                            <div style="font-size: 1.8rem; font-weight: 800; margin-top: 10px; color: #bc8cff;">{total_duration:.1f}s</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px; text-align: center;">
                            <div style="font-size: 0.8rem; color: #8b949e; text-transform: uppercase;">Datasets Processed</div>
                            <div style="font-size: 1.8rem; font-weight: 800; margin-top: 10px; color: white;">{len(datasets)}</div>
                        </div>
                    </div>
                </div>
    """
    
    # Check if ANY dataset has evaluation metrics
    has_eval_data = any(ds["metadata"].get("is_eval", True) for ds in datasets)
    
    if has_eval_data:
        html_content += """
                <div style="margin-top: 40px; background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 20px;">
                    <h3 style="margin-top: 0; border-bottom: 1px solid var(--card-border); padding-bottom: 15px; margin-bottom: 20px;">Accuracy Comparison Barplot</h3>
                    <div style="display: flex; gap: 40px; align-items: flex-end; height: 250px; padding: 20px 0; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 15px; justify-content: space-around;">
        """
        
        # Generate the barplot items
        for ds in datasets:
            if not ds["metadata"].get("is_eval", True):
                continue
            acc = ds["accuracy"]
            acc_pct = acc * 100
            bar_height = max(10, acc_pct) * 2 # scale up to fit in 250px container visually
            bar_color = "var(--success-green)" if acc >= 0.8 else ("#d29922" if acc >= 0.5 else "var(--error-red)")
            html_content += f"""
                            <div style="display: flex; flex-direction: column; align-items: center; gap: 10px; flex: 1; max-width: 120px;">
                                <div style="font-weight: 800; font-size: 1.1rem; color: {bar_color};">{acc_pct:.1f}%</div>
                                <div style="width: 100%; max-width: 80px; height: {bar_height}px; background-color: {bar_color}; border-radius: 6px 6px 0 0; box-shadow: 0 0 15px rgba(0,0,0,0.3); transition: transform 0.2s; cursor: pointer;" onmouseover="this.style.transform='scaleY(1.05)'" onmouseout="this.style.transform='scaleY(1)'" onclick="const tabs = document.querySelectorAll('.tab-link'); [...tabs].find(t => t.innerText === '{ds["name"]}').click()"></div>
                                <div style="font-size: 0.85rem; font-weight: 600; text-align: center; color: #c9d1d9; word-wrap: break-word;">{ds["name"]}</div>
                            </div>
            """
            
        html_content += """
                        </div>
                    </div>
        """
    
    html_content += """
                <table class="summary-table">
                    <thead>
                        <tr>
                            <th>Dataset Name</th>
                            <th>Clusters Processed</th>
                            <th>Type</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    for ds in datasets:
        is_ds_eval = ds["metadata"].get("is_eval", True)
        cluster_cnt = len(ds["mapping"])
        type_str = "Evaluation" if is_ds_eval else "Inference"
        
        html_content += f"""
                        <tr class="clickable-row" onclick="const tabs = document.querySelectorAll('.tab-link'); [...tabs].find(t => t.innerText === '{ds["name"]}').click()">
                            <td>{ds["name"]}</td>
                            <td>{cluster_cnt}</td>
                            <td>{type_str}</td>
                        </tr>
        """
        
    html_content += """
                    </tbody>
                </table>
            </div>
    """
    
    # Store traces for modal JS
    all_traces = {}
    all_eval_reasons = {}
    all_ann_reasons = {}
    
    # Generate Dataset Tabs
    for ds in datasets:
        is_ds_eval = ds["metadata"].get("is_eval", True)
        acc = ds["accuracy"]
        acc_class = "acc-high" if acc >= 0.8 else ("acc-med" if acc >= 0.5 else "acc-low")
        eval_acc = ds["eval_accuracy"]
        eval_acc_class = "acc-high" if eval_acc >= 0.8 else ("acc-med" if eval_acc >= 0.5 else "acc-low")
        
        acc_html = f"""
                            <span class="accuracy-badge {acc_class}" style="font-size: 0.9rem; padding: 5px 15px;">Naive Acc: {(acc * 100):.1f}%</span>
                            <span class="accuracy-badge {eval_acc_class}" style="font-size: 0.9rem; padding: 5px 15px; border: 1px solid #bc8cff; background: rgba(188,140,255,0.1);">Evaluator Acc: {(eval_acc * 100):.1f}%</span>
        """ if is_ds_eval else ""
        
        html_content += f"""
            <div id="ds_{ds["id"]}" class="tab-content">
                <div class="dataset-header">
                    <div style="font-size: 0.8rem; color: #8b949e; margin-bottom: 5px;">
                        {ds["metadata"].get("organism", "N/A")} | {ds["metadata"].get("tissue", "N/A")} | {ds["metadata"].get("disease", "N/A")}
                        { " | [TOY DATASET]" if ds["metadata"].get("is_toy") else "" }
                        { "" if is_ds_eval else " | [INFERENCE MODE]" }
                    </div>
                    <h1 style="margin-bottom: 10px;">{ds["name"]}</h1>
                    <div style="font-size: 0.9rem; margin-bottom: 20px;">
                        Model: <code style="color: var(--accent-blue);">{ds["metadata"].get("model_name", "N/A")}</code> | 
                        Path: <code style="color: #8b949e;">{ds["metadata"].get("data_path", "N/A")}</code>
                    </div>
                    <div style="display: flex; gap: 15px; justify-content: center; margin-bottom: 20px;">
                        <div style="background: rgba(255,255,255,0.05); padding: 10px 20px; border-radius: 8px; border: 1px solid var(--card-border);">
                            <div style="font-size: 0.7rem; color: #8b949e; text-transform: uppercase;">Duration</div>
                            <div style="font-size: 1.1rem; font-weight: 800;">{ds["metadata"].get("duration_seconds", 0):.1f}s</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.05); padding: 10px 20px; border-radius: 8px; border: 1px solid var(--card-border);">
                            <div style="font-size: 0.7rem; color: #8b949e; text-transform: uppercase;">Start Time</div>
                            <div style="font-size: 1.1rem; font-weight: 800;">{ds["metadata"].get("start_time", "Unknown").split('T')[1].split('.')[0] if 'T' in ds["metadata"].get("start_time", "") else "N/A"}</div>
                        </div>
                        <div style="display: flex; flex-direction: column; gap: 5px;">
                            {acc_html}
                        </div>
                    </div>
                </div>
        """
        
        if ds["umap_path"]:
            html_content += f"""
                <div class="umap-container">
                    <h3>Annotated UMAP</h3>
                    <img src="{ds["umap_path"]}" alt="UMAP for {ds["name"]}">
                </div>
            """
            
        html_content += f"""
                <div class="view-toggle-container">
                    <button class="view-toggle-btn active" onclick="switchView('table', '{ds['id']}')">Table View</button>
                    <button class="view-toggle-btn" onclick="switchView('cards', '{ds['id']}')">Card View</button>
                </div>

                <div id="view_table_{ds['id']}" style="background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 20px;">
                    <h3 style="margin-top: 0; border-bottom: 1px solid var(--card-border); padding-bottom: 15px; margin-bottom: 20px;">Cluster Details</h3>
                    <table class="summary-table">
                        <thead>
                            <tr>
                                { "<th>Status</th>" if is_ds_eval else "" }
                                <th>Cluster ID</th>
                                { "<th>Ground Truth</th>" if is_ds_eval else "" }
                                <th>Predicted Label</th>
                                <th>Confidence</th>
                                <th>Top DEGs</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        for cid, m in ds["mapping"].items():
            true_lbl = m.get("true", "Unknown")
            pred_lbl = m.get("pred", "Error")
            eval_result = ds["evaluator_results"].get(cid, {})
            is_match = eval_result.get("is_match", False)
            
            # Extract UMAP color for this cluster
            c_color = ds["cluster_colors"].get(cid, "#ffffff")
            color_swatch_html = f'<span class="color-swatch" style="background-color: {c_color};"></span>'
            
            pred_color = "white" if (is_match or not is_ds_eval) else "#f85149"
            
            raw_ann = ds["raw"].get(cid, {})
            if isinstance(raw_ann, str):
                confidence = 0.0
                reasoning = raw_ann
            else:
                confidence = raw_ann.get("confidence", 0.0) if isinstance(raw_ann, dict) else 0.0
                reasoning = raw_ann.get("reasoning_chain", "N/A") if isinstance(raw_ann, dict) else "N/A"
            
            cluster_genes = ds["degs"].get(cid, [])[:5] # limit to top 5 for table view
            genes_html_table = "".join([f'<span class="gene-chip">{g}</span>' for g in cluster_genes])
            
            all_traces[f"{ds['id']}_{cid}"] = ds["traces"].get(cid, [])
            all_eval_reasons[f"{ds['id']}_{cid}"] = eval_result
            all_ann_reasons[f"{ds['id']}_{cid}"] = reasoning
            
            status_td = f'<td style="text-align: center; font-size: 1.2rem;">{ "✅" if is_match else "❌" }</td>' if is_ds_eval else ""
            truth_td = f'<td>{true_lbl}</td>' if is_ds_eval else ""
            bio_btn = f'<button class="action-btn" style="flex: none; width: auto; padding: 8px 12px; background: rgba(188,140,255,0.1); border: 1px solid #bc8cff; color: #bc8cff;" onclick="openEvalReasonModal(\'{ds["id"]}_{cid}\')">Bio-Reasoning</button>' if is_ds_eval else ""
            ann_reason_btn = f'<button class="action-btn" style="flex: none; width: auto; padding: 8px 12px; margin-bottom: 5px; background: rgba(56, 139, 253, 0.1); border: 1px solid var(--accent-blue); color: var(--accent-blue);" onclick="openAnnReasonModal(\'{ds["id"]}_{cid}\')">Annotation Reason</button>'
            
            html_content += f"""
                            <tr>
                                {status_td}
                                <td><code>{cid}</code></td>
                                {truth_td}
                                <td style="color: {pred_color}">{color_swatch_html}{pred_lbl}</td>
                                <td>{(confidence * 100):.0f}%</td>
                                <td><div class="genes-list" style="background:none; padding:0; margin:0;">{genes_html_table}</div></td>
                                <td>
                                    {ann_reason_btn}
                                    <button class="action-btn" style="flex: none; width: auto; padding: 8px 12px; margin-bottom: 5px;" onclick="openTraceModal('{ds['id']}_{cid}', 'Cluster {cid} Trace')">View Trace</button>
                                    {bio_btn}
                                </td>
                            </tr>
            """
        
        html_content += """
                        </tbody>
                    </table>
                </div>
        """
        
        # Grid/Cards View (Initially hidden)
        html_content += f'<div id="view_cards_{ds["id"]}" class="cluster-grid hidden">\n'
        
        for cid, labels in ds["mapping"].items():
            true_lbl = labels.get("true", "Unknown")
            pred_lbl = labels.get("pred", "Error")
            eval_result = ds["evaluator_results"].get(cid, {})
            is_match = eval_result.get("is_match", False)
            match_class = "match" if is_match else "mismatch"
            
            c_color = ds["cluster_colors"].get(cid, "#ffffff")
            color_swatch_html = f'<span class="color-swatch" style="background-color: {c_color};"></span>'
            
            ann = ds["raw"].get(cid, {})
            if isinstance(ann, str):
                confidence = 0.0
                reasoning = ann
            else:
                confidence = ann.get("confidence", 0.0)
                reasoning = ann.get("reasoning_chain", "N/A")
                
            cluster_genes = ds["degs"].get(cid, [])
            genes_html = "".join([f'<span class="gene-chip">{g}</span>' for g in cluster_genes])
            
            trace_id = f"{ds['id']}_{cid}"
            trace_btn = f'<button class="action-btn" onclick="openTraceModal(\'{trace_id}\', \'Cluster {cid} Trace\')">Trace</button>' if all_traces.get(trace_id) else ""
            
            status_emoji = f'{"✅" if is_match else "❌"} ' if is_ds_eval else ""
            truth_tag = f'<div class="tag true"><span>True:</span> <span>{true_lbl}</span></div>' if is_ds_eval else ""
            bio_match_btn = f'<button class="action-btn" style="background: rgba(188,140,255,0.1); border: 1px solid #bc8cff; color: #bc8cff;" onclick="openEvalReasonModal(\'{trace_id}\')">Bio-Match</button>' if is_ds_eval else ""
            pred_class = f'{match_class}' if is_ds_eval else ""
            
            html_content += f"""
                <div class="cluster-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--card-border); padding-bottom: 10px;">
                        <span style="font-size: 1.2rem; font-weight: 800; color: white;">
                            {status_emoji}Cluster {cid}
                        </span>
                        <span style="font-size: 0.85rem; color: #8b949e;">Conf: {(confidence * 100):.1f}%</span>
                    </div>
                    
                    <div class="tag-row">
                        {truth_tag}
                        <div class="tag pred {pred_class}"><span>Pred:</span> <span>{color_swatch_html}{pred_lbl}</span></div>
                    </div>
                    
                    <div class="confidence-bar-container">
                        <div class="confidence-bar" style="width: {confidence * 100}%;"></div>
                    </div>
                    
                    <div class="reasoning">
                        <strong>Agent Reason:</strong><br/>
                        {reasoning}
                    </div>
                    
                    <div style="display: flex; gap: 5px; margin-top: auto;">
                        <button class="action-btn" onclick="toggleElement('genes_{trace_id}')">DEGs</button>
                        {trace_btn}
                        {bio_match_btn}
                    </div>
                    
                    <div id="genes_{trace_id}" class="genes-list hidden">{genes_html}</div>
                </div>
            """
            
        html_content += "</div></div>\n" # Close grid and tab-content
        
    html_content += f"""
        </div>
        
        <!-- Trace Modal -->
        <div id="traceModal" class="modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h2 id="traceModalTitle" style="margin: 0;">Trace Payload</h2>
                    <span class="close-modal" onclick="closeTraceModal()">&times;</span>
                </div>
                <div id="traceModalBody" class="trace-container">
                    <!-- Traces injected via JS -->
                </div>
            </div>
        </div>
        
        <script>
            // Store trace data for modal injection
            const traceData = {json.dumps(all_traces)};
            const evalReasonData = {json.dumps(all_eval_reasons)};
            const annReasonData = {json.dumps(all_ann_reasons)};
            
            function openTab(tabId, elClicked) {{
                // Hide all tabs
                document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.tab-link').forEach(el => el.classList.remove('active'));
                
                // Show requested tab
                document.getElementById(tabId).classList.add('active');
                
                // Highlight link
                if (elClicked) {{
                    elClicked.classList.add('active');
                }}
            }}
            
            function switchView(viewType, dsId) {{
                const tableCont = document.getElementById('view_table_' + dsId);
                const cardsCont = document.getElementById('view_cards_' + dsId);
                const btns = document.querySelectorAll('#ds_' + dsId + ' .view-toggle-btn');
                
                btns.forEach(b => b.classList.remove('active'));
                
                if (viewType === 'table') {{
                    tableCont.classList.remove('hidden');
                    cardsCont.classList.add('hidden');
                    btns[0].classList.add('active');
                }} else {{
                    tableCont.classList.add('hidden');
                    cardsCont.classList.remove('hidden');
                    btns[1].classList.add('active');
                }}
            }}
            
            function toggleElement(id) {{
                const el = document.getElementById(id);
                el.classList.toggle('hidden');
            }}
            
            // Modal Functions
            const modal = document.getElementById('traceModal');
            const modalTitle = document.getElementById('traceModalTitle');
            const modalBody = document.getElementById('traceModalBody');
            
            function openTraceModal(traceId, title) {{
                modalTitle.textContent = title;
                modalBody.innerHTML = ''; // Clear existing
                
                const traces = traceData[traceId] || [];
                traces.forEach(msg => {{
                    let content = '';
                    const agentType = msg.agent ? msg.agent.toLowerCase() : 'system';
                    const msgType = msg.type || '';
                    
                    if (msg.input) content += '**Input:**\\n' + JSON.stringify(msg.input, null, 2) + '\\n\\n';
                    if (msg.output) content += '**Output:**\\n' + JSON.stringify(msg.output, null, 2) + '\\n\\n';
                    if (msg.content) {{
                        if (typeof msg.content === 'object') content += JSON.stringify(msg.content, null, 2);
                        else content += msg.content;
                    }}
                    
                    const el = document.createElement('div');
                    el.className = `trace-message ${{agentType}}`;
                    el.innerHTML = `
                        <div class="trace-agent-name">${{msg.agent || 'System'}} ${{msgType}}</div>
                        <pre>${{content.trim()}}</pre>
                    `;
                    modalBody.appendChild(el);
                }});
                
                modal.classList.add('show');
            }}
            
            function openEvalReasonModal(traceId) {{
                modalTitle.textContent = "Evaluator Biological Reasoning";
                modalBody.innerHTML = '';
                const res = evalReasonData[traceId];
                if (res) {{
                    let content = '<div style="background: rgba(188,140,255,0.1); padding: 20px; border-radius: 10px; border: 1px solid #bc8cff; margin-top: 20px;">';
                    content += '<div style="font-size: 1.2rem; margin-bottom: 15px;">Match Status: ' + (res.is_match ? '<span style="color: #3fb950;">✅ BIOLOGICAL MATCH</span>' : '<span style="color: #f85149;">❌ NOT A MATCH</span>') + '</div>';
                    content += '<div style="line-height: 1.6;">' + res.explanation + '</div>';
                    content += '</div>';
                    modalBody.innerHTML = content;
                }}
                modal.classList.add('show');
            }}
            
            function openAnnReasonModal(traceId) {{
                modalTitle.textContent = "Agent Annotation Reasoning";
                modalBody.innerHTML = '';
                const explanation = annReasonData[traceId];
                if (explanation) {{
                    let content = '<div style="background: rgba(56, 139, 253, 0.1); padding: 20px; border-radius: 10px; border: 1px solid var(--accent-blue); margin-top: 20px;">';
                    content += '<div style="line-height: 1.6; font-family: monospace; white-space: pre-wrap; font-size: 1rem;">' + explanation + '</div>';
                    content += '</div>';
                    modalBody.innerHTML = content;
                }} else {{
                    modalBody.innerHTML = '<div style="color: var(--error-red);">No annotation reasoning provided by the agent.</div>';
                }}
                modal.classList.add('show');
            }}
            
            function closeTraceModal() {{
                modal.classList.remove('show');
            }}
            
            // Close modal if clicked outside content
            window.onclick = function(event) {{
                if (event.target == modal) {{
                    closeTraceModal();
                }}
            }}
        </script>
    </body>
    </html>
    """

    out_path = base_dir / "index.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    logger.info(f"Interactive HTML report generated at {out_path}")

if __name__ == "__main__":
    # Test script if run directly
    generate_html_report("eval_results")
