import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, BatchNorm
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import networkx as nx
import numpy as np
import requests
import json
import re
from typing import List, Dict, Any

# ============================
# 🔹 Helper: safe JSON loader
# ============================
def try_extract_json(text: str):
    """
    Try to extract a JSON array/object from `text`.
    Returns parsed JSON or raises ValueError.
    """
    if not text or not text.strip():
        raise ValueError("Empty text")
    # First try direct load
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find a JSON substring (handles code blocks or surrounding text)
    match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception as e:
            raise ValueError(f"Found JSON-like substring but failed to parse: {e}")

    raise ValueError("No JSON found")


# =========================================================
# 🔹 STEP 1: Robust LLM (Mistral via Ollama) client wrapper
# =========================================================
def mistral_prompt(prompt: str, retries: int = 3, timeout: int = 120) -> str:
    """
    Call local Ollama endpoint. Returns the model 'response' string.
    If Ollama is unreachable or returns invalid data, returns an empty string.
    """
    url = "http://localhost:11434/api/generate"
    payload = {"model": "mistral:7b", "prompt": prompt, "stream": False}
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            try:
                j = resp.json()
            except ValueError:
                # Some versions may return text — attempt to parse raw text
                j = json.loads(resp.text)

            # Ollama response typically places text in j["response"]
            text = j.get("response") if isinstance(j, dict) else None
            if text is None:
                # Fallback: maybe the server wrapped things differently
                text = j.get("output") if isinstance(j, dict) else None

            # final safety: ensure text is a string
            if text and isinstance(text, str) and text.strip():
                return text.strip()
            else:
                last_err = ValueError("Empty or missing 'response' field from Ollama")
                # small wait and retry
                time.sleep(0.5)
                continue

        except Exception as e:
            last_err = e
            # small backoff
            time.sleep(0.5)

    # If we reach here, the request failed repeatedly
    print(f"⚠️ mistral_prompt failed after {retries} attempts. Last error: {last_err}")
    return ""  # caller will handle fallback


def preprocess_logs_with_llm(raw_logs: str) -> List[Dict[str, Any]]:
    """
    Ask Mistral to structure logs into a JSON array of objects with numeric keys:
      ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'label'].
    If the LLM fails or returns invalid JSON, this function attempts a conservative fallback.
    """
    prompt = f"""
You are a cybersecurity log parser. Output ONLY a valid JSON array (no explanation).
Each item must be an object containing numeric keys:
["dur","spkts","dpkts","sbytes","dbytes","label"]
If a value is missing, use 0. If label is unknown, use 0.
Return only the JSON array, e.g.:
[{{"dur":0.1,"spkts":2,"dpkts":3,"sbytes":200,"dbytes":150,"label":0}}, ... ]

Logs:
{raw_logs}
"""
    print("📡 Sending logs to Mistral for structuring...")
    result_text = mistral_prompt(prompt)

    if not result_text:
        print("⚠️ No response from Mistral — falling back to heuristic parser.")
        return fallback_parse_logs(raw_logs)

    print("🔍 Raw Mistral response (first 500 chars):")
    print(result_text[:500])

    # Try to extract JSON
    try:
        parsed = try_extract_json(result_text)
        if isinstance(parsed, dict):
            # single object -> wrap into list
            parsed = [parsed]
        if not isinstance(parsed, list):
            raise ValueError("Parsed JSON is not a list")
        # Validate and sanitize entries
        cleaned = []
        for idx, item in enumerate(parsed):
            if not isinstance(item, dict):
                print(f"⚠️ skipping non-object entry at index {idx}")
                continue
            entry = {}
            for key in ["dur", "spkts", "dpkts", "sbytes", "dbytes", "label"]:
                val = item.get(key, 0)
                try:
                    if key == "label":
                        entry[key] = int(val)
                    else:
                        entry[key] = float(val)
                except Exception:
                    entry[key] = 0 if key != "label" else 0
            cleaned.append(entry)
        if len(cleaned) == 0:
            raise ValueError("No valid items after cleaning")
        return cleaned

    except Exception as e:
        print(f"⚠️ LLM parsing failed ({e}), using fallback heuristic parser.")
        return fallback_parse_logs(raw_logs)


def fallback_parse_logs(raw_logs: str) -> List[Dict[str, Any]]:
    """
    Basic heuristic parser that tries to extract numbers from text lines.
    This is intentionally conservative and returns a list of dicts with the expected keys.
    """
    lines = [l.strip() for l in raw_logs.strip().splitlines() if l.strip()]
    parsed = []
    for ln in lines:
        # Defaults
        dur = 0.0
        spkts = 0
        dpkts = 0
        sbytes = 0
        dbytes = 0
        label = 0

        # duration like "0.4s" or "lasting 0.2s"
        m = re.search(r'(\d+(\.\d+)?)\s*s\b', ln)
        if m:
            try:
                dur = float(m.group(1))
            except:
                pass

        # bytes or MB/KB mention
        mb_match = re.search(r'(\d+(\.\d+)?)\s*MB', ln, re.IGNORECASE)
        if mb_match:
            # approximate MB -> bytes (MB * 1,000,000)
            try:
                sbytes = int(float(mb_match.group(1)) * 1_000_000)
            except:
                pass

        # packets like "200 packets" or "200 pkts"
        pk_match = re.search(r'(\d+)\s*(packets|pkts|packets|pkt)', ln, re.IGNORECASE)
        if pk_match:
            try:
                spkts = int(pk_match.group(1))
            except:
                pass

        # other numbers fallback (take first numeric tokens)
        numbers = re.findall(r'\d+', ln)
        if numbers:
            # try to populate fields if still zero
            try:
                if spkts == 0 and len(numbers) >= 1:
                    spkts = int(numbers[0])
                if dpkts == 0 and len(numbers) >= 2:
                    dpkts = int(numbers[1])
            except:
                pass

        parsed.append({
            "dur": float(dur),
            "spkts": int(spkts),
            "dpkts": int(dpkts),
            "sbytes": int(sbytes),
            "dbytes": int(dbytes),
            "label": int(label)
        })
    # If nothing parsed, return a single default example
    if not parsed:
        return [{"dur": 0.1, "spkts": 2, "dpkts": 3, "sbytes": 200, "dbytes": 150, "label": 0}]
    return parsed


# =========================================================
# 🔹 STEP 2: GNN Model Definition (unchanged, minor fixes)
# =========================================================
class EdgeGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        # in_channels is edge_attr width (features per edge)
        self.nn1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels * hidden_channels)
        )
        self.conv1 = NNConv(in_channels, hidden_channels, self.nn1, aggr='mean')
        self.bn1 = BatchNorm(hidden_channels)
        self.drop1 = nn.Dropout(dropout)

        self.nn2 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels * hidden_channels)
        )
        self.conv2 = NNConv(hidden_channels, hidden_channels, self.nn2, aggr='mean')
        self.bn2 = BatchNorm(hidden_channels)
        self.drop2 = nn.Dropout(dropout)

        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop1(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.drop2(x)

        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        # Return predictions for source nodes of each edge (matching original behavior)
        return x[edge_index[0]]


# =========================================================
# 🔹 STEP 3: Convert JSON Logs → Graph → PyG Data (hardened)
# =========================================================
def build_graph_from_json(json_logs: List[Dict[str, Any]]) -> Data:
    if not isinstance(json_logs, list) or len(json_logs) == 0:
        raise ValueError("json_logs must be a non-empty list")

    # Ensure keys exist and are numeric
    entries = []
    for i, r in enumerate(json_logs):
        try:
            entry = {
                "dur": float(r.get("dur", 0.0)),
                "spkts": int(r.get("spkts", 0)),
                "dpkts": int(r.get("dpkts", 0)),
                "sbytes": int(r.get("sbytes", 0)),
                "dbytes": int(r.get("dbytes", 0)),
                "label": int(r.get("label", 0))
            }
            entries.append(entry)
        except Exception:
            # skip malformed entry
            print(f"⚠️ Skipping malformed log entry at index {i}: {r}")

    if len(entries) == 0:
        raise ValueError("No valid entries after cleaning logs")

    df_len = len(entries)
    src_nodes = [f"src_{i}" for i in range(df_len)]
    dst_nodes = [f"dst_{i}" for i in range(df_len)]
    edges = list(zip(src_nodes, dst_nodes))

    edge_features = np.array([[r["dur"], r["spkts"], r["dpkts"], r["sbytes"], r["dbytes"]] for r in entries], dtype=np.float32)
    edge_labels = np.array([r["label"] for r in entries], dtype=np.int64)

    G = nx.Graph()
    G.add_nodes_from(src_nodes)
    G.add_nodes_from(dst_nodes)
    for (src, dst, feat, lbl) in zip(src_nodes, dst_nodes, edge_features, edge_labels):
        G.add_edge(src, dst, features=feat, label=int(lbl))

    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    edge_u, edge_v, edge_feat, edge_lab = [], [], [], []

    for u, v, attr in G.edges(data=True):
        edge_u.append(node_mapping[u])
        edge_v.append(node_mapping[v])
        edge_feat.append(attr["features"])
        edge_lab.append(attr["label"])

    edge_index = torch.tensor([edge_u, edge_v], dtype=torch.long)
    edge_attr = torch.tensor(np.stack(edge_feat), dtype=torch.float)
    labels = torch.tensor(edge_lab, dtype=torch.long)

    num_nodes = len(node_mapping)
    node_features = torch.zeros((num_nodes, edge_attr.shape[1]), dtype=torch.float)
    for i in range(edge_index.shape[1]):
        u, v = edge_index[:, i]
        node_features[u] += edge_attr[i]
        node_features[v] += edge_attr[i]

    # Normalize node features safely
    norms = node_features.norm(dim=1, keepdim=True)
    norms[norms == 0] = 1.0
    node_features = node_features / norms

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=labels)


# =========================================================
# 🔹 STEP 4: Predict + Explain (hardened)
# =========================================================
def predict_with_gnn(model_path: str, data: Data) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = data.edge_attr.shape[1]
    model = EdgeGNN(in_channels=in_channels, hidden_channels=64, out_channels=2).to(device)

    if not os.path.exists(model_path):
        print(f"⚠️ Model file not found at {model_path}. Returning zeros.")
        # Return zeros matching number of edges
        return np.zeros(data.edge_index.shape[1], dtype=int)

    try:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    except Exception as e:
        print(f"⚠️ Failed to load model ({e}). Returning zeros.")
        return np.zeros(data.edge_index.shape[1], dtype=int)

    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device))
        preds = out.argmax(dim=1).cpu().numpy()
    return preds


def explain_with_llm(predictions: List[int], logs: List[Dict[str, Any]]) -> str:
    prompt = f"""
You are a cybersecurity analyst. Based on these structured network log entries (JSON):
{json.dumps(logs, indent=2)}

The GNN intrusion detector produced these binary predictions (1=intrusion, 0=normal):
{predictions}

Explain clearly what suspicious behavior (if any) is present and which log entries are most suspicious. Keep the answer brief and actionable. Also provide possible mitigation steps.
"""
    resp = mistral_prompt(prompt)
    if not resp:
        return "Mistral unavailable — no explanation generated."
    return resp


# =========================================================
# 🔹 MAIN PIPELINE
# =========================================================
if __name__ == "__main__":
    raw_logs = """
    10.0.0.5 connected to 192.168.1.10 sending 200 packets of 1MB each lasting 0.4s
    Suspicious IP 172.16.0.2 massive rapid small packets repeatedly to port 443
    High HTTP traffic between 10.0.0.8 and 10.0.0.12 lasting 0.2s
    """

    print("🔹 Preprocessing logs with Mistral...")
    structured_logs = preprocess_logs_with_llm(raw_logs)
    print("✅ Structured logs:")
    print(json.dumps(structured_logs, indent=2))

    print("🔹 Building graph from structured logs...")
    try:
        data = build_graph_from_json(structured_logs)
        print("✅ Graph built:", data)
    except Exception as e:
        print(f"❌ Failed to build graph: {e}")
        raise

    print("🔹 Running GNN inference...")
    model_path = r"/mnt/c/Users/Dell/Desktop/AIML_B_7th_Sem/edge_gnn_model.pth"
    preds = predict_with_gnn(model_path, data)
    print(f"Predictions: {preds}")

    print("🔹 Explaining with Mistral...")
    explanation = explain_with_llm(preds.tolist(), structured_logs)
    print("\n==============================")
    print("🧠 Intrusion Explanation:")
    print("==============================")
    print(explanation)