"""
admin_dashboard.py
Admin Dashboard cho chatbot tuyển sinh HaUI.
Mount vào Chainlit qua custom_auth hoặc chạy song song port 8001.

Chạy độc lập:
    uvicorn admin_dashboard:app --port 8001 --reload

Tính năng:
  1. Thống kê tổng quan (tổng câu hỏi, latency, sessions)
  2. Phân bố intent — biểu đồ
  3. Câu hỏi phổ biến nhất
  4. Giám sát latency theo ngày
  5. Log real-time (50 lượt gần nhất)
  6. Cập nhật dữ liệu JSON tuyển sinh
"""

import json
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import logger từ src
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.pipeline.logger import (
    init_db, get_stats_overview, get_intent_stats,
    get_popular_questions, get_latency_trend,
    get_recent_logs, get_slow_queries, get_hourly_traffic,
)

DATA_DIR   = Path(__file__).resolve().parent / "data" / "processed"
# PID file để track Chainlit process
CHAINLIT_PID_FILE = Path(__file__).resolve().parent / ".chainlit_pid"
NGANH_DIR  = DATA_DIR / "nganh"
NGANH_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".json", ".md", ".pdf", ".docx"}


def _rebuild_and_restart_chainlit():
    """
    Chạy trong background thread:
      1. Rebuild ChromaDB index
      2. Restart Chainlit process (kill + respawn)
    """
    project_root = Path(__file__).resolve().parent

    # ── Bước 1: Rebuild index ──────────────────────────────────────────────
    print("[Admin] Bắt đầu rebuild index...")
    try:
        result = subprocess.run(
            [sys.executable, "src/indexing/build_index.py"],
            cwd=str(project_root),
            capture_output=True, text=True, timeout=180
        )
        if result.returncode == 0:
            print("[Admin] ✅ Rebuild index xong")
        else:
            print(f"[Admin] ❌ Rebuild thất bại:\n{result.stderr[-300:]}")
            return
    except subprocess.TimeoutExpired:
        print("[Admin] ❌ Rebuild timeout")
        return
    except Exception as e:
        print(f"[Admin] ❌ Lỗi rebuild: {e}")
        return

    # ── Bước 2: Restart Chainlit ───────────────────────────────────────────
    print("[Admin] Đang restart Chainlit...")
    try:
        # Đọc PID file
        if CHAINLIT_PID_FILE.exists():
            pid = int(CHAINLIT_PID_FILE.read_text().strip())
            try:
                os.kill(pid, 15)   # SIGTERM — graceful shutdown
                import time; time.sleep(2)
            except ProcessLookupError:
                pass   # Process đã tắt rồi

        # Respawn Chainlit
        proc = subprocess.Popen(
            [sys.executable, "-m", "chainlit", "run", "app_chainlit.py",
             "--port", "8000", "--headless"],
            cwd=str(project_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        CHAINLIT_PID_FILE.write_text(str(proc.pid))
        print(f"[Admin] ✅ Chainlit khởi động lại (PID={proc.pid})")
    except Exception as e:
        print(f"[Admin] ❌ Lỗi restart Chainlit: {e}")
        print("[Admin] → Vui lòng restart Chainlit thủ công")

app = FastAPI(title="HaUI Chatbot Admin", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

init_db()

# ── API endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/overview")
def api_overview():
    return get_stats_overview()

@app.get("/api/intents")
def api_intents(days: int = 7):
    return get_intent_stats(days)

@app.get("/api/popular")
def api_popular(limit: int = 20):
    return get_popular_questions(limit)

@app.get("/api/latency")
def api_latency(days: int = 7):
    return get_latency_trend(days)

@app.get("/api/logs")
def api_logs(limit: int = 50):
    return get_recent_logs(limit)

@app.get("/api/slow")
def api_slow(threshold: int = 5000):
    return get_slow_queries(threshold)

@app.get("/api/traffic")
def api_traffic(days: int = 1):
    return get_hourly_traffic(days)

@app.get("/api/files")
def api_files():
    """Liệt kê các file JSON dữ liệu tuyển sinh."""
    files = []
    for f in DATA_DIR.glob("*.json"):
        stat = f.stat()
        files.append({
            "name"    : f.name,
            "size"    : stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
        })
    return sorted(files, key=lambda x: x["name"])

@app.get("/api/file/{filename}")
def api_get_file(filename: str):
    """Đọc nội dung file JSON."""
    path = DATA_DIR / filename
    if not path.exists() or not filename.endswith(".json"):
        raise HTTPException(404, "File không tồn tại")
    return json.loads(path.read_text(encoding="utf-8"))

@app.post("/api/file/{filename}")
async def api_update_file(filename: str, file: UploadFile = File(...)):
    """
    Upload file dữ liệu tuyển sinh:
      - .json  → lưu vào data/processed/
      - .md    → lưu vào data/processed/nganh/
      - .pdf   → extract text → lưu .md vào data/processed/nganh/
      - .docx  → extract text → lưu .md vào data/processed/nganh/
    Sau khi upload .md/.pdf/.docx cần rebuild ChromaDB index.
    """
    ext     = Path(filename).suffix.lower()
    raw     = await file.read()
    md_name = Path(filename).stem + ".md"

    if ext == ".json":
        path   = DATA_DIR / filename
        backup = DATA_DIR / f"{filename}.bak"
        if path.exists():
            shutil.copy(path, backup)
        try:
            json.loads(raw)
            path.write_bytes(raw)
            return {"ok": True, "message": f"✅ Đã cập nhật {filename}"}
        except json.JSONDecodeError as e:
            if backup.exists():
                shutil.copy(backup, path)
            raise HTTPException(400, f"JSON không hợp lệ: {e}")

    elif ext == ".md":
        path = NGANH_DIR / filename
        if path.exists():
            shutil.copy(path, NGANH_DIR / f"{filename}.bak")
        path.write_bytes(raw)
        # Tự động rebuild + restart Chainlit
        threading.Thread(target=_rebuild_and_restart_chainlit, daemon=True).start()
        return {"ok": True,
                "message": f"✅ Đã lưu {filename} — đang rebuild index và restart chatbot...",
                "rebuild": True, "auto": True}

    elif ext == ".pdf":
        try:
            import pypdf, io
            reader     = pypdf.PdfReader(io.BytesIO(raw))
            text       = "\n\n".join(p.extract_text() or "" for p in reader.pages)
            md_content = f"# {Path(filename).stem}\n\n{text}"
            path = NGANH_DIR / md_name
            path.write_text(md_content, encoding="utf-8")
            # Tự động rebuild + restart Chainlit
            threading.Thread(target=_rebuild_and_restart_chainlit, daemon=True).start()
            return {"ok": True,
                    "message": f"✅ Đã convert {filename} → {md_name} ({len(reader.pages)} trang) — đang rebuild index...",
                    "saved_as": md_name, "rebuild": True, "auto": True}
        except ImportError:
            raise HTTPException(500, "Cần cài: pip install pypdf")
        except Exception as e:
            raise HTTPException(500, f"Lỗi đọc PDF: {e}")

    elif ext == ".docx":
        try:
            import docx2txt, io
            text       = docx2txt.process(io.BytesIO(raw))
            md_content = f"# {Path(filename).stem}\n\n{text}"
            path = NGANH_DIR / md_name
            path.write_text(md_content, encoding="utf-8")
            # Tự động rebuild + restart Chainlit
            threading.Thread(target=_rebuild_and_restart_chainlit, daemon=True).start()
            return {"ok": True,
                    "message": f"✅ Đã convert {filename} → {md_name} — đang rebuild index...",
                    "saved_as": md_name, "rebuild": True, "auto": True}
        except ImportError:
            raise HTTPException(500, "Cần cài: pip install docx2txt")
        except Exception as e:
            raise HTTPException(500, f"Lỗi đọc DOCX: {e}")

    raise HTTPException(400, f"Không hỗ trợ định dạng {ext}. Chấp nhận: .json .md .pdf .docx")

@app.post("/api/rebuild")
async def api_rebuild():
    """Rebuild ChromaDB index sau khi upload văn bản mới."""
    import subprocess, sys
    try:
        result = subprocess.run(
            [sys.executable, "src/indexing/build_index.py"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            return {"ok": True, "message": "✅ Rebuild index thành công!", "log": result.stdout[-500:]}
        else:
            return {"ok": False, "message": "❌ Rebuild thất bại", "log": result.stderr[-500:]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "message": "❌ Timeout sau 120 giây"}
    except Exception as e:
        return {"ok": False, "message": f"❌ Lỗi: {e}"}


@app.post("/api/reload-chatbot")
async def api_reload():
    """
    Reload Retriever trong chatbot sau khi rebuild index.
    Không cần restart Chainlit — chỉ reload ChromaDB.
    """
    try:
        # Import chatbot module và reset retriever
        import importlib
        import sys

        # Xóa cache module retriever để load lại
        mods_to_reload = [k for k in sys.modules if "retriever" in k or "embedder" in k]
        for mod in mods_to_reload:
            del sys.modules[mod]

        # Reset global chatbot trong app_chainlit
        chainlit_mod = sys.modules.get("app_chainlit") or sys.modules.get("__main__")
        if chainlit_mod and hasattr(chainlit_mod, "_GLOBAL_CHATBOT"):
            chainlit_mod._GLOBAL_CHATBOT = None
            return {"ok": True, "message": "✅ Chatbot sẽ reload tự động ở câu hỏi tiếp theo"}

        return {"ok": True, "message": "✅ Module cache đã xóa — restart Chainlit để áp dụng"}
    except Exception as e:
        return {"ok": False, "message": f"❌ {e}"}

# ── Dashboard HTML ─────────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HaUI Chatbot — Admin Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {
    --red:     #C8102E;
    --red-dim: #9b0d22;
    --red-bg:  #fff0f2;
    --dark:    #0f1117;
    --card:    #ffffff;
    --border:  #e8eaf0;
    --text:    #1a1d2e;
    --muted:   #6b7280;
    --success: #16a34a;
    --warn:    #d97706;
    --info:    #2563eb;
    --shadow:  0 1px 3px rgba(0,0,0,.08), 0 4px 16px rgba(0,0,0,.06);
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Be Vietnam Pro', sans-serif;
    background: #f4f6fb;
    color: var(--text);
    min-height: 100vh;
  }

  /* ── Sidebar ── */
  .sidebar {
    position: fixed; top: 0; left: 0;
    width: 220px; height: 100vh;
    background: var(--dark);
    display: flex; flex-direction: column;
    padding: 0;
    z-index: 100;
  }
  .sidebar-logo {
    padding: 20px 20px 16px;
    border-bottom: 1px solid rgba(255,255,255,.08);
  }
  .sidebar-logo .badge {
    display: inline-block;
    background: var(--red);
    color: #fff;
    font-size: 10px; font-weight: 700;
    letter-spacing: .08em;
    padding: 2px 8px; border-radius: 4px;
    margin-bottom: 6px;
  }
  .sidebar-logo h2 {
    color: #fff; font-size: 15px; font-weight: 600;
    line-height: 1.3;
  }
  .sidebar-logo p { color: rgba(255,255,255,.4); font-size: 11px; margin-top: 2px; }

  .nav { flex: 1; padding: 12px 0; }
  .nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 20px;
    color: rgba(255,255,255,.55);
    font-size: 13px; font-weight: 500;
    cursor: pointer;
    border-left: 3px solid transparent;
    transition: all .15s;
  }
  .nav-item:hover  { color: #fff; background: rgba(255,255,255,.05); }
  .nav-item.active { color: #fff; border-left-color: var(--red); background: rgba(200,16,46,.12); }
  .nav-item .icon  { width: 18px; text-align: center; font-size: 15px; }

  .sidebar-footer {
    padding: 16px 20px;
    border-top: 1px solid rgba(255,255,255,.08);
    color: rgba(255,255,255,.3);
    font-size: 11px;
  }
  .status-dot {
    display: inline-block; width: 6px; height: 6px;
    border-radius: 50%; background: #22c55e;
    margin-right: 6px; animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%,100% { opacity: 1; } 50% { opacity: .4; }
  }

  /* ── Main ── */
  .main {
    margin-left: 220px;
    padding: 28px 32px;
    min-height: 100vh;
  }

  .page { display: none; }
  .page.active { display: block; }

  /* ── Header ── */
  .page-header {
    display: flex; justify-content: space-between; align-items: flex-end;
    margin-bottom: 24px;
  }
  .page-header h1 { font-size: 22px; font-weight: 700; }
  .page-header p  { font-size: 13px; color: var(--muted); margin-top: 3px; }
  .refresh-btn {
    padding: 8px 16px; border-radius: 8px;
    background: var(--red); color: #fff;
    border: none; font-size: 13px; font-weight: 500;
    cursor: pointer; font-family: inherit;
    transition: background .15s;
  }
  .refresh-btn:hover { background: var(--red-dim); }

  /* ── Stat cards ── */
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px; margin-bottom: 24px;
  }
  .stat-card {
    background: var(--card); border-radius: 12px;
    padding: 20px; box-shadow: var(--shadow);
    border: 1px solid var(--border);
    position: relative; overflow: hidden;
  }
  .stat-card::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--red);
  }
  .stat-card .label { font-size: 12px; color: var(--muted); font-weight: 500; text-transform: uppercase; letter-spacing: .06em; }
  .stat-card .value { font-size: 28px; font-weight: 700; margin: 6px 0 2px; line-height: 1; }
  .stat-card .sub   { font-size: 12px; color: var(--muted); }
  .stat-card .icon  { position: absolute; right: 16px; top: 16px; font-size: 24px; opacity: .12; }

  /* ── Chart + Table cards ── */
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
  .grid-3 { display: grid; grid-template-columns: 2fr 1fr; gap: 16px; margin-bottom: 20px; }

  .card {
    background: var(--card); border-radius: 12px;
    padding: 20px; box-shadow: var(--shadow);
    border: 1px solid var(--border);
  }
  .card-title {
    font-size: 14px; font-weight: 600;
    margin-bottom: 16px;
    display: flex; justify-content: space-between; align-items: center;
  }
  .card-title span { font-size: 11px; color: var(--muted); font-weight: 400; }
  .chart-wrap { position: relative; height: 220px; }

  /* ── Table ── */
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th {
    text-align: left; padding: 8px 12px;
    font-size: 11px; font-weight: 600;
    color: var(--muted); text-transform: uppercase; letter-spacing: .05em;
    border-bottom: 1px solid var(--border);
  }
  td { padding: 9px 12px; border-bottom: 1px solid #f3f4f6; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #fafbff; }

  .badge-intent {
    display: inline-block; padding: 2px 8px;
    border-radius: 4px; font-size: 11px; font-weight: 500;
    background: var(--red-bg); color: var(--red);
  }
  .badge-rule   { background: #f0fdf4; color: #16a34a; }
  .badge-claude { background: #eff6ff; color: #2563eb; }

  .latency-bar {
    display: flex; align-items: center; gap: 8px;
  }
  .latency-bar .bar {
    height: 6px; border-radius: 3px;
    background: linear-gradient(90deg, #22c55e, var(--red));
    transition: width .3s;
  }

  /* ── File manager ── */
  .file-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px,1fr)); gap: 12px; }
  .file-card {
    border: 1px solid var(--border); border-radius: 10px;
    padding: 16px; cursor: pointer;
    transition: all .15s; background: var(--card);
  }
  .file-card:hover { border-color: var(--red); box-shadow: 0 0 0 3px var(--red-bg); }
  .file-card .fname { font-size: 13px; font-weight: 600; margin-bottom: 4px; word-break: break-all; }
  .file-card .fmeta { font-size: 11px; color: var(--muted); }

  .upload-zone {
    border: 2px dashed var(--border); border-radius: 10px;
    padding: 32px; text-align: center;
    cursor: pointer; transition: all .2s;
    background: #fafbff;
    margin-bottom: 16px;
  }
  .upload-zone:hover, .upload-zone.drag { border-color: var(--red); background: var(--red-bg); }
  .upload-zone p { font-size: 13px; color: var(--muted); margin-top: 8px; }

  /* ── JSON editor ── */
  .json-editor {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; line-height: 1.6;
    width: 100%; min-height: 400px;
    border: 1px solid var(--border); border-radius: 8px;
    padding: 12px; resize: vertical;
    background: #fafbff; color: var(--text);
    outline: none;
  }
  .json-editor:focus { border-color: var(--red); }

  .btn {
    padding: 8px 18px; border-radius: 8px;
    font-size: 13px; font-weight: 500;
    cursor: pointer; border: none; font-family: inherit;
    transition: all .15s;
  }
  .btn-primary { background: var(--red); color: #fff; }
  .btn-primary:hover { background: var(--red-dim); }
  .btn-ghost { background: transparent; color: var(--muted); border: 1px solid var(--border); }
  .btn-ghost:hover { border-color: var(--red); color: var(--red); }

  .toast {
    position: fixed; bottom: 24px; right: 24px;
    padding: 12px 20px; border-radius: 8px;
    font-size: 13px; font-weight: 500;
    box-shadow: 0 4px 20px rgba(0,0,0,.15);
    z-index: 9999; transform: translateY(60px); opacity: 0;
    transition: all .3s;
  }
  .toast.show { transform: translateY(0); opacity: 1; }
  .toast.success { background: #16a34a; color: #fff; }
  .toast.error   { background: var(--red); color: #fff; }

  @media (max-width: 1100px) {
    .stat-grid { grid-template-columns: repeat(2,1fr); }
    .grid-2, .grid-3 { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<!-- Sidebar -->
<aside class="sidebar">
  <div class="sidebar-logo">
    <div class="badge">ADMIN</div>
    <h2>HaUI Chatbot</h2>
    <p>Dashboard quản trị</p>
  </div>
  <nav class="nav">
    <div class="nav-item active" onclick="showPage('overview')">
      <span class="icon">📊</span> Tổng quan
    </div>
    <div class="nav-item" onclick="showPage('questions')">
      <span class="icon">💬</span> Câu hỏi
    </div>
    <div class="nav-item" onclick="showPage('performance')">
      <span class="icon">⚡</span> Hiệu suất
    </div>
    <div class="nav-item" onclick="showPage('logs')">
      <span class="icon">📋</span> Logs
    </div>
    <div class="nav-item" onclick="showPage('data')">
      <span class="icon">🗄️</span> Dữ liệu
    </div>
  </nav>
  <div class="sidebar-footer">
    <span class="status-dot"></span>
    Hệ thống hoạt động
  </div>
</aside>

<!-- Main -->
<main class="main">

  <!-- ── OVERVIEW ── -->
  <div id="page-overview" class="page active">
    <div class="page-header">
      <div>
        <h1>Tổng quan</h1>
        <p>Thống kê hoạt động chatbot tuyển sinh HaUI</p>
      </div>
      <button class="refresh-btn" onclick="loadOverview()">↻ Làm mới</button>
    </div>

    <div class="stat-grid">
      <div class="stat-card">
        <div class="label">Tổng câu hỏi</div>
        <div class="value" id="stat-total">—</div>
        <div class="sub">Kể từ khi triển khai</div>
        <div class="icon">💬</div>
      </div>
      <div class="stat-card">
        <div class="label">Hôm nay</div>
        <div class="value" id="stat-today">—</div>
        <div class="sub">Câu hỏi trong ngày</div>
        <div class="icon">📅</div>
      </div>
      <div class="stat-card">
        <div class="label">Latency TB</div>
        <div class="value" id="stat-latency">—</div>
        <div class="sub">Mili-giây trước streaming</div>
        <div class="icon">⚡</div>
      </div>
      <div class="stat-card">
        <div class="label">Sessions</div>
        <div class="value" id="stat-sessions">—</div>
        <div class="sub">Người dùng khác nhau</div>
        <div class="icon">👤</div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-title">Phân bố Intent <span>7 ngày gần nhất</span></div>
        <div class="chart-wrap"><canvas id="chart-intent"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Traffic theo giờ <span>Hôm nay</span></div>
        <div class="chart-wrap"><canvas id="chart-traffic"></canvas></div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Latency theo ngày <span>7 ngày</span></div>
      <div class="chart-wrap"><canvas id="chart-latency"></canvas></div>
    </div>
  </div>

  <!-- ── QUESTIONS ── -->
  <div id="page-questions" class="page">
    <div class="page-header">
      <div><h1>Câu hỏi phổ biến</h1><p>Top 20 câu hỏi được hỏi nhiều nhất</p></div>
      <button class="refresh-btn" onclick="loadQuestions()">↻ Làm mới</button>
    </div>
    <div class="card">
      <table>
        <thead><tr>
          <th>#</th><th>Câu hỏi</th><th>Lần hỏi</th>
          <th>Latency TB</th><th>Lần cuối</th>
        </tr></thead>
        <tbody id="table-popular"></tbody>
      </table>
    </div>
  </div>

  <!-- ── PERFORMANCE ── -->
  <div id="page-performance" class="page">
    <div class="page-header">
      <div><h1>Hiệu suất hệ thống</h1><p>Latency và câu hỏi chậm</p></div>
      <button class="refresh-btn" onclick="loadPerformance()">↻ Làm mới</button>
    </div>
    <div class="card" style="margin-bottom:20px">
      <div class="card-title">Câu hỏi phản hồi chậm <span>> 5 giây</span></div>
      <table>
        <thead><tr>
          <th>Thời gian</th><th>Câu hỏi</th><th>Intent</th>
          <th>Latency</th><th>Method</th>
        </tr></thead>
        <tbody id="table-slow"></tbody>
      </table>
    </div>
  </div>

  <!-- ── LOGS ── -->
  <div id="page-logs" class="page">
    <div class="page-header">
      <div><h1>Log hội thoại</h1><p>50 lượt gần nhất</p></div>
      <button class="refresh-btn" onclick="loadLogs()">↻ Làm mới</button>
    </div>
    <div class="card">
      <table>
        <thead><tr>
          <th>Thời gian</th><th>Câu hỏi</th><th>Intent</th>
          <th>Method</th><th>Conf</th><th>Latency</th>
        </tr></thead>
        <tbody id="table-logs"></tbody>
      </table>
    </div>
  </div>

  <!-- ── DATA ── -->
  <div id="page-data" class="page">
    <div class="page-header">
      <div><h1>Quản lý dữ liệu</h1><p>Cập nhật file JSON tuyển sinh</p></div>
      <button class="refresh-btn" onclick="loadFiles()">↻ Làm mới</button>
    </div>

    <div class="card" style="margin-bottom:20px">
      <div class="card-title">Upload file JSON mới</div>
      <div class="upload-zone" id="upload-zone"
           onclick="document.getElementById('file-input').click()"
           ondragover="event.preventDefault();this.classList.add('drag')"
           ondragleave="this.classList.remove('drag')"
           ondrop="handleDrop(event)">
        <div style="font-size:32px">📂</div>
        <p><strong>Nhấn để chọn</strong> hoặc kéo thả file JSON vào đây</p>
        <p style="font-size:11px;margin-top:4px">
          .json (điểm chuẩn, học phí...) &nbsp;·&nbsp; .md (mô tả ngành) &nbsp;·&nbsp; .pdf .docx (văn bản tuyển sinh)
        </p>
        <p style="font-size:11px;color:#C8102E;margin-top:2px">⚠ Sau khi upload .md/.pdf/.docx cần rebuild index ChromaDB</p>
      </div>
      <input type="file" id="file-input" accept=".json,.md,.pdf,.docx" style="display:none"
             onchange="handleFileSelect(event)">
    </div>

    <div class="card">
      <div class="card-title">File dữ liệu hiện tại <span id="file-count"></span></div>
      <div class="file-grid" id="file-grid"></div>
    </div>

    <!-- JSON Editor Modal -->
    <div id="editor-modal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:200;padding:32px">
      <div style="background:#fff;border-radius:14px;max-width:900px;margin:0 auto;padding:24px;max-height:90vh;overflow:auto">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
          <h3 id="editor-title" style="font-size:15px;font-weight:600"></h3>
          <button class="btn btn-ghost" onclick="closeEditor()">✕ Đóng</button>
        </div>
        <textarea class="json-editor" id="json-content"></textarea>
        <div style="display:flex;gap:8px;margin-top:12px">
          <button class="btn btn-primary" onclick="saveJson()">💾 Lưu thay đổi</button>
          <button class="btn btn-ghost" onclick="closeEditor()">Hủy</button>
        </div>
      </div>
    </div>
  </div>

</main>

<!-- Toast -->
<div class="toast" id="toast"></div>

<script>
// ── State ──────────────────────────────────────────────────────────────
let charts = {};
let currentFile = null;

const INTENT_LABELS = {
  JSON_DIEM_CHUAN:      '📊 Điểm chuẩn',
  JSON_HOC_PHI:         '💰 Học phí',
  JSON_CHI_TIEU_TO_HOP: '📋 Chỉ tiêu',
  JSON_QUY_DOI_DIEM:    '🔢 Tính điểm',
  JSON_DAU_TRUOT:       '✅ Đậu/trượt',
  RAG_MO_TA_NGANH:      '🎓 Mô tả ngành',
  RAG_FAQ:              '❓ FAQ',
  RAG_TRUONG_HOC_BONG:  '🏫 Trường',
  UNKNOWN:              '🔍 Khác',
  GREETING:             '👋 Chào hỏi',
  OFF_TOPIC:            '🚫 Off-topic',
};

const COLORS = ['#C8102E','#2563eb','#16a34a','#d97706','#7c3aed','#0891b2','#be185d','#059669','#dc2626','#4f46e5'];

// ── Navigation ─────────────────────────────────────────────────────────
function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  event.currentTarget.classList.add('active');
  // Lazy load data
  if (name === 'overview')    loadOverview();
  if (name === 'questions')   loadQuestions();
  if (name === 'performance') loadPerformance();
  if (name === 'logs')        loadLogs();
  if (name === 'data')        loadFiles();
}

// ── API helpers ─────────────────────────────────────────────────────────
async function api(path) {
  const r = await fetch('/api/' + path);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function toast(msg, type='success') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'toast show ' + type;
  setTimeout(() => el.classList.remove('show'), 3000);
}

function fmt_ms(ms) {
  if (!ms) return '—';
  return ms < 1000 ? ms + 'ms' : (ms/1000).toFixed(1) + 's';
}

function fmt_time(ts) {
  if (!ts) return '—';
  return ts.slice(0,16).replace('T',' ');
}

function truncate(str, n=50) {
  if (!str) return '—';
  return str.length > n ? str.slice(0,n) + '…' : str;
}

// ── Overview ────────────────────────────────────────────────────────────
async function loadOverview() {
  try {
    const [ov, intents, latency, traffic] = await Promise.all([
      api('overview'), api('intents?days=7'),
      api('latency?days=7'), api('traffic?days=1'),
    ]);

    document.getElementById('stat-total').textContent    = (ov.total||0).toLocaleString();
    document.getElementById('stat-today').textContent    = (ov.today||0).toLocaleString();
    document.getElementById('stat-latency').textContent  = fmt_ms(ov.avg_latency);
    document.getElementById('stat-sessions').textContent = (ov.total_sessions||0).toLocaleString();

    renderIntentChart(intents);
    renderLatencyChart(latency);
    renderTrafficChart(traffic);
  } catch(e) { toast('Lỗi tải dữ liệu: ' + e.message, 'error'); }
}

function renderIntentChart(data) {
  const ctx = document.getElementById('chart-intent');
  if (charts.intent) charts.intent.destroy();
  charts.intent = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: data.map(d => INTENT_LABELS[d.intent] || d.intent),
      datasets: [{ data: data.map(d => d.count), backgroundColor: COLORS,
                   borderWidth: 2, borderColor: '#fff' }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'right', labels: { font:{size:11}, boxWidth:12 } } }
    }
  });
}

function renderLatencyChart(data) {
  const ctx = document.getElementById('chart-latency');
  if (charts.latency) charts.latency.destroy();
  charts.latency = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.map(d => d.date),
      datasets: [
        { label: 'TB (ms)', data: data.map(d => d.avg_latency),
          borderColor: '#C8102E', backgroundColor: 'rgba(200,16,46,.08)',
          fill: true, tension: .4, pointRadius: 4 },
        { label: 'Max (ms)', data: data.map(d => d.max_latency),
          borderColor: '#d97706', borderDash:[4,2],
          fill: false, tension: .4, pointRadius: 2 },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { font:{size:11} } } },
      scales: { y: { beginAtZero: true, grid:{color:'#f0f0f0'} } }
    }
  });
}

function renderTrafficChart(data) {
  const ctx = document.getElementById('chart-traffic');
  if (charts.traffic) charts.traffic.destroy();
  // Fill missing hours
  const hours = Array.from({length:24},(_,i)=>String(i).padStart(2,'0')+':00');
  const map   = Object.fromEntries(data.map(d=>[d.hour,d.count]));
  charts.traffic = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: hours,
      datasets: [{ label: 'Câu hỏi', data: hours.map(h=>map[h]||0),
                   backgroundColor: 'rgba(200,16,46,.7)', borderRadius: 4 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, grid:{color:'#f0f0f0'} } }
    }
  });
}

// ── Questions ───────────────────────────────────────────────────────────
async function loadQuestions() {
  try {
    const data = await api('popular?limit=20');
    const tbody = document.getElementById('table-popular');
    tbody.innerHTML = data.map((r,i) => `
      <tr>
        <td style="color:var(--muted);font-weight:600">${i+1}</td>
        <td>${truncate(r.user_message, 70)}</td>
        <td><strong>${r.count}</strong></td>
        <td>${fmt_ms(r.avg_latency)}</td>
        <td style="color:var(--muted)">${fmt_time(r.last_seen)}</td>
      </tr>`).join('');
  } catch(e) { toast('Lỗi: ' + e.message, 'error'); }
}

// ── Performance ─────────────────────────────────────────────────────────
async function loadPerformance() {
  try {
    const slow = await api('slow?threshold=5000');
    const tbody = document.getElementById('table-slow');
    if (!slow.length) {
      tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--muted);padding:32px">✅ Không có câu hỏi chậm > 5s</td></tr>';
      return;
    }
    tbody.innerHTML = slow.map(r => `
      <tr>
        <td style="color:var(--muted)">${fmt_time(r.timestamp)}</td>
        <td>${truncate(r.user_message, 60)}</td>
        <td><span class="badge-intent">${INTENT_LABELS[r.intent]||r.intent}</span></td>
        <td style="color:var(--red);font-weight:600">${fmt_ms(r.latency_ms)}</td>
        <td><span class="badge-${r.method==='rule'?'rule':'claude'}">${r.method}</span></td>
      </tr>`).join('');
  } catch(e) { toast('Lỗi: ' + e.message, 'error'); }
}

// ── Logs ────────────────────────────────────────────────────────────────
async function loadLogs() {
  try {
    const data = await api('logs?limit=50');
    const tbody = document.getElementById('table-logs');
    tbody.innerHTML = data.map(r => `
      <tr>
        <td style="color:var(--muted);white-space:nowrap">${fmt_time(r.timestamp)}</td>
        <td>${truncate(r.user_message, 55)}</td>
        <td><span class="badge-intent">${INTENT_LABELS[r.intent]||r.intent||'—'}</span></td>
        <td><span class="badge-${r.method==='rule'?'rule':'claude'}">${r.method||'—'}</span></td>
        <td style="color:var(--muted)">${r.confidence?Math.round(r.confidence*100)+'%':'—'}</td>
        <td>${fmt_ms(r.latency_ms)}</td>
      </tr>`).join('');
  } catch(e) { toast('Lỗi: ' + e.message, 'error'); }
}

// ── Data manager ─────────────────────────────────────────────────────────
async function loadFiles() {
  try {
    const data = await api('files');
    document.getElementById('file-count').textContent = data.length + ' file';
    const grid = document.getElementById('file-grid');
    grid.innerHTML = data.map(f => `
      <div class="file-card" onclick="openEditor('${f.name}', '${f.type}')">
        <div class="fname">${f.type==="json"?"🗃️":"📝"} ${f.name}</div>
        <div class="fmeta">${(f.size/1024).toFixed(1)} KB · ${f.modified}
          <span style="background:${f.type==="json"?"#eff6ff":"#f0fdf4"};color:${f.type==="json"?"#2563eb":"#16a34a"};padding:1px 6px;border-radius:3px;font-size:10px;margin-left:4px">${f.type.toUpperCase()}</span>
        </div>
      </div>`).join('');
  } catch(e) { toast('Lỗi: ' + e.message, 'error'); }
}

async function openEditor(filename, type) {
  try {
    const res = await api('file/' + filename);
    currentFile = filename;
    currentFileType = type || (filename.endsWith('.json') ? 'json' : 'md');
    document.getElementById('editor-title').textContent = '✏️ ' + filename;
    if (currentFileType === 'json') {
      document.getElementById('json-content').value = JSON.stringify(res.content, null, 2);
    } else {
      document.getElementById('json-content').value = res.content;
    }
    document.getElementById('editor-modal').style.display = 'block';
  } catch(e) { toast('Lỗi mở file: ' + e.message, 'error'); }
}

function closeEditor() {
  document.getElementById('editor-modal').style.display = 'none';
  currentFile = null;
}

let currentFileType = 'json';

async function saveJson() {
  if (!currentFile) return;
  const textContent = document.getElementById('json-content').value;
  try {
    let blob;
    if (currentFileType === 'json') {
      JSON.parse(textContent); // validate JSON
      blob = new Blob([textContent], {type:'application/json'});
    } else {
      blob = new Blob([textContent], {type:'text/markdown'});
    }
    const form = new FormData();
    form.append('file', blob, currentFile);
    const r   = await fetch('/api/file/' + currentFile, {method:'POST', body:form});
    const res = await r.json();
    if (res.ok) {
      toast(res.message);
      if (res.rebuild) {
        setTimeout(() => toast('⚠ Nhớ chạy: python src/indexing/build_index.py', 'error'), 2000);
      }
      closeEditor();
    } else toast('Lỗi: ' + res.message, 'error');
  } catch(e) { toast('Nội dung không hợp lệ: ' + e.message, 'error'); }
}

// Upload file mới
function handleFileSelect(event) {
  const file = event.target.files[0];
  if (file) uploadFile(file);
}

function handleDrop(event) {
  event.preventDefault();
  document.getElementById('upload-zone').classList.remove('drag');
  const file = event.dataTransfer.files[0];
  if (file) uploadFile(file);
}

async function uploadFile(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  if (!['json','md','pdf','docx'].includes(ext)) {
    toast('Chỉ chấp nhận .json .md .pdf .docx', 'error'); return;
  }
  const form = new FormData();
  form.append('file', file, file.name);
  try {
    const r   = await fetch('/api/file/' + file.name, {method:'POST', body:form});
    const res = await r.json();
    if (res.ok) {
      toast(res.message);
      if (res.auto) {
        // Đang rebuild tự động — hiện progress và poll trạng thái
        setTimeout(() => toast('⏳ Đang rebuild ChromaDB index...', 'success'), 1500);
        setTimeout(() => toast('⏳ Đang khởi động lại chatbot...', 'success'), 4000);
        setTimeout(() => toast('✅ Chatbot đã cập nhật dữ liệu mới!', 'success'), 12000);
      }
      loadFiles();
    } else toast('Lỗi: ' + res.message, 'error');
  } catch(e) { toast('Lỗi upload: ' + e.message, 'error'); }
}

// ── Init ─────────────────────────────────────────────────────────────────
loadOverview();
setInterval(loadOverview, 30000); // auto-refresh mỗi 30s
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML


# ── Chạy độc lập ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("Admin Dashboard: http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)