from __future__ import annotations

import io
from pathlib import Path
from huggingface_hub import hf_hub_download
from typing import Any, Dict

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image

from dataset import (
    SENTIMENT_LABELS,
    DataConfig,
    load_image_processor,
    load_tokenizer,
)
from model import (
    STREAM_DISPLAY_NAMES,
    VALID_MODES,
    ImplicitMABSAModel,
    ModelConfig,
)
from utils import get_device, load_checkpoint

CHECKPOINT_PATH = Path("saved_models/implicit_mabsa.pt")

app = FastAPI(title="ImplicitMABSA Live Demo", version="1.0.0")

# Global state loaded once at startup
STATE: Dict[str, Any] = {}


@app.on_event("startup")
def load_models() -> None:
    if not CHECKPOINT_PATH.is_file():
        hf_hub_download(
            repo_id="Garv3883/implicit-mabsa-weights",
            filename="implicit_mabsa.pt",
            local_dir="saved_models/"
        )   

    device = get_device(None)
    ckpt = load_checkpoint(str(CHECKPOINT_PATH))
    model_cfg = ModelConfig(**ckpt["model_config"])
    data_cfg = DataConfig(**ckpt["data_config"])

    tokenizer = load_tokenizer(model_cfg.text_model_name)
    image_processor = load_image_processor(model_cfg.vision_model_name)

    models: Dict[str, ImplicitMABSAModel] = {}
    for mode in VALID_MODES:
        if mode in ckpt["streams"]:
            m = ImplicitMABSAModel(model_cfg, modes=(mode,), pretrained=False)
            m.load_stream_state_dict(mode, ckpt["streams"][mode])
            m.to(device)
            m.eval()
            models[mode] = m

    STATE["device"] = device
    STATE["tokenizer"] = tokenizer
    STATE["image_processor"] = image_processor
    STATE["max_length"] = data_cfg.max_length
    STATE["models"] = models
    STATE["aspect_vocab"] = ckpt.get("aspect_vocab", [])


@torch.inference_mode()
def run_inference(text: str, aspect: str, image: Image.Image) -> Dict[str, Any]:
    device: torch.device = STATE["device"]
    tokenizer = STATE["tokenizer"]
    image_processor = STATE["image_processor"]
    max_length: int = STATE["max_length"]
    models: Dict[str, ImplicitMABSAModel] = STATE["models"]

    enc = tokenizer(
        text.strip(),
        aspect.strip().lower(),
        truncation="only_first",
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    token_type_ids = enc["token_type_ids"].to(device) if "token_type_ids" in enc else None

    pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)

    predictions: Dict[str, Any] = {}
    for mode, model in models.items():
        kwargs: Dict[str, Any] = {}
        if mode in ("text_only", "multimodal"):
            kwargs["input_ids"] = input_ids
            kwargs["attention_mask"] = attention_mask
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
        if mode in ("image_only", "multimodal"):
            kwargs["pixel_values"] = pixel_values

        out = model(mode, **kwargs)
        sent_probs = F.softmax(out["sentiment_logits"].float(), dim=-1).squeeze(0).cpu().tolist()
        pred_idx = int(torch.tensor(sent_probs).argmax().item())

        aspect_info = None
        if out["aspect_logits"] is not None:
            asp_probs = F.softmax(out["aspect_logits"].float(), dim=-1).squeeze(0).cpu().tolist()
            aspect_info = {
                "is_relevant": bool(asp_probs[1] >= 0.5),
                "relevance_score": round(asp_probs[1] * 100, 2),
            }

        predictions[mode] = {
            "stream_name": STREAM_DISPLAY_NAMES[mode],
            "predicted_sentiment": SENTIMENT_LABELS[pred_idx],
            "confidence": round(sent_probs[pred_idx] * 100, 2),
            "probabilities": {
                label: round(prob * 100, 2)
                for label, prob in zip(SENTIMENT_LABELS, sent_probs)
            },
            "aspect_relevance": aspect_info,
        }

    return {
        "input": {"text": text, "aspect": aspect},
        "device_used": str(device),
        "results": predictions,
    }


@app.post("/predict")
async def predict_endpoint(
    text: str = Form(..., description="Tweet or post text"),
    aspect: str = Form(..., description="Target or candidate aspect"),
    image: UploadFile = File(..., description="Accompanying image file"),
) -> Dict[str, Any]:
    try:
        raw_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

    return run_inference(text=text, aspect=aspect, image=pil_img)


@app.get("/", response_class=HTMLResponse)
def demo_ui() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="theme-color" content="#f8f7f3" />
  <title>ImplicitMABSA — Multimodal sentiment studio</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { color-scheme: light; --ink:#20241f; --muted:#777b72; --line:#e7e7df; --paper:#fff; --bg:#f8f7f3; --green:#315c43; --lime:#d8ef9a; --orange:#e8784e; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family:'DM Sans',sans-serif; -webkit-font-smoothing:antialiased; }
    .topbar { height:70px; border-bottom:1px solid var(--line); display:flex; align-items:center; justify-content:space-between; padding:0 max(28px,calc((100% - 1160px)/2)); background:rgba(248,247,243,.9); }
    .brand { display:flex; align-items:center; gap:11px; font-family:Manrope,sans-serif; font-weight:800; letter-spacing:-.5px; }
    .logo { width:34px;height:34px;border-radius:11px;background:var(--green);color:#fff;display:grid;place-items:center;font-size:18px; }
    .top-tag { color:var(--muted);font-size:13px;display:flex;align-items:center;gap:8px; }
    .dot { width:8px;height:8px;border-radius:50%;background:#63a779;box-shadow:0 0 0 3px #63a77922; }
    main { max-width:1160px;margin:0 auto;padding:57px 28px 70px; }
    .intro { display:flex;justify-content:space-between;align-items:end;margin-bottom:30px;gap:24px; }
    .eyebrow { text-transform:uppercase;letter-spacing:1.8px;font-size:11px;font-weight:700;color:var(--green);margin:0 0 13px; }
    h1,h2,h3,p { margin-top:0; }
    h1 { font:800 clamp(34px,5vw,53px)/1.08 Manrope,sans-serif;letter-spacing:-2.2px;margin:0 0 13px; }
    .intro p:last-child { color:var(--muted);font-size:16px;margin:0; }
    .model-pill { background:#eef1e9;border:1px solid #dfe5d8;color:var(--green);border-radius:999px;padding:10px 14px;font-size:12px;font-weight:700;white-space:nowrap; }
    .workspace { display:grid;grid-template-columns:minmax(0,1.05fr) minmax(340px,.95fr);gap:20px;align-items:start; }
    .panel { background:var(--paper);border:1px solid var(--line);border-radius:18px;box-shadow:0 8px 30px #29342408; }
    .form-panel { padding:27px; }
    .panel-heading { display:flex;align-items:center;gap:12px;margin-bottom:23px; }
    .step { width:29px;height:29px;border-radius:9px;background:#edf2e9;color:var(--green);display:grid;place-items:center;font-weight:800;font-size:13px; }
    .panel-heading h2 { font:700 17px Manrope,sans-serif;margin:0;letter-spacing:-.3px; }
    label { display:block;font-size:13px;font-weight:700;margin-bottom:8px; }
    .field { margin-bottom:19px; }
    textarea,input[type=text] { width:100%;border:1px solid #dedfd8;border-radius:10px;background:#fff;padding:13px 14px;color:var(--ink);font:14px 'DM Sans',sans-serif;outline:none;transition:.18s; }
    textarea { min-height:105px;resize:vertical;line-height:1.55; }
    textarea:focus,input[type=text]:focus { border-color:#78947f;box-shadow:0 0 0 3px #315c4315; }
    .hint { color:#92958d;font-size:11px;margin:7px 0 0; }
    .aspect-wrap { position:relative; }
    .aspect-wrap input { padding-right:70px; }
    .aspect-wrap span { position:absolute;right:12px;top:13px;color:#999c94;font-size:11px; }
    .upload { height:128px;border:1.5px dashed #cdd3c9;border-radius:12px;background:#fafbf8;display:flex;align-items:center;justify-content:center;cursor:pointer;text-align:center;transition:.18s;overflow:hidden;position:relative; }
    .upload:hover,.upload.drag { border-color:var(--green);background:#f3f7ef; }
    .upload input { position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%; }
    .upload-copy { pointer-events:none; }
    .upload-icon { font-size:23px;margin-bottom:5px; }
    .upload-title { font-size:13px;font-weight:700; }
    .upload-sub { font-size:11px;color:#92958d;margin-top:4px; }
    .preview { display:none;max-width:100%;max-height:128px;object-fit:contain; }
    .upload.has-image .upload-copy { display:none; }
    .upload.has-image .preview { display:block; }
    .actions { display:flex;gap:10px;align-items:center;margin-top:22px; }
    button { border:0;cursor:pointer;font:700 13px 'DM Sans',sans-serif; }
    .primary { flex:1;background:var(--green);color:#fff;padding:14px 18px;border-radius:10px;transition:.18s; }
    .primary:hover { background:#244a34;transform:translateY(-1px); }
    .primary:disabled { opacity:.65;cursor:wait;transform:none; }
    .sample { background:transparent;color:var(--green);padding:12px 8px;font-size:12px;white-space:nowrap; }
    .right-col { display:grid;gap:20px; }
    .guide { padding:25px 27px;background:#eef2e9;border:1px solid #e1e7db;border-radius:18px; }
    .guide h2 { font:700 16px Manrope,sans-serif;margin-bottom:7px; }
    .guide > p { font-size:12px;color:#737a70;line-height:1.6;margin-bottom:18px; }
    .stream-list { display:grid;gap:11px; }
    .stream { display:flex;gap:12px;align-items:center; }
    .stream-icon { width:31px;height:31px;border-radius:10px;background:#fff;display:grid;place-items:center;font-size:14px; }
    .stream strong { font-size:12px;display:block; }
    .stream small { color:#858b80;font-size:11px; }
    .results { padding:25px 27px;display:none; }
    .results-head { display:flex;justify-content:space-between;align-items:center;margin-bottom:16px; }
    .results-head h2 { font:700 16px Manrope,sans-serif;margin:0; }
    .device { font-size:10px;color:var(--muted);background:#f4f4ef;border-radius:999px;padding:6px 9px; }
    .grid { display:grid;gap:10px; }
    .stream-box { border:1px solid var(--line);border-radius:12px;padding:15px; }
    .result-top { display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px; }
    .stream-name { font-size:12px;font-weight:700; }
    .badge { padding:6px 10px;border-radius:999px;font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.35px; }
    .positive { background:#e6f4e9;color:#347448; }
    .neutral { background:#fff2d8;color:#936414; }
    .negative { background:#fce8e3;color:#b64c34; }
    .bar-label { display:flex;justify-content:space-between;color:#777b72;font-size:10px;margin-bottom:6px; }
    .bar { height:5px;border-radius:9px;background:#efefe9;overflow:hidden; }
    .bar i { display:block;height:100%;border-radius:9px;background:var(--green); }
    .meta { display:flex;justify-content:space-between;gap:10px;margin-top:10px;color:#858980;font-size:10px; }
    .error { color:#b64c34;font-size:12px;margin:12px 0 0;display:none; }
    footer { text-align:center;color:#999c94;font-size:11px;margin-top:30px; }
    @media(max-width:780px) { main{padding-top:38px}.workspace{grid-template-columns:1fr}.intro{align-items:start;flex-direction:column}.model-pill{display:none}.right-col{grid-row:auto} }
    @media(max-width:480px) { .topbar{padding:0 18px}.top-tag{font-size:11px}main{padding:30px 16px 45px}.form-panel,.guide,.results{padding:21px}.actions{align-items:stretch;flex-direction:column-reverse}.sample{padding:8px}.intro p:last-child{font-size:14px} }
  </style>
</head>
<body>
  <header class="topbar"><div class="brand"><span class="logo">✳</span> ImplicitMABSA</div><div class="top-tag"><i class="dot"></i> Model ready · Local inference</div></header>
  <main>
    <section class="intro"><div><p class="eyebrow">Multimodal sentiment studio</p><h1>Read between<br>the modalities.</h1><p>See how text and imagery shape the feeling behind every aspect.</p></div><div class="model-pill">DeBERTa-v3&nbsp; + &nbsp;ViT&nbsp; + &nbsp;Fusion</div></section>
    <div class="workspace">
      <section class="panel form-panel"><div class="panel-heading"><span class="step">01</span><h2>Build your analysis</h2></div>
        <form id="predictForm">
          <div class="field"><label for="text">Post text</label><textarea id="text" name="text" placeholder="What did they say? Add the post or tweet text here…" required></textarea><p class="hint">Include the context around the aspect you want to understand.</p></div>
          <div class="field"><label for="aspect">Aspect to analyze</label><div class="aspect-wrap"><input type="text" id="aspect" name="aspect" placeholder="e.g. battery, screen, service" required /><span>ASPECT</span></div></div>
          <div class="field"><label for="image">Post image</label><div class="upload" id="dropZone"><div class="upload-copy"><div class="upload-icon">▧</div><div class="upload-title">Drop an image here or browse</div><div class="upload-sub">PNG, JPG or WEBP</div></div><img class="preview" id="preview" alt="Selected image preview"/><input type="file" id="image" name="image" accept="image/*" required /></div></div>
          <div class="actions"><button type="submit" id="submitBtn" class="primary">✦ &nbsp; Analyze sentiment</button><button type="button" id="sampleBtn" class="sample">Try a sample ↗</button></div>
          <p class="error" id="errorMsg" role="alert"></p>
        </form>
      </section>
      <div class="right-col">
        <section class="guide"><p class="eyebrow">Three perspectives</p><h2>One post. Three signals.</h2><p>Compare what each model stream picks up, then see how fusion combines them.</p><div class="stream-list"><div class="stream"><span class="stream-icon">⌕</span><div><strong>Text-only</strong><small>Language and aspect context</small></div></div><div class="stream"><span class="stream-icon">▧</span><div><strong>Image-only</strong><small>Visual sentiment cues</small></div></div><div class="stream"><span class="stream-icon">✳</span><div><strong>Multimodal fusion</strong><small>Text + image together</small></div></div></div></section>
        <section id="output" class="panel results"><div class="results-head"><h2>Analysis results</h2><span class="device" id="deviceLabel"></span></div><div class="grid" id="resultsGrid"></div></section>
      </div>
    </div><footer>ImplicitMABSA · Aspect-based sentiment analysis</footer>
  </main>

  <script>
    const form = document.getElementById("predictForm");
    const btn = document.getElementById("submitBtn");
    const outBox = document.getElementById("output");
    const grid = document.getElementById("resultsGrid");
    const fileInput = document.getElementById("image");
    const dropZone = document.getElementById("dropZone");
    const errorMsg = document.getElementById("errorMsg");

    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      if (!file) return;
      document.getElementById("preview").src = URL.createObjectURL(file);
      dropZone.classList.add("has-image");
    });
    ["dragenter", "dragover"].forEach(name => dropZone.addEventListener(name, e => { e.preventDefault(); dropZone.classList.add("drag"); }));
    ["dragleave", "drop"].forEach(name => dropZone.addEventListener(name, e => { e.preventDefault(); dropZone.classList.remove("drag"); }));
    dropZone.addEventListener("drop", e => { if (e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; fileInput.dispatchEvent(new Event("change")); } });
    document.getElementById("sampleBtn").addEventListener("click", async () => {
      document.getElementById("text").value = "Well, that happened after two hours of use. My phone was completely dead before dinner.";
      document.getElementById("aspect").value = "battery";
      try {
        const res = await fetch("/sample-image");
        if (!res.ok) throw new Error("Sample image unavailable");
        const file = new File([await res.blob()], "sample.png", { type: "image/png" });
        const transfer = new DataTransfer(); transfer.items.add(file); fileInput.files = transfer.files;
        fileInput.dispatchEvent(new Event("change"));
      } catch (err) { errorMsg.textContent = "Add an image to finish the sample."; errorMsg.style.display = "block"; }
    });

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      btn.disabled = true;
      btn.textContent = "Analyzing post…";
      errorMsg.style.display = "none";
      grid.innerHTML = "";

      try {
        const formData = new FormData(form);
        const res = await fetch("/predict", { method: "POST", body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Prediction failed");

        for (const key of ["text_only", "image_only", "multimodal"]) {
          const item = data.results[key];
          if (!item) continue;
          const s = item.predicted_sentiment;
          grid.innerHTML += `
            <div class="stream-box">
              <div class="result-top"><span class="stream-name">${item.stream_name}</span><span class="badge ${s}">${s}</span></div>
              <div class="bar-label"><span>Confidence</span><strong>${item.confidence}%</strong></div><div class="bar"><i style="width:${item.confidence}%"></i></div>
              <div class="meta"><span>Positive ${item.probabilities.positive}% · Neutral ${item.probabilities.neutral}% · Negative ${item.probabilities.negative}%</span></div>
              ${item.aspect_relevance ? `<div class="meta"><span>Aspect match</span><strong>${item.aspect_relevance.is_relevant ? "Yes" : "No"} · ${item.aspect_relevance.relevance_score}%</strong></div>` : ""}
            </div>`;
        }
        document.getElementById("deviceLabel").textContent = data.device_used;
        outBox.style.display = "block";
        outBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
      } catch (err) {
        errorMsg.textContent = err.message;
        errorMsg.style.display = "block";
      } finally {
        btn.disabled = false;
        btn.textContent = "✦   Analyze sentiment";
      }
    });
  </script>
</body>
</html>"""


@app.get("/sample-image")
def sample_image() -> FileResponse:
    image_path = Path(__file__).parent / "saved_models" / "demo_battery_negative.png"
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Sample image not found")
    return FileResponse(image_path, media_type="image/png")
