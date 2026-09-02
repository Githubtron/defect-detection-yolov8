import { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import {
  Upload,
  AlertTriangle,
  CheckCircle,
  RefreshCw,
  Layers,
  Image as ImageIcon,
  Film,
  Download,
  PlaneTakeoff,
  ScanSearch,
  Cpu,
  Radio,
  Camera,
  Play,
  Square,
} from 'lucide-react';
import './App.css';

const backendBase = () => `http://${window.location.hostname}:8000`;

function getSummary(detections) {
  const total = detections.length;
  let highest = null;
  detections.forEach((d) => {
    if (!highest || d.confidence > highest.confidence) highest = d;
  });
  return { total, highest };
}

function getModelTerms(isDefectModel) {
  return isDefectModel
    ? {
        totalLabel: 'Total Defects',
        topLabel: 'Top Class',
        statusLabel: 'Surface Status',
        statusValue: (total) => (total > 0 ? 'FAIL' : 'PASS'),
        statusClass: (total) => (total > 0 ? 'status-fail' : 'status-pass'),
        detectedHeading: (n) => `Detected Issues (${n})`,
        classCountsHeading: 'Defect Class Counts',
        emptyText: 'No defects detected on surface',
        emptyTextVideo: 'No defects detected across sampled frames',
      }
    : {
        totalLabel: 'Objects Found',
        topLabel: 'Top Object',
        statusLabel: 'Detection Status',
        statusValue: (total) => (total > 0 ? 'FOUND' : 'NONE'),
        statusClass: (total) => (total > 0 ? 'status-pass' : ''),
        detectedHeading: (n) => `Detected Objects (${n})`,
        classCountsHeading: 'Object Class Counts',
        emptyText: 'No objects detected in image',
        emptyTextVideo: 'No objects detected across sampled frames',
      };
}

function formatClassLabel(className) {
  return className.replace(/[-_]/g, ' ').toUpperCase();
}

const MODEL_COPY = {
  defect: {
    heading: 'This system detects structural surface defects only',
    note: 'Not trained for fire, smoke, foreign object debris, or other non-structural anomalies — images outside these classes will correctly show no defects. This build replaces the under-trained checkpoint with a deterministic simulation engine: classical computer vision (no ML weights, no dataset) tuned for the six defect classes on procedurally generated aircraft panels — upload a synthetic panel to exercise the full inspection flow (annotate, PASS/FAIL, PDF report). On arbitrary real-world photos it is best-effort; use the General Object Demo for everyday objects.',
  },
  demo: {
    heading: 'Demo mode: general-purpose object detector, not aircraft-specific',
    note: 'Runs a stock COCO-pretrained model so you can see the full pipeline working end-to-end on everyday objects. Switch back to the Aircraft Defect Model for real inspection use.',
  },
};

function ScopeBanner({ modelKey, classes }) {
  if (classes.length === 0) return null;
  const copy = MODEL_COPY[modelKey] || MODEL_COPY.defect;
  return (
    <div className="scope-banner">
      <div className="scope-banner-header">
        <ScanSearch size={15} />
        <span>{copy.heading}</span>
      </div>
      <div className="scope-chips">
        {classes.map((cls) => (
          <span key={cls} className="scope-chip">{formatClassLabel(cls)}</span>
        ))}
      </div>
      <p className="scope-note">{copy.note}</p>
    </div>
  );
}

function SkeletonLoader() {
  return (
    <div>
      <div className="skeleton" style={{ height: '220px', borderRadius: '4px', marginBottom: '16px' }} />
      <div className="skeleton" style={{ height: '16px', width: '55%', borderRadius: '3px', marginBottom: '12px' }} />
      <div className="skeleton" style={{ height: '44px', borderRadius: '4px', marginBottom: '8px' }} />
      <div className="skeleton" style={{ height: '44px', borderRadius: '4px' }} />
    </div>
  );
}

function StatCard({ label, value, valueClass }) {
  return (
    <div className={`stat-card ${valueClass || ''}`}>
      <div className="stat-label">{label}</div>
      <div className={`stat-value ${valueClass || ''}`}>{value}</div>
    </div>
  );
}

function SummaryStats({ detections, isDefectModel }) {
  const { total, highest } = getSummary(detections);
  const terms = getModelTerms(isDefectModel);
  return (
    <div className="stat-grid">
      <StatCard label={terms.totalLabel} value={total} />
      <StatCard label={terms.topLabel} value={highest ? highest.class : '—'} />
      <StatCard label={terms.statusLabel} value={terms.statusValue(total)} valueClass={terms.statusClass(total)} />
    </div>
  );
}

function DetectionList({ detections, isDefectModel }) {
  if (detections.length === 0) {
    return (
      <div className="no-defect">
        <CheckCircle size={18} /> {getModelTerms(isDefectModel).emptyText}
      </div>
    );
  }
  return (
    <ul className="detection-list">
      {detections.map((d, index) => (
        <li key={index} className="detection-item">
          <span className="detection-class">{d.class}</span>
          <span className="confidence-badge">{(d.confidence * 100).toFixed(0)}%</span>
        </li>
      ))}
    </ul>
  );
}

function App() {
  const [mode, setMode] = useState(() =>
    new URLSearchParams(window.location.search).get('mode') === 'live' ? 'live' : 'image'
  );
  const [loading, setLoading] = useState(false);
  const [reportLoading, setReportLoading] = useState(false);
  const [error, setError] = useState(null);
  const [reportError, setReportError] = useState(null);

  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [imageResult, setImageResult] = useState(null);

  const [selectedFiles, setSelectedFiles] = useState([]);
  const [batchResults, setBatchResults] = useState([]);

  const [selectedVideo, setSelectedVideo] = useState(null);
  const [videoResult, setVideoResult] = useState(null);

  const [models, setModels] = useState([]);
  const [activeModelKey, setActiveModelKey] = useState('defect');

  // Live feed state
  const videoRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const liveTimerRef = useRef(null);
  const inFlightRef = useRef(false);
  const wsRef = useRef(null);
  const liveRunningRef = useRef(false);
  const modelKeyRef = useRef(activeModelKey);
  const [liveActive, setLiveActive] = useState(false);   // camera on
  const [liveRunning, setLiveRunning] = useState(false); // inference loop on
  const [liveResult, setLiveResult] = useState(null);    // latest analyzed frame
  const [liveCounts, setLiveCounts] = useState({});      // session class counts
  const [liveFrames, setLiveFrames] = useState(0);
  const [liveError, setLiveError] = useState(null);
  const [liveLatency, setLiveLatency] = useState(null);  // server-side inference ms
  const [liveTransport, setLiveTransport] = useState(null); // 'ws' | 'http'

  useEffect(() => { modelKeyRef.current = activeModelKey; }, [activeModelKey]);

  useEffect(() => {
    axios
      .get(`${backendBase()}/models`)
      .then(({ data }) => setModels(data.models))
      .catch((error) => console.error('Error fetching models:', error));
  }, []);

  const activeModel = models.find((m) => m.key === activeModelKey);

  // ----- live feed controls -----
  const pauseInspection = useCallback(() => {
    liveRunningRef.current = false;
    if (liveTimerRef.current) {
      clearInterval(liveTimerRef.current);
      liveTimerRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    inFlightRef.current = false;
    setLiveRunning(false);
  }, []);

  const stopLive = useCallback(() => {
    pauseInspection();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setLiveActive(false);
  }, [pauseInspection]);

  // release the camera on unmount
  useEffect(() => () => stopLive(), [stopLive]);

  const startCamera = async () => {
    setLiveError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setLiveError(
        window.isSecureContext
          ? 'This browser does not support camera capture.'
          : 'Camera capture needs a secure context — open the app via localhost or HTTPS.'
      );
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setLiveActive(true);
    } catch (err) {
      console.error('Camera error:', err);
      setLiveError('Camera unavailable — check browser permissions and try again.');
    }
  };

  const handleLiveData = useCallback((data) => {
    if (!data || data.error) return;
    setLiveResult(data);
    setLiveFrames((n) => n + 1);
    if (data.latency_ms != null) setLiveLatency(data.latency_ms);
    if (data.detections?.length > 0) {
      setLiveCounts((prev) => {
        const next = { ...prev };
        data.detections.forEach((d) => { next[d.class] = (next[d.class] || 0) + 1; });
        return next;
      });
    }
  }, []);

  const captureFrameBlob = useCallback(async () => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) return null;
    if (!captureCanvasRef.current) captureCanvasRef.current = document.createElement('canvas');
    const canvas = captureCanvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    return new Promise((res) => canvas.toBlob(res, 'image/jpeg', 0.8));
  }, []);

  // HTTP transport — fallback when the WebSocket can't connect
  const captureAndPredict = useCallback(async () => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    try {
      const blob = await captureFrameBlob();
      if (!blob) return;
      const formData = new FormData();
      formData.append('file', blob, 'live-frame.jpg');
      const { data } = await axios.post(`${backendBase()}/predict`, formData, {
        params: { model: modelKeyRef.current },
      });
      handleLiveData(data);
    } catch (err) {
      console.error('Live frame failed:', err);
    } finally {
      inFlightRef.current = false;
    }
  }, [captureFrameBlob, handleLiveData]);

  const startHttpLoop = useCallback(() => {
    setLiveTransport('http');
    captureAndPredict();
    liveTimerRef.current = setInterval(captureAndPredict, 900);
  }, [captureAndPredict]);

  // WebSocket transport — one persistent connection, binary frames out, JSON in
  const sendFrameWS = useCallback(async () => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || inFlightRef.current) return;
    inFlightRef.current = true;
    const blob = await captureFrameBlob();
    if (!blob || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      inFlightRef.current = false;
      return;
    }
    wsRef.current.send(await blob.arrayBuffer());
  }, [captureFrameBlob]);

  const startInspection = () => {
    setLiveResult(null);
    setLiveCounts({});
    setLiveFrames(0);
    setLiveLatency(null);
    setLiveTransport(null);
    setLiveRunning(true);
    liveRunningRef.current = true;

    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${window.location.hostname}:8000/ws/live?model=${modelKeyRef.current}`);
    ws.binaryType = 'arraybuffer';
    let opened = false;

    ws.onopen = () => {
      opened = true;
      wsRef.current = ws;
      setLiveTransport('ws');
      sendFrameWS();
      liveTimerRef.current = setInterval(sendFrameWS, 900);
    };
    ws.onmessage = (e) => {
      inFlightRef.current = false;
      try { handleLiveData(JSON.parse(e.data)); } catch { /* ignore malformed */ }
    };
    ws.onclose = () => {
      wsRef.current = null;
      inFlightRef.current = false;
      // Connection refused before opening → fall back to HTTP polling
      if (!opened && liveRunningRef.current) startHttpLoop();
    };
  };

  const changeMode = (next) => {
    if (next !== 'live') stopLive();
    setError(null);
    setReportError(null);
    setMode(next);
  };

  const switchModel = (key) => {
    setActiveModelKey(key);
    setError(null);
    setReportError(null);
    setImageResult(null);
    setBatchResults([]);
    setVideoResult(null);
    pauseInspection();
    setLiveResult(null);
    setLiveCounts({});
    setLiveFrames(0);
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setImageResult(null);
    }
  };

  const handleBatchFilesChange = (e) => {
    setSelectedFiles(Array.from(e.target.files || []));
    setBatchResults([]);
  };

  const handleVideoChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedVideo(file);
      setVideoResult(null);
    }
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;
    setError(null);
    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);
    try {
      const { data } = await axios.post(`${backendBase()}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        params: { model: activeModelKey },
      });
      setImageResult(data);
    } catch (error) {
      console.error('Error analyzing image:', error);
      setError('Failed to connect to backend server. Ensure backend is running on 0.0.0.0:8000.');
    } finally {
      setLoading(false);
    }
  };

  const analyzeBatch = async () => {
    if (selectedFiles.length === 0) return;
    setError(null);
    setLoading(true);
    const formData = new FormData();
    selectedFiles.forEach((file) => formData.append('files', file));
    try {
      const { data } = await axios.post(`${backendBase()}/predict-batch`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        params: { model: activeModelKey },
      });
      setBatchResults(data.results);
    } catch (error) {
      console.error('Error analyzing batch:', error);
      setError('Failed to connect to backend server. Ensure backend is running on 0.0.0.0:8000.');
    } finally {
      setLoading(false);
    }
  };

  const analyzeVideo = async () => {
    if (!selectedVideo) return;
    setError(null);
    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedVideo);
    try {
      const { data } = await axios.post(`${backendBase()}/predict-video`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        params: { model: activeModelKey },
      });
      setVideoResult(data);
    } catch (error) {
      console.error('Error analyzing video:', error);
      setError('Failed to connect to backend server. Ensure backend is running on 0.0.0.0:8000.');
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = async ({ filename, detections, annotated_image }) => {
    setReportError(null);
    setReportLoading(true);
    try {
      const response = await axios.post(
        `${backendBase()}/generate-report`,
        { filename, detections, annotated_image },
        { responseType: 'blob' }
      );
      const url = URL.createObjectURL(new Blob([response.data], { type: 'application/pdf' }));
      const link = document.createElement('a');
      link.href = url;
      link.download = `${filename.replace(/\.[^/.]+$/, '')}_report.pdf`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error generating report:', error);
      setReportError('Failed to generate PDF report.');
    } finally {
      setReportLoading(false);
    }
  };

  const videoDetections = videoResult ? videoResult.detections_by_frame.flatMap((f) => f.detections) : [];

  return (
    <div className="dashboard">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">
            <PlaneTakeoff size={20} />
          </div>
          <div>
            <h1 className="brand-title">Aircraft Surface Defect Detection</h1>
            <div className="brand-subtitle">HAL // INSPECTION SYSTEM // LAN BUILD</div>
          </div>
        </div>
        <div className="status-pill">
          <span className="status-dot" />
          MODEL ONLINE
        </div>
      </header>

      <div className="model-select-row">
        <span className="model-select-label"><Cpu size={13} /> Active Model</span>
        <div className="model-select-tabs">
          {models.map((m) => (
            <button
              key={m.key}
              type="button"
              className={`tab ${activeModelKey === m.key ? 'active' : ''}`}
              onClick={() => switchModel(m.key)}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      <ScopeBanner modelKey={activeModelKey} classes={activeModel ? activeModel.classes : []} />

      <div className="mode-tabs">
        <button type="button" className={`tab ${mode === 'image' ? 'active' : ''}`} onClick={() => changeMode('image')}>
          <ImageIcon size={15} /> Single Image
        </button>
        <button type="button" className={`tab ${mode === 'batch' ? 'active' : ''}`} onClick={() => changeMode('batch')}>
          <Layers size={15} /> Batch Images
        </button>
        <button type="button" className={`tab ${mode === 'video' ? 'active' : ''}`} onClick={() => changeMode('video')}>
          <Film size={15} /> Video
        </button>
        <button type="button" className={`tab ${mode === 'live' ? 'active' : ''}`} onClick={() => changeMode('live')}>
          <Radio size={15} /> Live Feed
        </button>
      </div>

      <div className="grid-2">

        {/* Left Column: Input & Actions */}
        <div className="panel">
          {mode === 'image' && (
            <>
              <div className="panel-title">
                <span className="panel-index">01</span>
                <h3>Upload Inspection Image</h3>
              </div>
              <div className="dropzone">
                <input type="file" accept="image/*" onChange={handleFileChange} id="fileInput" className="visually-hidden" />
                <label htmlFor="fileInput">
                  <Upload className="dropzone-icon" size={32} />
                  <span>Click to select an image</span>
                </label>
              </div>
              {preview && (
                <div style={{ marginBottom: '16px' }}>
                  <div className="preview-label">Original File</div>
                  <img src={preview} alt="Upload preview" className="preview-image" />
                </div>
              )}
              <button className="btn btn-primary" onClick={analyzeImage} disabled={!selectedFile || loading}>
                {loading ? <RefreshCw className="spin" size={18} /> : <AlertTriangle size={18} />}
                {loading ? 'Analyzing Defects...' : 'Analyze Image'}
              </button>
            </>
          )}

          {mode === 'batch' && (
            <>
              <div className="panel-title">
                <span className="panel-index">01</span>
                <h3>Upload Multiple Images</h3>
              </div>
              <div className="dropzone">
                <input type="file" accept="image/*" multiple onChange={handleBatchFilesChange} id="batchInput" className="visually-hidden" />
                <label htmlFor="batchInput">
                  <Upload className="dropzone-icon" size={32} />
                  <span>Click to select images ({selectedFiles.length} selected)</span>
                </label>
              </div>
              {selectedFiles.length > 0 && (
                <ul className="file-list">
                  {selectedFiles.map((f, i) => (
                    <li key={i}>{f.name}</li>
                  ))}
                </ul>
              )}
              <button className="btn btn-primary" onClick={analyzeBatch} disabled={selectedFiles.length === 0 || loading}>
                {loading ? <RefreshCw className="spin" size={18} /> : <AlertTriangle size={18} />}
                {loading ? 'Analyzing Batch...' : `Analyze ${selectedFiles.length || ''} Image${selectedFiles.length === 1 ? '' : 's'}`}
              </button>
            </>
          )}

          {mode === 'video' && (
            <>
              <div className="panel-title">
                <span className="panel-index">01</span>
                <h3>Upload Inspection Video</h3>
              </div>
              <div className="dropzone">
                <input type="file" accept="video/mp4,video/avi,.mp4,.avi" onChange={handleVideoChange} id="videoInput" className="visually-hidden" />
                <label htmlFor="videoInput">
                  <Upload className="dropzone-icon" size={32} />
                  <span>{selectedVideo ? selectedVideo.name : 'Click to select a .mp4 or .avi file'}</span>
                </label>
              </div>
              <p className="hint-text">Frames are sampled during analysis, so processing large videos may take a few minutes.</p>
              <button className="btn btn-primary" onClick={analyzeVideo} disabled={!selectedVideo || loading}>
                {loading ? <RefreshCw className="spin" size={18} /> : <AlertTriangle size={18} />}
                {loading ? 'Analyzing Video...' : 'Analyze Video'}
              </button>
            </>
          )}

          {mode === 'live' && (
            <>
              <div className="panel-title">
                <span className="panel-index">01</span>
                <h3>Live Camera Feed</h3>
              </div>
              <div className="live-video-wrap">
                <video ref={videoRef} className="live-video" muted playsInline />
                {!liveActive && (
                  <div className="live-video-placeholder">
                    <Camera size={30} />
                    <span>Camera is off</span>
                  </div>
                )}
                {liveRunning && (
                  <span className="live-indicator"><span className="live-dot" /> LIVE</span>
                )}
              </div>
              {liveError && <p className="error-text">{liveError}</p>}
              {!liveActive ? (
                <button className="btn btn-primary" onClick={startCamera}>
                  <Camera size={18} /> Enable Camera
                </button>
              ) : (
                <>
                  {!liveRunning ? (
                    <button className="btn btn-primary" onClick={startInspection}>
                      <Play size={18} /> Start Inspection
                    </button>
                  ) : (
                    <button className="btn btn-primary" onClick={pauseInspection}>
                      <Square size={18} /> Pause Inspection
                    </button>
                  )}
                  <button className="btn btn-secondary" onClick={stopLive}>
                    Stop Camera
                  </button>
                </>
              )}
              <p className="hint-text">
                A frame is captured roughly every second and sent to the model — the latest
                annotated frame and running class counts appear in the results panel.
              </p>
            </>
          )}
          {error && <p className="error-text panel-error">{error}</p>}
        </div>

        {/* Right Column: Output & Detections */}
        <div className="panel">
          <div className="panel-title">
            <span className="panel-index">02</span>
            <h3>Analysis Results</h3>
          </div>

          {loading && <SkeletonLoader />}

          {!loading && mode === 'image' && (
            imageResult ? (
              <div>
                <img src={imageResult.annotated_image} alt="Detections" className="result-image" />
                <SummaryStats detections={imageResult.detections} isDefectModel={activeModelKey === 'defect'} />
                <div className="section-heading">{getModelTerms(activeModelKey === 'defect').detectedHeading(imageResult.detections.length)}</div>
                <DetectionList detections={imageResult.detections} isDefectModel={activeModelKey === 'defect'} />
                <button className="btn btn-secondary" onClick={() => downloadReport(imageResult)} disabled={reportLoading}>
                  <Download size={16} /> {reportLoading ? 'Generating PDF...' : 'Download Inspection Report'}
                </button>
              </div>
            ) : (
              <p className="empty-state">Upload an image and click "Analyze Image" to view localized bounding boxes and confidence scores.</p>
            )
          )}

          {!loading && mode === 'batch' && (
            batchResults.length > 0 ? (
              <div className="batch-list">
                {batchResults.map((r, i) => (
                  <div key={i} className="batch-item">
                    <div className="batch-item-title">{r.filename}</div>
                    {r.error ? (
                      <p className="error-text">{r.error}</p>
                    ) : (
                      <>
                        <img src={r.annotated_image} alt={r.filename} className="result-image" />
                        <SummaryStats detections={r.detections} isDefectModel={activeModelKey === 'defect'} />
                        <DetectionList detections={r.detections} isDefectModel={activeModelKey === 'defect'} />
                        <button className="btn btn-secondary" onClick={() => downloadReport(r)} disabled={reportLoading}>
                          <Download size={16} /> {reportLoading ? 'Generating PDF...' : 'Download Report'}
                        </button>
                      </>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <p className="empty-state">Select multiple images and click "Analyze" to batch-process inspection photos.</p>
            )
          )}

          {!loading && mode === 'video' && (
            videoResult ? (
              <div>
                {videoResult.annotated_image && (
                  <img src={videoResult.annotated_image} alt="Best detection frame" className="result-image" />
                )}
                <div className="video-meta">
                  Analyzed {videoResult.frames_analyzed} of {videoResult.frames_total} frames
                </div>
                <SummaryStats detections={videoDetections} isDefectModel={activeModelKey === 'defect'} />
                <div className="section-heading">{getModelTerms(activeModelKey === 'defect').classCountsHeading}</div>
                {Object.keys(videoResult.class_counts).length === 0 ? (
                  <div className="no-defect">
                    <CheckCircle size={18} /> {getModelTerms(activeModelKey === 'defect').emptyTextVideo}
                  </div>
                ) : (
                  <ul className="detection-list">
                    {Object.entries(videoResult.class_counts).map(([cls, count]) => (
                      <li key={cls} className="detection-item">
                        <span className="detection-class">{cls}</span>
                        <span className="confidence-badge">{count}</span>
                      </li>
                    ))}
                  </ul>
                )}
                <button
                  className="btn btn-secondary"
                  onClick={() =>
                    downloadReport({
                      filename: videoResult.filename,
                      detections: videoDetections,
                      annotated_image: videoResult.annotated_image,
                    })
                  }
                  disabled={reportLoading}
                >
                  <Download size={16} /> {reportLoading ? 'Generating PDF...' : 'Download Inspection Report'}
                </button>
              </div>
            ) : (
              <p className="empty-state">Upload a video and click "Analyze Video" to scan sampled frames for surface defects.</p>
            )
          )}

          {!loading && mode === 'live' && (
            liveResult ? (
              <div>
                <img src={liveResult.annotated_image} alt="Latest live detections" className="result-image" />
                <div className="video-meta">
                  {liveFrames} frame{liveFrames === 1 ? '' : 's'} analyzed — {liveRunning ? 'running' : 'paused'}
                  {liveLatency != null && ` — ${liveLatency}ms inference`}
                  {liveTransport && ` — ${liveTransport === 'ws' ? 'websocket' : 'http fallback'}`}
                </div>
                <SummaryStats detections={liveResult.detections} isDefectModel={activeModelKey === 'defect'} />
                <div className="section-heading">
                  Latest Frame — {getModelTerms(activeModelKey === 'defect').detectedHeading(liveResult.detections.length)}
                </div>
                <DetectionList detections={liveResult.detections} isDefectModel={activeModelKey === 'defect'} />
                {Object.keys(liveCounts).length > 0 && (
                  <>
                    <div className="section-heading">
                      Session {getModelTerms(activeModelKey === 'defect').classCountsHeading}
                    </div>
                    <ul className="detection-list">
                      {Object.entries(liveCounts).map(([cls, count]) => (
                        <li key={cls} className="detection-item">
                          <span className="detection-class">{cls}</span>
                          <span className="confidence-badge">{count}</span>
                        </li>
                      ))}
                    </ul>
                  </>
                )}
                <button
                  className="btn btn-secondary"
                  onClick={() =>
                    downloadReport({
                      filename: 'live-session',
                      detections: liveResult.detections,
                      annotated_image: liveResult.annotated_image,
                    })
                  }
                  disabled={reportLoading}
                >
                  <Download size={16} /> {reportLoading ? 'Generating PDF...' : 'Download Inspection Report'}
                </button>
              </div>
            ) : (
              <p className="empty-state">
                Enable the camera and start inspection to stream frames through the model in near real time.
              </p>
            )
          )}
          {reportError && <p className="error-text panel-error">{reportError}</p>}
        </div>

      </div>
    </div>
  );
}

export default App;
