import { useState, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'

export default function UploadPage() {
  const navigate = useNavigate()
  const fileInputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const [totalArea, setTotalArea] = useState('450')

  const handleFile = useCallback((f) => {
    if (!f || !f.type.startsWith('image/')) return
    setFile(f)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(f)
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragActive(false)
    if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0])
  }, [handleFile])

  const handleAnalyze = () => {
    if (!file) return
    // Store file in sessionStorage as data URL for the loading page to pick up
    sessionStorage.setItem('demoMode', 'false')
    sessionStorage.setItem('totalArea', totalArea || '450')
    // We need to pass the file — use a global for simplicity (files can't go to sessionStorage)
    window.__uploadedFile = file
    navigate('/loading')
  }

  return (
    <div className="page" style={{ background: 'var(--primary-dark)' }}>
      {/* Top bar */}
      <div className="topbar">
        <button className="topbar__back" onClick={() => navigate('/')} id="back-btn">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15,18 9,12 15,6"/>
          </svg>
          Назад
        </button>
        <span className="topbar__title">Загрузка фото</span>
        <div style={{ width: 48 }} />
      </div>

      <div className="fade-in" style={{ padding: '0 20px 20px', display: 'flex', flexDirection: 'column', gap: 20, flex: 1 }}>
        {/* Upload zone */}
        <div
          id="upload-zone"
          className={`upload-zone ${dragActive ? 'upload-zone--active' : ''} ${file ? 'upload-zone--has-file' : ''}`}
          onClick={() => !file && fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={(e) => handleFile(e.target.files?.[0])}
          />

          {preview ? (
            <>
              <img src={preview} alt="Preview" className="upload-zone__preview" />
              <div className="upload-zone__overlay">
                <div style={{
                  padding: '6px 12px', borderRadius: 8,
                  background: 'rgba(30,48,68,0.8)', fontSize: 12, color: 'var(--text-secondary)',
                  display: 'flex', alignItems: 'center', gap: 6,
                }}>
                  📐 {file.name}
                </div>
                <div style={{ flex: 1 }} />
                <div style={{
                  padding: '6px 12px', borderRadius: 8,
                  background: 'rgba(30,48,68,0.8)', fontSize: 12, color: 'var(--text-secondary)',
                }}>
                  {(file.size / 1024 / 1024).toFixed(1)} MB
                </div>
              </div>
            </>
          ) : (
            <>
              <svg className="upload-zone__icon" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17,8 12,3 7,8"/><line x1="12" y1="3" x2="12" y2="15"/>
              </svg>
              <div className="upload-zone__title">
                {dragActive ? 'Отпустите файл' : 'Перетащите фото сюда'}
              </div>
              <div className="upload-zone__subtitle">или нажмите, чтобы выбрать файл</div>
              <div style={{ fontSize: 11, color: 'var(--text-secondary)', opacity: 0.6, marginTop: 8 }}>
                JPG, PNG до 50 MB
              </div>
            </>
          )}
        </div>

        {/* Area input */}
        <div className="card card--flat" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <label style={{ fontSize: 14, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 8 }}>
            <span>📐</span> Площадь фасада (м²)
          </label>
          <input
            type="number"
            className="input"
            value={totalArea}
            onChange={(e) => setTotalArea(e.target.value)}
            placeholder="450"
            id="area-input"
          />
          <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
            Используется для пересчёта повреждений в квадратные метры
          </div>
        </div>

        {/* Actions */}
        <div style={{ display: 'flex', gap: 14 }}>
          {file && (
            <button className="btn btn--outline" onClick={() => { setFile(null); setPreview(null) }} id="reset-btn">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="1,4 1,10 7,10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/>
              </svg>
              Сбросить
            </button>
          )}
          <button
            className="btn btn--primary btn--full"
            onClick={handleAnalyze}
            disabled={!file}
            id="analyze-btn"
            style={{ flex: 1 }}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="13,2 3,14 12,14 11,22 21,10 12,10"/>
            </svg>
            Анализировать
          </button>
        </div>

        {/* Tips */}
        <div className="alert alert--info">
          <span style={{ fontSize: 18, flexShrink: 0 }}>💡</span>
          <span>Для лучших результатов фотографируйте фасад прямо, при хорошем освещении, без перспективных искажений</span>
        </div>
      </div>
    </div>
  )
}
