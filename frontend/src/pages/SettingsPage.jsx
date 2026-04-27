import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { healthCheck } from '../api/apiService'

export default function SettingsPage() {
  const navigate = useNavigate()
  const [checking, setChecking] = useState(false)
  const [status, setStatus] = useState(null)

  const checkConnection = async () => {
    setChecking(true)
    setStatus(null)
    try {
      const data = await healthCheck()
      if (data.status === 'ok') {
        setStatus({ ok: true, msg: `Подключено (${data.device || 'unknown'})` })
      } else {
        setStatus({ ok: false, msg: 'Сервер ответил, но статус не OK' })
      }
    } catch (e) {
      setStatus({ ok: false, msg: `Ошибка: ${e.message}` })
    } finally {
      setChecking(false)
    }
  }

  return (
    <div className="page" style={{ background: 'var(--primary-dark)' }}>
      {/* Top bar */}
      <div className="topbar">
        <button className="topbar__back" onClick={() => navigate('/')} id="settings-back">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="15,18 9,12 15,6"/></svg>
          Назад
        </button>
        <span className="topbar__title">Настройки</span>
        <div style={{ width: 48 }} />
      </div>

      <div className="fade-in" style={{ padding: '0 20px 20px', display: 'flex', flexDirection: 'column', gap: 16 }}>
        {/* Server status */}
        <div className="card card--flat">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
            <span style={{ fontSize: 20 }}>🔗</span>
            <span style={{ fontSize: 16, fontWeight: 700 }}>Сервер API</span>
          </div>

          <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 12 }}>
            Бэкенд работает на том же сервере. Проверьте подключение:
          </div>

          <button
            className="btn btn--primary btn--full"
            onClick={checkConnection}
            disabled={checking}
            id="check-connection-btn"
          >
            {checking ? '⏳ Проверка...' : '📡 Проверить соединение'}
          </button>

          {status && (
            <div className={`alert ${status.ok ? 'alert--success' : 'alert--danger'}`} style={{ marginTop: 10 }}>
              <span>{status.ok ? '✅' : '❌'}</span>
              <span style={{ color: status.ok ? 'var(--success)' : 'var(--danger)' }}>{status.msg}</span>
            </div>
          )}
        </div>

        {/* Info */}
        <div className="alert alert--info" style={{ flexDirection: 'column', gap: 10 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span>❓</span>
            <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--info)' }}>Как развернуть сервер?</span>
          </div>
          <div style={{ fontSize: 12, lineHeight: 1.6 }}>
            1. Подключитесь к серверу по SSH<br/>
            2. Клонируйте репозиторий<br/>
            3. Запустите <code style={{ background: 'rgba(66,165,245,0.15)', padding: '2px 6px', borderRadius: 4 }}>./deploy/deploy.sh</code><br/>
            4. Откройте в браузере IP-адрес сервера
          </div>
        </div>

        {/* About */}
        <div className="card card--flat" style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>Facade Analyzer</div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>v2.0.0 — React + FastAPI</div>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 8, opacity: 0.6 }}>
            ML: Grounding DINO + SAM + CLIPSeg + SAM2
          </div>
        </div>
      </div>
    </div>
  )
}
