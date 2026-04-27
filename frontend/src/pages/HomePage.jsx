import { useNavigate } from 'react-router-dom'

const RECENT = [
  { address: 'ул. Абая, 45', date: '22 марта 2026', score: 72, status: 'Удовлетворительное' },
  { address: 'пр. Республики, 12', date: '20 марта 2026', score: 45, status: 'Требует ремонта' },
  { address: 'ул. Тулебаева, 88', date: '18 марта 2026', score: 91, status: 'Хорошее' },
]

function scoreColor(score) {
  if (score >= 80) return 'var(--success)'
  if (score >= 60) return 'var(--warning)'
  return 'var(--danger)'
}

export default function HomePage() {
  const navigate = useNavigate()

  return (
    <div className="page">
      <div className="bg-animated" />
      <div style={{ position: 'relative', zIndex: 1 }}>
        {/* Header */}
        <div style={{ padding: '20px 24px 8px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{
              padding: 12,
              borderRadius: 14,
              background: 'var(--accent-gradient)',
              boxShadow: '0 4px 16px rgba(255,109,0,0.4)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="4" y="2" width="16" height="20" rx="2"/><path d="M9 22v-4h6v4"/><path d="M8 6h.01"/><path d="M16 6h.01"/><path d="M12 6h.01"/><path d="M12 10h.01"/><path d="M12 14h.01"/><path d="M16 10h.01"/><path d="M16 14h.01"/><path d="M8 10h.01"/><path d="M8 14h.01"/>
              </svg>
            </div>
            <button className="btn--icon" onClick={() => navigate('/settings')} id="settings-btn" style={{
              padding: 10, borderRadius: 12,
              background: 'rgba(30,48,68,0.6)', border: '1px solid rgba(42,69,96,0.3)',
              cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--text-secondary)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="3"/><path d="M12 1v2m0 18v2M4.22 4.22l1.42 1.42m12.72 12.72 1.42 1.42M1 12h2m18 0h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
              </svg>
            </button>
          </div>

          <h1 style={{ fontSize: 38, fontWeight: 800, lineHeight: 1.1, letterSpacing: -1, marginTop: 24 }}>
            Анализ<br/>Фасадов
          </h1>
          <p style={{ fontSize: 15, color: 'var(--text-secondary)', lineHeight: 1.5, marginTop: 8, opacity: 0.8 }}>
            Оценка состояния фасадов зданий<br/>с помощью компьютерного зрения
          </p>
        </div>

        {/* Action Cards */}
        <div className="fade-in" style={{ padding: '24px 24px 8px', display: 'flex', flexDirection: 'column', gap: 14 }}>
          {/* Upload photo card */}
          <div
            className="card card--accent"
            id="upload-card"
            onClick={() => navigate('/upload')}
            style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 16 }}
          >
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17,8 12,3 7,8"/><line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 17, fontWeight: 700, color: '#fff' }}>Загрузить фото</div>
              <div style={{ fontSize: 13, color: 'rgba(255,255,255,0.75)', lineHeight: 1.3, marginTop: 4 }}>
                Загрузите фото фасада<br/>для мгновенного анализа
              </div>
            </div>
            <div style={{ padding: 12, borderRadius: 14, background: 'rgba(255,255,255,0.2)', display: 'flex' }}>
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="5" y1="12" x2="19" y2="12"/><polyline points="12,5 19,12 12,19"/>
              </svg>
            </div>
          </div>

          {/* Demo mode card */}
          <div
            className="card"
            id="demo-card"
            onClick={() => {
              sessionStorage.setItem('demoMode', 'true')
              navigate('/loading')
            }}
            style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 16 }}
          >
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.8)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"/><polygon points="10,8 16,12 10,16"/>
            </svg>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 17, fontWeight: 700 }}>Демо-режим</div>
              <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.3, marginTop: 4 }}>
                Посмотрите пример отчёта<br/>с тестовыми данными
              </div>
            </div>
            <div style={{ padding: 12, borderRadius: 14, background: 'rgba(255,109,0,0.15)', display: 'flex' }}>
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="5" y1="12" x2="19" y2="12"/><polyline points="12,5 19,12 12,19"/>
              </svg>
            </div>
          </div>
        </div>

        {/* Recent Analyses */}
        <div className="slide-up" style={{ padding: '24px 24px 32px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <span style={{ fontSize: 18, fontWeight: 700 }}>Недавние анализы</span>
            <span style={{ fontSize: 14, color: 'var(--accent)', fontWeight: 600, cursor: 'pointer' }}>Все →</span>
          </div>

          {RECENT.map((item, i) => {
            const color = scoreColor(item.score)
            return (
              <div key={i} style={{
                marginBottom: 10, padding: 14, borderRadius: 14,
                background: 'rgba(30,48,68,0.7)', border: '1px solid rgba(42,69,96,0.2)',
                display: 'flex', alignItems: 'center', gap: 14,
              }}>
                <div style={{
                  width: 48, height: 48, borderRadius: 12,
                  background: `color-mix(in srgb, ${color} 15%, transparent)`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="4" y="2" width="16" height="20" rx="2"/><path d="M9 22v-4h6v4"/><path d="M8 6h.01"/><path d="M16 6h.01"/>
                  </svg>
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 14, fontWeight: 600 }}>{item.address}</div>
                  <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2 }}>
                    {item.date} • {item.status}
                  </div>
                </div>
                <div style={{
                  padding: '6px 10px', borderRadius: 8,
                  background: `color-mix(in srgb, ${color} 15%, transparent)`,
                  fontSize: 14, fontWeight: 700, color,
                }}>
                  {item.score}%
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
