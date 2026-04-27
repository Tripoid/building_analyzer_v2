import { useNavigate } from 'react-router-dom'

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

        {/* Upload photo card */}
        <div className="fade-in" style={{ padding: '24px 24px 8px', display: 'flex', flexDirection: 'column', gap: 14 }}>
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
        </div>

        {/* Features */}
        <div className="slide-up" style={{ padding: '24px 24px 32px' }}>
          <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 16 }}>Возможности</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {[
              { icon: '🔍', title: 'Обнаружение дефектов', desc: 'Трещины, отслоения, высолы, биопоражение' },
              { icon: '🧱', title: 'Анализ материалов', desc: 'Определение состава и состояния материалов фасада' },
              { icon: '📊', title: 'Оценка состояния', desc: 'Балльная оценка с визуализацией на тепловой карте' },
              { icon: '🧮', title: 'Смета ремонта', desc: 'Автоматический расчёт стоимости в рублях' },
              { icon: '📄', title: 'PDF-отчёт', desc: 'Выгрузка красивого отчёта для заказчика' },
            ].map((f, i) => (
              <div key={i} style={{
                padding: 14, borderRadius: 14,
                background: 'rgba(30,48,68,0.7)', border: '1px solid rgba(42,69,96,0.2)',
                display: 'flex', alignItems: 'center', gap: 14,
              }}>
                <div style={{
                  width: 44, height: 44, borderRadius: 12,
                  background: 'rgba(255,109,0,0.1)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 22, flexShrink: 0,
                }}>{f.icon}</div>
                <div>
                  <div style={{ fontSize: 14, fontWeight: 600 }}>{f.title}</div>
                  <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2 }}>{f.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
