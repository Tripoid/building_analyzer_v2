import { formatCurrency } from '../api/apiService'

export default function CostBreakdownCard({ items }) {
  const total = items.reduce((sum, item) => sum + item.cost, 0)

  return (
    <div className="card card--flat" style={{ padding: 0, overflow: 'hidden' }}>
      {/* Header */}
      <div style={{
        padding: 16,
        background: 'linear-gradient(135deg, rgba(255,109,0,0.15), transparent)',
        display: 'flex',
        alignItems: 'center',
        gap: 12,
      }}>
        <div style={{
          padding: 8,
          borderRadius: 'var(--radius-sm)',
          background: 'rgba(255,109,0,0.2)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}>
          <span style={{ fontSize: 20 }}>🧮</span>
        </div>
        <span style={{ fontSize: 16, fontWeight: 600 }}>Смета ремонтных работ</span>
      </div>

      {/* Items */}
      {items.map((item, idx) => {
        const pct = Math.round(item.cost / total * 100)
        return (
          <div key={idx}>
            <div style={{ padding: '12px 16px', display: 'flex', alignItems: 'center', gap: 12 }}>
              <div style={{
                width: 32, height: 32,
                borderRadius: 'var(--radius-sm)',
                background: 'var(--primary-light)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: 'var(--accent)', fontWeight: 700, fontSize: 14,
                flexShrink: 0,
              }}>
                {idx + 1}
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 14, fontWeight: 600 }}>{item.category}</div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2 }}>{item.description}</div>
              </div>
              <div style={{ textAlign: 'right', flexShrink: 0 }}>
                <div style={{ fontSize: 15, fontWeight: 700 }}>{formatCurrency(item.cost)}</div>
                <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>{pct}%</div>
              </div>
            </div>
            {idx < items.length - 1 && (
              <div style={{ height: 1, background: 'rgba(42,69,96,0.3)', marginLeft: 60 }} />
            )}
          </div>
        )
      })}

      {/* Total */}
      <div style={{ padding: 16 }}>
        <div className="grand-total" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: 18, fontWeight: 700, color: '#fff' }}>Итого</span>
          <span style={{ fontSize: 22, fontWeight: 800, color: '#fff', letterSpacing: -0.5 }}>{formatCurrency(total)}</span>
        </div>
      </div>
    </div>
  )
}
