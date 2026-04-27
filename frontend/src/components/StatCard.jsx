export default function StatCard({ icon, value, label, color = 'var(--accent)' }) {
  return (
    <div className="card card--flat" style={{ borderColor: `color-mix(in srgb, ${color} 20%, transparent)` }}>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
        <div style={{
          padding: 10,
          borderRadius: 'var(--radius-md)',
          background: `color-mix(in srgb, ${color} 15%, transparent)`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}>
          <span style={{ fontSize: 24, color }}>{icon}</span>
        </div>
        <div style={{ fontSize: 22, fontWeight: 700, color, letterSpacing: -0.5 }}>{value}</div>
        <div style={{ fontSize: 12, color: 'var(--text-secondary)', textAlign: 'center', lineHeight: 1.4 }}>{label}</div>
      </div>
    </div>
  )
}
