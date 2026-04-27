import { useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { formatCurrency, getImageUrl } from '../api/apiService'
import StatCard from '../components/StatCard'
import DamageChart, { CHART_COLORS } from '../components/DamageChart'
import CostBreakdownCard from '../components/CostBreakdownCard'

const LAYER_NAMES = { finish: 'Финиш', base_plaster: 'Штукатурка', insulation: 'Утеплитель', structural: 'Несущий' }
const IMAGE_META = [
  { type: 'heatmap', title: 'Тепловая карта повреждений', desc: 'Визуализация плотности дефектов' },
  { type: 'defects', title: 'Выделенные дефекты', desc: 'Обнаружение и маркировка дефектов' },
  { type: 'segments', title: 'Сегментация материалов', desc: 'Разбивка по типам материалов' },
  { type: 'overlay', title: 'Зоны ремонта', desc: 'Области для первоочередного ремонта' },
]
const TABS = ['Снимки', 'Дефекты', 'Материалы', 'Смета', 'Ведомость']

function sevColor(s) { return s === 'Высокая' ? 'var(--danger)' : s === 'Средняя' ? 'var(--warning)' : 'var(--success)' }
function condColor(c) { return c === 'Хорошее' ? 'var(--success)' : c === 'Удовлетворительное' ? 'var(--warning)' : 'var(--danger)' }
function scoreColor(s) { return s >= 80 ? 'var(--success)' : s >= 60 ? 'var(--warning)' : 'var(--danger)' }

export default function ResultsPage() {
  const navigate = useNavigate()
  const [tab, setTab] = useState(0)
  const [imgIdx, setImgIdx] = useState(0)

  const result = useMemo(() => {
    try { return JSON.parse(sessionStorage.getItem('analysisResult')) } catch { return null }
  }, [])

  if (!result) return (
    <div className="page" style={{ display:'flex',alignItems:'center',justifyContent:'center',padding:40 }}>
      <div style={{ textAlign:'center' }}>
        <p style={{ marginBottom:16 }}>Нет данных анализа</p>
        <button className="btn btn--primary" onClick={() => navigate('/')}>На главную</button>
      </div>
    </div>
  )

  const sc = scoreColor(result.overall_score)
  const re = result.repair_estimate
  const sum = re?.summary || {}
  const circumference = 2 * Math.PI * 42

  return (
    <div className="page fade-in">
      {/* Score Header */}
      <div className="results-header">
        <div style={{ display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:20 }}>
          <button className="btn--icon" onClick={() => navigate('/')} id="back-btn" style={{ padding:8,borderRadius:10,background:'rgba(30,48,68,0.6)',border:'1px solid rgba(42,69,96,0.3)',cursor:'pointer',display:'flex' }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--text-primary)" strokeWidth="2"><polyline points="15,18 9,12 15,6"/></svg>
          </button>
          <div style={{ display:'flex',gap:8 }}>
            <button className="btn--icon" style={{ padding:8,borderRadius:10,background:'rgba(30,48,68,0.6)',border:'1px solid rgba(42,69,96,0.3)',cursor:'pointer',display:'flex' }}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--text-primary)" strokeWidth="2"><path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/><polyline points="16,6 12,2 8,6"/><line x1="12" y1="2" x2="12" y2="15"/></svg>
            </button>
            <button className="btn--icon" style={{ padding:8,borderRadius:10,background:'rgba(30,48,68,0.6)',border:'1px solid rgba(42,69,96,0.3)',cursor:'pointer',display:'flex' }}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--text-primary)" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7,10 12,15 17,10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            </button>
          </div>
        </div>
        <div style={{ display:'flex',alignItems:'center',gap:20 }}>
          <div className="score-circle" style={{ width:100,height:100 }}>
            <svg className="score-circle__svg" width="100" height="100" viewBox="0 0 100 100">
              <circle className="score-circle__track" cx="50" cy="50" r="42" strokeWidth="8"/>
              <circle className="score-circle__fill" cx="50" cy="50" r="42" strokeWidth="8" stroke={sc} strokeDasharray={circumference} strokeDashoffset={circumference * (1 - result.overall_score / 100)}/>
            </svg>
            <div className="score-circle__text">
              <div className="score-circle__value" style={{ color:sc }}>{Math.round(result.overall_score)}</div>
              <div className="score-circle__label" style={{ color:sc }}>баллов</div>
            </div>
          </div>
          <div>
            <span className="badge" style={{ background:`color-mix(in srgb, ${sc} 15%, transparent)`,color:sc }}>{result.overall_condition}</span>
            <div style={{ fontSize:13,color:'var(--text-secondary)',marginTop:10 }}>Площадь: {result.total_area_m2} м²</div>
            <div style={{ fontSize:13,color:'var(--text-secondary)',marginTop:4 }}>Повреждено: {result.damaged_area_m2} м² ({Math.round(result.damaged_area_m2/result.total_area_m2*100)}%)</div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="tabs">
        {TABS.map((t,i) => <button key={i} className={`tab ${tab===i?'tab--active':''}`} onClick={() => setTab(i)}>{t}</button>)}
      </div>

      {/* Tab content */}
      <div style={{ padding:20 }}>
        {tab === 0 && <ImagesTab result={result} imgIdx={imgIdx} setImgIdx={setImgIdx}/>}
        {tab === 1 && <DamageTab result={result}/>}
        {tab === 2 && <MaterialsTab result={result}/>}
        {tab === 3 && <CostTab result={result}/>}
        {tab === 4 && <RepairTab result={result}/>}
      </div>
    </div>
  )
}

function ImagesTab({ result, imgIdx, setImgIdx }) {
  const isDemo = !result.id || result.id.startsWith('mock')
  return (
    <div className="slide-up">
      <div style={{ borderRadius:20,overflow:'hidden',border:'1px solid rgba(42,69,96,0.3)',marginBottom:12,position:'relative',height:260,background:'var(--surface-card)' }}>
        {isDemo ? (
          <div style={{ display:'flex',alignItems:'center',justifyContent:'center',height:'100%',flexDirection:'column',gap:8 }}>
            <span style={{ fontSize:48 }}>{['🔥','🔍','🧱','🔧'][imgIdx]}</span>
            <div style={{ fontWeight:600 }}>{IMAGE_META[imgIdx].title}</div>
            <div style={{ fontSize:12,color:'var(--text-secondary)' }}>{IMAGE_META[imgIdx].desc}</div>
            <div className="badge badge--info" style={{ marginTop:4 }}>Демо-режим</div>
          </div>
        ) : (
          <img src={getImageUrl(result.id, IMAGE_META[imgIdx].type)} alt={IMAGE_META[imgIdx].title} style={{ width:'100%',height:'100%',objectFit:'cover' }}/>
        )}
      </div>
      <div style={{ display:'flex',justifyContent:'center',gap:6,marginBottom:24 }}>
        {IMAGE_META.map((_,i) => <div key={i} onClick={() => setImgIdx(i)} style={{ width:i===imgIdx?24:8,height:8,borderRadius:4,background:i===imgIdx?'var(--accent)':'var(--surface-light)',cursor:'pointer',transition:'all 0.3s' }}/>)}
      </div>
      <div className="stats-grid">
        <StatCard icon="🖼️" value={String(IMAGE_META.length)} label={'Обработанных\nснимков'} color="var(--info)"/>
        <StatCard icon="🐛" value={String(result.damages.length)} label={'Типов\nдефектов'} color="var(--danger)"/>
        <StatCard icon="📐" value={`${result.total_area_m2} м²`} label={'Общая\nплощадь'} color="var(--success)"/>
        <StatCard icon="⚠️" value={`${Math.round(result.damaged_area_m2/result.total_area_m2*100)}%`} label={'Площадь\nповреждений'} color="var(--warning)"/>
      </div>
    </div>
  )
}

function DamageTab({ result }) {
  const chartData = result.damages.map(d => ({ label: d.type_display, value: d.percentage }))
  return (
    <div className="slide-up">
      <DamageChart data={chartData} size={200}/>
      <div style={{ marginTop:24,display:'flex',flexDirection:'column',gap:12 }}>
        {result.damages.map((d,i) => {
          const color = CHART_COLORS[i % CHART_COLORS.length]
          const svc = sevColor(d.severity_display)
          return (
            <div key={i} className="card card--flat" style={{ borderColor:`${color}33` }}>
              <div style={{ display:'flex',alignItems:'center',gap:10,marginBottom:8 }}>
                <div style={{ width:12,height:12,borderRadius:3,background:color,flexShrink:0 }}/>
                <span style={{ flex:1,fontSize:15,fontWeight:600 }}>{d.type_display}</span>
                <span className="badge" style={{ background:`${svc}26`,color:svc,fontSize:11 }}>{d.severity_display}</span>
                <span style={{ fontSize:18,fontWeight:700,color }}>{d.percentage}%</span>
              </div>
              <div className="progress-bar" style={{ marginBottom:8 }}>
                <div style={{ height:'100%',width:`${d.percentage}%`,background:color,borderRadius:3,transition:'width 0.4s' }}/>
              </div>
              <div style={{ fontSize:12,color:'var(--text-secondary)',lineHeight:1.4 }}>{d.description}</div>
              {d.affected_layers?.length > 0 && (
                <div style={{ display:'flex',gap:6,marginTop:8,flexWrap:'wrap' }}>
                  {d.affected_layers.map(l => <span key={l} className="chip">{LAYER_NAMES[l]||l}</span>)}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

function MaterialsTab({ result }) {
  return (
    <div className="slide-up">
      <div className="card card--flat" style={{ marginBottom:16 }}>
        <div style={{ display:'flex',alignItems:'center',gap:10,marginBottom:16 }}><span style={{ fontSize:20 }}>🧱</span><span style={{ fontSize:16,fontWeight:700 }}>Состав фасада</span></div>
        {result.materials.map((m,i) => {
          const bc = condColor(m.condition)
          return (
            <div key={i} style={{ marginBottom:12 }}>
              <div style={{ display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:6 }}>
                <span style={{ fontSize:13 }}>{m.name_display}</span>
                <span style={{ fontSize:13,fontWeight:700,color:bc }}>{m.percentage}%</span>
              </div>
              <div className="progress-bar"><div style={{ height:'100%',width:`${m.percentage}%`,background:bc,borderRadius:3 }}/></div>
            </div>
          )
        })}
      </div>
      {result.materials.map((m,i) => {
        const cc = condColor(m.condition)
        return (
          <div key={i} className="card card--flat" style={{ marginBottom:10,display:'flex',alignItems:'center',gap:14,borderColor:`${cc}26` }}>
            <span style={{ fontSize:28 }}>{['🧱','🏗️','🪨','⚙️','🪵'][i] || '📦'}</span>
            <div style={{ flex:1 }}>
              <div style={{ fontSize:14,fontWeight:600 }}>{m.name_display}</div>
              <div style={{ fontSize:12,color:cc,fontWeight:500,marginTop:4,display:'flex',alignItems:'center',gap:4 }}>
                <span>{m.condition === 'Хорошее' ? '✅' : m.condition === 'Удовлетворительное' ? 'ℹ️' : '⚠️'}</span>{m.condition}
              </div>
            </div>
            <div style={{ padding:'6px 12px',borderRadius:8,background:'var(--primary-light)',fontSize:15,fontWeight:700 }}>{m.percentage}%</div>
          </div>
        )
      })}
    </div>
  )
}

function CostTab({ result }) {
  const re = result.repair_estimate; const sum = re?.summary || {}
  return (
    <div className="slide-up">
      <div className="stats-grid" style={{ marginBottom:16 }}>
        <StatCard icon="⏱️" value={String(sum.estimated_days||17)} label={'Дней\nработы'} color="var(--info)"/>
        <StatCard icon="👷" value={String(Math.round(sum.total_work_hours||135))} label={'Нормо-\nчасов'} color="var(--warning)"/>
      </div>
      <div className="alert alert--warning" style={{ marginBottom:16 }}>
        <span>ℹ️</span><span>Расчёт является предварительным и основан на средних рыночных ценах в регионе</span>
      </div>
      <CostBreakdownCard items={re?.costs_for_flutter || []}/>
    </div>
  )
}

function RepairTab({ result }) {
  const re = result.repair_estimate; const sum = re?.summary || {}
  return (
    <div className="slide-up">
      <div style={{ display:'flex',alignItems:'center',gap:10,marginBottom:4 }}><span style={{ fontSize:20 }}>📦</span><span style={{ fontSize:16,fontWeight:700 }}>Необходимые материалы</span></div>
      <div style={{ fontSize:12,color:'var(--text-secondary)',marginBottom:12 }}>{re?.materials?.length || 0} наименований • запас 10%</div>
      {(re?.materials||[]).map((m,i) => (
        <div key={i} className="card card--flat" style={{ marginBottom:8,display:'flex',alignItems:'center',gap:12 }}>
          <div style={{ flex:1 }}>
            <div style={{ fontSize:13,fontWeight:600 }}>{m.name_display}</div>
            <div style={{ fontSize:11,color:'var(--text-secondary)',marginTop:4 }}>{m.quantity} {m.unit} × {formatCurrency(m.price_per_unit)}</div>
          </div>
          <div style={{ fontSize:14,fontWeight:700,color:'var(--accent)' }}>{formatCurrency(m.total_cost)}</div>
        </div>
      ))}

      <div style={{ display:'flex',alignItems:'center',gap:10,marginTop:24,marginBottom:12 }}><span style={{ fontSize:20 }}>👷</span><span style={{ fontSize:16,fontWeight:700 }}>Работы</span></div>
      {(re?.labor||[]).map((l,i) => (
        <div key={i} className="card card--flat" style={{ marginBottom:8,display:'flex',alignItems:'center',gap:12,borderColor:'rgba(66,165,245,0.15)' }}>
          <div style={{ flex:1 }}>
            <div style={{ fontSize:13,fontWeight:600 }}>{l.name_display}</div>
            <div style={{ fontSize:11,color:'var(--text-secondary)',marginTop:4 }}>{l.quantity} {l.unit} • {Math.round(l.norm_hours)} ч</div>
          </div>
          <div style={{ fontSize:14,fontWeight:700,color:'var(--info)' }}>{formatCurrency(l.total_cost)}</div>
        </div>
      ))}

      <div className="grand-total" style={{ marginTop:24 }}>
        <div className="summary-row"><span>Материалы</span><span className="summary-row__amount">{formatCurrency(sum.materials_total||0)}</span></div>
        <div className="summary-row"><span>Работы</span><span className="summary-row__amount">{formatCurrency(sum.labor_total||0)}</span></div>
        <div className="summary-row"><span>Леса</span><span className="summary-row__amount">{formatCurrency(sum.scaffolding_total||0)}</span></div>
        <div className="summary-row"><span>НДС 12%</span><span className="summary-row__amount">{formatCurrency(sum.vat_amount||0)}</span></div>
        <div className="divider"/>
        <div className="summary-row summary-row--total"><span>Итого</span><span className="summary-row__amount">{formatCurrency(sum.grand_total||0)}</span></div>
      </div>
    </div>
  )
}
