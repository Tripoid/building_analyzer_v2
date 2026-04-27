import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { analyzeImage } from '../api/apiService'

const STEPS = [
  { label: 'Загрузка изображения...', icon: '☁️' },
  { label: 'Обнаружение фасада...', icon: '🔍' },
  { label: 'Анализ повреждений...', icon: '📊' },
  { label: 'Сегментация материалов...', icon: '🧱' },
  { label: 'Расчёт стоимости...', icon: '🧮' },
  { label: 'Формирование отчёта...', icon: '📋' },
]

export default function LoadingPage() {
  const navigate = useNavigate()
  const [step, setStep] = useState(0)
  const [error, setError] = useState(null)
  const started = useRef(false)

  useEffect(() => {
    if (started.current) return
    started.current = true

    const file = window.__uploadedFile
    if (!file) {
      setError('Файл не найден. Вернитесь на страницу загрузки.')
      return
    }

    const area = parseFloat(sessionStorage.getItem('totalArea') || '450')

    // Progress animation during analysis
    let c = 0
    const t = setInterval(() => {
      c++
      if (c < STEPS.length - 1) setStep(c)
      else clearInterval(t)
    }, 3000)

    analyzeImage(file, area)
      .then((r) => {
        clearInterval(t)
        setStep(STEPS.length - 1)
        sessionStorage.setItem('analysisResult', JSON.stringify(r))
        window.__uploadedFile = null
        setTimeout(() => navigate('/results', { replace: true }), 400)
      })
      .catch((e) => {
        clearInterval(t)
        window.__uploadedFile = null
        setError(e.message)
      })

    return () => clearInterval(t)
  }, [navigate])

  const progress = (step + 1) / STEPS.length

  if (error) return (
    <div className="page" style={{ display:'flex',alignItems:'center',justifyContent:'center',padding:40 }}>
      <div style={{ textAlign:'center',maxWidth:400 }}>
        <div style={{ fontSize:48,marginBottom:16 }}>⚠️</div>
        <h2 style={{ fontSize:20,fontWeight:700,marginBottom:8 }}>Ошибка анализа</h2>
        <p style={{ color:'var(--text-secondary)',fontSize:14,marginBottom:24 }}>{error}</p>
        <button className="btn btn--primary" onClick={() => navigate('/upload')}>Попробовать снова</button>
      </div>
    </div>
  )

  return (
    <div className="page" style={{ display:'flex',alignItems:'center',justifyContent:'center' }}>
      <div style={{ textAlign:'center',padding:40 }}>
        <div className="pulse-ring" style={{ margin:'0 auto 48px' }}>
          <div className="pulse-ring__outer" />
          <div className="pulse-ring__middle" />
          <div className="pulse-ring__inner">
            <span style={{ fontSize:36 }}>{STEPS[step].icon}</span>
          </div>
        </div>
        <div key={step} className="slide-up" style={{ marginBottom:12 }}>
          <div style={{ fontSize:18,fontWeight:600 }}>{STEPS[step].label}</div>
        </div>
        <div style={{ fontSize:13,color:'var(--text-secondary)',marginBottom:40 }}>Шаг {step+1} из {STEPS.length}</div>
        <div style={{ maxWidth:280,margin:'0 auto' }}>
          <div className="progress-bar"><div className="progress-bar__fill" style={{ width:`${progress*100}%` }} /></div>
          <div style={{ fontSize:13,fontWeight:600,color:'var(--accent)',marginTop:12 }}>{Math.round(progress*100)}%</div>
        </div>
      </div>
    </div>
  )
}
