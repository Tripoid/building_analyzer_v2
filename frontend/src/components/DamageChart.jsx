import { useEffect, useRef, useState } from 'react'

const COLORS = ['#EF5350', '#FFCA28', '#42A5F5', '#66BB6A', '#FF9E40']

export default function DamageChart({ data, size = 200 }) {
  const canvasRef = useRef(null)
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    let raf
    const start = performance.now()
    const duration = 1200

    function animate(now) {
      const elapsed = now - start
      const t = Math.min(elapsed / duration, 1)
      // easeOutCubic
      const p = 1 - Math.pow(1 - t, 3)
      setProgress(p)
      if (t < 1) raf = requestAnimationFrame(animate)
    }

    raf = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(raf)
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !data?.length) return

    const ctx = canvas.getContext('2d')
    const dpr = window.devicePixelRatio || 1
    canvas.width = size * dpr
    canvas.height = size * dpr
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, size, size)

    const cx = size / 2
    const cy = size / 2
    const radius = size / 2 - 8
    const innerRadius = radius * 0.55
    const lineWidth = radius - innerRadius

    const total = data.reduce((s, d) => s + d.value, 0)
    let startAngle = -Math.PI / 2

    data.forEach((item, i) => {
      const sweep = (item.value / total) * 2 * Math.PI * progress
      const midRadius = (radius + innerRadius) / 2

      ctx.beginPath()
      ctx.arc(cx, cy, midRadius, startAngle, startAngle + sweep)
      ctx.strokeStyle = COLORS[i % COLORS.length]
      ctx.lineWidth = lineWidth
      ctx.lineCap = 'butt'
      ctx.stroke()

      startAngle += sweep
    })

    // Inner circle
    ctx.beginPath()
    ctx.arc(cx, cy, innerRadius - 2, 0, Math.PI * 2)
    ctx.fillStyle = '#0D1B2A'
    ctx.fill()

    // Center text
    const mainValue = `${Math.round(data[0].value * progress)}%`
    ctx.fillStyle = '#ECEFF1'
    ctx.font = '700 28px Inter, sans-serif'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(mainValue, cx, cy - 6)

    ctx.fillStyle = '#90A4AE'
    ctx.font = '400 11px Inter, sans-serif'
    ctx.fillText(data[0].label, cx, cy + 16)
  }, [data, size, progress])

  return (
    <canvas
      ref={canvasRef}
      style={{ width: size, height: size, display: 'block', margin: '0 auto' }}
    />
  )
}

export { COLORS as CHART_COLORS }
