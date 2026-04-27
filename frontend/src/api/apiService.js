/**
 * API service for Facade Analyzer backend.
 * In dev mode, requests are proxied via Vite to localhost:9000.
 * In production, the React app is served by the same FastAPI server.
 */

const API_BASE = '' // same origin — proxied in dev, served in prod

export async function healthCheck() {
  const res = await fetch(`${API_BASE}/api/health`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function analyzeImage(file, totalAreaM2 = 450.0) {
  const formData = new FormData()
  formData.append('file', file)

  const res = await fetch(
    `${API_BASE}/api/analyze?total_area_m2=${totalAreaM2}`,
    { method: 'POST', body: formData }
  )

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Server error ${res.status}: ${text}`)
  }

  return res.json()
}

export function getImageUrl(analysisId, imageType) {
  return `${API_BASE}/api/results/${analysisId}/images/${imageType}`
}

/**
 * Format number as currency with thousands separator + ₽ (rubles)
 */
export function formatCurrency(amount) {
  return amount.toLocaleString('ru-RU', { maximumFractionDigits: 0 }) + ' ₽'
}

/**
 * Generate PDF report from analysis results
 */
export async function generatePdfReport(result) {
  const { jsPDF } = await import('https://cdn.jsdelivr.net/npm/jspdf@2.5.2/+esm')

  const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' })
  const W = 210, M = 15
  let y = M

  // Colors
  const DARK = [15, 23, 36]
  const ACCENT = [255, 109, 0]
  const WHITE = [255, 255, 255]
  const GRAY = [160, 170, 185]
  const GREEN = [76, 175, 80]
  const YELLOW = [255, 193, 7]
  const RED = [244, 67, 54]

  const scoreClr = result.overall_score >= 80 ? GREEN : result.overall_score >= 60 ? YELLOW : RED

  // ── Page 1: Header & Summary ──
  // Background
  doc.setFillColor(...DARK)
  doc.rect(0, 0, W, 297, 'F')

  // Orange accent bar
  doc.setFillColor(...ACCENT)
  doc.rect(0, 0, W, 6, 'F')

  // Title
  y = 20
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(24)
  doc.setTextColor(...WHITE)
  doc.text('Отчёт по анализу фасада', M, y)

  y += 10
  doc.setFontSize(11)
  doc.setTextColor(...GRAY)
  doc.text(`Дата: ${new Date().toLocaleDateString('ru-RU')}`, M, y)
  doc.text(`ID: ${result.id || '—'}`, W - M, y, { align: 'right' })

  // Score circle area
  y += 16
  doc.setFillColor(20, 32, 50)
  doc.roundedRect(M, y, W - 2 * M, 40, 4, 4, 'F')

  // Score
  doc.setFontSize(36)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(...scoreClr)
  doc.text(String(Math.round(result.overall_score)), M + 20, y + 27, { align: 'center' })

  doc.setFontSize(10)
  doc.text('баллов', M + 20, y + 34, { align: 'center' })

  // Condition & area
  doc.setFontSize(14)
  doc.setTextColor(...WHITE)
  doc.text(result.overall_condition || '', M + 45, y + 16)

  doc.setFontSize(11)
  doc.setTextColor(...GRAY)
  doc.text(`Площадь фасада: ${result.total_area_m2} м²`, M + 45, y + 24)
  doc.text(`Повреждено: ${result.damaged_area_m2} м² (${Math.round(result.damaged_area_m2 / result.total_area_m2 * 100)}%)`, M + 45, y + 32)

  // ── Damages section ──
  y += 50
  doc.setFillColor(...ACCENT)
  doc.rect(M, y, 4, 8, 'F')
  doc.setFontSize(16)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(...WHITE)
  doc.text('Обнаруженные дефекты', M + 8, y + 7)
  y += 14

  if (result.damages) {
    result.damages.forEach((d, i) => {
      if (y > 265) { doc.addPage(); doc.setFillColor(...DARK); doc.rect(0, 0, W, 297, 'F'); y = M }
      doc.setFillColor(20, 32, 50)
      doc.roundedRect(M, y, W - 2 * M, 18, 2, 2, 'F')

      doc.setFontSize(11)
      doc.setFont('helvetica', 'bold')
      doc.setTextColor(...WHITE)
      doc.text(`${i + 1}. ${d.type_display}`, M + 4, y + 7)

      const svClr = d.severity_display === 'Высокая' ? RED : d.severity_display === 'Средняя' ? YELLOW : GREEN
      doc.setTextColor(...svClr)
      doc.text(d.severity_display, W - M - 30, y + 7)

      doc.setTextColor(...ACCENT)
      doc.setFont('helvetica', 'bold')
      doc.text(`${d.percentage}%`, W - M - 4, y + 7, { align: 'right' })

      if (d.description) {
        doc.setFontSize(9)
        doc.setFont('helvetica', 'normal')
        doc.setTextColor(...GRAY)
        doc.text(d.description, M + 4, y + 14, { maxWidth: W - 2 * M - 8 })
      }

      y += 22
    })
  }

  // ── Materials section ──
  y += 4
  if (y > 250) { doc.addPage(); doc.setFillColor(...DARK); doc.rect(0, 0, W, 297, 'F'); y = M }
  doc.setFillColor(...ACCENT)
  doc.rect(M, y, 4, 8, 'F')
  doc.setFontSize(16)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(...WHITE)
  doc.text('Состав фасада', M + 8, y + 7)
  y += 14

  if (result.materials) {
    result.materials.forEach((m) => {
      if (y > 270) { doc.addPage(); doc.setFillColor(...DARK); doc.rect(0, 0, W, 297, 'F'); y = M }
      doc.setFontSize(11)
      doc.setFont('helvetica', 'normal')
      doc.setTextColor(...WHITE)
      doc.text(`• ${m.name_display}`, M + 4, y + 5)
      doc.setTextColor(...GRAY)
      doc.text(`${m.percentage}% — ${m.condition}`, W - M - 4, y + 5, { align: 'right' })

      // Progress bar
      doc.setFillColor(30, 48, 68)
      doc.roundedRect(M + 4, y + 8, W - 2 * M - 8, 3, 1, 1, 'F')
      const condClr = m.condition === 'Хорошее' ? GREEN : m.condition === 'Удовлетворительное' ? YELLOW : RED
      doc.setFillColor(...condClr)
      doc.roundedRect(M + 4, y + 8, (W - 2 * M - 8) * m.percentage / 100, 3, 1, 1, 'F')

      y += 16
    })
  }

  // ── Page 2+: Cost Estimate ──
  doc.addPage()
  doc.setFillColor(...DARK)
  doc.rect(0, 0, W, 297, 'F')
  doc.setFillColor(...ACCENT)
  doc.rect(0, 0, W, 6, 'F')

  y = 20
  doc.setFillColor(...ACCENT)
  doc.rect(M, y, 4, 8, 'F')
  doc.setFontSize(16)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(...WHITE)
  doc.text('Смета ремонтных работ', M + 8, y + 7)
  y += 16

  const re = result.repair_estimate
  if (re?.costs_for_flutter) {
    // Table header
    doc.setFillColor(20, 32, 50)
    doc.roundedRect(M, y, W - 2 * M, 10, 2, 2, 'F')
    doc.setFontSize(9)
    doc.setFont('helvetica', 'bold')
    doc.setTextColor(...ACCENT)
    doc.text('№', M + 4, y + 7)
    doc.text('Наименование', M + 14, y + 7)
    doc.text('Стоимость', W - M - 4, y + 7, { align: 'right' })
    y += 14

    re.costs_for_flutter.forEach((item, i) => {
      if (y > 270) { doc.addPage(); doc.setFillColor(...DARK); doc.rect(0, 0, W, 297, 'F'); y = M }
      const bg = i % 2 === 0 ? [18, 28, 42] : [22, 34, 50]
      doc.setFillColor(...bg)
      doc.rect(M, y, W - 2 * M, 14, 'F')

      doc.setFontSize(10)
      doc.setFont('helvetica', 'normal')
      doc.setTextColor(...GRAY)
      doc.text(String(i + 1), M + 4, y + 6)

      doc.setTextColor(...WHITE)
      doc.text(item.category, M + 14, y + 6, { maxWidth: 100 })
      doc.setFontSize(8)
      doc.setTextColor(...GRAY)
      doc.text(item.description, M + 14, y + 12, { maxWidth: 100 })

      doc.setFontSize(11)
      doc.setFont('helvetica', 'bold')
      doc.setTextColor(...WHITE)
      doc.text(formatCurrency(item.cost), W - M - 4, y + 8, { align: 'right' })

      y += 16
    })

    // Grand total
    y += 4
    const sum = re.summary || {}
    doc.setFillColor(...ACCENT)
    doc.roundedRect(M, y, W - 2 * M, 16, 3, 3, 'F')
    doc.setFontSize(14)
    doc.setFont('helvetica', 'bold')
    doc.setTextColor(...WHITE)
    doc.text('ИТОГО:', M + 8, y + 11)
    doc.setFontSize(16)
    doc.text(formatCurrency(sum.grand_total || 0), W - M - 8, y + 11, { align: 'right' })

    y += 24
    // Summary details
    doc.setFontSize(10)
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(...GRAY)
    doc.text(`Материалы: ${formatCurrency(sum.materials_total || 0)}`, M, y)
    doc.text(`Работы: ${formatCurrency(sum.labor_total || 0)}`, M + 70, y)
    y += 6
    doc.text(`Леса: ${formatCurrency(sum.scaffolding_total || 0)}`, M, y)
    doc.text(`НДС (20%): ${formatCurrency(sum.vat_amount || 0)}`, M + 70, y)
    y += 6
    doc.text(`Срок работ: ~${sum.estimated_days || '—'} дней (${Math.round(sum.total_work_hours || 0)} нормо-часов)`, M, y)
  }

  // Footer
  const addFooter = (pageNum) => {
    doc.setPage(pageNum)
    doc.setFontSize(8)
    doc.setTextColor(...GRAY)
    doc.text('Facade Analyzer v2.0 — автоматический отчёт', M, 290)
    doc.text(`Стр. ${pageNum}`, W - M, 290, { align: 'right' })
  }
  for (let p = 1; p <= doc.getNumberOfPages(); p++) addFooter(p)

  // Save
  const filename = `facade_report_${result.id || 'analysis'}_${new Date().toISOString().slice(0, 10)}.pdf`
  doc.save(filename)
  return filename
}
