/**
 * API service for Facade Analyzer backend.
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
  if (amount == null || isNaN(amount)) return '0 ₽'
  return Math.round(amount).toLocaleString('ru-RU') + ' ₽'
}

/**
 * Generate a clean, printable PDF report via browser print dialog.
 * Uses native browser rendering — full Cyrillic support, clean professional style.
 */
export function generatePdfReport(result) {
  const re = result.repair_estimate || {}
  const sum = re.summary || {}
  const date = new Date().toLocaleDateString('ru-RU', { day: 'numeric', month: 'long', year: 'numeric' })

  const scoreColor = result.overall_score >= 80 ? '#4CAF50' : result.overall_score >= 60 ? '#FFC107' : '#F44336'
  const sevColor = (s) => s === 'Высокая' ? '#F44336' : s === 'Средняя' ? '#FFC107' : '#4CAF50'
  const condColor = (c) => c === 'Хорошее' ? '#4CAF50' : c === 'Удовлетворительное' ? '#FFC107' : '#F44336'

  // Build damages rows
  const damagesRows = (result.damages || []).map((d, i) => `
    <tr>
      <td>${i + 1}</td>
      <td>${d.type_display}</td>
      <td style="color:${sevColor(d.severity_display)};font-weight:600">${d.severity_display}</td>
      <td style="text-align:right;font-weight:600">${d.percentage}%</td>
      <td>${d.description || ''}</td>
    </tr>
  `).join('')

  // Build materials rows
  const materialsRows = (result.materials || []).map((m, i) => `
    <tr>
      <td>${i + 1}</td>
      <td>${m.name_display}</td>
      <td style="text-align:right;font-weight:600">${m.percentage}%</td>
      <td style="color:${condColor(m.condition)}">${m.condition}</td>
    </tr>
  `).join('')

  // Build cost items rows
  const costRows = (re.costs_for_flutter || []).map((item, i) => `
    <tr>
      <td>${i + 1}</td>
      <td>${item.category}</td>
      <td style="color:#666">${item.description}</td>
      <td style="text-align:right;font-weight:600">${formatCurrency(item.cost)}</td>
    </tr>
  `).join('')

  // Build repair materials rows
  const repairMatRows = (re.materials || []).map((m, i) => `
    <tr>
      <td>${i + 1}</td>
      <td>${m.name_display}</td>
      <td style="text-align:center">${m.quantity} ${m.unit}</td>
      <td style="text-align:right">${formatCurrency(m.price_per_unit)}</td>
      <td style="text-align:right;font-weight:600">${formatCurrency(m.total_cost)}</td>
    </tr>
  `).join('')

  // Build labor rows
  const laborRows = (re.labor || []).map((l, i) => `
    <tr>
      <td>${i + 1}</td>
      <td>${l.name_display}</td>
      <td style="text-align:center">${l.quantity} ${l.unit}</td>
      <td style="text-align:right">${formatCurrency(l.price_per_unit)}</td>
      <td style="text-align:right;font-weight:600">${formatCurrency(l.total_cost)}</td>
    </tr>
  `).join('')

  const html = `<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <title>Отчёт — Анализ фасада</title>
  <style>
    @media print {
      @page { margin: 15mm; size: A4; }
      body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
      .no-break { page-break-inside: avoid; }
      .page-break { page-break-before: always; }
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
      font-size: 11pt;
      color: #1a1a1a;
      line-height: 1.5;
      background: #fff;
    }
    .header {
      border-bottom: 3px solid #1a1a1a;
      padding-bottom: 12px;
      margin-bottom: 20px;
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
    }
    .header h1 {
      font-size: 22pt;
      font-weight: 700;
      letter-spacing: -0.5px;
    }
    .header .meta {
      text-align: right;
      font-size: 9pt;
      color: #666;
    }
    h2 {
      font-size: 14pt;
      font-weight: 700;
      margin: 24px 0 10px;
      padding-bottom: 4px;
      border-bottom: 1.5px solid #ddd;
    }
    h2 .num {
      display: inline-block;
      width: 28px;
      height: 28px;
      line-height: 28px;
      text-align: center;
      background: #1a1a1a;
      color: #fff;
      border-radius: 4px;
      font-size: 12pt;
      margin-right: 8px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 8px 0 16px;
      font-size: 10pt;
    }
    th {
      background: #f5f5f5;
      font-weight: 600;
      text-align: left;
      padding: 8px 10px;
      border: 1px solid #ddd;
      font-size: 9pt;
      text-transform: uppercase;
      letter-spacing: 0.3px;
      color: #555;
    }
    td {
      padding: 7px 10px;
      border: 1px solid #eee;
      vertical-align: top;
    }
    tr:nth-child(even) { background: #fafafa; }
    .summary-box {
      background: #f8f8f8;
      border: 1.5px solid #ddd;
      border-radius: 6px;
      padding: 16px 20px;
      margin: 12px 0;
      display: flex;
      gap: 24px;
      flex-wrap: wrap;
    }
    .summary-item {
      flex: 1;
      min-width: 120px;
    }
    .summary-item .label {
      font-size: 9pt;
      color: #888;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .summary-item .value {
      font-size: 18pt;
      font-weight: 700;
      margin-top: 2px;
    }
    .total-row {
      background: #1a1a1a !important;
      color: #fff;
    }
    .total-row td {
      border-color: #333;
      font-weight: 700;
      font-size: 12pt;
      padding: 10px;
    }
    .score-badge {
      display: inline-block;
      padding: 4px 14px;
      border-radius: 4px;
      font-weight: 700;
      font-size: 12pt;
    }
    .footer {
      margin-top: 32px;
      padding-top: 12px;
      border-top: 1px solid #ddd;
      font-size: 8pt;
      color: #999;
      display: flex;
      justify-content: space-between;
    }
    .print-btn {
      position: fixed;
      top: 20px;
      right: 20px;
      padding: 12px 28px;
      background: #1a1a1a;
      color: #fff;
      border: none;
      border-radius: 6px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      z-index: 100;
    }
    .print-btn:hover { background: #333; }
    @media print { .print-btn { display: none; } }
  </style>
</head>
<body>
  <button class="print-btn" onclick="window.print()">📄 Сохранить PDF</button>

  <div class="header">
    <div>
      <h1>Отчёт по анализу фасада</h1>
    </div>
    <div class="meta">
      Дата: ${date}<br>
      ID: ${result.id || '—'}
    </div>
  </div>

  <!-- Summary -->
  <div class="summary-box no-break">
    <div class="summary-item">
      <div class="label">Оценка состояния</div>
      <div class="value">
        <span class="score-badge" style="background:${scoreColor}22;color:${scoreColor}">${Math.round(result.overall_score)} баллов</span>
      </div>
    </div>
    <div class="summary-item">
      <div class="label">Состояние</div>
      <div class="value" style="font-size:14pt">${result.overall_condition || '—'}</div>
    </div>
    <div class="summary-item">
      <div class="label">Площадь фасада</div>
      <div class="value" style="font-size:14pt">${result.total_area_m2} м²</div>
    </div>
    <div class="summary-item">
      <div class="label">Повреждено</div>
      <div class="value" style="font-size:14pt;color:${scoreColor}">${result.damaged_area_m2} м² (${Math.round(result.damaged_area_m2 / result.total_area_m2 * 100)}%)</div>
    </div>
  </div>

  <!-- 1. Damages -->
  <h2 class="no-break"><span class="num">1</span>Обнаруженные дефекты</h2>
  <table>
    <thead>
      <tr><th>№</th><th>Тип дефекта</th><th>Степень</th><th style="text-align:right">Доля</th><th>Описание</th></tr>
    </thead>
    <tbody>${damagesRows}</tbody>
  </table>

  <!-- 2. Materials -->
  <h2 class="no-break"><span class="num">2</span>Состав фасада</h2>
  <table>
    <thead>
      <tr><th>№</th><th>Материал</th><th style="text-align:right">Доля</th><th>Состояние</th></tr>
    </thead>
    <tbody>${materialsRows}</tbody>
  </table>

  <!-- 3. Cost Summary -->
  <h2 class="page-break no-break"><span class="num">3</span>Смета ремонтных работ</h2>
  <div class="summary-box no-break">
    <div class="summary-item">
      <div class="label">Срок работ</div>
      <div class="value" style="font-size:14pt">~${sum.estimated_days || '—'} дней</div>
    </div>
    <div class="summary-item">
      <div class="label">Нормо-часы</div>
      <div class="value" style="font-size:14pt">${Math.round(sum.total_work_hours || 0)} ч</div>
    </div>
    <div class="summary-item">
      <div class="label">Итоговая стоимость</div>
      <div class="value">${formatCurrency(sum.grand_total || 0)}</div>
    </div>
  </div>
  <table>
    <thead>
      <tr><th>№</th><th>Статья расходов</th><th>Описание</th><th style="text-align:right">Сумма</th></tr>
    </thead>
    <tbody>
      ${costRows}
      <tr class="total-row">
        <td colspan="3">ИТОГО</td>
        <td style="text-align:right">${formatCurrency(sum.grand_total || 0)}</td>
      </tr>
    </tbody>
  </table>

  <!-- 4. Materials Detail -->
  <h2 class="no-break"><span class="num">4</span>Ведомость материалов</h2>
  <table>
    <thead>
      <tr><th>№</th><th>Наименование</th><th style="text-align:center">Кол-во</th><th style="text-align:right">Цена за ед.</th><th style="text-align:right">Сумма</th></tr>
    </thead>
    <tbody>
      ${repairMatRows}
      <tr class="total-row">
        <td colspan="4">Итого материалы</td>
        <td style="text-align:right">${formatCurrency(sum.materials_total || 0)}</td>
      </tr>
    </tbody>
  </table>

  <!-- 5. Labor Detail -->
  <h2 class="no-break"><span class="num">5</span>Ведомость работ</h2>
  <table>
    <thead>
      <tr><th>№</th><th>Наименование</th><th style="text-align:center">Объём</th><th style="text-align:right">Цена за ед.</th><th style="text-align:right">Сумма</th></tr>
    </thead>
    <tbody>
      ${laborRows}
      <tr class="total-row">
        <td colspan="4">Итого работы</td>
        <td style="text-align:right">${formatCurrency(sum.labor_total || 0)}</td>
      </tr>
    </tbody>
  </table>

  <!-- Summary totals -->
  <div class="summary-box no-break" style="margin-top:20px">
    <div class="summary-item"><div class="label">Материалы</div><div class="value" style="font-size:12pt">${formatCurrency(sum.materials_total || 0)}</div></div>
    <div class="summary-item"><div class="label">Работы</div><div class="value" style="font-size:12pt">${formatCurrency(sum.labor_total || 0)}</div></div>
    <div class="summary-item"><div class="label">Леса</div><div class="value" style="font-size:12pt">${formatCurrency(sum.scaffolding_total || 0)}</div></div>
    <div class="summary-item"><div class="label">НДС 20%</div><div class="value" style="font-size:12pt">${formatCurrency(sum.vat_amount || 0)}</div></div>
    <div class="summary-item"><div class="label">ИТОГО</div><div class="value">${formatCurrency(sum.grand_total || 0)}</div></div>
  </div>

  <div class="footer">
    <span>Facade Analyzer v2.0 — автоматически сгенерированный отчёт</span>
    <span>${date}</span>
  </div>
</body>
</html>`

  const printWindow = window.open('', '_blank')
  printWindow.document.write(html)
  printWindow.document.close()
}
