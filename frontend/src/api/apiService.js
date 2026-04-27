/**
 * API service for Facade Analyzer backend.
 * In dev mode, requests are proxied via Vite to localhost:8000.
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
 * Format number as currency with thousands separator + ₸
 */
export function formatCurrency(amount) {
  return amount.toLocaleString('ru-RU', { maximumFractionDigits: 0 }) + ' ₸'
}

/**
 * Mock result for demo mode
 */
export function getMockResult() {
  return {
    id: 'mock-001',
    overall_score: 67.5,
    overall_condition: 'Удовлетворительное',
    total_area_m2: 450.0,
    damaged_area_m2: 146.25,
    damages: [
      { type: 'crack', type_display: 'Трещины', percentage: 35.0, severity: 'medium', severity_display: 'Средняя', description: 'Микро- и макротрещины в штукатурке, преимущественно в зоне окон', affected_layers: ['finish', 'base_plaster'], crack_depth: 'surface' },
      { type: 'peeling', type_display: 'Отслоение штукатурки', percentage: 25.0, severity: 'high', severity_display: 'Высокая', description: 'Отслоение финишного слоя на 2-4 этажах', affected_layers: ['finish', 'base_plaster'] },
      { type: 'efflorescence', type_display: 'Высолы', percentage: 20.0, severity: 'low', severity_display: 'Низкая', description: 'Белые солевые отложения в нижней части фасада', affected_layers: ['finish'] },
      { type: 'moss', type_display: 'Биопоражение', percentage: 12.0, severity: 'medium', severity_display: 'Средняя', description: 'Мох и грибок на северной стороне', affected_layers: ['finish'] },
      { type: 'spalling', type_display: 'Механические повреждения', percentage: 8.0, severity: 'low', severity_display: 'Низкая', description: 'Сколы и вмятины на цокольном уровне', affected_layers: ['finish', 'base_plaster', 'structural'] },
    ],
    materials: [
      { name: 'brick', name_display: 'Кирпич керамический', percentage: 45.0, condition: 'Хорошее' },
      { name: 'cement_plaster', name_display: 'Штукатурка цементная', percentage: 30.0, condition: 'Удовлетворительное' },
      { name: 'concrete', name_display: 'Бетон', percentage: 15.0, condition: 'Хорошее' },
      { name: 'rusty_metal', name_display: 'Металл (элементы)', percentage: 7.0, condition: 'Требует покраски' },
      { name: 'wood', name_display: 'Дерево (оконные рамы)', percentage: 3.0, condition: 'Плохое' },
    ],
    repair_estimate: {
      materials: [
        { name_display: 'Штукатурка цементная фасадная', unit: 'кг', quantity: 528.0, price_per_unit: 320, total_cost: 168960 },
        { name_display: 'Армирующая сетка фасадная', unit: 'м²', quantity: 36.3, price_per_unit: 650, total_cost: 23595 },
        { name_display: 'Грунтовка глубокого проникновения', unit: 'л', quantity: 9.9, price_per_unit: 1200, total_cost: 11880 },
        { name_display: 'Шпатлёвка фасадная', unit: 'кг', quantity: 29.7, price_per_unit: 850, total_cost: 25245 },
        { name_display: 'Краска фасадная', unit: 'л', quantity: 24.75, price_per_unit: 3500, total_cost: 86625 },
        { name_display: 'Антисоль (очиститель)', unit: 'л', quantity: 5.94, price_per_unit: 2200, total_cost: 13068 },
        { name_display: 'Гидрофобизатор фасадный', unit: 'л', quantity: 4.95, price_per_unit: 3800, total_cost: 18810 },
        { name_display: 'Антисептик фасадный', unit: 'л', quantity: 3.56, price_per_unit: 1800, total_cost: 6413 },
      ],
      labor: [
        { name_display: 'Восстановление штукатурного слоя', unit: 'м²', quantity: 30.0, price_per_unit: 4500, total_cost: 135000, norm_hours: 60.0 },
        { name_display: 'Заделка поверхностных трещин', unit: 'м.п.', quantity: 42.0, price_per_unit: 1500, total_cost: 63000, norm_hours: 21.0 },
        { name_display: 'Удаление высолов и гидрофобизация', unit: 'м²', quantity: 18.0, price_per_unit: 2000, total_cost: 36000, norm_hours: 14.4 },
        { name_display: 'Биоочистка и антисептирование', unit: 'м²', quantity: 10.8, price_per_unit: 1800, total_cost: 19440, norm_hours: 7.56 },
      ],
      summary: {
        materials_total: 354596,
        labor_total: 253440,
        scaffolding_total: 340000,
        subtotal: 948036,
        vat_amount: 113764,
        grand_total: 1061800,
        total_work_hours: 134.96,
        estimated_days: 17,
      },
      costs_for_flutter: [
        { category: 'Строительные материалы', description: '8 наименований с учётом запаса 10%', cost: 354596 },
        { category: 'Восстановление штукатурного слоя', description: '30.0 м²', cost: 135000 },
        { category: 'Заделка поверхностных трещин', description: '42.0 м.п.', cost: 63000 },
        { category: 'Удаление высолов и гидрофобизация', description: '18.0 м²', cost: 36000 },
        { category: 'Биоочистка и антисептирование', description: '10.8 м²', cost: 19440 },
        { category: 'Леса и оборудование', description: 'Монтаж/демонтаж строительных лесов', cost: 340000 },
        { category: 'НДС (12%)', description: 'Налог на добавленную стоимость', cost: 113764 },
      ],
    },
    processed_images: ['heatmap', 'defects', 'segments', 'overlay'],
  }
}
